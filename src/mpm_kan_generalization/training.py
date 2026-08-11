from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cases import CaseTable
from .config import (
    PARAMETER_NAMES,
    ParameterDomain,
    PhysicalConstants,
)
from .design import ExperimentDesign
from .model import ComsolCalibratedSoilKAN, ParametricSoilKAN
from .mpm import create_parametric_cloud
from .physics import (
    AugmentedState,
    GeneralizationLoss,
    ParametricBiotMPMProblem,
)


@dataclass(frozen=True)
class TrainingOptions:
    steps: int = 300
    float64_steps: int = 20
    learning_rate: float = 4.0e-4
    float64_learning_rate: float = 2.0e-5
    width: int = 24
    hidden_layers: int = 2
    n_basis: int = 7
    local_width: int = 16
    cases_per_step: int = 4
    background_per_case: int = 112
    annulus_per_case: int = 72
    boundary_points: int = 28
    al_update_every: int = 100
    validation_every: int = 50
    seed: int = 202416
    log_every: int = 20
    robust_case_alpha: float = 0.0
    robust_case_temperature: float = 0.75
    weight_decay: float = 1.0e-6
    minimum_lr_ratio: float = 0.08
    backbone: str = "kan"
    condition_random_field: bool = True
    particle_weight: float = 1.0
    p2g_weight: float = 0.15
    side_boundary_mode: str = "bloch"


def _combine_case_losses(
    losses: list[GeneralizationLoss],
    *,
    robust_alpha: float,
    temperature: float,
) -> GeneralizationLoss:
    """Blend the mean and a detached Group-DRO weighting.

    Standardising the sampled case totals makes the temperature portable
    across augmented-Lagrangian updates.  Detached weights prevent the
    optimiser from changing a case's weight instead of reducing its residual.
    """

    if not losses:
        raise ValueError("at least one case loss is required")
    totals = torch.stack([loss.total for loss in losses])
    count = totals.numel()
    uniform = torch.full_like(totals, 1.0 / count)
    alpha = float(min(max(robust_alpha, 0.0), 1.0))
    if alpha > 0.0 and count > 1:
        detached = totals.detach()
        standardised = (
            detached - detached.mean()
        ) / detached.std(unbiased=False).clamp_min(1.0e-6)
        robust = torch.softmax(
            standardised / max(float(temperature), 1.0e-3),
            dim=0,
        )
        weights = (1.0 - alpha) * uniform + alpha * robust
    else:
        weights = uniform
    fields: dict[str, torch.Tensor] = {}
    for name in GeneralizationLoss.__dataclass_fields__:
        values = torch.stack(
            [getattr(loss, name) for loss in losses]
        )
        fields[name] = torch.sum(weights * values)
    return GeneralizationLoss(**fields)


def _sampled_training_loss(
    problem: ParametricBiotMPMProblem,
    model: ParametricSoilKAN,
    cloud,
    indices: torch.Tensor,
    case_ids: torch.Tensor,
    augmented: AugmentedState,
    options: TrainingOptions,
) -> GeneralizationLoss:
    if options.robust_case_alpha <= 0.0:
        return problem.loss(model, indices, case_ids, augmented)
    case_losses = []
    point_cases = cloud.case_index[indices]
    for case in case_ids:
        local = indices[point_cases == case]
        case_losses.append(
            problem.loss(
                model,
                local,
                case.reshape(1),
                augmented,
            )
        )
    return _combine_case_losses(
        case_losses,
        robust_alpha=options.robust_case_alpha,
        temperature=options.robust_case_temperature,
    )


def _build_problem(
    parameters: np.ndarray,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    device: torch.device,
    dtype: torch.dtype,
    options: TrainingOptions,
    *,
    evaluation: bool,
) -> tuple[CaseTable, object, ParametricBiotMPMProblem]:
    table = CaseTable.from_numpy(
        parameters, constants, device, dtype
    )
    cloud = create_parametric_cloud(
        table,
        constants,
        base_x=12 if evaluation else 24,
        base_y=8 if evaluation else 14,
        annulus_radial=2 if evaluation else 3,
        annulus_theta=24 if evaluation else 48,
        p2g_x=24 if evaluation else 32,
        p2g_y=16 if evaluation else 20,
    )
    problem = ParametricBiotMPMProblem(
        table,
        cloud,
        domain,
        constants,
        particle_weight=options.particle_weight,
        p2g_weight=options.p2g_weight,
        boundary_points=options.boundary_points,
        side_boundary_mode=options.side_boundary_mode,
    )
    return table, cloud, problem


def _evaluate_physics(
    model: ParametricSoilKAN,
    parameters: np.ndarray,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: TrainingOptions,
    augmented: AugmentedState,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    table, cloud, problem = _build_problem(
        parameters,
        domain,
        constants,
        device,
        dtype,
        options,
        evaluation=True,
    )
    indices = torch.arange(
        cloud.n_particles, device=device
    )
    case_ids = torch.arange(table.n_cases, device=device)
    with torch.enable_grad():
        loss = problem.loss(
            model, indices, case_ids, augmented
        )
    result = loss.detached()
    result["training_objective_score_unaugmented"] = (
        problem.particle_weight
        * (
            problem.momentum_weight
            * result["particle_momentum"]
            + problem.fluid_weight * result["particle_fluid"]
        )
        + problem.p2g_weight
        * (
            problem.momentum_weight
            * result["p2g_momentum"]
            + problem.fluid_weight * result["p2g_fluid"]
        )
        + 10.0 * result["surface_traction"]
        + 15.0 * result["outer_boundary"]
        + 25.0 * result["pipe_flux"]
    )
    # Keep one invariant evaluation metric across every ablation. An
    # ablated term is still measured here so a variant cannot improve its
    # score merely by removing that term from its training objective.
    result["selection_score_unaugmented"] = (
        problem.momentum_weight
        * result["particle_momentum"]
        + problem.fluid_weight * result["particle_fluid"]
        + 0.15
        * (
            problem.momentum_weight
            * result["p2g_momentum"]
            + problem.fluid_weight * result["p2g_fluid"]
        )
        + 10.0 * result["surface_traction"]
        + 15.0 * result["outer_boundary"]
        + 25.0 * result["pipe_flux"]
    )
    result["cases"] = float(table.n_cases)
    result["particles"] = float(cloud.n_particles)
    return result


def _row(
    stage: str,
    step: int,
    loss,
    elapsed: float,
    augmented: AugmentedState,
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "step": float(step),
        "elapsed_seconds": elapsed,
        **loss.detached(),
        **augmented.as_dict(),
    }


def train(
    design: ExperimentDesign,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: TrainingOptions,
    device: torch.device,
    output_dir: Path,
    initial_checkpoint: Path | None = None,
) -> tuple[
    ParametricSoilKAN,
    AugmentedState,
    list[dict[str, object]],
]:
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    model = ParametricSoilKAN(
        domain,
        constants,
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.n_basis,
        local_width=options.local_width,
        backbone=options.backbone,
        condition_random_field=options.condition_random_field,
        dtype=torch.float32,
    ).to(device)
    if initial_checkpoint is not None:
        initial_state = torch.load(
            initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(initial_state["model"])
    table, cloud, problem = _build_problem(
        design.train,
        domain,
        constants,
        device,
        torch.float32,
        options,
        evaluation=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(options.steps, 1),
        eta_min=(
            options.learning_rate * options.minimum_lr_ratio
        ),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed)
    augmented = AugmentedState()
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()

    for step in range(1, options.steps + 1):
        indices, case_ids = cloud.sample(
            options.cases_per_step,
            options.background_per_case,
            options.annulus_per_case,
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _sampled_training_loss(
            problem,
            model,
            cloud,
            indices,
            case_ids,
            augmented,
            options,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 8.0
        )
        optimizer.step()
        scheduler.step()
        if (
            options.al_update_every > 0
            and step % options.al_update_every == 0
        ):
            augmented.update(
                {
                    "surface": float(
                        loss.surface_constraint.detach().cpu()
                    ),
                    "outer": float(
                        loss.outer_constraint.detach().cpu()
                    ),
                    "pipe": float(
                        loss.pipe_constraint.detach().cpu()
                    ),
                }
            )
        if (
            step == 1
            or step % options.log_every == 0
            or step == options.steps
        ):
            record = _row(
                "float32_adam",
                step,
                loss,
                time.perf_counter() - started,
                augmented,
            )
            history.append(record)
            print(json.dumps(record))
        if (
            options.validation_every > 0
            and (
                step % options.validation_every == 0
                or step == options.steps
            )
        ):
            validation = _evaluate_physics(
                model,
                design.validation,
                domain,
                constants,
                options,
                augmented,
                device,
                torch.float32,
            )
            record = {
                "stage": "validation_physics",
                "step": float(step),
                "elapsed_seconds": (
                    time.perf_counter() - started
                ),
                **validation,
            }
            history.append(record)
            print(json.dumps(record))
            selection_score = validation[
                "training_objective_score_unaugmented"
            ]
            if selection_score < best_validation:
                best_validation = selection_score
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model = model.to(dtype=torch.float64)
    table64, cloud64, problem64 = _build_problem(
        design.train,
        domain,
        constants,
        device,
        torch.float64,
        options,
        evaluation=False,
    )
    generator64 = torch.Generator(device=device)
    generator64.manual_seed(options.seed + 99)
    tail_indices, tail_cases = cloud64.sample(
        table64.n_cases,
        min(64, options.background_per_case),
        min(40, options.annulus_per_case),
        generator64,
    )
    optimizer64 = torch.optim.AdamW(
        model.parameters(),
        lr=options.float64_learning_rate,
        weight_decay=options.weight_decay,
    )
    scheduler64 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer64,
        T_max=max(options.float64_steps, 1),
        eta_min=(
            options.float64_learning_rate
            * options.minimum_lr_ratio
        ),
    )
    best_state64 = copy.deepcopy(model.state_dict())
    best_validation64 = _evaluate_physics(
        model,
        design.validation,
        domain,
        constants,
        options,
        augmented,
        device,
        torch.float64,
    )["training_objective_score_unaugmented"]
    for step in range(1, options.float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        # The deterministic tail already contains every training case.
        # A uniform full-batch objective is both stable and substantially
        # cheaper than rebuilding one higher-order graph per case.
        loss = problem64.loss(
            model, tail_indices, tail_cases, augmented
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 6.0
        )
        optimizer64.step()
        scheduler64.step()
        if (
            step == 1
            or step % 5 == 0
            or step == options.float64_steps
        ):
            record = _row(
                "float64_all_case_adam",
                step,
                loss,
                time.perf_counter() - started,
                augmented,
            )
            history.append(record)
            print(json.dumps(record))
        if step % 5 == 0 or step == options.float64_steps:
            validation = _evaluate_physics(
                model,
                design.validation,
                domain,
                constants,
                options,
                augmented,
                device,
                torch.float64,
            )
            record = {
                "stage": "float64_validation_physics",
                "step": float(step),
                "elapsed_seconds": (
                    time.perf_counter() - started
                ),
                **validation,
            }
            history.append(record)
            print(json.dumps(record))
            score = validation[
                "training_objective_score_unaugmented"
            ]
            if score < best_validation64:
                best_validation64 = score
                best_state64 = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state64)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoints" / "generalization.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "domain": domain.as_dict(),
            "constants": constants.as_dict(),
            "options": asdict(options),
            "design": {
                "train": design.train,
                "validation": design.validation,
                "blind": design.blind,
            },
            "augmented": augmented.as_dict(),
            "history": history,
            "training_labels": "none",
            "reference_data_used_in_training": False,
            "initial_checkpoint": (
                None
                if initial_checkpoint is None
                else str(initial_checkpoint.resolve())
            ),
            "side_boundary_mode": options.side_boundary_mode,
            "selection_metric": (
                "held-out variant-training-objective physics "
                "residual only"
            ),
        },
        checkpoint,
    )
    return model, augmented, history


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[
    ParametricSoilKAN,
    ExperimentDesign,
    ParameterDomain,
    PhysicalConstants,
    TrainingOptions,
    AugmentedState,
    list[dict[str, object]],
]:
    state = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    domain = ParameterDomain(**state["domain"])
    constants = PhysicalConstants(**state["constants"])
    options = TrainingOptions(**state["options"])
    design = ExperimentDesign(**state["design"])
    base_model = ParametricSoilKAN(
        domain,
        constants,
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.n_basis,
        local_width=options.local_width,
        backbone=options.backbone,
        condition_random_field=options.condition_random_field,
        dtype=torch.float64,
    ).to(device)
    if state.get("checkpoint_type") == "comsol_calibrated":
        architecture = state["calibration_architecture"]
        model = ComsolCalibratedSoilKAN(
            base_model,
            width=int(architecture["width"]),
            hidden_layers=int(architecture["hidden_layers"]),
            n_basis=int(architecture["n_basis"]),
            condition_wave_height=bool(
                architecture.get("condition_wave_height", True)
            ),
            condition_random_field=bool(
                architecture.get("condition_random_field", True)
            ),
            dtype=torch.float64,
        ).to(device)
    else:
        model = base_model
    model.load_state_dict(state["model"])
    augmented = AugmentedState(**state["augmented"])
    return (
        model,
        design,
        domain,
        constants,
        options,
        augmented,
        state.get("history", []),
    )


def _pipe_profiles(
    model: ParametricSoilKAN,
    parameters: np.ndarray,
    split: str,
    constants: PhysicalConstants,
    device: torch.device,
    *,
    points: int = 181,
) -> list[dict[str, float | str]]:
    table = CaseTable.from_numpy(
        parameters,
        constants,
        device,
        next(model.parameters()).dtype,
    )
    theta = torch.linspace(
        0.0,
        2.0 * torch.pi,
        points,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for case in range(table.n_cases):
            x = constants.radius * torch.cos(theta)
            y = (
                table.pipe_center_y[case]
                + constants.radius * torch.sin(theta)
            )
            xy = torch.stack((x, y), dim=1)
            parameter = table.parameters[case].expand(
                points, -1
            )
            pressure = model(
                xy,
                parameter,
                table.kappa[case].expand(points),
                table.pipe_center_y[case].expand(points),
            ).p.abs()
            absolute = (
                pressure * table.pressure_scale_pa[case]
            )
            for index in range(points):
                rows.append(
                    {
                        "split": split,
                        "case_index": float(case),
                        "theta_rad": float(theta[index].cpu()),
                        "normalized_pressure": float(
                            pressure[index].cpu()
                        ),
                        "absolute_pressure_pa": float(
                            absolute[index].cpu()
                        ),
                    }
                )
    return rows


def evaluate_and_save(
    model: ParametricSoilKAN,
    design: ExperimentDesign,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: TrainingOptions,
    augmented: AugmentedState,
    history: list[dict[str, object]],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    dtype = next(model.parameters()).dtype
    split_metrics = {
        name: _evaluate_physics(
            model,
            design.split(name),
            domain,
            constants,
            options,
            augmented,
            device,
            dtype,
        )
        for name in ("train", "validation", "blind")
    }
    blind_case_metrics: list[dict[str, object]] = []
    for index, parameters in enumerate(design.blind):
        physics = _evaluate_physics(
            model,
            parameters[None, :],
            domain,
            constants,
            options,
            augmented,
            device,
            dtype,
        )
        blind_case_metrics.append(
            {
                "case_index": index,
                "parameters": {
                    name: float(value)
                    for name, value in zip(
                        PARAMETER_NAMES, parameters
                    )
                },
                "physics": physics,
            }
        )
    worst_blind = max(
        blind_case_metrics,
        key=lambda row: row["physics"][
            "selection_score_unaugmented"
        ],
    )
    train_score = split_metrics["train"][
        "selection_score_unaugmented"
    ]
    metrics: dict[str, object] = {
        "experiment": (
            "single conditional MPM-KAN; no per-case retraining"
        ),
        "training_labels": "none",
        "known_reference_data_required_for_training": False,
        "reference_data_used_in_training": False,
        "validation_selection": (
            "held-out PDE and boundary residuals only"
        ),
        "external_accuracy_reference": (
            "not supplied; add FEM/COMSOL/experiment only "
            "for final blind accuracy evaluation"
        ),
        "splits": split_metrics,
        "blind_cases": blind_case_metrics,
        "worst_blind_case": {
            "case_index": worst_blind["case_index"],
            "physics_score": worst_blind["physics"][
                "selection_score_unaugmented"
            ],
        },
        "generalization_gap": {
            "validation_physics_score_over_train": (
                split_metrics["validation"][
                    "selection_score_unaugmented"
                ]
                / train_score
            ),
            "blind_physics_score_over_train": (
                split_metrics["blind"][
                    "selection_score_unaugmented"
                ]
                / train_score
            ),
        },
        "parameter_domain": domain.as_dict(),
        "random_field": {
            "kind": "four-mode differentiable lognormal field",
            "coefficient_columns": [
                f"xi_{index + 1}"
                for index in range(domain.random_modes)
            ],
            "coefficient_of_variation": domain.permeability_cov,
        },
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dtype": str(dtype),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    profiles: list[dict[str, float | str]] = []
    for name in ("train", "validation", "blind"):
        profiles.extend(
            _pipe_profiles(
                model,
                design.split(name),
                name,
                constants,
                device,
            )
        )
    profile_path = output_dir / "blind_pipe_profiles.csv"
    with profile_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(profiles[0])
        )
        writer.writeheader()
        writer.writerows(profiles)
    blind_profiles = [
        row for row in profiles if row["split"] == "blind"
    ]
    response_summary: list[dict[str, float]] = []
    for case in range(design.blind.shape[0]):
        selected = [
            row
            for row in blind_profiles
            if int(row["case_index"]) == case
        ]
        normalized = np.asarray(
            [row["normalized_pressure"] for row in selected],
            dtype=np.float64,
        )
        absolute = np.asarray(
            [row["absolute_pressure_pa"] for row in selected],
            dtype=np.float64,
        )
        response_summary.append(
            {
                "case_index": float(case),
                "normalized_pressure_min": float(
                    normalized.min()
                ),
                "normalized_pressure_max": float(
                    normalized.max()
                ),
                "absolute_pressure_pa_mean": float(
                    absolute.mean()
                ),
                "absolute_pressure_pa_max": float(
                    absolute.max()
                ),
            }
        )
    metrics["blind_response_summary"] = response_summary
    (output_dir / "generalization_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _plot_summary(metrics, profiles, output_dir)
    return metrics


def evaluate_selection_and_save(
    model: ParametricSoilKAN,
    design: ExperimentDesign,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: TrainingOptions,
    augmented: AugmentedState,
    history: list[dict[str, object]],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate only train/validation while blind cases remain sealed."""

    dtype = next(model.parameters()).dtype
    split_metrics = {
        name: _evaluate_physics(
            model,
            design.split(name),
            domain,
            constants,
            options,
            augmented,
            device,
            dtype,
        )
        for name in ("train", "validation")
    }
    validation_cases = []
    for index, parameters in enumerate(design.validation):
        physics = _evaluate_physics(
            model,
            parameters[None, :],
            domain,
            constants,
            options,
            augmented,
            device,
            dtype,
        )
        validation_cases.append(
            {
                "case_index": index,
                "physics_score": physics[
                    "selection_score_unaugmented"
                ],
            }
        )
    train_score = split_metrics["train"][
        "selection_score_unaugmented"
    ]
    scores = np.asarray(
        [row["physics_score"] for row in validation_cases],
        dtype=np.float64,
    )
    metrics: dict[str, object] = {
        "experiment": "hyperparameter selection; blind cases sealed",
        "training_labels": "none",
        "reference_data_used_in_training": False,
        "splits": split_metrics,
        "validation_cases": validation_cases,
        "selection": {
            "validation_over_train": (
                split_metrics["validation"][
                    "selection_score_unaugmented"
                ]
                / train_score
            ),
            "validation_case_mean": float(scores.mean()),
            "validation_case_p90": float(
                np.quantile(scores, 0.9)
            ),
            "validation_case_max": float(scores.max()),
        },
        "blind_evaluated": False,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dtype": str(dtype),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "selection_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def _plot_summary(
    metrics: dict[str, object],
    profiles: list[dict[str, float | str]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8)
    )
    names = ["train", "validation", "blind"]
    totals = [
        metrics["splits"][name][
            "selection_score_unaugmented"
        ]
        for name in names
    ]
    axes[0].bar(
        names, totals, color=("#2667ff", "#43aa8b", "#f8961e")
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Weighted physics score")
    axes[0].set_title("Held-out physics generalization")
    blind = [
        row for row in profiles if row["split"] == "blind"
    ]
    for case in range(
        min(4, 1 + int(max(
            row["case_index"] for row in blind
        )))
    ):
        selected = [
            row
            for row in blind
            if int(row["case_index"]) == case
        ]
        axes[1].plot(
            [row["theta_rad"] for row in selected],
            [row["normalized_pressure"] for row in selected],
            label=f"blind {case}",
        )
    axes[1].set_xlabel(r"$\theta$ (rad)")
    axes[1].set_ylabel(r"$|p|/P_0$")
    axes[1].set_title("One checkpoint, unseen cases")
    axes[1].legend()
    figure.suptitle(
        "Physics-only parametric MPM-KAN generalization"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "generalization_summary.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
