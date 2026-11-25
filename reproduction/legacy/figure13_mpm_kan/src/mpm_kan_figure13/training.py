from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
)
from .model import DepthHarmonicKAN
from .mpm import MaterialPointCloud2D
from .physics import (
    AugmentedBoundaryState,
    Figure13MPMKANProblem,
    LossBreakdown,
)


@dataclass
class TrainingOptions:
    float32_steps: int = 1200
    float32_learning_rate: float = 5.0e-4
    batch_particles: int = 1536
    float64_adam_steps: int = 80
    float64_learning_rate: float = 2.0e-5
    float64_lbfgs_steps: int = 40
    cells_x: int = 32
    cells_y: int = 64
    width: int = 20
    hidden_layers: int = 2
    kan_basis: int = 11
    momentum_weight: float = 10.0
    fluid_weight: float = 1000.0
    p2g_weight: float = 0.25
    roughness_weight: float = 1.0e-4
    boundary_points: int = 128
    al_update_every: int = 100
    rho_top: float = 160.0
    rho_bottom: float = 320.0
    seed: int = 202413
    log_every: int = 100


def make_cloud_and_problem(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaterialPointCloud2D, Figure13MPMKANProblem]:
    cloud = MaterialPointCloud2D.create(
        wave,
        geometry,
        device,
        dtype,
        cells_x=options.cells_x,
        cells_y=options.cells_y,
    )
    problem = Figure13MPMKANProblem(
        wave,
        soil,
        geometry,
        cloud,
        momentum_weight=options.momentum_weight,
        fluid_weight=options.fluid_weight,
        p2g_weight=options.p2g_weight,
        roughness_weight=options.roughness_weight,
        boundary_points=options.boundary_points,
    )
    return cloud, problem


def _history_row(
    stage: str,
    step: int,
    loss: LossBreakdown,
    augmented: AugmentedBoundaryState,
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "step": float(step),
        **loss.detached(),
        **augmented.as_dict(),
    }


def train(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    output_dir: Path,
) -> tuple[
    DepthHarmonicKAN,
    torch.dtype,
    list[dict[str, float | str]],
    AugmentedBoundaryState,
]:
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    coefficients = dimensionless_coefficients(
        wave, soil, geometry
    )
    model = DepthHarmonicKAN(
        coefficients["kappa"],
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        dtype=torch.float32,
    ).to(device)
    cloud32, problem32 = make_cloud_and_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float32,
    )
    augmented = AugmentedBoundaryState(
        rho_top=options.rho_top,
        rho_bottom=options.rho_bottom,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=options.float32_learning_rate,
    )
    history: list[dict[str, float | str]] = []
    previous_constraints: tuple[float, float] | None = None
    started = time.perf_counter()

    for step in range(1, options.float32_steps + 1):
        indices = cloud32.sample_indices(
            options.batch_particles, generator
        )
        optimizer.zero_grad(set_to_none=True)
        loss = problem32.loss(model, indices, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=15.0
        )
        optimizer.step()
        if step % options.al_update_every == 0:
            previous_constraints = augmented.update(
                float(loss.top_constraint.detach().cpu()),
                float(loss.bottom_constraint.detach().cpu()),
                previous_constraints,
            )
        if (
            step == 1
            or step % options.log_every == 0
            or step == options.float32_steps
        ):
            row = _history_row(
                "float32_adam", step, loss, augmented
            )
            history.append(row)
            print(
                f"Float32 Adam {step:5d}/{options.float32_steps}: "
                f"total={row['total']:.4e}, "
                f"momentum={row['particle_momentum']:.3e}, "
                f"fluid={row['particle_fluid']:.3e}, "
                f"c_top={row['top_constraint']:.3e}, "
                f"c_bottom={row['bottom_constraint']:.3e}"
            )

    del optimizer
    model = model.to(dtype=torch.float64)
    cloud64, problem64 = make_cloud_and_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float64,
    )
    full_indices = torch.arange(
        cloud64.n_particles,
        device=device,
        dtype=torch.long,
    )
    optimizer64 = torch.optim.Adam(
        model.parameters(),
        lr=options.float64_learning_rate,
    )
    for step in range(1, options.float64_adam_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        loss = problem64.loss(model, full_indices, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )
        optimizer64.step()
        if step % 20 == 0:
            previous_constraints = augmented.update(
                float(loss.top_constraint.detach().cpu()),
                float(loss.bottom_constraint.detach().cpu()),
                previous_constraints,
            )
        if (
            step == 1
            or step % 20 == 0
            or step == options.float64_adam_steps
        ):
            row = _history_row(
                "float64_full_adam", step, loss, augmented
            )
            history.append(row)
            print(
                f"Float64 full Adam "
                f"{step:4d}/{options.float64_adam_steps}: "
                f"total={row['total']:.4e}, "
                f"momentum={row['particle_momentum']:.3e}, "
                f"fluid={row['particle_fluid']:.3e}"
            )

    if options.float64_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.4,
            max_iter=options.float64_lbfgs_steps,
            max_eval=options.float64_lbfgs_steps * 2,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=40,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = problem64.loss(
                model, full_indices, augmented
            )
            value.total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % 10 == 0:
                row = _history_row(
                    "float64_full_lbfgs",
                    evaluations,
                    value,
                    augmented,
                )
                history.append(row)
                print(
                    f"Float64 LBFGS eval {evaluations:4d}: "
                    f"total={row['total']:.4e}, "
                    f"momentum={row['particle_momentum']:.3e}, "
                    f"fluid={row['particle_fluid']:.3e}"
                )
            return value.total

        lbfgs.step(closure)

    final = problem64.loss(
        model, full_indices, augmented
    )
    history.append(
        {
            **_history_row(
                "float64_full_final",
                options.float64_adam_steps,
                final,
                augmented,
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    checkpoint = output_dir / "checkpoints" / "figure13.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "options": asdict(options),
            "history": history,
            "dtype": str(torch.float64),
            "augmented_boundary": augmented.as_dict(),
            "cloud": cloud64.summary(),
            "coefficients": coefficients,
            "training_data": "physics_only_no_figure13_csv",
        },
        checkpoint,
    )
    return (
        model,
        torch.float64,
        history,
        augmented,
    )


def load_checkpoint(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    checkpoint: Path,
    device: torch.device,
) -> tuple[
    DepthHarmonicKAN,
    TrainingOptions,
    torch.dtype,
    list[dict[str, float | str]],
    AugmentedBoundaryState,
]:
    state = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    options = TrainingOptions(**state["options"])
    dtype = (
        torch.float64
        if "float64" in state.get("dtype", "")
        else torch.float32
    )
    coefficients = dimensionless_coefficients(
        wave, soil, geometry
    )
    model = DepthHarmonicKAN(
        coefficients["kappa"],
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(state["model"])
    augmented = AugmentedBoundaryState(
        **state["augmented_boundary"]
    )
    return (
        model,
        options,
        dtype,
        list(state.get("history", [])),
        augmented,
    )


def refine_full_batch(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    model: DepthHarmonicKAN,
    options: TrainingOptions,
    history: list[dict[str, float | str]],
    augmented: AugmentedBoundaryState,
    device: torch.device,
    output_dir: Path,
    *,
    adam_steps: int = 240,
    learning_rate: float = 5.0e-5,
    lbfgs_steps: int = 80,
) -> tuple[
    DepthHarmonicKAN,
    torch.dtype,
    list[dict[str, float | str]],
    AugmentedBoundaryState,
]:
    """Continue a checkpoint using only full-batch float64 physics losses."""

    started = time.perf_counter()
    model = model.to(device=device, dtype=torch.float64)
    cloud, problem = make_cloud_and_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float64,
    )
    indices = torch.arange(
        cloud.n_particles, device=device, dtype=torch.long
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    previous_constraints: tuple[float, float] | None = None
    for step in range(1, adam_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss(model, indices, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )
        optimizer.step()
        if step % 60 == 0:
            previous_constraints = augmented.update(
                float(loss.top_constraint.detach().cpu()),
                float(loss.bottom_constraint.detach().cpu()),
                previous_constraints,
            )
        if step == 1 or step % 40 == 0 or step == adam_steps:
            row = _history_row(
                "float64_resume_adam", step, loss, augmented
            )
            history.append(row)
            print(
                f"Float64 resume Adam {step:4d}/{adam_steps}: "
                f"total={row['total']:.4e}, "
                f"momentum={row['particle_momentum']:.3e}, "
                f"fluid={row['particle_fluid']:.3e}"
            )

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.35,
            max_iter=lbfgs_steps,
            max_eval=lbfgs_steps * 2,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            history_size=60,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = problem.loss(model, indices, augmented)
            value.total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % 10 == 0:
                row = _history_row(
                    "float64_resume_lbfgs",
                    evaluations,
                    value,
                    augmented,
                )
                history.append(row)
                print(
                    f"Float64 resume LBFGS eval {evaluations:4d}: "
                    f"total={row['total']:.4e}, "
                    f"momentum={row['particle_momentum']:.3e}, "
                    f"fluid={row['particle_fluid']:.3e}"
                )
            return value.total

        lbfgs.step(closure)

    final = problem.loss(model, indices, augmented)
    history.append(
        {
            **_history_row(
                "float64_resume_final",
                adam_steps,
                final,
                augmented,
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    checkpoint = output_dir / "checkpoints" / "figure13.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "options": asdict(options),
            "history": history,
            "dtype": str(torch.float64),
            "augmented_boundary": augmented.as_dict(),
            "cloud": cloud.summary(),
            "coefficients": dimensionless_coefficients(
                wave, soil, geometry
            ),
            "training_data": "physics_only_no_figure13_csv",
            "refinement": {
                "adam_steps": adam_steps,
                "learning_rate": learning_rate,
                "lbfgs_steps": lbfgs_steps,
            },
        },
        checkpoint,
    )
    return model, torch.float64, history, augmented
