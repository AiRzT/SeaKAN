from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .adaptive_mpm import AdaptiveMaterialPointCloud
from .adaptive_training import (
    AdaptiveTrainingOptions,
    load_adaptive_checkpoint,
)
from .config import Geometry, SoilParameters, WaveParameters
from .model import PipeHarmonicKAN, TopGapLocalizedKAN
from .strong_mpm import (
    AdaptiveStrongMPMProblem,
    AugmentedBoundaryState,
    StrongLossBreakdown,
)
from .training import TrainingOptions


@dataclass
class TopGapTrainingOptions:
    """Fourth-level refinement, augmented boundaries, and float64 finish."""

    float32_steps: int = 60
    float32_learning_rate: float = 2.5e-5
    float64_adam_steps: int = 10
    float64_learning_rate: float = 3.0e-6
    float64_lbfgs_steps: int = 4
    batch_background: int = 320
    batch_middle: int = 480
    batch_fine: int = 704
    batch_gap: int = 1024
    level_weight_background: float = 0.12
    level_weight_middle: float = 0.18
    level_weight_fine: float = 0.25
    level_weight_gap: float = 0.45
    momentum_weight: float = 10.0
    fluid_weight: float = 1000.0
    p2g_weight: float = 0.25
    roughness_weight: float = 1.0e-3
    boundary_points: int = 192
    al_update_every: int = 40
    al_progress_ratio: float = 0.90
    al_penalty_growth: float = 1.5
    al_maximum_penalty: float = 5000.0
    rho_top: float = 160.0
    rho_outer: float = 160.0
    rho_pipe: float = 320.0
    adapter_width: int = 10
    adapter_basis: int = 9
    freeze_base: bool = True
    x_support_radii: float = 0.60
    y_support_center_gap: float = 1.40
    distillation_steps: int = 800
    distillation_learning_rate: float = 1.0e-3
    seed: int = 202409
    log_every: int = 20

    @property
    def batch_per_level(self) -> tuple[int, int, int, int]:
        return (
            self.batch_background,
            self.batch_middle,
            self.batch_fine,
            self.batch_gap,
        )

    @property
    def level_weights(self) -> tuple[float, float, float, float]:
        return (
            self.level_weight_background,
            self.level_weight_middle,
            self.level_weight_fine,
            self.level_weight_gap,
        )


def make_top_gap_problem(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    cloud: AdaptiveMaterialPointCloud,
    options: TopGapTrainingOptions,
) -> AdaptiveStrongMPMProblem:
    return AdaptiveStrongMPMProblem(
        wave,
        soil,
        geometry,
        cloud,
        level_weights=options.level_weights,
        momentum_weight=options.momentum_weight,
        fluid_weight=options.fluid_weight,
        p2g_weight=options.p2g_weight,
        boundary_weight=0.0,
        pipe_flux_weight=0.0,
        roughness_weight=options.roughness_weight,
        boundary_points=options.boundary_points,
    )


def _constraints(loss: StrongLossBreakdown) -> dict[str, float]:
    return {
        "top": float(loss.top_constraint.detach().cpu()),
        "outer": float(loss.outer_constraint.detach().cpu()),
        "pipe": float(loss.pipe_constraint.detach().cpu()),
    }


def _history_row(
    stage: str,
    step: int,
    loss: StrongLossBreakdown,
    augmented: AugmentedBoundaryState,
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "step": float(step),
        **loss.detached(),
        **augmented.as_dict(),
    }


def _update_augmented(
    augmented: AugmentedBoundaryState,
    loss: StrongLossBreakdown,
    previous: dict[str, float] | None,
    options: TopGapTrainingOptions,
) -> dict[str, float]:
    current = _constraints(loss)
    augmented.update(
        current,
        previous,
        progress_ratio=options.al_progress_ratio,
        penalty_growth=options.al_penalty_growth,
        maximum_penalty=options.al_maximum_penalty,
    )
    return current


def _distill_physics_teacher(
    model: TopGapLocalizedKAN,
    teacher: PipeHarmonicKAN,
    cloud: AdaptiveMaterialPointCloud,
    options: TopGapTrainingOptions,
    history: list[dict[str, float | str]],
) -> None:
    if options.distillation_steps <= 0:
        return
    gap_xy = cloud.xy[cloud.level == len(cloud.counts) - 1]
    radius = model.geometry.radius_over_height
    theta = torch.linspace(
        0.25 * torch.pi,
        0.75 * torch.pi,
        257,
        dtype=gap_xy.dtype,
        device=gap_xy.device,
    )[:-1]
    pipe_xy = torch.stack(
        (
            radius * torch.cos(theta),
            model.geometry.center_y + radius * torch.sin(theta),
        ),
        dim=1,
    )
    xy = torch.cat((gap_xy, pipe_xy), dim=0)
    with torch.no_grad():
        base_fields = model.base(xy)
        teacher_fields = teacher(xy)
        teacher_envelope = torch.exp(
            -(
                xy[:, 0]
                / (
                    options.x_support_radii
                    * model.geometry.radius_over_height
                )
            ).pow(4)
            - (
                (1.0 - xy[:, 1])
                / (
                    options.y_support_center_gap
                    * (1.0 - model.geometry.center_y)
                )
            ).pow(4)
        )
        target_u = base_fields.u + teacher_envelope * (
            teacher_fields.u - base_fields.u
        )
        target_w = base_fields.w + teacher_envelope * (
            teacher_fields.w - base_fields.w
        )
        target_p = base_fields.p + teacher_envelope * (
            teacher_fields.p - base_fields.p
        )
    optimizer = torch.optim.Adam(
        model.local_kan.parameters(),
        lr=options.distillation_learning_rate,
    )
    for step in range(1, options.distillation_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xy)
        loss = (
            (prediction.p - target_p).abs().square().mean()
            + 0.10 * (prediction.u - target_u).abs().square().mean()
            + 0.10 * (prediction.w - target_w).abs().square().mean()
        )
        loss.backward()
        optimizer.step()
        if (
            step == 1
            or step % 100 == 0
            or step == options.distillation_steps
        ):
            value = float(loss.detach().cpu())
            history.append(
                {
                    "stage": "physics_teacher_distillation",
                    "step": float(step),
                    "field_mse": value,
                }
            )
            print(
                f"Physics-teacher distillation "
                f"{step:4d}/{options.distillation_steps}: "
                f"field MSE={value:.4e}"
            )


def train_top_gap_augmented(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    cloud_float32: AdaptiveMaterialPointCloud,
    cloud_float64: AdaptiveMaterialPointCloud,
    baseline_checkpoint: Path,
    adaptive_checkpoint: Path,
    teacher_checkpoint: Path | None,
    output_dir: Path,
    options: TopGapTrainingOptions,
    device: torch.device,
) -> tuple[
    TopGapLocalizedKAN,
    TrainingOptions,
    AdaptiveTrainingOptions,
    torch.dtype,
    list[dict[str, float | str]],
    AugmentedBoundaryState,
]:
    base_model, baseline_options, adaptive_options, _, _ = load_adaptive_checkpoint(
        wave,
        soil,
        geometry,
        baseline_checkpoint,
        adaptive_checkpoint,
        device,
    )
    base_model = base_model.to(dtype=torch.float32)
    model = TopGapLocalizedKAN(
        base_model,
        geometry,
        width=options.adapter_width,
        n_basis=options.adapter_basis,
        dtype=torch.float32,
        freeze_base=options.freeze_base,
        x_support_radii=options.x_support_radii,
        y_support_center_gap=options.y_support_center_gap,
    ).to(device)
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed)
    augmented = AugmentedBoundaryState(
        rho_top=options.rho_top,
        rho_outer=options.rho_outer,
        rho_pipe=options.rho_pipe,
    )
    history: list[dict[str, float | str]] = []
    started = time.perf_counter()
    if teacher_checkpoint is not None:
        teacher, _, _, _, _ = load_adaptive_checkpoint(
            wave,
            soil,
            geometry,
            baseline_checkpoint,
            adaptive_checkpoint,
            device,
        )
        teacher_state = torch.load(
            teacher_checkpoint, map_location=device, weights_only=False
        )
        teacher.load_state_dict(teacher_state["model"])
        teacher = teacher.to(dtype=torch.float32)
        teacher.eval()
        _distill_physics_teacher(
            model, teacher, cloud_float32, options, history
        )
        del teacher

    problem_float32 = make_top_gap_problem(
        wave, soil, geometry, cloud_float32, options
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=options.float32_learning_rate
    )
    previous_constraints: dict[str, float] | None = None
    for step in range(1, options.float32_steps + 1):
        indices = cloud_float32.sample_indices(
            options.batch_per_level, generator
        )
        optimizer.zero_grad(set_to_none=True)
        loss = problem_float32.loss(model, indices, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=12.0)
        optimizer.step()
        if step % options.al_update_every == 0:
            previous_constraints = _update_augmented(
                augmented, loss, previous_constraints, options
            )
        if (
            step == 1
            or step % options.log_every == 0
            or step == options.float32_steps
        ):
            row = _history_row(
                "top_gap_augmented_float32", step, loss, augmented
            )
            history.append(row)
            print(
                f"Top-gap float32 {step:4d}/{options.float32_steps}: "
                f"total={row['total']:.4e}, gap-weight=0.45, "
                f"c_top={row['top_constraint']:.3e}, "
                f"c_pipe={row['pipe_constraint']:.3e}"
            )

    del optimizer
    model = model.to(dtype=torch.float64)
    problem_float64 = make_top_gap_problem(
        wave, soil, geometry, cloud_float64, options
    )
    full_indices = torch.arange(
        cloud_float64.n_particles, device=device, dtype=torch.long
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer64 = torch.optim.Adam(
        trainable, lr=options.float64_learning_rate
    )
    for step in range(1, options.float64_adam_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        loss = problem_float64.loss(model, full_indices, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        optimizer64.step()
        if step % max(5, options.al_update_every // 4) == 0:
            previous_constraints = _update_augmented(
                augmented, loss, previous_constraints, options
            )
        if (
            step == 1
            or step % max(5, options.log_every // 2) == 0
            or step == options.float64_adam_steps
        ):
            row = _history_row(
                "full_batch_float64_adam", step, loss, augmented
            )
            history.append(row)
            print(
                f"Full-batch float64 Adam "
                f"{step:3d}/{options.float64_adam_steps}: "
                f"total={row['total']:.4e}, "
                f"c_top={row['top_constraint']:.3e}, "
                f"c_pipe={row['pipe_constraint']:.3e}"
            )

    if options.float64_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            trainable,
            lr=0.35,
            max_iter=options.float64_lbfgs_steps,
            max_eval=options.float64_lbfgs_steps * 2,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=30,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = problem_float64.loss(model, full_indices, augmented)
            value.total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % 5 == 0:
                row = _history_row(
                    "full_batch_float64_lbfgs",
                    evaluations,
                    value,
                    augmented,
                )
                history.append(row)
                print(
                    f"Full-batch float64 LBFGS eval {evaluations:3d}: "
                    f"total={row['total']:.4e}, "
                    f"c_top={row['top_constraint']:.3e}, "
                    f"c_pipe={row['pipe_constraint']:.3e}"
                )
            return value.total

        lbfgs.step(closure)

    final = problem_float64.loss(model, full_indices, augmented)
    previous_constraints = _update_augmented(
        augmented, final, previous_constraints, options
    )
    history.append(
        {
            **_history_row(
                "top_gap_augmented_float64_final",
                options.float64_adam_steps,
                final,
                augmented,
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    checkpoint = output_dir / "checkpoints" / "figure9_top_gap_augmented.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "baseline_options": asdict(baseline_options),
            "adaptive_options": asdict(adaptive_options),
            "top_gap_options": asdict(options),
            "history": history,
            "dtype": str(torch.float64),
            "cloud": cloud_float64.summary(),
            "augmented_boundary": augmented.as_dict(),
            "final_constraints": previous_constraints,
            "source_adaptive_checkpoint": str(adaptive_checkpoint),
            "source_teacher_checkpoint": (
                str(teacher_checkpoint)
                if teacher_checkpoint is not None
                else None
            ),
            "localized_adapter": True,
        },
        checkpoint,
    )
    return (
        model,
        baseline_options,
        adaptive_options,
        torch.float64,
        history,
        augmented,
    )


def load_top_gap_checkpoint(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    baseline_checkpoint: Path,
    adaptive_checkpoint: Path,
    top_gap_checkpoint: Path,
    device: torch.device,
) -> tuple[
    TopGapLocalizedKAN,
    TrainingOptions,
    AdaptiveTrainingOptions,
    TopGapTrainingOptions,
    torch.dtype,
    list[dict[str, float | str]],
    AugmentedBoundaryState,
]:
    state = torch.load(
        top_gap_checkpoint, map_location=device, weights_only=False
    )
    base_model, baseline_options, adaptive_options, _, _ = load_adaptive_checkpoint(
        wave,
        soil,
        geometry,
        baseline_checkpoint,
        adaptive_checkpoint,
        device,
    )
    options = TopGapTrainingOptions(**state["top_gap_options"])
    base_model = base_model.to(dtype=torch.float64)
    model = TopGapLocalizedKAN(
        base_model,
        geometry,
        width=options.adapter_width,
        n_basis=options.adapter_basis,
        dtype=torch.float64,
        freeze_base=options.freeze_base,
        x_support_radii=options.x_support_radii,
        y_support_center_gap=options.y_support_center_gap,
    ).to(device)
    model.load_state_dict(state["model"])
    augmented = AugmentedBoundaryState(**state["augmented_boundary"])
    return (
        model,
        baseline_options,
        adaptive_options,
        options,
        torch.float64,
        list(state.get("history", [])),
        augmented,
    )
