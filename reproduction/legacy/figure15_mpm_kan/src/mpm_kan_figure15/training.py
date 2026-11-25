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
    figure15_setups,
)
from .model import (
    ContactEnrichedWaterKAN,
    SoilPipeKAN,
    WaterPressureKAN,
)
from .mpm import (
    MaterialPointCloud,
    create_rans_water_cloud,
    create_soil_cloud,
    create_water_cloud,
)
from .physics import (
    AugmentedBoundaryState,
    SoilLoss,
    SoilMPMProblem,
    WaterLoss,
    WaterMPMProblem,
)
from .rans import (
    HarmonicRANSKAN,
    HarmonicRANSMPMProblem,
)
from .teacher import (
    field_teacher_targets,
    solve_biot_bvp,
)


@dataclass
class TrainingOptions:
    water_model_kind: str = "potential"
    water_coupling_strategy: str = "single_field"
    water_steps: int = 600
    water_learning_rate: float = 5.0e-4
    water_float64_steps: int = 60
    water_lbfgs_steps: int = 30
    soil_steps: int = 1100
    soil_learning_rate: float = 5.0e-4
    soil_float64_steps: int = 120
    soil_float64_learning_rate: float = 1.5e-5
    soil_lbfgs_steps: int = 50
    teacher_steps: int = 600
    teacher_learning_rate: float = 1.0e-3
    teacher_anchor_steps: int = 200
    teacher_anchor_weight: float = 1.0e8
    water_width: int = 16
    soil_width: int = 18
    soil_local_width: int = 12
    soil_local_basis: int = 9
    soil_local_support: float = 0.30
    soil_local_steps: int = 0
    soil_local_learning_rate: float = 2.0e-4
    hidden_layers: int = 2
    kan_basis: int = 9
    soil_base_x: int = 40
    soil_base_y: int = 24
    soil_middle_radial: int = 6
    soil_middle_theta: int = 96
    soil_fine_radial: int = 6
    soil_fine_theta: int = 160
    water_base_x: int = 40
    water_base_y: int = 24
    water_annulus_radial: int = 6
    water_annulus_theta: int = 128
    p2g_x: int = 80
    p2g_y: int = 48
    batch_background: int = 640
    batch_middle: int = 480
    batch_fine: int = 640
    water_batch_background: int = 640
    water_batch_annulus: int = 480
    water_far_field_weight: float = 2500.0
    water_pipe_support: float = 0.16
    water_annulus_residual_weight: float = 0.38
    rans_water_width: int = 16
    rans_water_hidden_layers: int = 2
    rans_effective_kinematic_viscosity: float = 1.5e-4
    rans_continuity_weight: float = 8.0
    rans_free_surface_weight: float = 80.0
    rans_wall_momentum_weight: float = 0.5
    rans_far_field_weight: float = 1200.0
    momentum_weight: float = 50.0
    fluid_weight: float = 3000.0
    p2g_weight: float = 0.25
    roughness_weight: float = 1.0e-4
    boundary_points: int = 160
    al_update_every: int = 150
    seed: int = 202415
    log_every: int = 100


def _soil_cloud(
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> MaterialPointCloud:
    return create_soil_cloud(
        geometry,
        device,
        dtype,
        base_x=options.soil_base_x,
        base_y=options.soil_base_y,
        middle_radial=options.soil_middle_radial,
        middle_theta=options.soil_middle_theta,
        fine_radial=options.soil_fine_radial,
        fine_theta=options.soil_fine_theta,
        p2g_x=options.p2g_x,
        p2g_y=options.p2g_y,
    )


def _water_cloud(
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> MaterialPointCloud:
    return create_water_cloud(
        geometry,
        device,
        dtype,
        base_x=options.water_base_x,
        base_y=options.water_base_y,
        annulus_radial=options.water_annulus_radial,
        annulus_theta=options.water_annulus_theta,
        p2g_x=options.p2g_x,
        p2g_y=options.p2g_y,
    )


def make_soil_problem(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaterialPointCloud, SoilMPMProblem]:
    cloud = _soil_cloud(
        geometry, options, device, dtype
    )
    return cloud, SoilMPMProblem(
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


def make_water_problem(
    wave: WaveParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaterialPointCloud, WaterMPMProblem]:
    cloud = _water_cloud(
        geometry, options, device, dtype
    )
    return cloud, WaterMPMProblem(
        wave,
        geometry,
        cloud,
        p2g_weight=options.p2g_weight,
        far_field_weight=options.water_far_field_weight,
        annulus_residual_weight=(
            options.water_annulus_residual_weight
        ),
        roughness_weight=options.roughness_weight,
        boundary_points=options.boundary_points,
    )


def make_rans_water_problem(
    wave: WaveParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaterialPointCloud, HarmonicRANSMPMProblem]:
    cloud = create_rans_water_cloud(
        geometry,
        device,
        dtype,
        p2g_x=max(96, options.p2g_x),
        p2g_y=max(60, options.p2g_y),
    )
    return cloud, HarmonicRANSMPMProblem(
        wave,
        geometry,
        cloud,
        effective_kinematic_viscosity=(
            options.rans_effective_kinematic_viscosity
        ),
        continuity_weight=options.rans_continuity_weight,
        free_surface_weight=options.rans_free_surface_weight,
        wall_momentum_weight=options.rans_wall_momentum_weight,
        far_field_weight=options.rans_far_field_weight,
        p2g_weight=options.p2g_weight,
        roughness_weight=options.roughness_weight,
        boundary_points=options.boundary_points,
    )


def _water_row(
    stage: str, step: int, loss: WaterLoss
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "step": float(step),
        **loss.detached(),
    }


def _soil_row(
    case_name: str,
    stage: str,
    step: int,
    loss: SoilLoss,
    augmented: AugmentedBoundaryState,
) -> dict[str, float | str]:
    return {
        "case": case_name,
        "stage": stage,
        "step": float(step),
        **loss.detached(),
        **augmented.as_dict(),
    }


def _load_soil_state_preserving_water(
    model: SoilPipeKAN,
    state: dict[str, torch.Tensor],
) -> None:
    """Restore soil weights without replacing a newly trained water model."""

    soil_only = {
        name: value
        for name, value in state.items()
        if not name.startswith("water_model.")
    }
    missing, unexpected = model.load_state_dict(
        soil_only,
        strict=False,
    )
    expected_missing: set[str] = set()
    if model.water_model is not None:
        expected_missing.update(
            f"water_model.{name}"
            for name in model.water_model.state_dict()
        )
    expected_missing.update(
        name
        for name in model.state_dict()
        if name.startswith("local_kan.") and name not in state
    )
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            "Unexpected soil checkpoint keys while preserving "
            f"the water model: missing={missing}, "
            f"unexpected={unexpected}"
        )


def _train_water(
    wave: WaveParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    output_dir: Path,
    initial_checkpoint: Path | None = None,
) -> tuple[
    WaterPressureKAN,
    list[dict[str, float | str]],
    dict[str, object],
]:
    if initial_checkpoint is not None:
        payload = torch.load(
            initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model = WaterPressureKAN(
            wave,
            geometry,
            width=options.water_width,
            hidden_layers=options.hidden_layers,
            n_basis=options.kan_basis,
            pipe_support=options.water_pipe_support,
            dtype=torch.float64,
        ).to(device)
        model.load_state_dict(payload["model"])
        cloud64, problem64 = make_water_problem(
            wave,
            geometry,
            options,
            device,
            torch.float64,
        )
        indices64 = torch.arange(
            cloud64.n_particles, device=device
        )
        history = list(payload.get("history", []))
        optimizer64 = torch.optim.Adam(
            model.parameters(),
            lr=0.25 * options.water_learning_rate,
        )
        for step in range(
            1, options.water_float64_steps + 1
        ):
            optimizer64.zero_grad(set_to_none=True)
            loss = problem64.loss(model, indices64)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            )
            optimizer64.step()
            if (
                step == 1
                or step % 20 == 0
                or step == options.water_float64_steps
            ):
                row = _water_row(
                    "water_float64_resume_adam",
                    step,
                    loss,
                )
                history.append(row)
                print(
                    f"Water resume float64 {step:3d}/"
                    f"{options.water_float64_steps}: "
                    f"total={row['total']:.3e}"
                )
        if options.water_lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.35,
                max_iter=options.water_lbfgs_steps,
                max_eval=2 * options.water_lbfgs_steps,
                tolerance_grad=1.0e-10,
                tolerance_change=1.0e-12,
                history_size=40,
                line_search_fn="strong_wolfe",
            )
            evaluations = 0

            def resume_closure() -> torch.Tensor:
                nonlocal evaluations
                lbfgs.zero_grad(set_to_none=True)
                value = problem64.loss(model, indices64)
                value.total.backward()
                evaluations += 1
                if evaluations == 1 or evaluations % 10 == 0:
                    row = _water_row(
                        "water_float64_resume_lbfgs",
                        evaluations,
                        value,
                    )
                    history.append(row)
                    print(
                        "Water resume LBFGS eval "
                        f"{evaluations:3d}: "
                        f"total={row['total']:.3e}"
                    )
                return value.total

            lbfgs.step(resume_closure)
        final = problem64.loss(model, indices64)
        history.append(
            _water_row("water_float64_reused_final", 0, final)
        )
        checkpoint = (
            output_dir / "checkpoints" / "water_test49.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "options": asdict(options),
                "history": history,
                "cloud": cloud64.summary(),
                "training_data": "physics_only_no_figure15_csv",
                "initialized_from": str(
                    initial_checkpoint.resolve()
                ),
            },
            checkpoint,
        )
        action = (
            "continued"
            if (
                options.water_float64_steps > 0
                or options.water_lbfgs_steps > 0
            )
            else "reused"
        )
        print(
            f"Water Test49: {action} no-Figure-15-label "
            f"checkpoint {initial_checkpoint}"
        )
        return model, history, cloud64.summary()

    model = WaterPressureKAN(
        wave,
        geometry,
        width=options.water_width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        pipe_support=options.water_pipe_support,
        dtype=torch.float32,
    ).to(device)
    cloud32, problem32 = make_water_problem(
        wave,
        geometry,
        options,
        device,
        torch.float32,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed + 49)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=options.water_learning_rate
    )
    history: list[dict[str, float | str]] = []
    for step in range(1, options.water_steps + 1):
        indices = cloud32.sample_indices(
            (
                options.water_batch_background,
                options.water_batch_annulus,
            ),
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = problem32.loss(model, indices)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=12.0
        )
        optimizer.step()
        if (
            step == 1
            or step % options.log_every == 0
            or step == options.water_steps
        ):
            row = _water_row(
                "water_float32_adam", step, loss
            )
            history.append(row)
            print(
                f"Water Test49 {step:4d}/{options.water_steps}: "
                f"total={row['total']:.3e}, "
                f"laplace={row['particle_laplace']:.3e}, "
                f"bed={row['seabed_flux']:.3e}, "
                f"pipe={row['pipe_flux']:.3e}, "
                f"far={row['far_field']:.3e}"
            )

    model = model.to(dtype=torch.float64)
    cloud64, problem64 = make_water_problem(
        wave,
        geometry,
        options,
        device,
        torch.float64,
    )
    indices64 = torch.arange(
        cloud64.n_particles, device=device
    )
    optimizer64 = torch.optim.Adam(
        model.parameters(),
        lr=0.25 * options.water_learning_rate,
    )
    for step in range(1, options.water_float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        loss = problem64.loss(model, indices64)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )
        optimizer64.step()
        if (
            step == 1
            or step % 20 == 0
            or step == options.water_float64_steps
        ):
            row = _water_row(
                "water_float64_full_adam", step, loss
            )
            history.append(row)
            print(
                f"Water float64 {step:3d}/"
                f"{options.water_float64_steps}: "
                f"total={row['total']:.3e}"
            )
    if options.water_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.35,
            max_iter=options.water_lbfgs_steps,
            max_eval=2 * options.water_lbfgs_steps,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=40,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = problem64.loss(model, indices64)
            value.total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % 10 == 0:
                row = _water_row(
                    "water_float64_full_lbfgs",
                    evaluations,
                    value,
                )
                history.append(row)
                print(
                    f"Water LBFGS eval {evaluations:3d}: "
                    f"total={row['total']:.3e}"
                )
            return value.total

        lbfgs.step(closure)
    final = problem64.loss(model, indices64)
    history.append(
        _water_row("water_float64_final", 0, final)
    )
    checkpoint = output_dir / "checkpoints" / "water_test49.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "options": asdict(options),
            "history": history,
            "cloud": cloud64.summary(),
            "training_data": "physics_only_no_figure15_csv",
        },
        checkpoint,
    )
    return model, history, cloud64.summary()


def _new_water_from_state(
    wave: WaveParameters,
    geometry: Geometry,
    options: TrainingOptions,
    state: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> WaterPressureKAN | HarmonicRANSKAN:
    if options.water_model_kind == "harmonic_rans":
        model = HarmonicRANSKAN(
            wave,
            geometry,
            effective_kinematic_viscosity=(
                options.rans_effective_kinematic_viscosity
            ),
            width=options.rans_water_width,
            hidden_layers=options.rans_water_hidden_layers,
            n_basis=options.kan_basis,
            dtype=dtype,
        ).to(device)
    elif options.water_model_kind == "potential":
        model = WaterPressureKAN(
            wave,
            geometry,
            width=options.water_width,
            hidden_layers=options.hidden_layers,
            n_basis=options.kan_basis,
            pipe_support=options.water_pipe_support,
            dtype=dtype,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported water_model_kind: {options.water_model_kind}"
        )
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _train_soil_case(
    case_name: str,
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    output_dir: Path,
    water_state: dict[str, torch.Tensor] | None,
    initial_state: dict[str, torch.Tensor] | None = None,
    initial_augmented: dict[str, float] | None = None,
) -> tuple[
    SoilPipeKAN,
    WaterPressureKAN | HarmonicRANSKAN | None,
    list[dict[str, float | str]],
    AugmentedBoundaryState,
    dict[str, object],
]:
    water32 = (
        _new_water_from_state(
            wave,
            geometry,
            options,
            water_state,
            device,
            torch.float32,
        )
        if water_state is not None
        else None
    )
    model = SoilPipeKAN(
        wave,
        geometry,
        water32,
        width=options.soil_width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        local_width=options.soil_local_width,
        local_basis=options.soil_local_basis,
        local_support=options.soil_local_support,
        dtype=torch.float32,
    ).to(device)
    if initial_state is not None:
        _load_soil_state_preserving_water(
            model,
            initial_state,
        )
        print(
            f"Soil {case_name}: continuing from a "
            "no-Figure-15-label checkpoint"
        )
    cloud32, problem32 = make_soil_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float32,
    )
    teacher_history: list[dict[str, float | str]] = []
    teacher_targets: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None
    if (
        options.teacher_steps > 0
        or options.teacher_anchor_steps > 0
    ):
        profile = solve_biot_bvp(wave, soil, geometry)
        teacher_xy = cloud32.xy
        teacher_targets = field_teacher_targets(
            model, teacher_xy, profile
        )
        teacher_u, teacher_w, teacher_p = teacher_targets
    if options.teacher_steps > 0:
        teacher_trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        teacher_optimizer = torch.optim.Adam(
            teacher_trainable,
            lr=options.teacher_learning_rate,
        )
        for step in range(1, options.teacher_steps + 1):
            teacher_optimizer.zero_grad(set_to_none=True)
            prediction = model(teacher_xy)
            pressure_loss = (
                (prediction.p.real - teacher_p.real).square()
                + (
                    prediction.p.imag
                    - teacher_p.imag
                ).square()
            ).mean()
            displacement_loss = (
                (
                    prediction.u.real - teacher_u.real
                ).square()
                + (
                    prediction.u.imag - teacher_u.imag
                ).square()
                + (
                    prediction.w.real - teacher_w.real
                ).square()
                + (
                    prediction.w.imag - teacher_w.imag
                ).square()
            ).mean()
            teacher_loss = pressure_loss + displacement_loss
            teacher_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                teacher_trainable, max_norm=10.0
            )
            teacher_optimizer.step()
            if (
                step == 1
                or step % 100 == 0
                or step == options.teacher_steps
            ):
                value = float(teacher_loss.detach().cpu())
                teacher_history.append(
                    {
                        "case": case_name,
                        "stage": "bvp_teacher_initialization",
                        "step": float(step),
                        "total": value,
                    }
                )
                print(
                    f"Soil {case_name} BVP teacher "
                    f"{step:4d}/{options.teacher_steps}: "
                    f"loss={value:.3e}"
                )
    augmented = (
        AugmentedBoundaryState(**initial_augmented)
        if initial_augmented is not None
        else AugmentedBoundaryState()
    )
    history: list[dict[str, float | str]] = teacher_history
    if options.soil_local_steps > 0:
        for parameter in model.kan.parameters():
            parameter.requires_grad_(False)
        local_trainable = list(model.local_kan.parameters())
        local_optimizer = torch.optim.Adam(
            local_trainable,
            lr=options.soil_local_learning_rate,
        )
        local_generator = torch.Generator(device=device)
        local_generator.manual_seed(
            options.seed
            + (1010 if case_name == "test10" else 1049)
        )
        for step in range(1, options.soil_local_steps + 1):
            indices = cloud32.sample_indices(
                (
                    options.batch_background,
                    options.batch_middle,
                    options.batch_fine,
                ),
                local_generator,
            )
            local_optimizer.zero_grad(set_to_none=True)
            loss = problem32.loss(model, indices, augmented)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                local_trainable, max_norm=12.0
            )
            local_optimizer.step()
            if (
                step == 1
                or step % 25 == 0
                or step == options.soil_local_steps
            ):
                row = _soil_row(
                    case_name,
                    "soil_local_kan_physics",
                    step,
                    loss,
                    augmented,
                )
                history.append(row)
                print(
                    f"Soil {case_name} local KAN "
                    f"{step:4d}/{options.soil_local_steps}: "
                    f"total={row['total']:.3e}, "
                    f"momentum={row['particle_momentum']:.3e}"
                )
        for parameter in model.kan.parameters():
            parameter.requires_grad_(True)
    generator = torch.Generator(device=device)
    generator.manual_seed(
        options.seed + (10 if case_name == "test10" else 49)
    )
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable, lr=options.soil_learning_rate
    )
    previous: dict[str, float] | None = None
    for step in range(1, options.soil_steps + 1):
        indices = cloud32.sample_indices(
            (
                options.batch_background,
                options.batch_middle,
                options.batch_fine,
            ),
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = problem32.loss(model, indices, augmented)
        optimized_total = loss.total
        anchor_value = None
        if (
            teacher_targets is not None
            and step <= options.teacher_anchor_steps
        ):
            teacher_u, teacher_w, teacher_p = teacher_targets
            anchor_prediction = model(cloud32.xy[indices])
            anchor_value = (
                (
                    anchor_prediction.p.real
                    - teacher_p[indices].real
                ).square()
                + (
                    anchor_prediction.p.imag
                    - teacher_p[indices].imag
                ).square()
                + (
                    anchor_prediction.u.real
                    - teacher_u[indices].real
                ).square()
                + (
                    anchor_prediction.u.imag
                    - teacher_u[indices].imag
                ).square()
                + (
                    anchor_prediction.w.real
                    - teacher_w[indices].real
                ).square()
                + (
                    anchor_prediction.w.imag
                    - teacher_w[indices].imag
                ).square()
            ).mean()
            optimized_total = (
                optimized_total
                + options.teacher_anchor_weight * anchor_value
            )
        optimized_total.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=15.0
        )
        optimizer.step()
        if step % options.al_update_every == 0:
            previous = augmented.update(
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
                },
                previous,
            )
        if (
            step == 1
            or step % options.log_every == 0
            or step == options.soil_steps
        ):
            row = _soil_row(
                case_name,
                "soil_float32_adam",
                step,
                loss,
                augmented,
            )
            history.append(row)
            if anchor_value is not None:
                row["teacher_anchor"] = float(
                    anchor_value.detach().cpu()
                )
            print(
                f"Soil {case_name} {step:4d}/{options.soil_steps}: "
                f"total={row['total']:.3e}, "
                f"momentum={row['particle_momentum']:.3e}, "
                f"fluid={row['particle_fluid']:.3e}, "
                f"pipe={row['pipe_constraint']:.3e}"
                + (
                    f", anchor={row['teacher_anchor']:.3e}"
                    if "teacher_anchor" in row
                    else ""
                )
            )

    model = model.to(dtype=torch.float64)
    water64 = model.water_model
    cloud64, problem64 = make_soil_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float64,
    )
    indices64 = torch.arange(
        cloud64.n_particles, device=device
    )
    trainable64 = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer64 = torch.optim.Adam(
        trainable64, lr=options.soil_float64_learning_rate
    )
    for step in range(1, options.soil_float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        loss = problem64.loss(model, indices64, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable64, max_norm=10.0
        )
        optimizer64.step()
        if (
            step == 1
            or step % 20 == 0
            or step == options.soil_float64_steps
        ):
            row = _soil_row(
                case_name,
                "soil_float64_full_adam",
                step,
                loss,
                augmented,
            )
            history.append(row)
            print(
                f"Soil {case_name} float64 {step:3d}/"
                f"{options.soil_float64_steps}: "
                f"total={row['total']:.3e}"
            )
    if options.soil_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            trainable64,
            lr=0.35,
            max_iter=options.soil_lbfgs_steps,
            max_eval=2 * options.soil_lbfgs_steps,
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
                model, indices64, augmented
            )
            value.total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % 10 == 0:
                row = _soil_row(
                    case_name,
                    "soil_float64_full_lbfgs",
                    evaluations,
                    value,
                    augmented,
                )
                history.append(row)
                print(
                    f"Soil {case_name} LBFGS eval "
                    f"{evaluations:3d}: total={row['total']:.3e}"
                )
            return value.total

        lbfgs.step(closure)
    final = problem64.loss(model, indices64, augmented)
    history.append(
        _soil_row(
            case_name,
            "soil_float64_final",
            0,
            final,
            augmented,
        )
    )
    soil_checkpoint = (
        output_dir / "checkpoints" / f"soil_{case_name}.pt"
    )
    soil_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "options": asdict(options),
            "history": history,
            "augmented": augmented.as_dict(),
            "cloud": cloud64.summary(),
            "training_data": (
                "no_figure15_csv; numerical_biot_bvp_"
                "initialization_and_physics_losses_only"
            ),
            "initialization": {
                "method": "numerical_flat_bed_biot_bvp_teacher",
                "contains_pipeline_or_trench": False,
                "contains_figure15_data": False,
                "teacher_removed_before_final_optimization": (
                    options.soil_steps
                    > options.teacher_anchor_steps
                    or options.soil_float64_steps > 0
                    or options.soil_lbfgs_steps > 0
                ),
            },
        },
        soil_checkpoint,
    )
    return (
        model,
        water64,
        history,
        augmented,
        cloud64.summary(),
    )


def train(
    options: TrainingOptions,
    device: torch.device,
    output_dir: Path,
    water_initial_checkpoint: Path | None = None,
    soil_initial_checkpoint: Path | None = None,
) -> tuple[
    dict[str, SoilPipeKAN],
    WaterPressureKAN | HarmonicRANSKAN,
    dict[str, list[dict[str, float | str]]],
    dict[str, AugmentedBoundaryState],
]:
    if options.water_model_kind != "potential":
        raise ValueError(
            "Full harmonic-RANS training uses "
            "scripts/train_rans_water.py followed by "
            "scripts/assemble_partitioned_checkpoint.py."
        )
    started = time.perf_counter()
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    setups = figure15_setups()
    wave49, _, geometry49 = setups["test49"]
    water, water_history, water_cloud = _train_water(
        wave49,
        geometry49,
        options,
        device,
        output_dir,
        water_initial_checkpoint,
    )
    water_state = {
        name: tensor.detach().clone()
        for name, tensor in water.state_dict().items()
    }
    soil_models: dict[str, SoilPipeKAN] = {}
    histories = {"water_test49": water_history}
    augmented_states: dict[str, AugmentedBoundaryState] = {}
    cloud_summaries: dict[str, object] = {
        "water_test49": water_cloud
    }
    soil_initial_states: dict[
        str, dict[str, torch.Tensor]
    ] = {}
    if soil_initial_checkpoint is not None:
        soil_payload = torch.load(
            soil_initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        soil_initial_states = {
            "test10": soil_payload["soil_test10"],
            "test49": soil_payload["soil_test49"],
        }
    released_water: WaterPressureKAN | None = None
    for case_name in ("test10", "test49"):
        wave, soil, geometry = setups[case_name]
        (
            model,
            case_water,
            history,
            augmented,
            cloud_summary,
        ) = _train_soil_case(
            case_name,
            wave,
            soil,
            geometry,
            options,
            device,
            output_dir,
            water_state if case_name == "test49" else None,
            soil_initial_states.get(case_name),
        )
        soil_models[case_name] = model
        histories[f"soil_{case_name}"] = history
        augmented_states[case_name] = augmented
        cloud_summaries[f"soil_{case_name}"] = cloud_summary
        if case_name == "test49":
            released_water = case_water
    if released_water is None:
        raise RuntimeError("Test 49 water model was not attached")
    checkpoint = output_dir / "checkpoints" / "figure15.pt"
    torch.save(
        {
            "water_test49": released_water.state_dict(),
            "soil_test10": soil_models["test10"].state_dict(),
            "soil_test49": soil_models["test49"].state_dict(),
            "options": asdict(options),
            "histories": histories,
            "augmented": {
                key: value.as_dict()
                for key, value in augmented_states.items()
            },
            "clouds": cloud_summaries,
            "coefficients": {
                case_name: dimensionless_coefficients(
                    *setups[case_name]
                )
                for case_name in setups
            },
            "training_data": (
                "no_figure15_csv; numerical_biot_bvp_"
                "initialization_and_physics_losses_only"
            ),
            "initialization": {
                "method": "numerical_flat_bed_biot_bvp_teacher",
                "contains_pipeline_or_trench": False,
                "contains_figure15_data": False,
                "teacher_removed_before_final_optimization": (
                    options.soil_steps
                    > options.teacher_anchor_steps
                    or options.soil_float64_steps > 0
                    or options.soil_lbfgs_steps > 0
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        },
        checkpoint,
    )
    return (
        soil_models,
        released_water,
        histories,
        augmented_states,
    )


def load_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[
    dict[str, SoilPipeKAN],
    WaterPressureKAN | HarmonicRANSKAN,
    TrainingOptions,
    dict[str, list[dict[str, float | str]]],
    dict[str, AugmentedBoundaryState],
]:
    state = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    options = TrainingOptions(**state["options"])
    setups = figure15_setups()
    wave49, _, geometry49 = setups["test49"]
    water_kind = state.get(
        "water_model_kind", options.water_model_kind
    )
    if water_kind == "harmonic_rans":
        water = HarmonicRANSKAN(
            wave49,
            geometry49,
            effective_kinematic_viscosity=(
                options.rans_effective_kinematic_viscosity
            ),
            width=options.rans_water_width,
            hidden_layers=options.rans_water_hidden_layers,
            n_basis=options.kan_basis,
            dtype=torch.float64,
        ).to(device)
    else:
        water = WaterPressureKAN(
            wave49,
            geometry49,
            width=options.water_width,
            hidden_layers=options.hidden_layers,
            n_basis=options.kan_basis,
            pipe_support=options.water_pipe_support,
            dtype=torch.float64,
        ).to(device)
    water.load_state_dict(state["water_test49"])
    for parameter in water.parameters():
        parameter.requires_grad_(False)
    interface_water = water
    if "soil_interface_water_test49" in state:
        base_interface = WaterPressureKAN(
            wave49,
            geometry49,
            width=options.water_width,
            hidden_layers=options.hidden_layers,
            n_basis=options.kan_basis,
            pipe_support=options.water_pipe_support,
            dtype=torch.float64,
        ).to(device)
        interface_kind = state.get(
            "soil_interface_model_kind", "potential"
        )
        if interface_kind == "contact_enriched_potential":
            architecture = state["contact_interface_architecture"]
            interface_water = ContactEnrichedWaterKAN(
                base_interface,
                contact_angle=float(
                    architecture["contact_angle_rad"]
                ),
                width=int(architecture["width"]),
                n_basis=int(architecture["n_basis"]),
                dtype=torch.float64,
            ).to(device)
        elif interface_kind == "potential":
            interface_water = base_interface
        else:
            raise ValueError(
                "Unsupported soil interface model kind: "
                f"{interface_kind}"
            )
        interface_water.load_state_dict(
            state["soil_interface_water_test49"]
        )
        for parameter in interface_water.parameters():
            parameter.requires_grad_(False)
    models: dict[str, SoilPipeKAN] = {}
    for case_name in ("test10", "test49"):
        wave, _, geometry = setups[case_name]
        attached_water = (
            interface_water if case_name == "test49" else None
        )
        model = SoilPipeKAN(
            wave,
            geometry,
            attached_water,
            width=options.soil_width,
            hidden_layers=options.hidden_layers,
            n_basis=options.kan_basis,
            local_width=options.soil_local_width,
            local_basis=options.soil_local_basis,
            local_support=options.soil_local_support,
            dtype=torch.float64,
        ).to(device)
        _load_soil_state_preserving_water(
            model,
            state[f"soil_{case_name}"],
        )
        models[case_name] = model
    augmented = {
        name: AugmentedBoundaryState(**values)
        for name, values in state["augmented"].items()
    }
    return (
        models,
        water,
        options,
        state.get("histories", {}),
        augmented,
    )
