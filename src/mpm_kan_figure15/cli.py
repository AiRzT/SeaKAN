from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from .config import (
    dimensionless_coefficients,
    figure15_setups,
)
from .coupling import contact_diagnostics
from .data import (
    calculate_metrics,
    load_figure15_data,
    plot_figure15,
    save_json,
    save_profiles,
)
from .model import ContactEnrichedWaterKAN
from .physics import WaterMPMProblem
from .rans import HarmonicRANSKAN
from .training import (
    TrainingOptions,
    load_checkpoint,
    make_rans_water_problem,
    make_soil_problem,
    make_water_problem,
    train,
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def project_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    return (
        candidate
        if (candidate / "pyproject.toml").is_file()
        else Path.cwd()
    )


def packaged_resource(relative_path: str) -> Path:
    return Path(
        str(
            files("mpm_kan_figure15").joinpath(
                "resources", relative_path
            )
        )
    )


def _data_path(
    explicit: str | None, root: Path, filename: str
) -> Path:
    if explicit:
        return Path(explicit).resolve()
    local = root / "data" / filename
    return local if local.is_file() else packaged_resource(filename)


def _checkpoint_path(
    explicit: str | None, output_dir: Path, root: Path
) -> Path:
    if explicit:
        return Path(explicit).resolve()
    candidates = (
        output_dir / "checkpoints" / "figure15.pt",
        root / "artifacts" / "checkpoints" / "figure15.pt",
        packaged_resource("artifacts/checkpoints/figure15.pt"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def main() -> None:
    args = build_parser().parse_args()
    root = project_root()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "outputs" / "figure15"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    test10_data = _data_path(
        args.test10_data, root, "figure15a.csv"
    )
    test49_data = _data_path(
        args.test49_data, root, "figure15b.csv"
    )
    device = select_device(args.device)
    options = TrainingOptions(
        water_model_kind=args.water_model_kind,
        water_coupling_strategy=args.water_coupling_strategy,
        water_steps=args.water_steps,
        water_learning_rate=args.water_learning_rate,
        water_float64_steps=args.water_float64_steps,
        water_lbfgs_steps=args.water_lbfgs_steps,
        soil_steps=args.soil_steps,
        soil_learning_rate=args.soil_learning_rate,
        soil_float64_steps=args.soil_float64_steps,
        soil_float64_learning_rate=args.soil_float64_learning_rate,
        soil_lbfgs_steps=args.soil_lbfgs_steps,
        teacher_steps=args.teacher_steps,
        teacher_learning_rate=args.teacher_learning_rate,
        teacher_anchor_steps=args.teacher_anchor_steps,
        teacher_anchor_weight=args.teacher_anchor_weight,
        water_width=args.water_width,
        soil_width=args.soil_width,
        soil_local_width=args.soil_local_width,
        soil_local_basis=args.soil_local_basis,
        soil_local_support=args.soil_local_support,
        soil_local_steps=args.soil_local_steps,
        soil_local_learning_rate=args.soil_local_learning_rate,
        hidden_layers=args.hidden_layers,
        kan_basis=args.kan_basis,
        soil_base_x=args.soil_base_x,
        soil_base_y=args.soil_base_y,
        soil_middle_radial=args.soil_middle_radial,
        soil_middle_theta=args.soil_middle_theta,
        soil_fine_radial=args.soil_fine_radial,
        soil_fine_theta=args.soil_fine_theta,
        water_base_x=args.water_base_x,
        water_base_y=args.water_base_y,
        water_annulus_radial=args.water_annulus_radial,
        water_annulus_theta=args.water_annulus_theta,
        p2g_x=args.p2g_x,
        p2g_y=args.p2g_y,
        batch_background=args.batch_background,
        batch_middle=args.batch_middle,
        batch_fine=args.batch_fine,
        water_batch_background=args.water_batch_background,
        water_batch_annulus=args.water_batch_annulus,
        water_far_field_weight=args.water_far_field_weight,
        water_pipe_support=args.water_pipe_support,
        water_annulus_residual_weight=(
            args.water_annulus_residual_weight
        ),
        rans_continuity_weight=args.rans_continuity_weight,
        rans_free_surface_weight=args.rans_free_surface_weight,
        rans_wall_momentum_weight=args.rans_wall_momentum_weight,
        rans_far_field_weight=args.rans_far_field_weight,
        momentum_weight=args.momentum_weight,
        fluid_weight=args.fluid_weight,
        p2g_weight=args.p2g_weight,
        roughness_weight=args.roughness_weight,
        boundary_points=args.boundary_points,
        al_update_every=args.al_update_every,
        seed=args.seed,
        log_every=args.log_every,
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        "Solver: water-domain MPM-KAN coupled one-way to strong-form "
        "Biot soil-domain MPM-KAN."
    )
    print(
        "Training labels: none. Figure 15 CSV files remain unopened "
        "until all physics training returns."
    )
    checkpoint = _checkpoint_path(
        args.checkpoint, output_dir, root
    )
    if args.evaluate_only:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )
        (
            soil_models,
            water_model,
            used_options,
            histories,
            augmented,
        ) = load_checkpoint(checkpoint, device)
    else:
        (
            soil_models,
            water_model,
            histories,
            augmented,
        ) = train(
            options,
            device,
            output_dir,
            (
                Path(args.water_initial_checkpoint).resolve()
                if args.water_initial_checkpoint
                else None
            ),
            (
                Path(args.soil_initial_checkpoint).resolve()
                if args.soil_initial_checkpoint
                else None
            ),
        )
        used_options = options
        checkpoint = (
            output_dir / "checkpoints" / "figure15.pt"
        )
    print(
        "Water closure: "
        f"{used_options.water_model_kind}; coupling: "
        f"{used_options.water_coupling_strategy}."
    )

    setups = figure15_setups()
    predictions: dict[str, dict[str, np.ndarray]] = {}
    physics_losses: dict[str, object] = {}
    cloud_summaries: dict[str, object] = {}
    for case_name in ("test10", "test49"):
        wave, soil, geometry = setups[case_name]
        cloud, problem = make_soil_problem(
            wave,
            soil,
            geometry,
            used_options,
            device,
            torch.float64,
        )
        model = soil_models[case_name]
        profile = problem.evaluate_pipe(
            model,
            water_model if case_name == "test49" else None,
            n_theta=args.theta_points,
        )
        predictions[case_name] = {
            key: value.cpu().numpy()
            for key, value in profile.items()
        }
        indices = torch.arange(
            cloud.n_particles, device=device
        )
        physics_losses[case_name] = problem.loss(
            model, indices, augmented[case_name]
        ).detached()
        cloud_summaries[case_name] = cloud.summary()
    wave49, _, geometry49 = setups["test49"]
    if used_options.water_model_kind == "harmonic_rans":
        water_cloud, water_problem = make_rans_water_problem(
            wave49,
            geometry49,
            used_options,
            device,
            torch.float64,
        )
    else:
        water_cloud, water_problem = make_water_problem(
            wave49,
            geometry49,
            used_options,
            device,
            torch.float64,
        )
    water_indices = torch.arange(
        water_cloud.n_particles, device=device
    )
    physics_losses["water_test49"] = water_problem.loss(
        water_model, water_indices
    ).detached()
    cloud_summaries["water_test49"] = water_cloud.summary()

    # Deliberate post-training validation-data access.
    extracted, quality = load_figure15_data(
        test10_data, test49_data
    )
    metrics = calculate_metrics(
        predictions, extracted, quality
    )
    test49_pressure = predictions["test49"]["combined_pressure"]
    test49_theta = predictions["test49"]["theta"]
    midpoint = int(np.argmin(np.abs(test49_theta - np.pi)))
    interface_model = soil_models["test49"].water_model
    contact_is_trained = isinstance(
        interface_model, ContactEnrichedWaterKAN
    ) and isinstance(water_model, HarmonicRANSKAN)
    support_angle = (
        interface_model.contact_angle
        if isinstance(interface_model, ContactEnrichedWaterKAN)
        else None
    )
    raw_contact = (
        contact_diagnostics(
            water_model, interface_model, geometry49
        ).as_dict()
        if contact_is_trained
        else None
    )
    metrics["contact_continuity"] = {
        "method": (
            "raw network output from physics-trained compact "
            "contact-corner KANs"
        ),
        "enabled": contact_is_trained,
        "postprocessing_applied": False,
        "figure15_training_data_used": False,
        "support_definition": (
            "trained local support angle stored in the checkpoint"
        ),
        "support_angle_rad": support_angle,
        "support_angle_deg": (
            float(np.degrees(support_angle))
            if support_angle is not None
            else None
        ),
        "raw_complex_contact_diagnostics": raw_contact,
        "visual_amplitude_jump_theta_0_2pi": float(
            abs(test49_pressure[0] - test49_pressure[-1])
        ),
        "visual_amplitude_jump_theta_pi_one_grid_step": float(
            abs(
                test49_pressure[midpoint]
                - test49_pressure[midpoint + 1]
            )
        ),
        "grid_step_rad": float(
            test49_theta[midpoint + 1]
            - test49_theta[midpoint]
        ),
    }
    metrics["physics_losses"] = physics_losses
    metrics["clouds"] = cloud_summaries
    metrics["prediction_ranges"] = {
        case_name: {
            key: [
                float(np.nanmin(values[key])),
                float(np.nanmax(values[key])),
            ]
            for key in (
                "combined_pressure",
                "pore_pressure",
                "hydrodynamic_pressure",
            )
            if bool(np.isfinite(values[key]).any())
        }
        for case_name, values in predictions.items()
    }
    plot_figure15(
        output_dir / "figure15_mpm_kan_pinn.png",
        predictions,
        extracted,
    )
    save_profiles(
        output_dir / "figure15_profiles.csv",
        predictions,
        extracted,
    )
    save_json(metrics, output_dir / "figure15_metrics.json")
    save_json(
        {"histories": histories},
        output_dir / "training_history.json",
    )
    run_config = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dtype_finish": "torch.float64",
        "training_labels": "none",
        "water_model_kind": used_options.water_model_kind,
        "water_coupling_strategy": (
            used_options.water_coupling_strategy
        ),
        "contact_corner_training": {
            "enabled": contact_is_trained,
            "model": (
                type(interface_model).__name__
                if interface_model is not None
                else None
            ),
            "quantity": "complex harmonic pressure",
            "objectives": (
                "water-domain physics plus complex-pressure "
                "contact continuity"
            ),
            "postprocessing_applied": False,
            "figure15_training_data_used": False,
            "support_definition": (
                "checkpoint contact_angle with quintic C2 compact "
                "support"
            ),
            "support_angle_rad": support_angle,
            "support_angle_deg": (
                float(np.degrees(support_angle))
                if support_angle is not None
                else None
            ),
        },
        "figure15_csv_loaded_after_training": True,
        "test10_data": str(test10_data),
        "test49_data": str(test49_data),
        "checkpoint": str(checkpoint),
        "water_initial_checkpoint": (
            str(Path(args.water_initial_checkpoint).resolve())
            if args.water_initial_checkpoint
            else None
        ),
        "soil_initial_checkpoint": (
            str(Path(args.soil_initial_checkpoint).resolve())
            if args.soil_initial_checkpoint
            else None
        ),
        "training_options": asdict(used_options),
        "setups": {
            case_name: {
                "wave": asdict(wave),
                "soil": asdict(soil),
                "geometry": geometry.as_dict(),
                "coefficients": dimensionless_coefficients(
                    wave, soil, geometry
                ),
            }
            for case_name, (wave, soil, geometry) in setups.items()
        },
        "evaluate_only": args.evaluate_only,
        "theta_points": args.theta_points,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    print("Completed Figure 15 reproduction.")
    for case_name, sources in (
        ("test10", ("experimental", "pore_numerical")),
        (
            "test49",
            (
                "experimental",
                "pore_numerical",
                "hydrodynamic_numerical",
            ),
        ),
    ):
        for source in sources:
            value = metrics["mpm_kan"][case_name][source][
                "relative_l2"
            ]
            print(
                f"{case_name} vs {source}: "
                f"relative L2={100.0 * value:.3f}%"
            )
    print(
        f"Figure: {output_dir / 'figure15_mpm_kan_pinn.png'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Zhu et al. (2024) Figure 15 with a "
            "coupled water-seabed MPM-KAN PINN."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version="mpm-kan-figure15 1.3.0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--water-model-kind",
        choices=("potential", "harmonic_rans"),
        default="potential",
    )
    parser.add_argument(
        "--water-coupling-strategy",
        default="single_field",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test10-data", default=None)
    parser.add_argument("--test49-data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--water-initial-checkpoint",
        default=None,
        help=(
            "Reuse a no-label water checkpoint; nonzero float64 or "
            "L-BFGS water steps continue its physics training."
        ),
    )
    parser.add_argument(
        "--soil-initial-checkpoint",
        default=None,
        help=(
            "Continue both soil models from a no-Figure-15-label "
            "Figure 15 checkpoint."
        ),
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--water-steps", type=int, default=600)
    parser.add_argument(
        "--water-learning-rate", type=float, default=5.0e-4
    )
    parser.add_argument(
        "--water-float64-steps", type=int, default=60
    )
    parser.add_argument("--water-lbfgs-steps", type=int, default=30)
    parser.add_argument("--soil-steps", type=int, default=1100)
    parser.add_argument(
        "--soil-learning-rate", type=float, default=5.0e-4
    )
    parser.add_argument(
        "--soil-float64-steps", type=int, default=120
    )
    parser.add_argument(
        "--soil-float64-learning-rate",
        type=float,
        default=1.5e-5,
    )
    parser.add_argument("--soil-lbfgs-steps", type=int, default=50)
    parser.add_argument("--teacher-steps", type=int, default=600)
    parser.add_argument(
        "--teacher-learning-rate",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--teacher-anchor-steps", type=int, default=200
    )
    parser.add_argument(
        "--teacher-anchor-weight",
        type=float,
        default=1.0e8,
    )
    parser.add_argument("--water-width", type=int, default=16)
    parser.add_argument("--soil-width", type=int, default=18)
    parser.add_argument("--soil-local-width", type=int, default=12)
    parser.add_argument("--soil-local-basis", type=int, default=9)
    parser.add_argument(
        "--soil-local-support", type=float, default=0.30
    )
    parser.add_argument("--soil-local-steps", type=int, default=0)
    parser.add_argument(
        "--soil-local-learning-rate",
        type=float,
        default=2.0e-4,
    )
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--kan-basis", type=int, default=9)
    parser.add_argument("--soil-base-x", type=int, default=40)
    parser.add_argument("--soil-base-y", type=int, default=24)
    parser.add_argument(
        "--soil-middle-radial", type=int, default=6
    )
    parser.add_argument(
        "--soil-middle-theta", type=int, default=96
    )
    parser.add_argument(
        "--soil-fine-radial", type=int, default=6
    )
    parser.add_argument(
        "--soil-fine-theta", type=int, default=160
    )
    parser.add_argument("--water-base-x", type=int, default=40)
    parser.add_argument("--water-base-y", type=int, default=24)
    parser.add_argument(
        "--water-annulus-radial", type=int, default=6
    )
    parser.add_argument(
        "--water-annulus-theta", type=int, default=128
    )
    parser.add_argument("--p2g-x", type=int, default=80)
    parser.add_argument("--p2g-y", type=int, default=48)
    parser.add_argument("--batch-background", type=int, default=640)
    parser.add_argument("--batch-middle", type=int, default=480)
    parser.add_argument("--batch-fine", type=int, default=640)
    parser.add_argument(
        "--water-batch-background", type=int, default=640
    )
    parser.add_argument(
        "--water-batch-annulus", type=int, default=480
    )
    parser.add_argument(
        "--water-far-field-weight",
        type=float,
        default=2500.0,
    )
    parser.add_argument(
        "--water-pipe-support",
        type=float,
        default=0.16,
        help="Dimensionless support radius of the hard pipe map.",
    )
    parser.add_argument(
        "--water-annulus-residual-weight",
        type=float,
        default=0.38,
        help="Water Laplace-loss share assigned to pipe-annulus points.",
    )
    parser.add_argument(
        "--rans-continuity-weight", type=float, default=8.0
    )
    parser.add_argument(
        "--rans-free-surface-weight", type=float, default=80.0
    )
    parser.add_argument(
        "--rans-wall-momentum-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--rans-far-field-weight", type=float, default=1200.0
    )
    parser.add_argument(
        "--momentum-weight", type=float, default=50.0
    )
    parser.add_argument(
        "--fluid-weight", type=float, default=3000.0
    )
    parser.add_argument("--p2g-weight", type=float, default=0.25)
    parser.add_argument(
        "--roughness-weight", type=float, default=1.0e-4
    )
    parser.add_argument("--boundary-points", type=int, default=160)
    parser.add_argument("--al-update-every", type=int, default=150)
    parser.add_argument("--seed", type=int, default=202415)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--theta-points", type=int, default=721)
    return parser


if __name__ == "__main__":
    main()
