from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from .config import dimensionless_coefficients, figure13_setup
from .data import (
    calculate_metrics,
    load_figure13_data,
    plot_figure13,
    save_json,
    save_profiles,
)
from .reference import solve_reference
from .training import (
    TrainingOptions,
    load_checkpoint,
    make_cloud_and_problem,
    refine_full_batch,
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
            files("mpm_kan_figure13").joinpath(
                "resources", relative_path
            )
        )
    )


def _resolve_data_file(
    explicit: str | None,
    root: Path,
    filename: str,
) -> Path:
    if explicit:
        return Path(explicit).resolve()
    local = root / "data" / filename
    return (
        local
        if local.is_file()
        else packaged_resource(filename)
    )


def _resolve_checkpoint(
    explicit: str | None,
    output_dir: Path,
    root: Path,
) -> Path:
    if explicit:
        return Path(explicit).resolve()
    candidates = (
        output_dir / "checkpoints" / "figure13.pt",
        root / "artifacts" / "checkpoints" / "figure13.pt",
        packaged_resource("artifacts/checkpoints/figure13.pt"),
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
        else root / "outputs" / "figure13"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pressure_data = _resolve_data_file(
        args.pressure_data, root, "figure13a.csv"
    )
    stress_data = _resolve_data_file(
        args.stress_data, root, "figure13b.csv"
    )
    device = select_device(args.device)
    wave, soil, geometry = figure13_setup()
    coefficients = dimensionless_coefficients(
        wave, soil, geometry
    )
    requested_options = TrainingOptions(
        float32_steps=args.float32_steps,
        float32_learning_rate=args.float32_learning_rate,
        batch_particles=args.batch_particles,
        float64_adam_steps=args.float64_adam_steps,
        float64_learning_rate=args.float64_learning_rate,
        float64_lbfgs_steps=args.float64_lbfgs_steps,
        cells_x=args.cells_x,
        cells_y=args.cells_y,
        width=args.width,
        hidden_layers=args.hidden_layers,
        kan_basis=args.kan_basis,
        momentum_weight=args.momentum_weight,
        fluid_weight=args.fluid_weight,
        p2g_weight=args.p2g_weight,
        roughness_weight=args.roughness_weight,
        boundary_points=args.boundary_points,
        al_update_every=args.al_update_every,
        rho_top=args.rho_top,
        rho_bottom=args.rho_bottom,
        seed=args.seed,
        log_every=args.log_every,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        "Physics: strong-form 2-D harmonic u-w-p Biot equations "
        "on MPM particles with P2G residual projection."
    )
    print(
        "Training labels: none. Figure 13 CSV files and the independent "
        "BVP reference remain withheld until training is complete."
    )

    checkpoint_path = _resolve_checkpoint(
        args.checkpoint, output_dir, root
    )
    if args.evaluate_only or args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        (
            model,
            used_options,
            dtype,
            history,
            augmented,
        ) = load_checkpoint(
            wave,
            soil,
            geometry,
            checkpoint_path,
            device,
        )
        if args.resume:
            used_options = replace(
                used_options,
                cells_x=(
                    args.resume_cells_x
                    if args.resume_cells_x is not None
                    else used_options.cells_x
                ),
                cells_y=(
                    args.resume_cells_y
                    if args.resume_cells_y is not None
                    else used_options.cells_y
                ),
                momentum_weight=(
                    args.resume_momentum_weight
                    if args.resume_momentum_weight is not None
                    else used_options.momentum_weight
                ),
                fluid_weight=(
                    args.resume_fluid_weight
                    if args.resume_fluid_weight is not None
                    else used_options.fluid_weight
                ),
                p2g_weight=(
                    args.resume_p2g_weight
                    if args.resume_p2g_weight is not None
                    else used_options.p2g_weight
                ),
            )
            (
                model,
                dtype,
                history,
                augmented,
            ) = refine_full_batch(
                wave,
                soil,
                geometry,
                model,
                used_options,
                history,
                augmented,
                device,
                output_dir,
                adam_steps=args.resume_adam_steps,
                learning_rate=args.resume_learning_rate,
                lbfgs_steps=args.resume_lbfgs_steps,
            )
            checkpoint_path = (
                output_dir / "checkpoints" / "figure13.pt"
            )
    else:
        (
            model,
            dtype,
            history,
            augmented,
        ) = train(
            wave,
            soil,
            geometry,
            requested_options,
            device,
            output_dir,
        )
        used_options = requested_options
        checkpoint_path = (
            output_dir / "checkpoints" / "figure13.pt"
        )

    cloud, problem = make_cloud_and_problem(
        wave,
        soil,
        geometry,
        used_options,
        device,
        dtype,
    )
    full_indices = torch.arange(
        cloud.n_particles, device=device, dtype=torch.long
    )
    profile_tensors = problem.evaluate_profile(
        model, n_depth=args.depth_points
    )
    prediction = {
        key: value.cpu().numpy()
        for key, value in profile_tensors.items()
    }
    final_loss = problem.loss(
        model, full_indices, augmented
    ).detached()

    # Both verification sources are deliberately accessed only after training.
    reference_result = solve_reference(
        wave,
        soil,
        geometry,
        n_depth=max(1001, args.depth_points),
    )
    reference = reference_result.profile()
    extracted, quality = load_figure13_data(
        pressure_data, stress_data
    )
    metrics = calculate_metrics(
        prediction,
        extracted,
        quality,
        reference,
    )
    metrics["final_physics_loss"] = final_loss
    metrics["independent_bvp_reference"] = {
        "maximum_rms_residual": (
            reference_result.maximum_rms_residual
        ),
        "mesh_nodes": reference_result.mesh_nodes,
    }
    metrics["prediction_range"] = {
        quantity: [
            float(np.min(prediction[quantity])),
            float(np.max(prediction[quantity])),
        ]
        for quantity in (
            "pressure",
            "vertical_effective_stress",
        )
    }

    save_profiles(
        output_dir / "figure13_profiles.csv",
        prediction,
        extracted,
        reference,
    )
    save_json(metrics, output_dir / "figure13_metrics.json")
    save_json(
        {"history": history},
        output_dir / "training_history.json",
    )
    plot_figure13(
        output_dir / "figure13_mpm_kan_pinn.png",
        prediction,
        extracted,
        reference,
    )
    run_config = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dtype": str(dtype),
        "training_labels": "none",
        "figure13_data_loaded_after_training": True,
        "independent_reference_loaded_after_training": True,
        "pressure_data": str(pressure_data),
        "stress_data": str(stress_data),
        "checkpoint": str(checkpoint_path),
        "wave": asdict(wave),
        "soil": asdict(soil),
        "geometry": geometry.as_dict(),
        "dimensionless_coefficients": coefficients,
        "requested_training": asdict(requested_options),
        "checkpoint_training": asdict(used_options),
        "evaluate_only": args.evaluate_only,
        "resume": args.resume,
        "requested_resume": {
            "adam_steps": args.resume_adam_steps,
            "learning_rate": args.resume_learning_rate,
            "lbfgs_steps": args.resume_lbfgs_steps,
            "cells_x": args.resume_cells_x,
            "cells_y": args.resume_cells_y,
            "momentum_weight": args.resume_momentum_weight,
            "fluid_weight": args.resume_fluid_weight,
            "p2g_weight": args.resume_p2g_weight,
        },
        "depth_points": args.depth_points,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print("Completed Figure 13 reproduction.")
    for quantity in (
        "pressure",
        "vertical_effective_stress",
    ):
        print(f"{quantity}:")
        for source in (
            "numerical",
            "analytical",
            "independent_bvp_reference",
        ):
            value = metrics["mpm_kan"][quantity][source][
                "relative_l2"
            ]
            print(f"  vs {source}: relative L2={100.0 * value:.3f}%")
    print(
        f"Figure: {output_dir / 'figure13_mpm_kan_pinn.png'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Zhu et al. (2024) Figure 13 with a "
            "physics-only MPM-KAN PINN."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version="mpm-kan-figure13 1.0.0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pressure-data", default=None)
    parser.add_argument("--stress-data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a checkpoint with float64 full-batch physics",
    )

    parser.add_argument("--float32-steps", type=int, default=1200)
    parser.add_argument(
        "--float32-learning-rate", type=float, default=5.0e-4
    )
    parser.add_argument("--batch-particles", type=int, default=1536)
    parser.add_argument("--float64-adam-steps", type=int, default=80)
    parser.add_argument(
        "--float64-learning-rate", type=float, default=2.0e-5
    )
    parser.add_argument("--float64-lbfgs-steps", type=int, default=40)
    parser.add_argument("--cells-x", type=int, default=32)
    parser.add_argument("--cells-y", type=int, default=64)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--kan-basis", type=int, default=11)
    parser.add_argument("--momentum-weight", type=float, default=10.0)
    parser.add_argument("--fluid-weight", type=float, default=1000.0)
    parser.add_argument("--p2g-weight", type=float, default=0.25)
    parser.add_argument(
        "--roughness-weight", type=float, default=1.0e-4
    )
    parser.add_argument("--boundary-points", type=int, default=128)
    parser.add_argument("--al-update-every", type=int, default=100)
    parser.add_argument("--rho-top", type=float, default=160.0)
    parser.add_argument("--rho-bottom", type=float, default=320.0)
    parser.add_argument("--seed", type=int, default=202413)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--depth-points", type=int, default=501)
    parser.add_argument("--resume-adam-steps", type=int, default=240)
    parser.add_argument(
        "--resume-learning-rate", type=float, default=5.0e-5
    )
    parser.add_argument("--resume-lbfgs-steps", type=int, default=80)
    parser.add_argument("--resume-cells-x", type=int, default=None)
    parser.add_argument("--resume-cells-y", type=int, default=None)
    parser.add_argument(
        "--resume-momentum-weight", type=float, default=None
    )
    parser.add_argument(
        "--resume-fluid-weight", type=float, default=None
    )
    parser.add_argument(
        "--resume-p2g-weight", type=float, default=None
    )
    return parser


if __name__ == "__main__":
    main()
