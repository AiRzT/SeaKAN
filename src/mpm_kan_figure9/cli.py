from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from .config import dimensionless_coefficients, figure9_setup
from .data import (
    calculate_metrics,
    load_figure9_data,
    plot_field,
    plot_polar,
    save_json,
    save_profiles,
)
from .physics import MPMKANPipeProblem
from .reference import solve_reference
from .training import (
    TrainingOptions,
    load_checkpoint,
    make_problem,
    save_history,
    train,
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def project_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


def packaged_resource(relative_path: str) -> Path:
    return Path(
        str(files("mpm_kan_figure9").joinpath("resources", relative_path))
    )


def main() -> None:
    args = build_parser().parse_args()
    root = project_root()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "outputs" / "figure9"
    )
    data_path = (
        Path(args.data_file).resolve()
        if args.data_file
        else (
            root / "data" / "figure9.csv"
            if (root / "data" / "figure9.csv").is_file()
            else packaged_resource("figure9.csv")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    wave, soil, geometry = figure9_setup()
    coefficients = dimensionless_coefficients(wave, soil, geometry)

    baseline = TrainingOptions(
        steps=args.baseline_steps,
        learning_rate=args.baseline_learning_rate,
        lbfgs_steps=0,
        grid_cells_x=args.baseline_grid_x,
        grid_cells_y=args.baseline_grid_y,
        width=args.width,
        hidden_layers=args.hidden_layers,
        kan_basis=args.kan_basis,
        momentum_weight=args.momentum_weight,
        fluid_weight=args.fluid_weight,
        boundary_weight=args.boundary_weight,
        pipe_flux_weight=args.pipe_flux_weight,
        strong_weight=0.0,
        roughness_weight=args.roughness_weight,
        strong_stride=args.strong_stride,
        boundary_points=args.boundary_points,
        seed=args.seed,
        log_every=args.log_every,
    )
    refinement = replace(
        baseline,
        steps=args.refinement_steps,
        learning_rate=args.refinement_learning_rate,
        lbfgs_steps=args.refinement_lbfgs_steps,
        grid_cells_x=args.refinement_grid_x,
        grid_cells_y=args.refinement_grid_y,
        strong_weight=args.strong_weight,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Precision: {args.precision}")
    print(
        f"MPM grid: {baseline.grid_cells_x}x{baseline.grid_cells_y} -> "
        f"{refinement.grid_cells_x}x{refinement.grid_cells_y}"
    )
    print(
        "Physics: 2-D harmonic u-w-p Biot equations, MPM weak quadrature, "
        "impermeable rigid circular pipe."
    )
    print("Digitized Figure 9 data are withheld until training finishes.")

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else output_dir / "checkpoints" / "figure9.pt"
    )
    if args.evaluate_only:
        if not checkpoint_path.is_file():
            fallback = root / "artifacts" / "checkpoints" / "figure9.pt"
            if fallback.is_file():
                checkpoint_path = fallback
            else:
                packaged = packaged_resource("artifacts/checkpoints/figure9.pt")
                if packaged.is_file():
                    checkpoint_path = packaged
                else:
                    raise FileNotFoundError(
                        f"Checkpoint not found: {checkpoint_path}"
                    )
        model, used_options, dtype, history = load_checkpoint(
            wave, soil, geometry, checkpoint_path, device
        )
    else:
        if args.resume:
            model, history = train(
                wave,
                soil,
                geometry,
                refinement,
                device,
                dtype,
                output_dir,
                resume=True,
            )
        else:
            train(
                wave,
                soil,
                geometry,
                baseline,
                device,
                dtype,
                output_dir,
                resume=False,
            )
            model, history = train(
                wave,
                soil,
                geometry,
                refinement,
                device,
                dtype,
                output_dir,
                resume=True,
            )
        used_options = refinement
        checkpoint_path = output_dir / "checkpoints" / "figure9.pt"

    _, problem = make_problem(
        wave, soil, geometry, used_options, device, dtype
    )
    pipe_tensor = problem.evaluate_pipe(model, n_theta=args.theta_points)
    prediction = {
        key: value.cpu().numpy() for key, value in pipe_tensor.items()
    }
    field_tensor = problem.evaluate_field(
        model, nx=args.field_x, ny=args.field_y
    )
    field = {key: value.cpu().numpy() for key, value in field_tensor.items()}

    # Independent numerical check and digitized curves are loaded after training.
    reference_result = solve_reference(
        wave,
        soil,
        geometry,
        nx=args.reference_x,
        ny=args.reference_y,
        n_theta=args.theta_points,
    )
    reference = reference_result.profile()
    extracted, quality = load_figure9_data(data_path)
    metrics = calculate_metrics(prediction, extracted, quality, reference)
    metrics["final_physics_loss"] = problem.loss(model).detached()
    metrics["finite_difference_reference"] = {
        "degrees_of_freedom": reference_result.degrees_of_freedom,
        "relative_linear_residual": reference_result.relative_linear_residual,
    }
    metrics["prediction_range"] = {
        "minimum": float(np.min(prediction["normalized_pressure"])),
        "maximum": float(np.max(prediction["normalized_pressure"])),
    }

    save_profiles(
        output_dir / "figure9_profiles.csv",
        prediction,
        extracted,
        reference,
    )
    save_json(metrics, output_dir / "figure9_metrics.json")
    save_history(history, output_dir / "training_history.json")
    plot_polar(
        output_dir / "figure9_mpm_kan_pinn.png",
        prediction,
        extracted,
        reference,
    )
    plot_field(
        output_dir / "figure9_pressure_field.png", field, geometry
    )
    np.savez_compressed(
        output_dir / "figure9_field.npz",
        x=field["x"],
        y=field["y"],
        normalized_pressure=field["pressure"],
    )
    run_config = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "precision": str(dtype),
        "data_file": str(data_path),
        "checkpoint": str(checkpoint_path),
        "data_loaded_after_training": True,
        "wave": asdict(wave),
        "soil": asdict(soil),
        "geometry": geometry.as_dict(),
        "geometry_note": (
            "Table 5 buried depth 0.167 m is interpreted as mudline-to-pipe-center "
            "distance, consistent with the Figure 9 definition sketch."
        ),
        "dimensionless_coefficients": coefficients,
        "requested_baseline": asdict(baseline),
        "requested_refinement": asdict(refinement),
        "checkpoint_training": asdict(used_options),
        "evaluate_only": args.evaluate_only,
        "reference_grid": [args.reference_x, args.reference_y],
        "theta_points": args.theta_points,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("Completed Figure 9 reproduction.")
    for source in (
        "paper_this_study",
        "jeng_numerical",
        "experimental",
        "finite_difference_reference",
    ):
        value = metrics["mpm_kan"][source]["relative_l2"]
        print(f"MPM-KAN vs {source}: relative L2={100.0 * value:.3f}%")
    print(f"Figure: {output_dir / 'figure9_mpm_kan_pinn.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Zhu et al. (2024) Figure 9 with a 2-D MPM-KAN PINN."
    )
    parser.add_argument("--version", action="version", version="mpm-kan-figure9 1.2.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--baseline-steps", type=int, default=1200)
    parser.add_argument("--baseline-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--baseline-grid-x", type=int, default=48)
    parser.add_argument("--baseline-grid-y", type=int, default=20)
    parser.add_argument("--refinement-steps", type=int, default=600)
    parser.add_argument("--refinement-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--refinement-lbfgs-steps", type=int, default=40)
    parser.add_argument("--refinement-grid-x", type=int, default=64)
    parser.add_argument("--refinement-grid-y", type=int, default=24)
    parser.add_argument("--width", type=int, default=18)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--kan-basis", type=int, default=9)

    parser.add_argument("--momentum-weight", type=float, default=20.0)
    parser.add_argument("--fluid-weight", type=float, default=2000.0)
    parser.add_argument("--boundary-weight", type=float, default=80.0)
    parser.add_argument("--pipe-flux-weight", type=float, default=120.0)
    parser.add_argument("--strong-weight", type=float, default=1.0)
    parser.add_argument("--roughness-weight", type=float, default=1.0e-2)
    parser.add_argument("--strong-stride", type=int, default=8)
    parser.add_argument("--boundary-points", type=int, default=128)

    parser.add_argument("--seed", type=int, default=202409)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--theta-points", type=int, default=721)
    parser.add_argument("--field-x", type=int, default=241)
    parser.add_argument("--field-y", type=int, default=81)
    parser.add_argument("--reference-x", type=int, default=257)
    parser.add_argument("--reference-y", type=int, default=65)
    return parser


if __name__ == "__main__":
    main()
