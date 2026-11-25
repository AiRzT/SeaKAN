from __future__ import annotations

import argparse
import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from .figure8 import FIGURE8_WAVE_HEIGHT, figure8_setup
from .data import (
    calculate_figure8_metrics,
    load_figure8_data,
    plot_figure8,
    save_figure8_profiles,
    save_json,
)
from .mpm import MaterialPointGrid1D
from .physics import MPMKANBiotProblem
from .reference import solve_reference
from .training import (
    TrainingOptions,
    load_case_checkpoint,
    save_history,
    train_case,
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
    return Path(str(files("mpm_kan_figure8").joinpath("resources", relative_path)))


def main() -> None:
    args = build_parser().parse_args()
    device = select_device(args.device)
    root = project_root()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "outputs" / "figure8"
    )
    data_path = (
        Path(args.data_file).resolve()
        if args.data_file
        else (
            root / "data" / "figure8.csv"
            if (root / "data" / "figure8.csv").is_file()
            else packaged_resource("figure8.csv")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wave, soil, case = figure8_setup()
    base_options = TrainingOptions(
        steps=args.baseline_steps,
        learning_rate=args.baseline_learning_rate,
        lbfgs_steps=args.baseline_lbfgs_steps,
        grid_cells=args.baseline_grid_cells,
        particles_per_cell=args.particles_per_cell,
        width=args.width,
        hidden_layers=args.hidden_layers,
        kan_basis=args.kan_basis,
        momentum_weight=1.0,
        boundary_weight=args.boundary_weight,
        fluid_weight=args.fluid_weight,
        strong_weight=0.05,
        roughness_weight=1.0e-7,
        seed=args.seed,
        log_every=args.log_every,
    )
    refinement_options = replace(
        base_options,
        steps=args.refinement_steps,
        learning_rate=args.refinement_learning_rate,
        lbfgs_steps=args.refinement_lbfgs_steps,
        grid_cells=args.refinement_grid_cells,
        momentum_weight=args.momentum_weight,
        strong_weight=args.strong_weight,
        roughness_weight=args.roughness_weight,
    )

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Figure 8 data: {data_path}")
    print(
        f"Linear dispersion: k={wave.wave_number:.10g} 1/m, "
        f"L={wave.wavelength:.10g} m"
    )

    if args.evaluate_only:
        checkpoint_dir = (
            Path(args.checkpoint_dir).resolve()
            if args.checkpoint_dir
            else output_dir
        )
        if not (checkpoint_dir / "checkpoints" / "figure8.pt").is_file():
            packaged_artifacts = root / "artifacts"
            if (packaged_artifacts / "checkpoints" / "figure8.pt").is_file():
                checkpoint_dir = packaged_artifacts
            else:
                installed_artifacts = packaged_resource("artifacts")
                if (
                    installed_artifacts / "checkpoints" / "figure8.pt"
                ).is_file():
                    checkpoint_dir = installed_artifacts
        model, history, used_options = load_case_checkpoint(
            case, device, checkpoint_dir
        )
    elif args.resume:
        model, history = train_case(
            case,
            wave,
            soil,
            refinement_options,
            device,
            output_dir,
            resume=True,
        )
        used_options = refinement_options
    else:
        train_case(
            case,
            wave,
            soil,
            base_options,
            device,
            output_dir,
            resume=False,
        )
        model, history = train_case(
            case,
            wave,
            soil,
            refinement_options,
            device,
            output_dir,
            resume=True,
        )
        used_options = refinement_options

    grid = MaterialPointGrid1D.create(
        used_options.grid_cells,
        used_options.particles_per_cell,
        device=device,
        dtype=torch.float64,
    )
    problem = MPMKANBiotProblem(
        wave,
        soil,
        case,
        grid,
        momentum_weight=used_options.momentum_weight,
        boundary_weight=used_options.boundary_weight,
        fluid_weight=used_options.fluid_weight,
        strong_weight=used_options.strong_weight,
        roughness_weight=used_options.roughness_weight,
    )
    tensor_profile = problem.evaluate(model, n_points=args.evaluation_points)
    prediction = {
        name: value.detach().cpu().numpy()
        for name, value in tensor_profile.items()
    }
    reference = solve_reference(
        wave, soil, case, n_depth=args.reference_points
    )
    extracted, quality = load_figure8_data(data_path)
    metrics = calculate_figure8_metrics(
        prediction, reference, extracted, quality
    )
    metrics["final_physics_loss"] = problem.loss(model).detached()

    save_figure8_profiles(
        prediction,
        reference,
        extracted,
        output_dir / "figure8_profiles.csv",
    )
    save_json(metrics, output_dir / "figure8_metrics.json")
    save_history({"figure8": history}, output_dir / "training_history.json")
    plot_figure8(
        prediction,
        reference,
        extracted,
        output_dir / "figure8_mpm_kan_pinn.png",
    )
    run_config = {
        "device": str(device),
        "data_file": str(data_path),
        "checkpoint_source": (
            str(checkpoint_dir) if args.evaluate_only else str(output_dir)
        ),
        "table4": {
            "wave_period_s": wave.period,
            "wave_height_m": FIGURE8_WAVE_HEIGHT,
            "water_depth_m": wave.water_depth,
            "wave_length_m": wave.wavelength,
            "wave_number_per_m": wave.wave_number,
            "poisson_ratio": soil.poisson_ratio,
            "shear_modulus_pa": soil.shear_modulus,
            "porosity": soil.porosity,
            "seabed_thickness_m": soil.thickness,
            "permeability_m_per_s": case.permeability,
            "saturation": case.saturation,
        },
        "normalization": (
            "P0=1 because the linear harmonic problem is normalized by "
            "the mudline pressure amplitude"
        ),
        "baseline_training": vars(base_options),
        "refinement_training": vars(refinement_options),
        "checkpoint_training": vars(used_options),
        "evaluation_points": args.evaluation_points,
        "reference_points": args.reference_points,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("Completed.")
    for source in ("analytical", "paper_numerical", "experimental"):
        value = metrics["mpm_kan"][source]["relative_l2"]
        print(f"MPM-KAN vs {source}: relative L2={100.0 * value:.3f}%")
    print(
        "MPM-KAN vs independent FD: relative L2="
        f"{100.0 * metrics['mpm_kan']['finite_difference_reference']['relative_l2']:.3f}%"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Zhu et al. (2024) Figure 8 with MPM-KAN PINNs."
    )
    parser.add_argument(
        "--version", action="version", version="mpm-kan-figure8 1.0.0"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory containing checkpoints/figure8.pt for evaluation. "
            "Defaults to the output directory, then packaged artifacts."
        ),
    )
    parser.add_argument("--baseline-steps", type=int, default=3200)
    parser.add_argument("--baseline-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--baseline-lbfgs-steps", type=int, default=250)
    parser.add_argument("--baseline-grid-cells", type=int, default=48)
    parser.add_argument("--refinement-steps", type=int, default=5000)
    parser.add_argument("--refinement-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--refinement-lbfgs-steps", type=int, default=400)
    parser.add_argument("--refinement-grid-cells", type=int, default=64)
    parser.add_argument("--particles-per-cell", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--kan-basis", type=int, default=10)
    parser.add_argument("--momentum-weight", type=float, default=10.0)
    parser.add_argument("--fluid-weight", type=float, default=500.0)
    parser.add_argument("--boundary-weight", type=float, default=500.0)
    parser.add_argument("--strong-weight", type=float, default=0.5)
    parser.add_argument("--roughness-weight", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=202407)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--evaluation-points", type=int, default=501)
    parser.add_argument("--reference-points", type=int, default=801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


if __name__ == "__main__":
    main()
