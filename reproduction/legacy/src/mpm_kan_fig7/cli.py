from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import SoilParameters, WaveParameters, fig7_cases
from .extracted import (
    calculate_extracted_metrics,
    load_extracted_figure7,
    plot_extracted_comparison,
)
from .mpm import MaterialPointGrid1D
from .physics import MPMKANBiotProblem
from .plotting import (
    calculate_metrics,
    plot_figure7,
    save_metrics,
    save_profiles,
)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = select_device(args.device)
    root = Path(__file__).resolve().parents[1]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "outputs" / "figure7"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = fig7_cases()
    requested_keys = args.cases or list(all_cases)
    unknown = sorted(set(requested_keys) - set(all_cases))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Choose from {sorted(all_cases)}")
    cases = {key: all_cases[key] for key in requested_keys}
    options = TrainingOptions(
        steps=args.steps,
        learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_steps,
        grid_cells=args.grid_cells,
        particles_per_cell=args.particles_per_cell,
        width=args.width,
        hidden_layers=args.hidden_layers,
        kan_basis=args.kan_basis,
        momentum_weight=args.momentum_weight,
        boundary_weight=args.boundary_weight,
        fluid_weight=args.fluid_weight,
        strong_weight=args.strong_weight,
        roughness_weight=args.roughness_weight,
        seed=args.seed,
        log_every=args.log_every,
    )
    wave = WaveParameters()
    soil = SoilParameters()
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Cases: {', '.join(cases)}")

    predictions: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, dict[str, np.ndarray]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    final_residuals: dict[str, dict[str, float]] = {}
    used_case_options: dict[str, dict[str, object]] = {}
    for key, case in cases.items():
        if args.evaluate_only:
            model, history, case_options = load_case_checkpoint(
                case, device, Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir
            )
        else:
            case_options = options
            if case.permeability <= 1.0e-4:
                case_options = replace(
                    options,
                    steps=args.fine_steps,
                    lbfgs_steps=args.fine_lbfgs_steps,
                )
            model, history = train_case(
                case,
                wave,
                soil,
                case_options,
                device,
                output_dir,
                resume=args.resume,
            )
        used_case_options[key] = dict(vars(case_options))
        grid = MaterialPointGrid1D.create(
            case_options.grid_cells,
            case_options.particles_per_cell,
            device=device,
            dtype=torch.float64,
        )
        problem = MPMKANBiotProblem(
            wave,
            soil,
            case,
            grid,
            momentum_weight=case_options.momentum_weight,
            boundary_weight=case_options.boundary_weight,
            fluid_weight=case_options.fluid_weight,
            strong_weight=case_options.strong_weight,
            roughness_weight=case_options.roughness_weight,
        )
        tensor_profile = problem.evaluate(model, n_points=args.evaluation_points)
        predictions[key] = {
            name: value.detach().cpu().numpy()
            for name, value in tensor_profile.items()
        }
        references[key] = solve_reference(
            wave, soil, case, n_depth=args.reference_points
        )
        histories[key] = history
        final_residuals[key] = problem.loss(model).detached()

    metrics = calculate_metrics(predictions, references)
    for key in cases:
        metrics[key].update(
            {f"final_{name}": value for name, value in final_residuals[key].items()}
        )
    save_profiles(
        predictions, references, cases, output_dir / "figure7_profiles.csv"
    )
    save_metrics(metrics, output_dir / "metrics.json")
    save_history(histories, output_dir / "training_history.json")
    plot_figure7(
        predictions,
        references,
        cases,
        output_dir / "figure7_mpm_kan_pinn.png",
    )
    extracted_data_dir = (
        Path(args.extracted_data_dir).resolve()
        if args.extracted_data_dir
        else root.parent / "figure7_data"
    )
    if extracted_data_dir.is_dir():
        extracted, quality = load_extracted_figure7(extracted_data_dir)
        extracted_metrics = calculate_extracted_metrics(
            predictions, extracted, quality
        )
        save_metrics(
            extracted_metrics, output_dir / "extracted_data_metrics.json"
        )
        plot_extracted_comparison(
            predictions,
            extracted,
            cases,
            output_dir / "figure7_extracted_comparison.png",
        )
        print(f"Extracted Figure 7 validation: {extracted_data_dir}")
        for key in cases:
            values = extracted_metrics["cases"][key]
            formatted = ", ".join(
                f"{field}={100.0 * values[field]['analytical']['relative_l2']:.2f}%"
                for field in ("pore_pressure", "sigma_x_eff", "sigma_z_eff", "tau_xz")
            )
            print(f"  {key}: {formatted}")
    elif args.extracted_data_dir:
        raise FileNotFoundError(
            f"Extracted Figure 7 data directory not found: {extracted_data_dir}"
        )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "cases": requested_keys,
                "training": vars(options),
                "case_training": used_case_options,
                "fine_case_steps": args.fine_steps,
                "fine_case_lbfgs_steps": args.fine_lbfgs_steps,
                "evaluation_points": args.evaluation_points,
                "reference_points": args.reference_points,
                "extracted_data_dir": (
                    str(extracted_data_dir) if extracted_data_dir.is_dir() else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Completed.")
    for key, values in metrics.items():
        print(f"{key}: mean relative L2={values['mean_relative_l2']:.4e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an MPM-KAN physics-informed solver for Zhu et al. Figure 7."
    )
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument(
        "--fine-steps",
        type=int,
        default=3200,
        help="Adam target step for the two low-permeability fine-sand cases.",
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument(
        "--fine-lbfgs-steps",
        type=int,
        default=250,
        help="L-BFGS steps for the two low-permeability fine-sand cases.",
    )
    parser.add_argument("--grid-cells", type=int, default=48)
    parser.add_argument("--particles-per-cell", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--kan-basis", type=int, default=10)
    parser.add_argument(
        "--momentum-weight",
        type=float,
        default=1.0,
        help="Relative weight for weak and strong mechanical momentum residuals.",
    )
    parser.add_argument("--boundary-weight", type=float, default=500.0)
    parser.add_argument("--fluid-weight", type=float, default=500.0)
    parser.add_argument("--strong-weight", type=float, default=0.05)
    parser.add_argument("--roughness-weight", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=202407)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--evaluation-points", type=int, default=401)
    parser.add_argument("--reference-points", type=int, default=401)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="Read evaluation checkpoints from this run directory.")
    parser.add_argument(
        "--extracted-data-dir",
        default=None,
        help=(
            "Directory containing figure7a.csv through figure7d.csv. "
            "Defaults to ../figure7_data when present."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load existing checkpoints and regenerate profiles, metrics, and plots.",
    )
    return parser


if __name__ == "__main__":
    main()
