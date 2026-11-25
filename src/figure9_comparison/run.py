from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import pandas as pd
import torch

from mpm_kan_figure9.config import figure9_setup

from .common import physics_constants
from .evaluation import evaluate_all, plot_comparison, write_isolation_manifest
from .pino import PINOOptions, train_pino
from .pointwise import PointwiseOptions, refine_pointwise_lbfgs, train_pointwise


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
DEFAULT_OUTPUT = ROOT / "outputs"
BENCHMARK = WORKSPACE / "figure9_mpm_kan" / "data" / "figure9.csv"
SEAKAN_PROFILE = (
    WORKSPACE
    / "figure9_mpm_kan"
    / "artifacts"
    / "top_gap_augmented"
    / "figure9_top_gap_augmented_profiles.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["deepxde_pinn", "gpinn_rar", "xpinn", "pino"],
        default=["deepxde_pinn", "gpinn_rar", "xpinn", "pino"],
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--pino-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202409)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--refine-only", action="store_true")
    parser.add_argument("--lbfgs-steps", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the released comparison. Run with the usual "
            "conda environment rather than the CPU-only base interpreter."
        )
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    wave, soil, geometry = figure9_setup()
    constants = physics_constants(wave, soil, geometry)
    runs: list[dict[str, object]] = []
    if args.refine_only:
        for method in args.methods:
            if method == "pino":
                continue
            runs.append(
                refine_pointwise_lbfgs(
                    method,
                    constants,
                    args.output,
                    device=device,
                    max_iter=args.lbfgs_steps,
                    interior_points=256 if args.smoke else 3072,
                )
            )
    elif not args.evaluate_only:
        point_steps = 3 if args.smoke else args.steps
        pino_epochs = 2 if args.smoke else args.pino_epochs
        for method in args.methods:
            if method == "pino":
                result = train_pino(
                    args.output,
                    options=PINOOptions(
                        epochs=pino_epochs,
                        seed=args.seed,
                        nx=33 if args.smoke else 65,
                        ny=17 if args.smoke else 25,
                    ),
                    device=device,
                )
            else:
                result = train_pointwise(
                    method,
                    constants,
                    args.output,
                    options=PointwiseOptions(
                        steps=point_steps,
                        batch_size=128 if args.smoke else 1536,
                        boundary_points=24 if args.smoke else 96,
                        gpinn_points=16 if args.smoke else 192,
                        rar_pool=256 if args.smoke else 4096,
                        rar_add=32 if args.smoke else 256,
                        seed=args.seed,
                    ),
                    device=device,
                )
            runs.append(result)
        pd.DataFrame(runs).to_csv(args.output / "run_summary.csv", index=False)
        (args.output / "run_summary.json").write_text(
            json.dumps(runs, indent=2), encoding="utf-8"
        )

    required = [args.output / method / "profile.csv" for method in args.methods]
    if all(path.is_file() for path in required) and all(
        (args.output / method / "profile.csv").is_file()
        for method in ("deepxde_pinn", "gpinn_rar", "xpinn", "pino")
    ):
        metrics, profiles = evaluate_all(
            args.output, BENCHMARK, SEAKAN_PROFILE
        )
        plot_comparison(args.output, metrics, profiles, BENCHMARK)
        write_isolation_manifest(args.output, BENCHMARK, SEAKAN_PROFILE)
        print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
