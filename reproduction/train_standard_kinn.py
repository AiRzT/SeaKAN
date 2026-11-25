"""Train the full Table 6 KINN comparison from a fresh initialization."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "exp1_kinn_and_ablation"))
import exp1_kinn_and_ablation as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    experiment.OUT_DIR = args.output.resolve()
    constants = experiment.physics_constants(*experiment.figure9_setup())
    benchmark = experiment.load_benchmark(experiment.BENCHMARK_CSV)
    experiment.run_standard_kinn(constants, benchmark, torch.device(args.device))


if __name__ == "__main__":
    main()
