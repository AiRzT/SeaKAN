"""Run a short physics-only Standard KINN training and pressure evaluation on CPU."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "exp1_kinn_and_ablation"))
import exp1_kinn_and_ablation as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    experiment.OUT_DIR = args.output.resolve()
    torch.set_num_threads(2)
    constants = experiment.physics_constants(*experiment.figure9_setup())
    reference = experiment.load_benchmark(experiment.BENCHMARK_CSV)
    result = experiment.run_standard_kinn(constants, reference, torch.device("cpu"), steps=3,
                                           batch_size=48, boundary_pts=16, lbfgs_stage1=0, lbfgs_stage2=0)
    if result["parameters"] != 13131 or not np.isfinite(result["metrics"]["experimental"]["relative_l2"]):
        raise RuntimeError("Training/evaluation smoke failed.")
    print("CPU smoke passed (3 Adam updates, no L-BFGS). See verify_tier1.py for published metrics.")


if __name__ == "__main__":
    main()
