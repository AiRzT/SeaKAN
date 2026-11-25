"""Retrain the selected single-field SeaKAN adapter from its three archived physics stages."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from verify_tier1 import DATA, LEGACY, figure9_setup, load_benchmark, metrics
from mpm_kan_figure9.adaptive_mpm import AdaptiveMaterialPointCloud
from mpm_kan_figure9.top_gap_training import TopGapTrainingOptions, train_top_gap_augmented
from mpm_kan_figure9.training import make_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = DATA / "selected_training"
    config = json.loads((data / "config.json").read_text("utf-8"))
    options = TopGapTrainingOptions(**config["selected"]["top_gap_options"])
    # The original source seeds after constructing the adapter. Also seed the
    # process here to make the previously unrecorded initial RNG state explicit.
    torch.manual_seed(options.seed)
    wave, soil, geometry = figure9_setup()
    cloud_args = config["selected"]["cloud_parameters"]
    clouds = [AdaptiveMaterialPointCloud.create(geometry, device, dtype, **cloud_args)
              for dtype in (torch.float32, torch.float64)]
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        checkpoints = []
        for stage in ("weak", "adaptive", "teacher"):
            path = Path(tmp) / f"{stage}.pt"
            with np.load(data / f"{stage}.npz", allow_pickle=False) as weights:
                state = {k: torch.from_numpy(weights[k]) for k in weights}
            torch.save({"model": state, **config[stage]}, path)
            checkpoints.append(path)
        model, baseline_options, _, dtype, history, _ = train_top_gap_augmented(
            wave, soil, geometry, *clouds, *checkpoints, out, options, device)
    _, problem = make_problem(wave, soil, geometry, baseline_options, device, dtype)
    prediction = problem.evaluate_pipe(model.eval(), n_theta=721)
    curve = tuple(prediction[key].detach().cpu().numpy()
                  for key in ("theta", "normalized_pressure"))
    with (out / "profile.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_rad", "normalized_pressure"])
        writer.writerows(zip(*curve, strict=True))
    references = load_benchmark(LEGACY / "figure9_mpm_kan/data/figure9.csv")
    names = ("exp_l2_pct", "exp_r2", "fem_l2_pct", "fem_r2")
    result = dict(zip(names, metrics(curve, references), strict=True))
    result.update(device=str(device), initial_seed=options.seed,
                  initialized_from=["weak", "adaptive", "teacher"],
                  elapsed_seconds=history[-1]["elapsed_seconds"])
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
