"""Recompute the manuscript's buried-pipe metrics from the released numerical arrays."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "tier1_benchmarks"
LEGACY = ROOT / "legacy"
sys.path.insert(0, str(LEGACY / "figure9_mpm_kan" / "src"))
sys.path.insert(0, str(LEGACY / "figure9_model_comparison" / "src"))
from mpm_kan_figure9.config import dimensionless_coefficients, figure9_setup
from mpm_kan_figure9.model import PipeHarmonicKAN, TopGapLocalizedKAN
from figure9_comparison.evaluation import error_metrics, load_benchmark, periodic_interpolate

REFERENCES = ("experimental", "zhu_numerical")
# Printed Table 6: experimental/fem relative L2 (%) and R2, in that order.
TABLE6 = {
    "seakan": (5.41, 0.9894, 12.81, 0.9111),
    "deepxde_pinn": (251.62, -21.9018, 175.96, -15.7584),
    "gpinn_rar": (251.51, -21.8829, 177.02, -15.9620),
    "xpinn": (209.39, -14.8596, 146.36, -10.5943),
    "standard_kinn": (242.99, -20.3591, 165.83, -13.8856),
    "pino": (36.37, 0.5216, 29.17, 0.5394),
}
# Printed Table 10: mean and sample SD of Exp L2, Exp R2, FEM L2, FEM R2.
TABLE10 = {
    "full_seakan": (10.91, 2.73, .955, .020, 16.80, 1.85, .846, .033),
    "with_strong_p2g": (10.92, 2.55, .955, .019, 16.61, 1.75, .850, .030),
    "no_local_branch": (13.53, 1.67, .933, .016, 20.10, .84, .781, .018),
    "no_weak_precond": (13.84, 1.10, .930, .011, 10.65, 1.80, .937, .022),
    "no_crown_refinement": (24.73, 9.23, .758, .158, 26.38, 5.29, .613, .149),
    "mlp_backbone": (159.93, 2.80, -8.254, .324, 106.47, 2.54, -5.138, .292),
}


def seakan_profile():
    wave, soil, geometry = figure9_setup()
    kappa = dimensionless_coefficients(wave, soil, geometry)["kappa"]
    base = PipeHarmonicKAN(geometry, kappa, dtype=torch.float64)
    model = TopGapLocalizedKAN(base, geometry, dtype=torch.float64)
    with np.load(DATA / "seakan_single_field.npz", allow_pickle=False) as weights:
        model.load_state_dict({name: torch.from_numpy(weights[name]) for name in weights})
    model.eval()
    angle = torch.linspace(0., 2 * np.pi, 721, dtype=torch.float64)
    radius = geometry.radius_over_height
    xy = torch.stack((radius * torch.cos(angle), geometry.center_y + radius * torch.sin(angle)), dim=1)
    with torch.no_grad():
        pressure = model(xy).p.abs().numpy()
    return angle.numpy(), pressure


def read_profile(path):
    values = np.genfromtxt(path, delimiter=",", names=True)
    return values["theta_rad"], values["normalized_pressure"]


def metrics(profile, references):
    values = []
    for key in REFERENCES:
        target_theta, target = references[key]
        prediction = periodic_interpolate(*profile, target_theta)
        scores = error_metrics(prediction, target)
        values.extend((100 * scores["relative_l2"], scores["r2"]))
    return values


def check_printed(label, calculated, printed, precision):
    for value, expected, digits in zip(calculated, printed, precision, strict=True):
        if not np.isclose(value, expected, rtol=0., atol=.5 * 10**-digits + 1e-9):
            raise ValueError(f"{label}: calculated {value} differs from printed {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the recomputed metrics as CSV files in this directory.")
    args = parser.parse_args()
    ref = load_benchmark(LEGACY / "figure9_mpm_kan/data/figure9.csv")
    assert tuple(len(ref[name][0]) for name in REFERENCES) == (8, 163)
    row6, row10 = [], []
    for method, printed in TABLE6.items():
        profile = seakan_profile() if method == "seakan" else read_profile(DATA / "profiles" / f"{method}.csv")
        if method == "seakan":
            archived = np.genfromtxt(LEGACY / "figure9_mpm_kan/artifacts/top_gap_augmented/figure9_top_gap_augmented_profiles.csv",
                                     delimiter=",", names=True, dtype=None, encoding="utf-8")
            archived = archived[archived["source"] == "mpm_kan_pinn"]
            if len(archived) != len(profile[0]):
                raise ValueError("The stored and regenerated SeaKAN angular grids differ in size")
            ref_curve = periodic_interpolate(*profile, archived["theta_rad"])
            np.testing.assert_allclose(ref_curve, archived["normalized_pressure"], atol=1e-10, rtol=0.)
        result = metrics(profile, ref)
        check_printed(method, result, printed, [2, 4, 2, 4])
        row6.append((method, *result))
    for variant, printed in TABLE10.items():
        scores = [metrics(read_profile(DATA / "ablation" / variant / f"seed_{seed}" / "profile.csv"), ref)
                  for seed in (202409, 202410, 202411)]
        mean, sd = np.mean(scores, axis=0), np.std(scores, axis=0, ddof=1)
        result = np.stack((mean, sd), axis=1).ravel()
        check_printed(variant, result, printed, [2, 2, 3, 3, 2, 2, 3, 3])
        row10.append((variant, *result))
    for row in row6:
        print(f"{row[0]:22s} Exp L2 {row[1]:8.3f}%  FEM L2 {row[3]:8.3f}%")
    print("Verified six models (Table 6) and six three-seed variants (Table 10).")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        names = ["exp_l2_pct", "exp_r2", "fem_l2_pct", "fem_r2"]
        for fn, head, rows in [("table6", names, row6),
                               ("table10", [f"{n}_{s}" for n in names for s in ("mean", "sd")], row10)]:
            with (args.output / f"{fn}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["model", *head])
                writer.writerows(rows)


if __name__ == "__main__":
    main()
