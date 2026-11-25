"""Recompute evaluation metrics from the released model weights and numerical fields."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exp2_mf_seakan_and_costs import (
    CleanParametricSeaKAN, PressureCorrectorKAN, PhysicalConstants, ParameterDomain,
    evaluate_field_list, generate_design_chart, invert_comsol_case_clean, load_all_comsol_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--inversions", action="store_true", help="Run the published 8-start optimizer for all 10 cases.")
    parser.add_argument("--noise", action="store_true", help="Repeat the two held-out sensor-noise studies.")
    parser.add_argument("--chart", action="store_true", help="Re-evaluate all 1,600 parameter pairs and spatial pressure gradients.")
    parser.add_argument("--report", type=Path, help="Save the computed metrics and comparisons as JSON.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    assets = Path(__file__).resolve().parent
    constants, domain = PhysicalConstants(), ParameterDomain()
    cases = load_all_comsol_cases(constants)
    test = cases["blind_sealed"] + cases["prospective"]
    heterogeneous = cases["heterogeneous_stress"]
    archive = json.loads((assets / "exp2_complete_results.json").read_text(encoding="utf-8"))
    checks = {}

    def compare(name, computed, reference, tolerance):
        maximum = float(np.max(np.abs(np.asarray(computed) - np.asarray(reference))))
        checks[name] = {"max_abs_difference": maximum, "tolerance": tolerance,
                        "pass": bool(np.isfinite(maximum) and maximum <= tolerance)}

    trunk = CleanParametricSeaKAN(constants, domain)
    mf = PressureCorrectorKAN(constants, domain, trunk).to(device)
    mf.load_state_dict(torch.load(assets / "selected_mf32_model.pt", map_location=device, weights_only=True))
    mf.eval()
    report = {"field_metrics": {}, "trunk_seed_test_amplitude_L2_pct": {}, "comparisons": checks}
    field_sets = (
        ("mf32_validation_10cases", cases["validation"], mf.forward_pressure),
        ("mf32_test_8cases", test, mf.forward_pressure),
        ("mf32_het_stress_2cases", heterogeneous, mf.forward_pressure),
        ("physics_trunk_validation_10cases", cases["validation"], lambda *x: mf.base(*x)[2]),
        ("physics_trunk_test_8cases", test, lambda *x: mf.base(*x)[2]),
    )
    for label, group, predict in field_sets:
        result = evaluate_field_list(predict, group, device)
        report["field_metrics"][label] = {k: v for k, v in result.items() if k != "cases"}
        for metric in ("amplitude_l2", "complex_l2", "amplitude_r2", "complex_r2"):
            compare(label + "_" + metric, [r[metric] for r in result["cases"]],
                    [r[metric] for r in archive[label]["cases"]], 0.00002)
        print(f"{label}: amplitude relative L2 = {100 * result['mean_amplitude_l2']:.5f}%, "
              f"R2 = {result['mean_amplitude_r2']:.6f}", flush=True)
    for seed_ref in archive["physics_trunk_3seed"]:
        model = CleanParametricSeaKAN(constants, domain).to(device).eval()
        seed = seed_ref["seed"]
        model.load_state_dict(torch.load(assets / f"clean_trunk_seed_{seed}.pt", map_location=device, weights_only=True))
        v = evaluate_field_list(lambda *x: model(*x)[2], test, device)
        error_pct = 100 * v["mean_amplitude_l2"]
        report["trunk_seed_test_amplitude_L2_pct"][seed] = error_pct
        compare(f"trunk_{seed}_amp_L2_pct", error_pct, seed_ref["test8_amp_l2_pct"], 0.002)
    if args.chart:
        elapsed, chart = generate_design_chart(mf, constants, device)
        with np.load(assets / "design_chart_1600.npz") as expected:
            for key in ("k_grid", "bD_grid", "C_gamma", "C_uplift", "i_max", "p0_pa"):
                compare("chart_" + key, chart[key], expected[key], 0.0002)
        report["chart"] = {"shape": list(chart["i_max"].shape), "elapsed_seconds": elapsed,
                           "min_i_max": float(chart["i_max"].min()), "max_i_max": float(chart["i_max"].max()),
                           "p0_Pa": float(chart["p0_pa"])}
        print("Recomputed the complete 40 x 40 design grid.", flush=True)
    if args.inversions:
        reference = {v["case_id"]: v for v in archive["inverse_comsol_cases"]}
        report["inversions"] = {}
        for case in test + heterogeneous:
            result = invert_comsol_case_clean(mf, case, constants, domain, device)
            name, error = case["case_id"], result["E_k_pct"]
            report["inversions"][name] = {"k_m_per_s": result["k_inv"], "error_pct": error}
            compare("inverse_" + name + "_error_pp", error, reference[name]["E_k_pct"], 0.1)
            print(f"{name}: k = {result['k_inv']:.8g}, error = {error:.4f}%", flush=True)
    if args.noise:
        reference = {(r["case_id"], r["noise_pct"]): r for r in archive["inverse_noise_study"]}
        for case in cases["blind_sealed"]:
            for level in (0.0, 0.01, 0.02, 0.05):
                errors = []
                for rep in range(1 if level == 0 else 5):
                    seed = 8000 + 100 * int(100 * level) + rep
                    row = invert_comsol_case_clean(mf, case, constants, domain, device, noise_rel=level, noise_seed=seed)
                    errors.append(row["E_k_pct"])
                expected = reference[case["case_id"], 100 * level]
                prefix = f"noise_{case['case_id']}_{level:g}"
                compare(prefix + "_mean_pp", np.mean(errors), expected["E_k_pct_mean"], 0.1)
                compare(prefix + "_sd_pp", np.std(errors, ddof=1) if level else 0.0, expected["E_k_pct_std"], 0.1)
    failed = [name for name, result in checks.items() if not result["pass"]]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if failed:
        raise SystemExit("Outside tolerance: " + ", ".join(failed))
    print(f"Passed {len(checks)} archived-result comparisons.")


if __name__ == "__main__":
    main()
