"""Train the selected Table 7 configuration from scratch without loading any weights."""
import argparse
import json
from pathlib import Path

import torch

from exp2_mf_seakan_and_costs import (
    PhysicalConstants, ParameterDomain, evaluate_field_list, load_all_comsol_cases,
    train_clean_physics_trunk, train_surrogate_variant,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for this independent run.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    constants, domain = PhysicalConstants(), ParameterDomain()
    cases = load_all_comsol_cases(constants)
    test = cases["blind_sealed"] + cases["prospective"]
    # The released full study selected this trunk using validation complex-field L2.
    trunk, train_seconds, _ = train_clean_physics_trunk(constants, domain, cases["validation"], device, seed=202418)
    torch.save(trunk.state_dict(), args.output_dir / "trunk.pt")
    trunk_test = evaluate_field_list(lambda *x: trunk(*x)[2], test, device)
    model, calibration_seconds = train_surrogate_variant(
        constants, domain, trunk, cases["train"], cases["validation"], device, steps=800, seed=202438)
    torch.save(model.state_dict(), args.output_dir / "mf32.pt")
    result = {
        "trunk_seed": 202418, "corrector_seed": 202438, "device": str(device),
        "physics_trunk_seconds": train_seconds, "calibration_seconds": calibration_seconds,
        "trunk_test": trunk_test,
        "mf32_validation": evaluate_field_list(model.forward_pressure, cases["validation"], device),
        "mf32_test": evaluate_field_list(model.forward_pressure, test, device),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Eight-case amplitude L2: trunk %.5f%%; MF32 %.5f%%" % (
        100 * result["trunk_test"]["mean_amplitude_l2"], 100 * result["mf32_test"]["mean_amplitude_l2"]))


if __name__ == "__main__":
    main()
