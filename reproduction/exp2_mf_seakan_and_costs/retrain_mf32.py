"""Train the 32-field MF-SeaKAN configuration, optionally reusing a physics trunk."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from exp2_mf_seakan_and_costs import (
    CleanParametricSeaKAN, PhysicalConstants, ParameterDomain, evaluate_field_list, load_all_comsol_cases,
    train_clean_physics_trunk, train_surrogate_variant,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for this independent run.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--trunk-checkpoint", type=Path,
                        help="Reuse a physics trunk trained in an earlier run; the corrector is initialized anew.")
    parser.add_argument("--corrector-seed", type=int, choices=(202438, 202439, 202440), default=202438)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    constants, domain = PhysicalConstants(), ParameterDomain()
    cases = load_all_comsol_cases(constants)
    test = cases["blind_sealed"] + cases["prospective"]
    if len(cases["train"]) != 32:
        raise ValueError(f"Expected 32 calibration fields, found {len(cases['train'])}.")
    if args.trunk_checkpoint:
        trunk_path = args.trunk_checkpoint
        trunk = CleanParametricSeaKAN(constants, domain).to(device)
        trunk.load_state_dict(torch.load(trunk_path, map_location=device, weights_only=True))
        trunk.eval()
        train_seconds = None
    else:
        # The released full study selected seed 202418 using validation complex-field L2.
        trunk_path = args.output_dir / "trunk.pt"
        trunk, train_seconds, _ = train_clean_physics_trunk(constants, domain, cases["validation"], device, seed=202418)
        torch.save(trunk.state_dict(), trunk_path)
    trunk_test = evaluate_field_list(lambda *x: trunk(*x)[2], test, device)
    model, calibration_seconds = train_surrogate_variant(
        constants, domain, trunk, cases["train"], cases["validation"], device, steps=800, seed=args.corrector_seed)
    torch.save(model.state_dict(), args.output_dir / "mf32.pt")
    result = {
        "trunk_seed": None if args.trunk_checkpoint else 202418,
        "trunk_sha256": hashlib.sha256(trunk_path.read_bytes()).hexdigest(),
        "reused_trunk": bool(args.trunk_checkpoint),
        "corrector_seed": args.corrector_seed, "device": str(device),
        "calibration_cases": [c["case_id"] for c in cases["train"]],
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
