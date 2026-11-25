from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mpm_kan_figure15.config import figure15_setups
from mpm_kan_figure15.training import (
    TrainingOptions,
    _train_soil_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach a no-label harmonic RANS checkpoint and continue the "
            "Test 49 Biot MPM-KAN solve."
        )
    )
    parser.add_argument("--water-checkpoint", type=Path, required=True)
    parser.add_argument("--soil-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soil-steps", type=int, default=300)
    parser.add_argument("--soil-float64-steps", type=int, default=80)
    parser.add_argument("--soil-lbfgs-steps", type=int, default=20)
    parser.add_argument("--soil-learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--soil-float64-learning-rate", type=float, default=1.0e-5
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This continuation requires a CUDA GPU.")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    old = torch.load(
        args.soil_checkpoint,
        map_location=device,
        weights_only=False,
    )
    rans = torch.load(
        args.water_checkpoint,
        map_location=device,
        weights_only=False,
    )
    if rans.get("training_data") != "physics_only_no_figure15_csv":
        raise ValueError("RANS checkpoint provenance is not physics-only")
    architecture = rans["architecture"]
    option_values = dict(old["options"])
    option_values.update(
        {
            "water_model_kind": "harmonic_rans",
            "rans_water_width": int(architecture["width"]),
            "rans_water_hidden_layers": int(
                architecture["hidden_layers"]
            ),
            "rans_effective_kinematic_viscosity": float(
                rans["effective_kinematic_viscosity"]
            ),
            "soil_steps": args.soil_steps,
            "soil_learning_rate": args.soil_learning_rate,
            "soil_float64_steps": args.soil_float64_steps,
            "soil_float64_learning_rate": (
                args.soil_float64_learning_rate
            ),
            "soil_lbfgs_steps": args.soil_lbfgs_steps,
            "teacher_steps": 0,
            "teacher_anchor_steps": 0,
            "soil_local_steps": 0,
            "log_every": 50,
        }
    )
    options = TrainingOptions(**option_values)
    wave, soil, geometry = figure15_setups()["test49"]
    (
        model49,
        water49,
        history49,
        augmented49,
        cloud49,
    ) = _train_soil_case(
        "test49",
        wave,
        soil,
        geometry,
        options,
        device,
        args.output,
        rans["model"],
        old["soil_test49"],
        old["augmented"].get("test49"),
    )
    if water49 is None:
        raise RuntimeError("RANS water was not attached to Test 49 soil")
    histories = dict(old.get("histories", {}))
    histories["water_test49"] = rans.get("history", [])
    histories["soil_test49"] = history49
    augmented = dict(old["augmented"])
    augmented["test49"] = augmented49.as_dict()
    clouds = dict(old.get("clouds", {}))
    clouds["water_test49"] = rans["cloud"]
    clouds["soil_test49"] = cloud49
    checkpoint = checkpoint_dir / "figure15.pt"
    torch.save(
        {
            "water_model_kind": "harmonic_rans",
            "water_test49": water49.state_dict(),
            "soil_test10": old["soil_test10"],
            "soil_test49": model49.state_dict(),
            "options": asdict(options),
            "histories": histories,
            "augmented": augmented,
            "clouds": clouds,
            "coefficients": old.get("coefficients", {}),
            "training_data": (
                "no_figure15_csv; harmonic_RANS_and_strong_Biot_"
                "MPM-KAN_physics_losses_only"
            ),
            "water_training": {
                "source_checkpoint": str(
                    args.water_checkpoint.resolve()
                ),
                "training_data": rans["training_data"],
                "physics": rans["physics"],
                "effective_kinematic_viscosity": rans[
                    "effective_kinematic_viscosity"
                ],
                "hard_no_slip": architecture["hard_no_slip"],
                "cuda_device": rans["device"],
            },
            "soil_continuation": {
                "source_checkpoint": str(
                    args.soil_checkpoint.resolve()
                ),
                "training_data": (
                    "physics_only_no_figure15_csv"
                ),
            },
        },
        checkpoint,
    )
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "cuda_device": torch.cuda.get_device_name(device),
        "water_model_kind": "harmonic_rans",
        "training_data": "physics_only_no_figure15_csv",
        "soil_test49_final": history49[-1],
        "clouds": {
            "water_test49": rans["cloud"],
            "soil_test49": cloud49,
        },
    }
    (args.output / "coupling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
