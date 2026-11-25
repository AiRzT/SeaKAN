from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mpm_kan_figure15.config import figure15_setups
from mpm_kan_figure15.coupling import contact_diagnostics
from mpm_kan_figure15.model import WaterPressureKAN
from mpm_kan_figure15.rans import HarmonicRANSKAN
from mpm_kan_figure15.training import (
    TrainingOptions,
    _train_soil_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the Test 49 strong Biot MPM-KAN solve after "
            "physics-only contact-interface training."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soil-steps", type=int, default=300)
    parser.add_argument("--soil-float64-steps", type=int, default=80)
    parser.add_argument("--soil-lbfgs-steps", type=int, default=20)
    parser.add_argument("--soil-learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--soil-float64-learning-rate",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument("--soil-local-steps", type=int, default=0)
    parser.add_argument(
        "--soil-local-learning-rate", type=float, default=5.0e-5
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Soil continuation requires a CUDA GPU.")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    source = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if "soil_interface_water_test49" not in source:
        raise ValueError(
            "Checkpoint has no trained potential interface model."
        )
    source_options = TrainingOptions(**source["options"])
    option_values = asdict(source_options)
    option_values.update(
        {
            "water_model_kind": "potential",
            "soil_steps": args.soil_steps,
            "soil_float64_steps": args.soil_float64_steps,
            "soil_lbfgs_steps": args.soil_lbfgs_steps,
            "soil_learning_rate": args.soil_learning_rate,
            "soil_float64_learning_rate": (
                args.soil_float64_learning_rate
            ),
            "soil_local_steps": args.soil_local_steps,
            "soil_local_learning_rate": (
                args.soil_local_learning_rate
            ),
            "teacher_steps": 0,
            "teacher_anchor_steps": 0,
            "log_every": 50,
        }
    )
    training_options = TrainingOptions(**option_values)
    wave, soil, geometry = figure15_setups()["test49"]
    (
        soil_model,
        interface_model,
        soil_history,
        augmented,
        cloud,
    ) = _train_soil_case(
        "test49",
        wave,
        soil,
        geometry,
        training_options,
        device,
        args.output,
        source["soil_interface_water_test49"],
        source["soil_test49"],
        source["augmented"].get("test49"),
    )
    if not isinstance(interface_model, WaterPressureKAN):
        raise TypeError("Expected a potential interface model.")

    rans_model = HarmonicRANSKAN(
        wave,
        geometry,
        effective_kinematic_viscosity=(
            source_options.rans_effective_kinematic_viscosity
        ),
        width=source_options.rans_water_width,
        hidden_layers=source_options.rans_water_hidden_layers,
        n_basis=source_options.kan_basis,
        dtype=torch.float64,
    ).to(device)
    rans_model.load_state_dict(source["water_test49"])
    diagnostics = contact_diagnostics(
        rans_model, interface_model, geometry
    )

    released_values = asdict(source_options)
    released_values["water_model_kind"] = "harmonic_rans"
    released_values["water_coupling_strategy"] = (
        "trained_complex_contact_partitioned_rans_potential"
    )
    released_options = TrainingOptions(**released_values)
    histories = dict(source.get("histories", {}))
    histories["soil_test49_contact_continuation"] = soil_history
    augmented_states = dict(source["augmented"])
    augmented_states["test49"] = augmented.as_dict()
    clouds = dict(source.get("clouds", {}))
    clouds["soil_test49_contact_continuation"] = cloud
    payload = dict(source)
    payload.update(
        {
            "water_model_kind": "harmonic_rans",
            "water_test49": rans_model.state_dict(),
            "soil_interface_water_test49": (
                interface_model.state_dict()
            ),
            "soil_test49": soil_model.state_dict(),
            "options": asdict(released_options),
            "histories": histories,
            "augmented": augmented_states,
            "clouds": clouds,
            "coupling_strategy": (
                "trained_complex_contact_partitioned_rans_potential"
            ),
            "training_data": (
                "no_figure15_csv; contact-coupled water physics and "
                "strong Biot MPM-KAN physics only"
            ),
            "soil_contact_continuation": {
                "source_checkpoint": str(
                    args.checkpoint.resolve()
                ),
                "figure15_data_used": False,
                "arguments": {
                    name: (
                        str(value.resolve())
                        if isinstance(value, Path)
                        else value
                    )
                    for name, value in vars(args).items()
                },
                "contact_after_soil_training": (
                    diagnostics.as_dict()
                ),
            },
        }
    )
    checkpoint = args.output / "checkpoints" / "figure15.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "device": torch.cuda.get_device_name(device),
        "training_data": "physics_only_no_figure15_csv",
        "contact": diagnostics.as_dict(),
        "soil_final": soil_history[-1],
        "cloud": cloud,
    }
    (args.output / "soil_contact_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
