from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mpm_kan_figure15.training import TrainingOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the partitioned Test 49 wall/interface checkpoint "
            "without opening Figure 15 validation data."
        )
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--rans-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    rans = torch.load(
        args.rans_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if rans.get("training_data") != "physics_only_no_figure15_csv":
        raise ValueError("The RANS checkpoint is not physics-only")
    architecture = rans["architecture"]
    option_values = dict(base["options"])
    retained_coupling = option_values.get(
        "water_coupling_strategy",
        "partitioned_rans_wall_potential_porous_interface",
    )
    option_values.update(
        {
            "water_model_kind": "harmonic_rans",
            "water_coupling_strategy": retained_coupling,
            "rans_water_width": int(architecture["width"]),
            "rans_water_hidden_layers": int(
                architecture["hidden_layers"]
            ),
            "rans_effective_kinematic_viscosity": float(
                rans["effective_kinematic_viscosity"]
            ),
            "rans_continuity_weight": float(
                rans.get("loss_weights", {}).get("continuity", 8.0)
            ),
            "rans_free_surface_weight": float(
                rans.get("loss_weights", {}).get("free_surface", 80.0)
            ),
            "rans_wall_momentum_weight": float(
                rans.get("loss_weights", {}).get("wall_momentum", 0.5)
            ),
            "rans_far_field_weight": float(
                rans.get("loss_weights", {}).get("far_field", 1200.0)
            ),
        }
    )
    options = TrainingOptions(**option_values)
    histories = dict(base.get("histories", {}))
    histories["water_test49_rans_wall"] = rans.get("history", [])
    clouds = dict(base.get("clouds", {}))
    clouds["water_test49_rans_wall"] = rans["cloud"]
    interface_state = base.get(
        "soil_interface_water_test49", base["water_test49"]
    )
    payload = {
        **base,
        "water_model_kind": "harmonic_rans",
        "coupling_strategy": base.get(
            "coupling_strategy",
            "partitioned_rans_wall_potential_porous_interface",
        ),
        "water_test49": rans["model"],
        "soil_interface_water_test49": interface_state,
        "options": asdict(options),
        "histories": histories,
        "clouds": clouds,
        "training_data": (
            "no_figure15_csv; physics-only harmonic-RANS wall "
            "refinement with the existing porous-interface model"
        ),
        "water_training": {
            **base.get("water_training", {}),
            "wall_pressure_checkpoint": str(
                args.rans_checkpoint.resolve()
            ),
            "porous_interface_checkpoint": str(
                args.base_checkpoint.resolve()
            ),
            "wall_training_data": rans["training_data"],
            "interface_training_data": base["training_data"],
            "physics": rans["physics"],
            "effective_kinematic_viscosity": rans[
                "effective_kinematic_viscosity"
            ],
            "hard_no_slip": architecture["hard_no_slip"],
            "cuda_device": rans["device"],
            "loss_weights": rans.get("loss_weights", {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
