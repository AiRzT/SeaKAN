from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mpm_kan_figure15.config import figure15_setups
from mpm_kan_figure15.coupling import (
    complex_contact_loss,
    contact_diagnostics,
)
from mpm_kan_figure15.model import (
    ContactEnrichedWaterKAN,
    WaterPressureKAN,
)
from mpm_kan_figure15.training import (
    TrainingOptions,
    load_checkpoint,
    make_water_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train compact contact-corner KAN enrichments with water "
            "physics and complex-pressure continuity losses."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--float64-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--contact-weight-start", type=float, default=250.0)
    parser.add_argument("--contact-weight-end", type=float, default=6000.0)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--basis", type=int, default=7)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20241550)
    return parser.parse_args()


def ramp(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return end
    fraction = (step - 1) / (total - 1)
    return float(
        math.exp(
            (1.0 - fraction) * math.log(start)
            + fraction * math.log(end)
        )
    )


def record(
    stage: str,
    step: int,
    started: float,
    weight: float,
    contact: torch.Tensor,
    physics,
    rans,
    interface,
    geometry,
) -> dict[str, object]:
    return {
        "stage": stage,
        "step": step,
        "elapsed_seconds": time.perf_counter() - started,
        "contact_weight": weight,
        "contact_loss": float(contact.detach().cpu()),
        "interface_physics_total": float(
            physics.total.detach().cpu()
        ),
        **contact_diagnostics(
            rans, interface, geometry
        ).as_dict(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Contact enrichment requires a CUDA GPU.")
    if args.angle_scale <= 0.0:
        raise ValueError("angle-scale must be positive.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    models, rans, options, histories, _ = load_checkpoint(
        args.checkpoint, device
    )
    base = models["test49"].water_model
    if not isinstance(base, WaterPressureKAN):
        raise TypeError(
            "Source must contain the un-enriched potential interface."
        )
    wave, _, geometry = figure15_setups()["test49"]
    contact_angle = (
        args.angle_scale
        * float(rans.boundary_layer_thickness)
        / geometry.radius_over_height
    )
    interface = ContactEnrichedWaterKAN(
        base,
        contact_angle=contact_angle,
        width=args.width,
        n_basis=args.basis,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.float32)
    rans = rans.to(dtype=torch.float32)
    for parameter in rans.parameters():
        parameter.requires_grad_(False)
    for parameter in interface.base_model.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter
        for parameter in interface.parameters()
        if parameter.requires_grad
    ]
    initial = contact_diagnostics(rans, interface, geometry)
    print(f"Initial contact: {initial.as_dict()}")
    cloud, problem = make_water_problem(
        wave,
        geometry,
        options,
        device,
        torch.float32,
    )
    optimizer = torch.optim.Adam(
        trainable, lr=args.learning_rate
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = cloud.sample_indices(
            (
                options.water_batch_background,
                options.water_batch_annulus,
            ),
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        physics = problem.loss(interface, indices)
        contact = complex_contact_loss(
            rans, interface, geometry
        )
        weight = ramp(
            args.contact_weight_start,
            args.contact_weight_end,
            step,
            max(args.steps, 1),
        )
        total = physics.total + weight * contact
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        if (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        ):
            row = record(
                "contact_enrichment_float32_adam",
                step,
                started,
                weight,
                contact,
                physics,
                rans,
                interface,
                geometry,
            )
            history.append(row)
            print(json.dumps(row))

    interface = interface.to(dtype=torch.float64)
    rans = rans.to(dtype=torch.float64)
    cloud64, problem64 = make_water_problem(
        wave,
        geometry,
        options,
        device,
        torch.float64,
    )
    indices64 = torch.arange(
        cloud64.n_particles, device=device
    )
    trainable64 = [
        parameter
        for parameter in interface.parameters()
        if parameter.requires_grad
    ]
    optimizer64 = torch.optim.Adam(
        trainable64, lr=0.15 * args.learning_rate
    )
    for step in range(1, args.float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        physics = problem64.loss(interface, indices64)
        contact = complex_contact_loss(
            rans, interface, geometry
        )
        total = (
            physics.total
            + args.contact_weight_end * contact
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable64, 8.0)
        optimizer64.step()
        if (
            step == 1
            or step % 10 == 0
            or step == args.float64_steps
        ):
            row = record(
                "contact_enrichment_float64_full_adam",
                step,
                started,
                args.contact_weight_end,
                contact,
                physics,
                rans,
                interface,
                geometry,
            )
            history.append(row)
            print(json.dumps(row))

    final_physics = problem64.loss(interface, indices64)
    final = contact_diagnostics(rans, interface, geometry)
    option_values = asdict(options)
    option_values["water_coupling_strategy"] = (
        "physics_trained_local_contact_kan"
    )
    released_options = TrainingOptions(**option_values)
    released_histories = dict(histories)
    released_histories["contact_enrichment_training"] = history
    payload = dict(source)
    payload.update(
        {
            "water_model_kind": "harmonic_rans",
            "water_test49": rans.state_dict(),
            "soil_interface_model_kind": (
                "contact_enriched_potential"
            ),
            "soil_interface_water_test49": (
                interface.state_dict()
            ),
            "contact_interface_architecture": {
                "contact_angle_rad": contact_angle,
                "contact_angle_deg": math.degrees(contact_angle),
                "width": args.width,
                "n_basis": args.basis,
                "horizontal_support": (
                    interface.horizontal_support
                ),
                "vertical_support": interface.vertical_support,
            },
            "options": asdict(released_options),
            "histories": released_histories,
            "coupling_strategy": (
                "physics_trained_local_contact_kan"
            ),
            "training_data": (
                "no_figure15_csv; local KAN enrichment trained with "
                "water physics and complex contact continuity"
            ),
            "contact_enrichment_training": {
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
                "initial": initial.as_dict(),
                "final": final.as_dict(),
                "final_interface_physics": (
                    final_physics.detached()
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
        "architecture": payload["contact_interface_architecture"],
        "initial_contact": initial.as_dict(),
        "final_contact": final.as_dict(),
        "final_interface_physics": final_physics.detached(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "contact_enrichment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "contact_enrichment_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
