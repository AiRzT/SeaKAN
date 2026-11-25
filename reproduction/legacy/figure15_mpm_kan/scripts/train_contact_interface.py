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
from mpm_kan_figure15.model import WaterPressureKAN
from mpm_kan_figure15.training import (
    TrainingOptions,
    load_checkpoint,
    make_rans_water_problem,
    make_water_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train pressure continuity between the partitioned RANS wall "
            "and potential porous-interface MPM-KAN models without using "
            "Figure 15 observations."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--float64-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--rans-learning-rate-scale", type=float, default=0.0)
    parser.add_argument("--contact-weight-start", type=float, default=250.0)
    parser.add_argument("--contact-weight-end", type=float, default=4000.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20241549)
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Contact training requires a CUDA GPU.")
    if args.steps < 0 or args.float64_steps < 0:
        raise ValueError("Training step counts must be non-negative.")
    if args.contact_weight_start <= 0.0:
        raise ValueError("Contact weights must be positive.")
    if args.contact_weight_end <= 0.0:
        raise ValueError("Contact weights must be positive.")
    if args.rans_learning_rate_scale < 0.0:
        raise ValueError("RANS learning-rate scale cannot be negative.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    models, rans_model, options, histories, _ = load_checkpoint(
        args.checkpoint, device
    )
    interface_model = models["test49"].water_model
    if not isinstance(interface_model, WaterPressureKAN):
        raise TypeError(
            "The source checkpoint has no potential interface model."
        )
    wave, _, geometry = figure15_setups()["test49"]
    initial = contact_diagnostics(
        rans_model, interface_model, geometry
    )
    print(f"Initial contact: {initial.as_dict()}")

    rans_trainable = args.rans_learning_rate_scale > 0.0
    for parameter in interface_model.parameters():
        parameter.requires_grad_(True)
    for parameter in rans_model.parameters():
        parameter.requires_grad_(rans_trainable)

    interface_model = interface_model.to(dtype=torch.float32)
    rans_model = rans_model.to(dtype=torch.float32)
    interface_cloud, interface_problem = make_water_problem(
        wave,
        geometry,
        options,
        device,
        torch.float32,
    )
    rans_cloud = None
    rans_problem = None
    if rans_trainable:
        rans_cloud, rans_problem = make_rans_water_problem(
            wave,
            geometry,
            options,
            device,
            torch.float32,
        )
    parameter_groups: list[dict[str, object]] = [
        {
            "params": list(interface_model.parameters()),
            "lr": args.learning_rate,
        }
    ]
    if rans_trainable:
        parameter_groups.append(
            {
                "params": list(rans_model.parameters()),
                "lr": (
                    args.learning_rate
                    * args.rans_learning_rate_scale
                ),
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        interface_indices = interface_cloud.sample_indices(
            (
                options.water_batch_background,
                options.water_batch_annulus,
            ),
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        interface_loss = interface_problem.loss(
            interface_model, interface_indices
        )
        contact = complex_contact_loss(
            rans_model, interface_model, geometry
        )
        weight = ramp(
            args.contact_weight_start,
            args.contact_weight_end,
            step,
            max(args.steps, 1),
        )
        total = interface_loss.total + weight * contact
        rans_loss = None
        if rans_trainable:
            assert rans_cloud is not None
            assert rans_problem is not None
            rans_indices = rans_cloud.sample_indices(
                (900, 560, 800), generator
            )
            rans_loss = rans_problem.loss(
                rans_model, rans_indices
            )
            total = total + rans_loss.total
        total.backward()
        parameters = list(interface_model.parameters())
        if rans_trainable:
            parameters.extend(rans_model.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, 12.0)
        optimizer.step()
        if (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        ):
            diagnostics = contact_diagnostics(
                rans_model, interface_model, geometry
            )
            record: dict[str, object] = {
                "stage": "contact_float32_adam",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "contact_weight": weight,
                "contact_loss": float(contact.detach().cpu()),
                "interface_physics_total": float(
                    interface_loss.total.detach().cpu()
                ),
                **diagnostics.as_dict(),
            }
            if rans_loss is not None:
                record["rans_physics_total"] = float(
                    rans_loss.total.detach().cpu()
                )
            history.append(record)
            print(json.dumps(record))

    interface_model = interface_model.to(dtype=torch.float64)
    rans_model = rans_model.to(dtype=torch.float64)
    interface_cloud64, interface_problem64 = make_water_problem(
        wave,
        geometry,
        options,
        device,
        torch.float64,
    )
    interface_indices64 = torch.arange(
        interface_cloud64.n_particles, device=device
    )
    rans_indices64 = None
    rans_problem64 = None
    if rans_trainable:
        rans_cloud64, rans_problem64 = make_rans_water_problem(
            wave,
            geometry,
            options,
            device,
            torch.float64,
        )
        rans_indices64 = torch.arange(
            rans_cloud64.n_particles, device=device
        )
    parameter_groups64: list[dict[str, object]] = [
        {
            "params": list(interface_model.parameters()),
            "lr": 0.15 * args.learning_rate,
        }
    ]
    if rans_trainable:
        parameter_groups64.append(
            {
                "params": list(rans_model.parameters()),
                "lr": (
                    0.15
                    * args.learning_rate
                    * args.rans_learning_rate_scale
                ),
            }
        )
    optimizer64 = torch.optim.Adam(parameter_groups64)
    for step in range(1, args.float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        interface_loss = interface_problem64.loss(
            interface_model, interface_indices64
        )
        contact = complex_contact_loss(
            rans_model, interface_model, geometry
        )
        total = (
            interface_loss.total
            + args.contact_weight_end * contact
        )
        rans_loss = None
        if rans_trainable:
            assert rans_problem64 is not None
            assert rans_indices64 is not None
            rans_loss = rans_problem64.loss(
                rans_model, rans_indices64
            )
            total = total + rans_loss.total
        total.backward()
        parameters = list(interface_model.parameters())
        if rans_trainable:
            parameters.extend(rans_model.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, 8.0)
        optimizer64.step()
        if (
            step == 1
            or step % 10 == 0
            or step == args.float64_steps
        ):
            diagnostics = contact_diagnostics(
                rans_model, interface_model, geometry
            )
            record = {
                "stage": "contact_float64_full_adam",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "contact_weight": args.contact_weight_end,
                "contact_loss": float(contact.detach().cpu()),
                "interface_physics_total": float(
                    interface_loss.total.detach().cpu()
                ),
                **diagnostics.as_dict(),
            }
            if rans_loss is not None:
                record["rans_physics_total"] = float(
                    rans_loss.total.detach().cpu()
                )
            history.append(record)
            print(json.dumps(record))

    final_interface_loss = interface_problem64.loss(
        interface_model, interface_indices64
    )
    final_contact = contact_diagnostics(
        rans_model, interface_model, geometry
    )
    option_values = asdict(options)
    option_values["water_coupling_strategy"] = (
        "trained_complex_contact_partitioned_rans_potential"
    )
    released_options = TrainingOptions(**option_values)
    released_histories = dict(histories)
    released_histories["contact_interface_training"] = history
    payload = dict(source)
    payload.update(
        {
            "water_model_kind": "harmonic_rans",
            "water_test49": rans_model.state_dict(),
            "soil_interface_water_test49": (
                interface_model.state_dict()
            ),
            "options": asdict(released_options),
            "histories": released_histories,
            "coupling_strategy": (
                "trained_complex_contact_partitioned_rans_potential"
            ),
            "training_data": (
                "no_figure15_csv; strong_water_physics_plus_"
                "complex_contact_pressure_continuity"
            ),
            "contact_training": {
                "source_checkpoint": str(
                    args.checkpoint.resolve()
                ),
                "figure15_data_used": False,
                "constraint": (
                    "complex RANS/potential pressure equality at both "
                    "water-soil-pipe contact corners"
                ),
                "arguments": vars(args) | {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "output": str(args.output.resolve()),
                },
                "initial": initial.as_dict(),
                "final": final_contact.as_dict(),
                "interface_final_physics": (
                    final_interface_loss.detached()
                ),
            },
        }
    )
    output_checkpoint = checkpoint_dir / "figure15.pt"
    torch.save(payload, output_checkpoint)
    summary = {
        "checkpoint": str(output_checkpoint.resolve()),
        "device": torch.cuda.get_device_name(device),
        "training_data": "physics_only_no_figure15_csv",
        "initial_contact": initial.as_dict(),
        "final_contact": final_contact.as_dict(),
        "interface_final_physics": (
            final_interface_loss.detached()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "contact_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "contact_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
