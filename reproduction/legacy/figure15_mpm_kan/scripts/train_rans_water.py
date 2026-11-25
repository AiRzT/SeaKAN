from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from mpm_kan_figure15.config import figure15_setups
from mpm_kan_figure15.mpm import create_rans_water_cloud
from mpm_kan_figure15.rans import (
    HarmonicRANSKAN,
    HarmonicRANSMPMProblem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the no-label harmonic RANS MPM-KAN water model."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--float64-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--effective-viscosity", type=float, default=5.0e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--basis", type=int, default=9)
    parser.add_argument("--continuity-weight", type=float, default=8.0)
    parser.add_argument("--free-surface-weight", type=float, default=80.0)
    parser.add_argument("--wall-momentum-weight", type=float, default=0.5)
    parser.add_argument("--far-field-weight", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=202449)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def make_problem(
    device: torch.device,
    dtype: torch.dtype,
    effective_viscosity: float,
    continuity_weight: float,
    free_surface_weight: float,
    wall_momentum_weight: float,
    far_field_weight: float,
):
    wave, _, geometry = figure15_setups()["test49"]
    cloud = create_rans_water_cloud(geometry, device, dtype)
    problem = HarmonicRANSMPMProblem(
        wave,
        geometry,
        cloud,
        effective_kinematic_viscosity=effective_viscosity,
        continuity_weight=continuity_weight,
        free_surface_weight=free_surface_weight,
        wall_momentum_weight=wall_momentum_weight,
        far_field_weight=far_field_weight,
    )
    return wave, geometry, cloud, problem


def row(stage: str, step: int, loss, elapsed: float) -> dict[str, object]:
    return {
        "stage": stage,
        "step": step,
        "elapsed_seconds": elapsed,
        **loss.detached(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This optimization run requires a CUDA GPU.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    (
        wave,
        geometry,
        cloud32,
        problem32,
    ) = make_problem(
        device,
        torch.float32,
        args.effective_viscosity,
        args.continuity_weight,
        args.free_surface_weight,
        args.wall_momentum_weight,
        args.far_field_weight,
    )
    model = HarmonicRANSKAN(
        wave,
        geometry,
        effective_kinematic_viscosity=args.effective_viscosity,
        width=args.width,
        hidden_layers=args.hidden_layers,
        n_basis=args.basis,
        dtype=torch.float32,
    ).to(device)
    history: list[dict[str, object]] = []
    if args.initial_checkpoint is not None:
        initial = torch.load(
            args.initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(initial["model"])
        history.extend(initial.get("history", []))
        print(
            "Continuing physics-only RANS checkpoint "
            f"{args.initial_checkpoint.resolve()}"
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    start = time.perf_counter()
    requested = (900, 560, 800)
    for step in range(1, args.steps + 1):
        indices = cloud32.sample_indices(requested, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = problem32.loss(model, indices)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()
        if (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        ):
            record = row(
                "rans_float32_adam",
                step,
                loss,
                time.perf_counter() - start,
            )
            history.append(record)
            print(
                f"RANS {step:4d}/{args.steps}: "
                f"total={record['total']:.3e}, "
                f"mom={record['particle_momentum']:.3e}, "
                f"div={record['particle_continuity']:.3e}, "
                f"pipe={record['pipe_no_slip']:.3e}, "
                f"wall={record['wall_momentum']:.3e}"
            )

    model = model.to(dtype=torch.float64)
    _, _, cloud64, problem64 = make_problem(
        device,
        torch.float64,
        args.effective_viscosity,
        args.continuity_weight,
        args.free_surface_weight,
        args.wall_momentum_weight,
        args.far_field_weight,
    )
    indices64 = torch.arange(cloud64.n_particles, device=device)
    optimizer64 = torch.optim.Adam(
        model.parameters(), lr=0.15 * args.learning_rate
    )
    for step in range(1, args.float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        loss = problem64.loss(model, indices64)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer64.step()
        if (
            step == 1
            or step % 10 == 0
            or step == args.float64_steps
        ):
            record = row(
                "rans_float64_full_adam",
                step,
                loss,
                time.perf_counter() - start,
            )
            history.append(record)
            print(
                f"RANS float64 {step:3d}/{args.float64_steps}: "
                f"total={record['total']:.3e}"
            )

    final = problem64.loss(model, indices64)
    history.append(
        row(
            "rans_float64_final",
            0,
            final,
            time.perf_counter() - start,
        )
    )
    checkpoint = args.output / "water_test49_rans.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": {
                "width": args.width,
                "hidden_layers": args.hidden_layers,
                "n_basis": args.basis,
                "hard_no_slip": True,
            },
            "effective_kinematic_viscosity": (
                args.effective_viscosity
            ),
            "history": history,
            "cloud": cloud64.summary(),
            "training_data": "physics_only_no_figure15_csv",
            "loss_weights": {
                "continuity": args.continuity_weight,
                "free_surface": args.free_surface_weight,
                "wall_momentum": args.wall_momentum_weight,
                "far_field": args.far_field_weight,
            },
            "initialized_from": (
                str(args.initial_checkpoint.resolve())
                if args.initial_checkpoint is not None
                else None
            ),
            "physics": (
                "strong harmonic incompressible RANS with constant "
                "eddy-viscosity closure"
            ),
            "device": torch.cuda.get_device_name(device),
        },
        checkpoint,
    )
    theta = torch.linspace(
        0.0,
        math.pi,
        721,
        dtype=torch.float64,
        device=device,
    )
    radius = geometry.radius_over_height
    pipe = torch.stack(
        (
            radius * torch.cos(theta),
            geometry.pipe_center_y + radius * torch.sin(theta),
        ),
        dim=1,
    )
    with torch.no_grad():
        pressure = model(pipe).q.abs().cpu().numpy()
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "training_data": "physics_only_no_figure15_csv",
        "cuda_device": torch.cuda.get_device_name(device),
        "effective_kinematic_viscosity": args.effective_viscosity,
        "pressure_min": float(pressure.min()),
        "pressure_max": float(pressure.max()),
        "pressure_mean": float(pressure.mean()),
        "cloud": cloud64.summary(),
        "final_physics_loss": final.detached(),
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
