from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import (
    DeepXDEFNN,
    PhysicsConstants,
    TorchMLP,
    abs2,
    boundary_loss,
    complex_gradient,
    hard_fields,
    parameter_count,
    pipe_profile,
    sample_interior,
    strong_residuals,
)


@dataclass
class PointwiseOptions:
    steps: int = 1200
    batch_size: int = 1536
    learning_rate: float = 1.0e-3
    width: int = 64
    depth: int = 4
    boundary_points: int = 96
    log_every: int = 50
    rar_every: int = 200
    rar_pool: int = 4096
    rar_add: int = 256
    gpinn_points: int = 192
    seed: int = 202409


class PiecewiseXPINN(nn.Module):
    def __init__(self, width: int, depth: int, split_normalized: float):
        super().__init__()
        self.nets = nn.ModuleList([TorchMLP(width, depth) for _ in range(3)])
        self.split_normalized = split_normalized

    def forward(self, normalized_xy: torch.Tensor) -> torch.Tensor:
        x = normalized_xy[:, :1]
        left = self.nets[0](normalized_xy)
        center = self.nets[1](normalized_xy)
        right = self.nets[2](normalized_xy)
        return torch.where(
            x < -self.split_normalized,
            left,
            torch.where(x > self.split_normalized, right, center),
        )


def _physics_loss(
    model: nn.Module,
    xy: torch.Tensor,
    constants: PhysicsConstants,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    xy = xy.detach().requires_grad_(True)
    residuals = strong_residuals(model, xy, constants)
    loss = abs2(residuals[0]).mean() + abs2(residuals[1]).mean()
    loss = loss + 20.0 * abs2(residuals[2]).mean()
    return loss, residuals


def _gradient_residual_loss(
    residuals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    xy: torch.Tensor,
) -> torch.Tensor:
    loss = torch.zeros((), device=xy.device, dtype=xy.dtype)
    for index, residual in enumerate(residuals):
        gradient = complex_gradient(residual, xy)
        weight = 20.0 if index == 2 else 1.0
        loss = loss + weight * abs2(gradient).mean()
    return loss


def _rar_select(
    model: nn.Module,
    constants: PhysicsConstants,
    n_pool: int,
    n_add: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pool = sample_interior(n_pool, constants, device, dtype)
    scores: list[torch.Tensor] = []
    for chunk in pool.split(512):
        points = chunk.detach().requires_grad_(True)
        residuals = strong_residuals(model, points, constants)
        score = abs2(residuals[0]) + abs2(residuals[1]) + 20.0 * abs2(residuals[2])
        scores.append(score.detach())
    score = torch.cat(scores)
    indices = torch.topk(score, min(n_add, score.numel()), largest=True).indices
    return pool[indices].detach()


def _interface_loss(
    model: PiecewiseXPINN,
    constants: PhysicsConstants,
    n: int = 128,
) -> torch.Tensor:
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    split = model.split_normalized * constants.half_width
    losses: list[torch.Tensor] = []
    for x_value, left_index, right_index in (
        (-split, 0, 1),
        (split, 1, 2),
    ):
        y = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
        x = torch.full_like(y, x_value)
        outside = (
            x.square() + (y - constants.center_y).square()
            > (1.003 * constants.radius) ** 2
        )
        xy = torch.stack((x[outside], y[outside]), dim=1).requires_grad_(True)
        left_fields = hard_fields(model.nets[left_index], xy, constants)
        right_fields = hard_fields(model.nets[right_index], xy, constants)
        continuity = torch.zeros((), device=device, dtype=dtype)
        flux = torch.zeros((), device=device, dtype=dtype)
        for left_value, right_value in zip(left_fields, right_fields):
            continuity = continuity + abs2(left_value - right_value).mean()
            left_gradient = complex_gradient(left_value, xy)
            right_gradient = complex_gradient(right_value, xy)
            flux = flux + abs2(left_gradient[:, 0] - right_gradient[:, 0]).mean()
        losses.append(continuity + 0.2 * flux)
    return sum(losses)


def train_pointwise(
    method: str,
    constants: PhysicsConstants,
    output_dir: Path,
    *,
    options: PointwiseOptions,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    if method not in {"deepxde_pinn", "gpinn_rar", "xpinn"}:
        raise ValueError(f"Unknown pointwise method: {method}")
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)

    if method == "deepxde_pinn":
        model: nn.Module = DeepXDEFNN(options.width, options.depth)
    elif method == "gpinn_rar":
        model = TorchMLP(options.width, options.depth)
    else:
        split_normalized = 0.35 / constants.half_width
        model = PiecewiseXPINN(40, options.depth, split_normalized)
    model = model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=options.steps, eta_min=options.learning_rate * 0.05
    )
    generator = torch.Generator(device=device).manual_seed(options.seed + 17)
    rar_points = torch.empty((0, 2), device=device, dtype=dtype)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for step in range(1, options.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        xy = sample_interior(
            options.batch_size, constants, device, dtype, generator=generator
        )
        if method == "gpinn_rar" and rar_points.numel():
            n_adaptive = min(rar_points.shape[0], options.batch_size // 3)
            selection = torch.randint(
                rar_points.shape[0], (n_adaptive,), device=device, generator=generator
            )
            xy[:n_adaptive] = rar_points[selection]
        xy = xy.detach().requires_grad_(True)
        residuals = strong_residuals(model, xy, constants)
        physics = (
            abs2(residuals[0]).mean()
            + abs2(residuals[1]).mean()
            + 20.0 * abs2(residuals[2]).mean()
        )
        bc, bc_parts = boundary_loss(model, constants, options.boundary_points)
        gradient = torch.zeros((), device=device, dtype=dtype)
        interface = torch.zeros((), device=device, dtype=dtype)
        if method == "gpinn_rar":
            gp_xy = xy[: options.gpinn_points].detach().clone().requires_grad_(True)
            gp_residuals = strong_residuals(model, gp_xy, constants)
            gradient = _gradient_residual_loss(gp_residuals, gp_xy)
        elif method == "xpinn":
            interface = _interface_loss(model, constants)
        total = physics + bc + 0.02 * gradient + 20.0 * interface
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        scheduler.step()

        if (
            method == "gpinn_rar"
            and step % options.rar_every == 0
            and step < options.steps
        ):
            selected = _rar_select(
                model,
                constants,
                options.rar_pool,
                options.rar_add,
                device,
                dtype,
            )
            rar_points = torch.cat((rar_points, selected), dim=0)[-2048:]

        if step == 1 or step % options.log_every == 0 or step == options.steps:
            row = {
                "step": step,
                "total": float(total.detach().cpu()),
                "physics": float(physics.detach().cpu()),
                "boundary": float(bc.detach().cpu()),
                "gradient": float(gradient.detach().cpu()),
                "interface": float(interface.detach().cpu()),
                "top_traction": float(bc_parts["top_traction"].detach().cpu()),
                "outer_flux": float(bc_parts["outer_flux"].detach().cpu()),
                "pipe_flux": float(bc_parts["pipe_flux"].detach().cpu()),
                "rar_points": int(rar_points.shape[0]),
            }
            history.append(row)
            print(
                f"{method} {step:4d}/{options.steps}: "
                f"loss={row['total']:.3e}, pde={row['physics']:.3e}, "
                f"bc={row['boundary']:.3e}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    theta, pressure = pipe_profile(model, constants)
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        method_dir / "profile.csv",
        np.column_stack((theta, pressure)),
        delimiter=",",
        header="theta_rad,normalized_pressure",
        comments="",
    )
    (method_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "method": method,
            "options": asdict(options),
            "elapsed_seconds": elapsed,
            "parameters": parameter_count(model),
            "device": str(device),
            "dtype": str(dtype),
        },
        method_dir / "checkpoint.pt",
    )
    result = {
        "method": method,
        "elapsed_seconds": elapsed,
        "parameters": parameter_count(model),
        "device": str(device),
        "dtype": str(dtype),
        "steps": options.steps,
        "seed": options.seed,
        "target_data_used_for_training": False,
        "profile": str(method_dir / "profile.csv"),
    }
    (method_dir / "run_metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def refine_pointwise_lbfgs(
    method: str,
    constants: PhysicsConstants,
    output_dir: Path,
    *,
    device: torch.device,
    max_iter: int = 120,
    interior_points: int = 3072,
) -> dict[str, object]:
    """Apply the same deterministic full-batch L-BFGS tail to each PINN."""
    method_dir = output_dir / method
    state = torch.load(method_dir / "checkpoint.pt", map_location=device, weights_only=False)
    options = PointwiseOptions(**state["options"])
    if method == "deepxde_pinn":
        model: nn.Module = DeepXDEFNN(options.width, options.depth)
    elif method == "gpinn_rar":
        model = TorchMLP(options.width, options.depth)
    elif method == "xpinn":
        model = PiecewiseXPINN(40, options.depth, 0.35 / constants.half_width)
    else:
        raise ValueError(method)
    model = model.to(device=device, dtype=torch.float64)
    model.load_state_dict(state["model"])
    torch.manual_seed(options.seed + 991)
    xy = sample_interior(
        interior_points, constants, device, torch.float64
    ).detach()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0
    started = time.perf_counter()
    last: dict[str, float] = {}

    def closure() -> torch.Tensor:
        nonlocal evaluations, last
        optimizer.zero_grad(set_to_none=True)
        physics, _ = _physics_loss(model, xy, constants)
        bc, _ = boundary_loss(model, constants, options.boundary_points)
        interface = (
            _interface_loss(model, constants) if method == "xpinn" else torch.zeros_like(physics)
        )
        total = physics + bc + 20.0 * interface
        total.backward()
        evaluations += 1
        last = {
            "total": float(total.detach().cpu()),
            "physics": float(physics.detach().cpu()),
            "boundary": float(bc.detach().cpu()),
            "interface": float(interface.detach().cpu()),
        }
        if evaluations == 1 or evaluations % 20 == 0:
            print(
                f"{method} LBFGS eval {evaluations:4d}: "
                f"loss={last['total']:.3e}, pde={last['physics']:.3e}, "
                f"bc={last['boundary']:.3e}",
                flush=True,
            )
        return total

    optimizer.step(closure)
    elapsed = time.perf_counter() - started
    theta, pressure = pipe_profile(model, constants)
    np.savetxt(
        method_dir / "profile.csv",
        np.column_stack((theta, pressure)),
        delimiter=",",
        header="theta_rad,normalized_pressure",
        comments="",
    )
    state.update(
        {
            "model": model.state_dict(),
            "lbfgs_steps": max_iter,
            "lbfgs_evaluations": evaluations,
            "lbfgs_elapsed_seconds": elapsed,
            "lbfgs_final": last,
            "dtype": "torch.float64",
        }
    )
    torch.save(state, method_dir / "checkpoint.pt")
    metadata_path = method_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous_lbfgs = int(
        metadata.get("lbfgs_steps_total", metadata.get("lbfgs_steps", 0))
    )
    metadata["lbfgs_steps"] = max_iter
    metadata["lbfgs_steps_total"] = previous_lbfgs + max_iter
    metadata["lbfgs_evaluations"] = evaluations
    metadata["lbfgs_elapsed_seconds"] = elapsed
    metadata["elapsed_seconds"] = float(metadata["elapsed_seconds"]) + elapsed
    metadata["dtype_finish"] = "torch.float64"
    metadata["lbfgs_final"] = last
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
