from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Fig7Case, SoilParameters, WaveParameters
from .model import HarmonicKAN
from .mpm import MaterialPointGrid1D
from .physics import MPMKANBiotProblem


@dataclass
class TrainingOptions:
    steps: int = 1600
    learning_rate: float = 2.0e-3
    lbfgs_steps: int = 150
    grid_cells: int = 48
    particles_per_cell: int = 2
    width: int = 16
    hidden_layers: int = 2
    kan_basis: int = 10
    momentum_weight: float = 1.0
    boundary_weight: float = 500.0
    fluid_weight: float = 500.0
    strong_weight: float = 0.05
    roughness_weight: float = 1.0e-7
    seed: int = 202407
    log_every: int = 200


def train_case(
    case: Fig7Case,
    wave: WaveParameters,
    soil: SoilParameters,
    options: TrainingOptions,
    device: torch.device,
    output_dir: Path,
    resume: bool = False,
) -> tuple[HarmonicKAN, list[dict[str, float]]]:
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    dtype = torch.float64
    grid = MaterialPointGrid1D.create(
        options.grid_cells, options.particles_per_cell, device=device, dtype=dtype
    )
    model = HarmonicKAN(
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        dtype=dtype,
    ).to(device)
    problem = MPMKANBiotProblem(
        wave,
        soil,
        case,
        grid,
        momentum_weight=options.momentum_weight,
        boundary_weight=options.boundary_weight,
        fluid_weight=options.fluid_weight,
        strong_weight=options.strong_weight,
        roughness_weight=options.roughness_weight,
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{case.key}.pt"
    history: list[dict[str, float]] = []
    start_step = 0
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        history = list(payload.get("history", []))
        start_step = int(payload.get("step", 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, options.steps), eta_min=options.learning_rate * 0.05
    )
    started = time.perf_counter()
    for step in range(start_step, options.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = problem.loss(model)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        scheduler.step()

        if step == start_step or (step + 1) % options.log_every == 0 or step + 1 == options.steps:
            record = losses.detached()
            record["step"] = float(step + 1)
            record["learning_rate"] = float(scheduler.get_last_lr()[0])
            record["elapsed_seconds"] = float(time.perf_counter() - started)
            history.append(record)
            print(
                f"[{case.key}] step={step + 1:5d} "
                f"loss={record['total']:.3e} "
                f"mom={record['momentum']:.3e} "
                f"fluid={record['fluid']:.3e} "
                f"strong={record['strong']:.3e} "
                f"bc={record['boundary']:.3e}"
            )

    if options.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.6,
            max_iter=options.lbfgs_steps,
            max_eval=options.lbfgs_steps * 2,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            current = problem.loss(model)
            current.total.backward()
            return current.total

        lbfgs.step(closure)
        final_losses = problem.loss(model).detached()
        final_losses["step"] = float(options.steps + options.lbfgs_steps)
        final_losses["learning_rate"] = 0.0
        final_losses["elapsed_seconds"] = float(time.perf_counter() - started)
        history.append(final_losses)
        print(
            f"[{case.key}] LBFGS done loss={final_losses['total']:.3e} "
            f"mom={final_losses['momentum']:.3e} "
            f"fluid={final_losses['fluid']:.3e} "
            f"strong={final_losses['strong']:.3e} "
            f"bc={final_losses['boundary']:.3e}"
        )

    torch.save(
        {
            "case": case.as_dict(),
            "model": model.state_dict(),
            "history": history,
            "step": options.steps,
            "options": vars(options),
        },
        checkpoint_path,
    )
    return model, history


def load_case_checkpoint(
    case: Fig7Case,
    device: torch.device,
    output_dir: Path,
) -> tuple[HarmonicKAN, list[dict[str, float]], TrainingOptions]:
    checkpoint_path = output_dir / "checkpoints" / f"{case.key}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {case.key}: {checkpoint_path}"
        )
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    option_values = vars(TrainingOptions())
    option_values.update(payload.get("options", {}))
    options = TrainingOptions(**option_values)
    model = HarmonicKAN(
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        dtype=torch.float64,
    ).to(device)
    model.load_state_dict(payload["model"])
    return model, list(payload.get("history", [])), options


def save_history(
    histories: dict[str, list[dict[str, float]]], output_path: Path
) -> None:
    output_path.write_text(
        json.dumps(histories, indent=2, ensure_ascii=False), encoding="utf-8"
    )
