from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import Geometry, SoilParameters, WaveParameters, dimensionless_coefficients
from .model import PipeHarmonicKAN
from .mpm import MaterialPointGrid2D
from .physics import MPMKANPipeProblem


@dataclass
class TrainingOptions:
    steps: int = 1800
    learning_rate: float = 1.5e-3
    lbfgs_steps: int = 120
    grid_cells_x: int = 64
    grid_cells_y: int = 24
    width: int = 18
    hidden_layers: int = 2
    kan_basis: int = 9
    momentum_weight: float = 1.0
    fluid_weight: float = 20.0
    boundary_weight: float = 80.0
    pipe_flux_weight: float = 120.0
    strong_weight: float = 0.02
    roughness_weight: float = 1.0e-6
    strong_stride: int = 3
    boundary_points: int = 128
    seed: int = 202409
    log_every: int = 100


def make_problem(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaterialPointGrid2D, MPMKANPipeProblem]:
    grid = MaterialPointGrid2D.create(
        options.grid_cells_x,
        options.grid_cells_y,
        geometry,
        device=device,
        dtype=dtype,
    )
    problem = MPMKANPipeProblem(
        wave,
        soil,
        geometry,
        grid,
        momentum_weight=options.momentum_weight,
        fluid_weight=options.fluid_weight,
        boundary_weight=options.boundary_weight,
        pipe_flux_weight=options.pipe_flux_weight,
        strong_weight=options.strong_weight,
        roughness_weight=options.roughness_weight,
        strong_stride=options.strong_stride,
        boundary_points=options.boundary_points,
    )
    return grid, problem


def train(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    options: TrainingOptions,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    *,
    resume: bool = False,
) -> tuple[PipeHarmonicKAN, list[dict[str, float]]]:
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    coefficients = dimensionless_coefficients(wave, soil, geometry)
    model = PipeHarmonicKAN(
        geometry,
        coefficients["kappa"],
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        dtype=dtype,
    ).to(device)
    _, problem = make_problem(wave, soil, geometry, options, device, dtype)
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "checkpoints" / "figure9.pt"
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        history = list(state.get("history", []))

    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    started = time.perf_counter()
    for step in range(1, options.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        breakdown = problem.loss(model)
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        optimizer.step()
        if step == 1 or step % options.log_every == 0 or step == options.steps:
            row = {"stage": "adam", "step": len(history) + 1, **breakdown.detached()}
            history.append(row)
            print(
                f"Adam {step:5d}/{options.steps}: total={row['total']:.4e}, "
                f"weak_m={row['momentum']:.3e}, weak_p={row['fluid']:.3e}, "
                f"pipe_flux={row['pipe_flux']:.3e}"
            )

    if options.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.6,
            max_iter=options.lbfgs_steps,
            max_eval=options.lbfgs_steps * 2,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = problem.loss(model)
            value.total.backward()
            evaluations += 1
            if evaluations == 1 or evaluations % max(10, options.log_every // 2) == 0:
                row = {
                    "stage": "lbfgs",
                    "step": len(history) + 1,
                    **value.detached(),
                }
                history.append(row)
                print(
                    f"LBFGS eval {evaluations:4d}: total={row['total']:.4e}, "
                    f"weak_m={row['momentum']:.3e}, weak_p={row['fluid']:.3e}"
                )
            return value.total

        lbfgs.step(closure)

    elapsed = time.perf_counter() - started
    final = problem.loss(model).detached()
    history.append(
        {
            "stage": "final",
            "step": len(history) + 1,
            "elapsed_seconds": elapsed,
            **final,
        }
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "options": asdict(options),
            "history": history,
            "dtype": str(dtype),
            "device": str(device),
            "coefficients": coefficients,
        },
        checkpoint_path,
    )
    return model, history


def load_checkpoint(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PipeHarmonicKAN, TrainingOptions, torch.dtype, list[dict[str, float]]]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    options = TrainingOptions(**state["options"])
    dtype = torch.float64 if "float64" in state.get("dtype", "") else torch.float32
    coefficients = dimensionless_coefficients(wave, soil, geometry)
    model = PipeHarmonicKAN(
        geometry,
        coefficients["kappa"],
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.kan_basis,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(state["model"])
    return model, options, dtype, list(state.get("history", []))


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
