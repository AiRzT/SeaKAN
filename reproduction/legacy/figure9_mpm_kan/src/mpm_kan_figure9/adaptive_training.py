from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .adaptive_mpm import AdaptiveMaterialPointCloud
from .config import Geometry, SoilParameters, WaveParameters
from .model import PipeHarmonicKAN
from .strong_mpm import AdaptiveStrongMPMProblem
from .training import TrainingOptions, load_checkpoint


@dataclass
class AdaptiveTrainingOptions:
    steps: int = 900
    learning_rate: float = 1.0e-4
    batch_background: int = 384
    batch_middle: int = 640
    batch_fine: int = 896
    level_weight_background: float = 0.20
    level_weight_middle: float = 0.32
    level_weight_fine: float = 0.48
    momentum_weight: float = 10.0
    fluid_weight: float = 800.0
    p2g_weight: float = 0.20
    boundary_weight: float = 80.0
    pipe_flux_weight: float = 160.0
    roughness_weight: float = 1.0e-3
    boundary_points: int = 160
    seed: int = 202409
    log_every: int = 100

    @property
    def batch_per_level(self) -> tuple[int, int, int]:
        return (
            self.batch_background,
            self.batch_middle,
            self.batch_fine,
        )

    @property
    def level_weights(self) -> tuple[float, float, float]:
        return (
            self.level_weight_background,
            self.level_weight_middle,
            self.level_weight_fine,
        )


def make_adaptive_problem(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    cloud: AdaptiveMaterialPointCloud,
    options: AdaptiveTrainingOptions,
) -> AdaptiveStrongMPMProblem:
    return AdaptiveStrongMPMProblem(
        wave,
        soil,
        geometry,
        cloud,
        level_weights=options.level_weights,
        momentum_weight=options.momentum_weight,
        fluid_weight=options.fluid_weight,
        p2g_weight=options.p2g_weight,
        boundary_weight=options.boundary_weight,
        pipe_flux_weight=options.pipe_flux_weight,
        roughness_weight=options.roughness_weight,
        boundary_points=options.boundary_points,
    )


def train_adaptive_strong(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    cloud: AdaptiveMaterialPointCloud,
    baseline_checkpoint: Path,
    output_dir: Path,
    options: AdaptiveTrainingOptions,
    device: torch.device,
    *,
    resume: bool = False,
) -> tuple[
    PipeHarmonicKAN,
    TrainingOptions,
    torch.dtype,
    list[dict[str, float | str]],
]:
    adaptive_checkpoint = output_dir / "checkpoints" / "figure9_adaptive_strong.pt"
    if resume and adaptive_checkpoint.is_file():
        state = torch.load(
            adaptive_checkpoint, map_location=device, weights_only=False
        )
        baseline_options = TrainingOptions(**state["baseline_options"])
        dtype = (
            torch.float64
            if "float64" in state.get("dtype", "")
            else torch.float32
        )
        model, _, _, _ = load_checkpoint(
            wave, soil, geometry, baseline_checkpoint, device
        )
        model.load_state_dict(state["model"])
        history: list[dict[str, float | str]] = list(
            state.get("history", [])
        )
    else:
        model, baseline_options, dtype, _ = load_checkpoint(
            wave, soil, geometry, baseline_checkpoint, device
        )
        history = []

    problem = make_adaptive_problem(
        wave, soil, geometry, cloud, options
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    started = time.perf_counter()
    for step in range(1, options.steps + 1):
        indices = cloud.sample_indices(options.batch_per_level, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss(model, indices)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)
        optimizer.step()
        if step == 1 or step % options.log_every == 0 or step == options.steps:
            row: dict[str, float | str] = {
                "stage": "adaptive_strong_adam",
                "step": float(step),
                **loss.detached(),
            }
            history.append(row)
            print(
                f"Adaptive strong {step:5d}/{options.steps}: "
                f"total={row['total']:.4e}, "
                f"particle_m={row['particle_momentum']:.3e}, "
                f"particle_p={row['particle_fluid']:.3e}, "
                f"pipe_flux={row['pipe_flux']:.3e}"
            )

    full_indices = torch.arange(cloud.n_particles, device=device)
    final = problem.loss(model, full_indices).detached()
    history.append(
        {
            "stage": "adaptive_strong_full_final",
            "step": float(options.steps),
            "elapsed_seconds": time.perf_counter() - started,
            **final,
        }
    )
    adaptive_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "baseline_options": asdict(baseline_options),
            "adaptive_options": asdict(options),
            "history": history,
            "dtype": str(dtype),
            "cloud": cloud.summary(),
        },
        adaptive_checkpoint,
    )
    return model, baseline_options, dtype, history


def load_adaptive_checkpoint(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    baseline_checkpoint: Path,
    adaptive_checkpoint: Path,
    device: torch.device,
) -> tuple[
    PipeHarmonicKAN,
    TrainingOptions,
    AdaptiveTrainingOptions,
    torch.dtype,
    list[dict[str, float | str]],
]:
    state = torch.load(
        adaptive_checkpoint, map_location=device, weights_only=False
    )
    model, baseline_options, dtype, _ = load_checkpoint(
        wave, soil, geometry, baseline_checkpoint, device
    )
    model.load_state_dict(state["model"])
    adaptive_options = AdaptiveTrainingOptions(**state["adaptive_options"])
    return (
        model,
        baseline_options,
        adaptive_options,
        dtype,
        list(state.get("history", [])),
    )
