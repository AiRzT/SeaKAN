from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mpm_kan_figure9.config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
    figure9_setup,
)
from mpm_kan_figure9.reference import _sample_wall, solve_reference

from .common import parameter_count


@dataclass
class PINOOptions:
    epochs: int = 500
    learning_rate: float = 2.0e-3
    batch_size: int = 4
    width: int = 32
    modes_x: int = 12
    modes_y: int = 8
    layers: int = 4
    nx: int = 65
    ny: int = 25
    seed: int = 202409
    log_every: int = 25


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int):
        super().__init__()
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1.0 / width
        self.weight_low = nn.Parameter(
            scale
            * torch.randn(width, width, modes_y, modes_x, dtype=torch.cfloat)
        )
        self.weight_high = nn.Parameter(
            scale
            * torch.randn(width, width, modes_y, modes_x, dtype=torch.cfloat)
        )

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(value)
        output = torch.zeros(
            value.shape[0],
            value.shape[1],
            value.shape[-2],
            value.shape[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=value.device,
        )
        my = min(self.modes_y, spectrum.shape[-2] // 2)
        mx = min(self.modes_x, spectrum.shape[-1])
        output[:, :, :my, :mx] = self.multiply(
            spectrum[:, :, :my, :mx], self.weight_low[:, :, :my, :mx]
        )
        output[:, :, -my:, :mx] = self.multiply(
            spectrum[:, :, -my:, :mx], self.weight_high[:, :, :my, :mx]
        )
        return torch.fft.irfft2(output, s=value.shape[-2:])


class PINO2d(nn.Module):
    def __init__(self, options: PINOOptions):
        super().__init__()
        self.lift = nn.Conv2d(8, options.width, 1)
        self.spectral = nn.ModuleList(
            [
                SpectralConv2d(options.width, options.modes_y, options.modes_x)
                for _ in range(options.layers)
            ]
        )
        self.local = nn.ModuleList(
            [nn.Conv2d(options.width, options.width, 1) for _ in range(options.layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(options.width, 64, 1), nn.GELU(), nn.Conv2d(64, 6, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.lift(inputs)
        for spectral, local in zip(self.spectral, self.local):
            value = torch.nn.functional.gelu(spectral(value) + local(value))
        return self.project(value)


def _parameter_family() -> list[tuple[WaveParameters, SoilParameters, Geometry]]:
    wave, soil, geometry = figure9_setup()
    family: list[tuple[WaveParameters, SoilParameters, Geometry]] = []
    for period_factor in (0.88, 1.0, 1.12):
        for permeability_factor in (0.75, 1.0, 1.25):
            for depth_factor in (0.88, 1.0, 1.12):
                if (
                    period_factor == 1.0
                    and permeability_factor == 1.0
                    and depth_factor == 1.0
                ):
                    continue
                family.append(
                    (
                        replace(
                            wave,
                            period=wave.period * period_factor,
                            wavelength=wave.wavelength * period_factor,
                        ),
                        replace(
                            soil,
                            permeability=soil.permeability * permeability_factor,
                        ),
                        replace(
                            geometry,
                            pipe_center_depth=(
                                geometry.pipe_center_depth * depth_factor
                            ),
                        ),
                    )
                )
    return family


def _case_arrays(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    reference = solve_reference(wave, soil, geometry, nx=nx, ny=ny, n_theta=181)
    coefficients = dimensionless_coefficients(wave, soil, geometry)
    xx, yy = np.meshgrid(reference.x, reference.y)
    active = np.isfinite(reference.pressure)
    inputs = np.stack(
        (
            xx / geometry.half_width_over_height,
            2.0 * yy - 1.0,
            np.full_like(xx, coefficients["kappa"] / 4.1519288509842704),
            np.full_like(xx, coefficients["mass_pressure"] / 170.6161523852406),
            np.full_like(xx, coefficients["mass_strain"] / 66.37335976021267),
            np.full_like(xx, geometry.center_y),
            np.full_like(xx, geometry.radius_over_height),
            active.astype(float),
        ),
        axis=0,
    ).astype(np.float32)
    outputs = np.stack(
        (
            reference.displacement_x.real,
            reference.displacement_x.imag,
            reference.displacement_y.real,
            reference.displacement_y.imag,
            reference.pressure.real,
            reference.pressure.imag,
        ),
        axis=0,
    )
    outputs = np.nan_to_num(outputs).astype(np.float32)
    metadata = {
        "period": wave.period,
        "wavelength": wave.wavelength,
        "permeability": soil.permeability,
        "pipe_center_depth": geometry.pipe_center_depth,
        "kappa": coefficients["kappa"],
        "mass_pressure": coefficients["mass_pressure"],
        "mass_strain": coefficients["mass_strain"],
        "linear_residual": reference.relative_linear_residual,
    }
    return inputs, outputs, metadata


def generate_training_family(cache_path: Path, options: PINOOptions) -> dict[str, object]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "inputs": data["inputs"],
                "outputs": data["outputs"],
                "metadata_json": str(data["metadata_json"]),
            }
    inputs, outputs, metadata = [], [], []
    for index, case in enumerate(_parameter_family(), start=1):
        print(f"PINO reference case {index:02d}/26", flush=True)
        case_inputs, case_outputs, case_metadata = _case_arrays(
            *case, options.nx, options.ny
        )
        inputs.append(case_inputs)
        outputs.append(case_outputs)
        metadata.append(case_metadata)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata)
    np.savez_compressed(
        cache_path,
        inputs=np.stack(inputs),
        outputs=np.stack(outputs),
        metadata_json=np.asarray(metadata_json),
    )
    return {
        "inputs": np.stack(inputs),
        "outputs": np.stack(outputs),
        "metadata_json": metadata_json,
    }


def _physics_loss(prediction: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    mask = inputs[:, 7:8]
    interior = mask[:, :, 1:-1, 1:-1]
    p = torch.complex(prediction[:, 4], prediction[:, 5])
    u = torch.complex(prediction[:, 0], prediction[:, 1])
    w = torch.complex(prediction[:, 2], prediction[:, 3])
    ny, nx = p.shape[-2:]
    half = 2.767554479418886
    dx = 2.0 * half / (nx - 1)
    dy = 1.0 / (ny - 1)

    def dx1(value: torch.Tensor) -> torch.Tensor:
        return (value[:, 1:-1, 2:] - value[:, 1:-1, :-2]) / (2.0 * dx)

    def dy1(value: torch.Tensor) -> torch.Tensor:
        return (value[:, 2:, 1:-1] - value[:, :-2, 1:-1]) / (2.0 * dy)

    def dx2(value: torch.Tensor) -> torch.Tensor:
        return (
            value[:, 1:-1, 2:]
            - 2.0 * value[:, 1:-1, 1:-1]
            + value[:, 1:-1, :-2]
        ) / dx**2

    def dy2(value: torch.Tensor) -> torch.Tensor:
        return (
            value[:, 2:, 1:-1]
            - 2.0 * value[:, 1:-1, 1:-1]
            + value[:, :-2, 1:-1]
        ) / dy**2

    def cross(value: torch.Tensor) -> torch.Tensor:
        return (
            value[:, 2:, 2:]
            - value[:, 2:, :-2]
            - value[:, :-2, 2:]
            + value[:, :-2, :-2]
        ) / (4.0 * dx * dy)

    lam = 1.9411764705882353
    r_u = (2.0 + lam) * dx2(u) + dy2(u) + (1.0 + lam) * cross(w) - dx1(p)
    r_w = (1.0 + lam) * cross(u) + dx2(w) + (2.0 + lam) * dy2(w) - dy1(p)
    mass_p = inputs[:, 3, 0, 0] * 170.6161523852406
    mass_e = inputs[:, 4, 0, 0] * 66.37335976021267
    eps_v = dx1(u) + dy1(w)
    r_p = dx2(p) + dy2(p) + 1j * mass_p[:, None, None] * p[:, 1:-1, 1:-1]
    r_p = (r_p + 1j * mass_e[:, None, None] * eps_v) / (
        mass_p[:, None, None] + mass_e[:, None, None]
    )
    valid = interior * mask[:, :, 1:-1, 2:] * mask[:, :, 1:-1, :-2]
    valid = valid * mask[:, :, 2:, 1:-1] * mask[:, :, :-2, 1:-1]
    valid = valid[:, 0]
    denominator = valid.sum().clamp_min(1.0)
    return (
        ((r_u.abs().square() + r_w.abs().square() + 20.0 * r_p.abs().square()) * valid).sum()
        / denominator
    )


def train_pino(
    output_dir: Path,
    *,
    options: PINOOptions,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    method_dir = output_dir / "pino"
    method_dir.mkdir(parents=True, exist_ok=True)
    family = generate_training_family(method_dir / "training_family.npz", options)
    train_inputs = torch.from_numpy(family["inputs"]).to(device)
    train_outputs = torch.from_numpy(family["outputs"]).to(device)
    model = PINO2d(options).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=1.0e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, options.epochs, eta_min=options.learning_rate * 0.05
    )
    generator = torch.Generator(device=device).manual_seed(options.seed + 29)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, options.epochs + 1):
        permutation = torch.randperm(
            train_inputs.shape[0], device=device, generator=generator
        )
        data_total = physics_total = 0.0
        batches = 0
        for indices in permutation.split(options.batch_size):
            optimizer.zero_grad(set_to_none=True)
            inputs = train_inputs[indices]
            target = train_outputs[indices]
            prediction = model(inputs)
            mask = inputs[:, 7:8]
            scale = torch.tensor(
                [0.1, 0.1, 0.1, 0.1, 1.0, 1.0], device=device
            )[None, :, None, None]
            data = (((prediction - target) / scale).square() * mask).sum()
            data = data / (mask.sum().clamp_min(1.0) * prediction.shape[1])
            physics = _physics_loss(prediction, inputs)
            total = data + 2.0e-3 * physics
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            data_total += float(data.detach())
            physics_total += float(physics.detach())
            batches += 1
        scheduler.step()
        if epoch == 1 or epoch % options.log_every == 0 or epoch == options.epochs:
            row = {
                "epoch": epoch,
                "data_loss": data_total / batches,
                "physics_loss": physics_total / batches,
            }
            history.append(row)
            print(
                f"pino {epoch:4d}/{options.epochs}: data={row['data_loss']:.3e}, "
                f"physics={row['physics_loss']:.3e}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    wave, soil, geometry = figure9_setup()
    target_inputs, _, target_metadata = _case_arrays(
        wave, soil, geometry, options.nx, options.ny
    )
    with torch.no_grad():
        prediction = model(torch.from_numpy(target_inputs[None]).to(device))[0]
    prediction = prediction.cpu().numpy()
    pressure = prediction[4] + 1j * prediction[5]
    active = target_inputs[7].astype(bool)
    pressure = np.where(active, pressure, np.nan + 0j)
    x = np.linspace(-geometry.half_width_over_height, geometry.half_width_over_height, options.nx)
    y = np.linspace(0.0, 1.0, options.ny)
    theta = np.linspace(0.0, 2.0 * math.pi, 721)
    wall = _sample_wall(
        theta,
        pressure,
        active,
        x,
        y,
        geometry,
        max(x[1] - x[0], y[1] - y[0]),
    )
    np.savetxt(
        method_dir / "profile.csv",
        np.column_stack((theta, wall)),
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
            "options": asdict(options),
            "elapsed_seconds": elapsed,
        },
        method_dir / "checkpoint.pt",
    )
    metadata = {
        "method": "pino",
        "elapsed_seconds": elapsed,
        "parameters": parameter_count(model),
        "device": str(device),
        "dtype": "torch.float32",
        "epochs": options.epochs,
        "seed": options.seed,
        "target_data_used_for_training": False,
        "training_supervision": "26 independent finite-difference parameter cases",
        "held_out_target": target_metadata,
        "profile": str(method_dir / "profile.csv"),
    }
    (method_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
