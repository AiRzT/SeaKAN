from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from .config import ParameterDomain, PhysicalConstants, solve_wave_number
from .inverse import (
    _logit,
    initial_latent_from_fraction,
    read_case_parameters,
)
from .metrics import regression_metrics
from .model import ParametricSoilKAN
from .training import load_checkpoint


@dataclass(frozen=True)
class FieldInverseOptions:
    invert: str = "both"
    sensor_layout: str = "fixed"
    sensor_count: int = 20
    cycles: int = 1
    adam_steps: int = 400
    lbfgs_steps: int = 40
    learning_rate: float = 4.0e-2
    starts: int = 6
    grid_points: int = 11
    noise_fraction: float = 0.0
    seed: int = 202421


@dataclass(frozen=True)
class FieldEstimate:
    start: int
    permeability_m_per_s: float
    burial_center_depth_over_diameter: float
    loss: float
    complex_relative_l2: float
    complex_mse: float
    complex_rmse: float
    amplitude_relative_l2: float
    amplitude_mse: float
    amplitude_rmse: float
    amplitude_r2: float | None
    elapsed_seconds: float


def default_field_sensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A fixed 4 x 5 array that remains outside every candidate pipe."""

    x = (-0.8, -0.4, 0.4, 0.8)
    y = (0.15, 0.35, 0.55, 0.75, 0.90)
    return torch.tensor(
        [(x_value, y_value) for y_value in y for x_value in x],
        device=device,
        dtype=dtype,
    )


def fisher_optimal_sensors(
    model: ParametricSoilKAN,
    base_parameters: torch.Tensor,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    *,
    count: int = 64,
    minimum_spacing: float = 0.075,
) -> torch.Tensor:
    """Select a robust permeability-sensitive layout without using truth k."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x_values = np.linspace(-1.6, 1.6, 97)
    y_values = np.linspace(0.04, 0.96, 57)
    burial = base_parameters[0, 3]
    centre = float(
        (
            1.0
            - burial
            * constants.pipe_diameter_m
            / constants.seabed_thickness_m
        )
        .detach()
        .cpu()
    )
    clearance = constants.radius + 0.025
    candidates = np.asarray(
        [
            (x_value, y_value)
            for y_value in y_values
            for x_value in x_values
            if x_value**2 + (y_value - centre) ** 2 > clearance**2
        ],
        dtype=np.float64,
    )
    xy = torch.as_tensor(candidates, device=device, dtype=dtype)
    low, high = np.log10(domain.permeability_m_per_s)
    squared_sensitivities: list[np.ndarray] = []
    with torch.no_grad():
        for fraction in (0.15, 0.50, 0.85):
            latent = math.log(fraction / (1.0 - fraction))
            increment = 0.025

            def permeability(value: float) -> float:
                bounded = 1.0 / (1.0 + math.exp(-value))
                return 10.0 ** (low + (high - low) * bounded)

            plus = predict_field_pressure(
                model,
                base_parameters,
                xy,
                torch.tensor(
                    permeability(latent + increment),
                    device=device,
                    dtype=dtype,
                ),
                burial,
                constants,
            )
            minus = predict_field_pressure(
                model,
                base_parameters,
                xy,
                torch.tensor(
                    permeability(latent - increment),
                    device=device,
                    dtype=dtype,
                ),
                burial,
                constants,
            )
            derivative = (plus - minus) / (2.0 * increment)
            squared_sensitivities.append(
                derivative.abs().square().cpu().numpy()
            )
    robust_score = np.mean(squared_sensitivities, axis=0)
    order = np.argsort(robust_score)[::-1]
    selected: list[int] = []
    for index in order:
        if all(
            np.linalg.norm(candidates[index] - candidates[prior])
            >= minimum_spacing
            for prior in selected
        ):
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"only {len(selected)} separated sensor locations were found"
        )
    return torch.as_tensor(
        candidates[selected], device=device, dtype=dtype
    )


def _permeability_from_latent(
    latent: torch.Tensor,
    domain: ParameterDomain,
) -> torch.Tensor:
    fraction = torch.sigmoid(latent[0])
    low, high = np.log10(domain.permeability_m_per_s)
    return torch.pow(10.0, low + (high - low) * fraction)


def _both_from_latent(
    latent: torch.Tensor,
    domain: ParameterDomain,
) -> tuple[torch.Tensor, torch.Tensor]:
    permeability = _permeability_from_latent(latent, domain)
    fraction = torch.sigmoid(latent[1])
    low, high = domain.burial_center_depth_over_diameter
    burial = low + (high - low) * fraction
    return permeability, burial


def predict_field_pressure(
    model: ParametricSoilKAN,
    base_parameters: torch.Tensor,
    xy: torch.Tensor,
    permeability: torch.Tensor,
    burial: torch.Tensor,
    constants: PhysicalConstants,
) -> torch.Tensor:
    count = xy.shape[0]
    fixed = base_parameters.expand(count, -1)
    parameters = torch.cat(
        (
            fixed[:, :2],
            permeability.expand(count, 1),
            burial.expand(count, 1),
            fixed[:, 4:],
        ),
        dim=1,
    )
    period = float(base_parameters[0, 1].detach().cpu())
    wave_number = solve_wave_number(
        np.asarray([period], dtype=np.float64), constants
    )[0]
    kappa = torch.tensor(
        wave_number * constants.seabed_thickness_m,
        device=xy.device,
        dtype=xy.dtype,
    )
    pipe_center_y = (
        1.0
        - burial
        * constants.pipe_diameter_m
        / constants.seabed_thickness_m
    )
    return model(
        xy,
        parameters,
        kappa.expand(count),
        pipe_center_y.expand(count),
    ).p


def synthetic_field_observations(
    model: ParametricSoilKAN,
    parameters: np.ndarray,
    constants: PhysicalConstants,
    noise_fraction: float,
    seed: int,
    *,
    xy: torch.Tensor | None = None,
    cycles: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if cycles < 1:
        raise ValueError("cycles must be positive")
    if xy is None:
        xy = default_field_sensors(device=device, dtype=dtype)
    base = torch.as_tensor(
        parameters[None, :], device=device, dtype=dtype
    )
    with torch.no_grad():
        pressure = predict_field_pressure(
            model,
            base,
            xy,
            base[0, 2],
            base[0, 3],
            constants,
        )
    if noise_fraction > 0.0:
        generator = np.random.default_rng(seed)
        rms = float(torch.sqrt(pressure.abs().square().mean()).cpu())
        standard_deviation = (
            noise_fraction * rms / math.sqrt(2.0 * cycles)
        )
        noise = generator.normal(
            0.0, standard_deviation, size=(pressure.numel(), 2)
        )
        noise_tensor = torch.complex(
            torch.as_tensor(noise[:, 0], device=device, dtype=dtype),
            torch.as_tensor(noise[:, 1], device=device, dtype=dtype),
        )
        pressure = pressure + noise_tensor
    return xy, pressure


def comsol_field_observations(
    reference: Path,
    xy: torch.Tensor,
    constants: PhysicalConstants,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Interpolate an independent COMSOL complex-pressure export."""

    values = np.loadtxt(reference, delimiter=",", comments="%")
    if values.ndim != 2 or values.shape[1] < 7:
        raise ValueError(
            "COMSOL reference must contain x, y, real(p), imag(p), "
            "and abs(p) columns"
        )
    nodes = values[:, 2:4]
    pressure = values[:, 4] + 1j * values[:, 5]
    exported_amplitude = values[:, 6]
    query_dimensionless = xy.detach().cpu().numpy()
    query = query_dimensionless * constants.seabed_thickness_m
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    real_interp = mtri.LinearTriInterpolator(
        triangulation, pressure.real
    )(query[:, 0], query[:, 1])
    imag_interp = mtri.LinearTriInterpolator(
        triangulation, pressure.imag
    )(query[:, 0], query[:, 1])
    amplitude_interp = mtri.LinearTriInterpolator(
        triangulation, exported_amplitude
    )(query[:, 0], query[:, 1])
    if (
        np.ma.is_masked(real_interp)
        and np.any(np.ma.getmaskarray(real_interp))
    ) or (
        np.ma.is_masked(imag_interp)
        and np.any(np.ma.getmaskarray(imag_interp))
    ):
        raise ValueError("one or more sensor locations lie outside COMSOL data")
    interpolated = np.asarray(real_interp) + 1j * np.asarray(imag_interp)
    amplitude = np.asarray(amplitude_interp)
    distances = np.sqrt(
        np.sum((query[:, None, :] - nodes[None, :, :]) ** 2, axis=2)
    ).min(axis=1)
    consistency = np.linalg.norm(np.abs(interpolated) - amplitude) / max(
        np.linalg.norm(amplitude), 1.0e-12
    )
    target = torch.as_tensor(
        interpolated, device=xy.device, dtype=torch.complex128
    )
    if xy.dtype == torch.float32:
        target = target.to(torch.complex64)
    quality: dict[str, float | int | str] = {
        "reference_file": str(reference.resolve()),
        "comsol_solution_nodes": int(values.shape[0]),
        "mean_nearest_node_distance_m": float(distances.mean()),
        "maximum_nearest_node_distance_m": float(distances.max()),
        "complex_to_exported_amplitude_l2": float(consistency),
        "interpolation": "linear Delaunay interpolation of COMSOL nodes",
    }
    return target, quality


def _fractions_to_latent(
    fractions: np.ndarray,
    invert: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if invert == "permeability":
        values = _logit(np.asarray(fractions[:1], dtype=np.float64))
        return torch.tensor(
            values,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
    return initial_latent_from_fraction(
        fractions, device=device, dtype=dtype
    )


def _candidate(
    latent: torch.Tensor,
    invert: str,
    known_burial: torch.Tensor,
    domain: ParameterDomain,
) -> tuple[torch.Tensor, torch.Tensor]:
    if invert == "permeability":
        return _permeability_from_latent(latent, domain), known_burial
    return _both_from_latent(latent, domain)


def coarse_grid_initialisations(
    model: ParametricSoilKAN,
    base: torch.Tensor,
    xy: torch.Tensor,
    target: torch.Tensor,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: FieldInverseOptions,
) -> list[np.ndarray]:
    grid = np.linspace(0.03, 0.97, options.grid_points)
    fractions = (
        [np.asarray([value, 0.5]) for value in grid]
        if options.invert == "permeability"
        else [np.asarray(value) for value in product(grid, grid)]
    )
    scale = target.abs().square().mean().clamp_min(1.0e-12)
    scored: list[tuple[float, np.ndarray]] = []
    with torch.no_grad():
        for fraction in fractions:
            latent = _fractions_to_latent(
                fraction,
                options.invert,
                device=xy.device,
                dtype=xy.dtype,
            )
            permeability, burial = _candidate(
                latent, options.invert, base[0, 3], domain
            )
            prediction = predict_field_pressure(
                model,
                base,
                xy,
                permeability,
                burial,
                constants,
            )
            loss = (prediction - target).abs().square().mean() / scale
            scored.append((float(loss.cpu()), fraction))
    scored.sort(key=lambda item: item[0])
    selected: list[np.ndarray] = []
    for _, fraction in scored:
        if all(
            np.linalg.norm(fraction - existing) > 0.08
            for existing in selected
        ):
            selected.append(fraction)
        if len(selected) == options.starts:
            break
    return selected


def field_identifiability(
    model: ParametricSoilKAN,
    base: torch.Tensor,
    xy: torch.Tensor,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    invert: str,
) -> dict[str, object]:
    low, high = np.log10(domain.permeability_m_per_s)
    permeability_fraction = (
        math.log10(float(base[0, 2].cpu())) - low
    ) / (high - low)
    burial_low, burial_high = domain.burial_center_depth_over_diameter
    burial_fraction = (
        float(base[0, 3].cpu()) - burial_low
    ) / (burial_high - burial_low)
    latent = _fractions_to_latent(
        np.asarray([permeability_fraction, burial_fraction]),
        invert,
        device=xy.device,
        dtype=xy.dtype,
    )

    def response(value: torch.Tensor) -> torch.Tensor:
        permeability, burial = _candidate(
            value, invert, base[0, 3], domain
        )
        pressure = predict_field_pressure(
            model,
            base,
            xy,
            permeability,
            burial,
            constants,
        )
        return torch.view_as_real(pressure).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(response, latent)
    singular = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
    condition = (
        float(singular.max() / singular.min())
        if singular.size > 1
        else 1.0
    )
    return {
        "singular_values": singular.tolist(),
        "condition_number": condition,
        "rank": int(np.count_nonzero(singular > 1.0e-12)),
        "observation_components": "pressure real and imaginary parts",
    }


def invert_field_parameters(
    model: ParametricSoilKAN,
    parameters: np.ndarray,
    xy: torch.Tensor,
    target: torch.Tensor,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: FieldInverseOptions,
) -> tuple[FieldEstimate, list[FieldEstimate], list[dict[str, float]]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    base = torch.as_tensor(
        parameters[None, :], device=device, dtype=dtype
    )
    scale = target.abs().square().mean().clamp_min(1.0e-12)
    initialisations = coarse_grid_initialisations(
        model, base, xy, target, domain, constants, options
    )
    estimates: list[FieldEstimate] = []
    best_history: list[dict[str, float]] = []

    for start_index, fraction in enumerate(initialisations):
        latent = _fractions_to_latent(
            fraction, options.invert, device=device, dtype=dtype
        )
        optimizer = torch.optim.Adam(
            [latent], lr=options.learning_rate
        )
        history: list[dict[str, float]] = []
        started = time.perf_counter()

        def objective() -> tuple[torch.Tensor, torch.Tensor]:
            permeability, burial = _candidate(
                latent, options.invert, base[0, 3], domain
            )
            prediction = predict_field_pressure(
                model,
                base,
                xy,
                permeability,
                burial,
                constants,
            )
            loss = (prediction - target).abs().square().mean() / scale
            return loss, prediction

        for step in range(options.adam_steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective()
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 20 == 0:
                permeability, burial = _candidate(
                    latent, options.invert, base[0, 3], domain
                )
                history.append(
                    {
                        "iteration": float(step + 1),
                        "loss": float(loss.detach().cpu()),
                        "permeability_m_per_s": float(
                            permeability.detach().cpu()
                        ),
                        "burial_ratio": float(burial.detach().cpu()),
                    }
                )

        lbfgs = torch.optim.LBFGS(
            [latent],
            lr=0.8,
            max_iter=options.lbfgs_steps,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss, _ = objective()
            loss.backward()
            return loss

        lbfgs.step(closure)
        loss, prediction = objective()
        permeability, burial = _candidate(
            latent, options.invert, base[0, 3], domain
        )
        complex_relative_l2 = float(
            (
                torch.linalg.vector_norm(prediction - target)
                / torch.linalg.vector_norm(target).clamp_min(1.0e-12)
            )
            .detach()
            .cpu()
        )
        complex_mse = float(
            (prediction - target).abs().square().mean().detach().cpu()
        )
        amplitude_metrics = regression_metrics(
            target.abs().detach().cpu().numpy(),
            prediction.abs().detach().cpu().numpy(),
        )
        estimate = FieldEstimate(
            start=start_index,
            permeability_m_per_s=float(permeability.detach().cpu()),
            burial_center_depth_over_diameter=float(burial.detach().cpu()),
            loss=float(loss.detach().cpu()),
            complex_relative_l2=complex_relative_l2,
            complex_mse=complex_mse,
            complex_rmse=math.sqrt(complex_mse),
            amplitude_relative_l2=amplitude_metrics.relative_l2,
            amplitude_mse=amplitude_metrics.mse,
            amplitude_rmse=amplitude_metrics.rmse,
            amplitude_r2=amplitude_metrics.r2,
            elapsed_seconds=time.perf_counter() - started,
        )
        estimates.append(estimate)
        if estimate.loss <= min(item.loss for item in estimates):
            history.append(
                {
                    "iteration": float(
                        options.adam_steps + options.lbfgs_steps
                    ),
                    "loss": estimate.loss,
                    "permeability_m_per_s": (
                        estimate.permeability_m_per_s
                    ),
                    "burial_ratio": (
                        estimate.burial_center_depth_over_diameter
                    ),
                }
            )
            best_history = history
    estimates.sort(key=lambda item: item.loss)
    return estimates[0], estimates, best_history


def _write_outputs(
    output: Path,
    result: dict[str, object],
    estimates: list[FieldEstimate],
    history: list[dict[str, float]],
    xy: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "field_inverse_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output / "field_inverse_multistart.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(estimates[0]).keys())
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(asdict(estimate))
    if history:
        with (output / "field_inverse_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    xy_numpy = xy.detach().cpu().numpy()
    target_numpy = target.detach().cpu().numpy()
    prediction_numpy = prediction.detach().cpu().numpy()
    with (output / "field_inverse_sensors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "x_over_h",
                "y_over_h",
                "observed_real",
                "observed_imag",
                "predicted_real",
                "predicted_imag",
            )
        )
        for point, observed, estimated in zip(
            xy_numpy, target_numpy, prediction_numpy, strict=True
        ):
            writer.writerow(
                (
                    point[0],
                    point[1],
                    observed.real,
                    observed.imag,
                    estimated.real,
                    estimated.imag,
                )
            )


def _plot(
    path: Path,
    target: torch.Tensor,
    prediction: torch.Tensor,
    history: list[dict[str, float]],
    best: FieldEstimate,
) -> None:
    target_numpy = target.detach().cpu().numpy()
    prediction_numpy = prediction.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    index = np.arange(target_numpy.size)
    axes[0].plot(
        index,
        np.abs(target_numpy),
        "o",
        color="#6A51A3",
        markerfacecolor="white",
        label="Observed amplitude",
    )
    axes[0].plot(
        index,
        np.abs(prediction_numpy),
        "-",
        color="#0072B2",
        linewidth=1.6,
        label="Inverse prediction",
    )
    axes[0].set_xlabel("Sensor index")
    axes[0].set_ylabel(r"$|p|/p_0$")
    axes[0].legend(frameon=False, fontsize=8)
    iteration = [row["iteration"] for row in history]
    loss = [row["loss"] for row in history]
    axes[1].semilogy(iteration, loss, color="#D55E00", linewidth=1.6)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Complex-pressure loss")
    axes[1].text(
        0.04,
        0.06,
        (
            rf"$k={best.permeability_m_per_s:.3e}$ m/s"
            "\n"
            rf"$b/D={best.burial_center_depth_over_diameter:.3f}$"
            "\n"
            rf"$L_2={100.0 * best.complex_relative_l2:.2f}\%$"
        ),
        transform=axes[1].transAxes,
        fontsize=8,
        va="bottom",
    )
    for axis in axes:
        axis.grid(True, color="#C9D0D8", linewidth=0.45)
        axis.tick_params(direction="in", top=True, right=True)
    figure.tight_layout()
    figure.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    model, _, domain, constants, _, _, _ = load_checkpoint(
        args.checkpoint, device
    )
    parameters = read_case_parameters(args.cases, args.case_id)
    options = FieldInverseOptions(
        invert=args.invert,
        sensor_layout=args.sensor_layout,
        sensor_count=args.sensors,
        cycles=args.cycles,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        learning_rate=args.learning_rate,
        starts=args.starts,
        grid_points=args.grid_points,
        noise_fraction=args.noise,
        seed=args.seed,
    )
    base_for_layout = torch.as_tensor(
        parameters[None, :],
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    if options.sensor_layout == "fisher":
        if options.invert != "permeability":
            raise ValueError(
                "the Fisher layout requires known burial and "
                "--invert permeability"
            )
        xy_layout = fisher_optimal_sensors(
            model,
            base_for_layout,
            domain,
            constants,
            count=options.sensor_count,
        )
    else:
        xy_layout = default_field_sensors(
            device=device, dtype=next(model.parameters()).dtype
        )
    data_quality: dict[str, float | int | str]
    if args.reference is None:
        xy, target = synthetic_field_observations(
            model,
            parameters,
            constants,
            options.noise_fraction,
            options.seed,
            xy=xy_layout,
            cycles=options.cycles,
        )
        mode = "frozen-model synthetic field inverse"
        data_quality = {
            "source": "frozen SeaKAN self-consistency observations"
        }
    else:
        if options.noise_fraction != 0.0 or options.cycles != 1:
            raise ValueError(
                "--noise and --cycles are synthetic-observation options"
            )
        xy = xy_layout
        target, data_quality = comsol_field_observations(
            args.reference, xy, constants
        )
        mode = "frozen-model independent COMSOL field inverse"
    best, estimates, history = invert_field_parameters(
        model,
        parameters,
        xy,
        target,
        domain,
        constants,
        options,
    )
    base = torch.as_tensor(
        parameters[None, :], device=device, dtype=xy.dtype
    )
    with torch.no_grad():
        prediction = predict_field_pressure(
            model,
            base,
            xy,
            torch.tensor(
                best.permeability_m_per_s, device=device, dtype=xy.dtype
            ),
            torch.tensor(
                best.burial_center_depth_over_diameter,
                device=device,
                dtype=xy.dtype,
            ),
            constants,
        )
    identifiability = field_identifiability(
        model, base, xy, domain, constants, options.invert
    )
    result = {
        "case_id": args.case_id,
        "mode": mode,
        "observation_layout": (
            f"{xy.shape[0]} {options.sensor_layout} interior "
            "complex-pressure sensors"
        ),
        "invert": options.invert,
        "forward_weights_frozen": True,
        "reference_used_in_forward_training": False,
        "reference_used_for_sensor_selection": False,
        "observation_source": (
            None if args.reference is None else str(args.reference.resolve())
        ),
        "observation_data_quality": data_quality,
        "truth": {
            "permeability_m_per_s": float(parameters[2]),
            "burial_center_depth_over_diameter": float(parameters[3]),
        },
        "best_estimate": asdict(best),
        "parameter_relative_errors": {
            "permeability": abs(
                best.permeability_m_per_s - parameters[2]
            )
            / parameters[2],
            "burial_ratio": abs(
                best.burial_center_depth_over_diameter - parameters[3]
            )
            / parameters[3],
        },
        "identifiability": identifiability,
        "options": asdict(options),
        "effective_noise_fraction": (
            options.noise_fraction / math.sqrt(options.cycles)
            if args.reference is None
            else None
        ),
        "cost": {
            "coarse_grid_evaluations": (
                options.grid_points
                if options.invert == "permeability"
                else options.grid_points**2
            ),
            "multistart_total_seconds": sum(
                item.elapsed_seconds for item in estimates
            ),
            "mean_seconds_per_start": float(
                np.mean([item.elapsed_seconds for item in estimates])
            ),
            "frozen_network_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_inverse_variables": (
                1 if options.invert == "permeability" else 2
            ),
        },
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    _write_outputs(
        args.output, result, estimates, history, xy, target, prediction
    )
    _plot(
        args.output / "field_inverse_result.png",
        target,
        prediction,
        history,
        best,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inverse identification from a frozen SeaKAN using sparse "
            "interior complex-pressure sensors."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/generalization.pt"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "outputs/strict_multiseed_20260728/accuracy_reference_cases.csv"
        ),
    )
    parser.add_argument("--case-id", default="blind_004")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="independent COMSOL complex-pressure domain CSV",
    )
    parser.add_argument(
        "--invert", choices=("both", "permeability"), default="both"
    )
    parser.add_argument(
        "--sensor-layout", choices=("fixed", "fisher"), default="fixed"
    )
    parser.add_argument("--sensors", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--grid-points", type=int, default=11)
    parser.add_argument("--adam-steps", type=int, default=400)
    parser.add_argument("--lbfgs-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=4.0e-2)
    parser.add_argument("--seed", type=int, default=202421)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/inverse/field")
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
