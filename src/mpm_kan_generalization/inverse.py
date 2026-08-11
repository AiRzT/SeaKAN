from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ParameterDomain, PhysicalConstants, solve_wave_number
from .metrics import regression_metrics
from .model import ParametricSoilKAN
from .training import load_checkpoint


@dataclass(frozen=True)
class InverseOptions:
    sensor_count: int = 24
    adam_steps: int = 600
    lbfgs_steps: int = 60
    learning_rate: float = 5.0e-2
    starts: int = 8
    noise_fraction: float = 0.0
    prior_weight: float = 0.0
    seed: int = 202420


@dataclass(frozen=True)
class PipeObservations:
    theta_rad: np.ndarray
    normalized_pressure: np.ndarray
    source: str
    case_id: str


@dataclass(frozen=True)
class InverseEstimate:
    start: int
    permeability_m_per_s: float
    burial_center_depth_over_diameter: float
    loss: float
    relative_l2: float
    rmse: float
    r2: float | None
    elapsed_seconds: float
    iterations: int


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


def bounded_inverse_parameters(
    latent: torch.Tensor,
    domain: ParameterDomain,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map two unconstrained values to log-permeability and burial bounds."""

    fraction = torch.sigmoid(latent)
    log_bounds = (
        math.log10(domain.permeability_m_per_s[0]),
        math.log10(domain.permeability_m_per_s[1]),
    )
    log_permeability = (
        log_bounds[0]
        + (log_bounds[1] - log_bounds[0]) * fraction[0]
    )
    burial = (
        domain.burial_center_depth_over_diameter[0]
        + (
            domain.burial_center_depth_over_diameter[1]
            - domain.burial_center_depth_over_diameter[0]
        )
        * fraction[1]
    )
    return torch.pow(10.0, log_permeability), burial


def initial_latent_from_fraction(
    fractions: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = _logit(np.asarray(fractions, dtype=np.float64))
    return torch.tensor(
        values,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )


def _fixed_kappa(
    period_s: float,
    constants: PhysicalConstants,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    wave_number = solve_wave_number(
        np.asarray([period_s], dtype=np.float64), constants
    )[0]
    return torch.tensor(
        wave_number * constants.seabed_thickness_m,
        device=device,
        dtype=dtype,
    )


def predict_pipe_pressure(
    model: ParametricSoilKAN,
    base_parameters: torch.Tensor,
    theta: torch.Tensor,
    permeability: torch.Tensor,
    burial: torch.Tensor,
    constants: PhysicalConstants,
) -> torch.Tensor:
    """Evaluate the frozen conditional model on a candidate pipe boundary."""

    count = theta.numel()
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
    pipe_center_y = (
        1.0
        - burial
        * constants.pipe_diameter_m
        / constants.seabed_thickness_m
    )
    x = constants.radius * torch.cos(theta)
    y = pipe_center_y + constants.radius * torch.sin(theta)
    xy = torch.stack((x, y), dim=1)
    kappa = _fixed_kappa(
        float(base_parameters[0, 1].detach().cpu()),
        constants,
        device=theta.device,
        dtype=theta.dtype,
    )
    return model(
        xy,
        parameters,
        kappa.expand(count),
        pipe_center_y.expand(count),
    ).p.abs()


def select_sensor_rows(
    theta: np.ndarray,
    pressure: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 4:
        raise ValueError("at least four sensors are required")
    if theta.size != pressure.size:
        raise ValueError("theta and pressure arrays must have equal length")
    order = np.argsort(np.mod(theta, 2.0 * np.pi))
    theta = np.asarray(theta, dtype=np.float64)[order]
    pressure = np.asarray(pressure, dtype=np.float64)[order]
    unique, indices = np.unique(
        np.round(np.mod(theta, 2.0 * np.pi), 12),
        return_index=True,
    )
    theta = theta[indices]
    pressure = pressure[indices]
    count = min(count, unique.size)
    selected = np.linspace(0, unique.size - 1, count, dtype=int)
    return theta[selected], pressure[selected]


def read_case_parameters(path: Path, case_id: str) -> np.ndarray:
    names = (
        "wave_height_m",
        "wave_period_s",
        "permeability_m_per_s",
        "burial_center_depth_over_diameter",
        "xi_1",
        "xi_2",
        "xi_3",
        "xi_4",
    )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("case_id") == case_id:
                return np.asarray(
                    [float(row[name]) for name in names],
                    dtype=np.float64,
                )
    raise ValueError(f"case {case_id!r} was not found in {path}")


def read_reference_observations(
    path: Path,
    case_id: str,
    sensor_count: int,
) -> PipeObservations:
    theta: list[float] = []
    pressure: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("case_id") != case_id:
                continue
            theta.append(float(row["theta_rad"]))
            pressure.append(float(row["reference_normalized_pressure"]))
    if not theta:
        raise ValueError(f"case {case_id!r} was not found in {path}")
    selected_theta, selected_pressure = select_sensor_rows(
        np.asarray(theta), np.asarray(pressure), sensor_count
    )
    return PipeObservations(
        theta_rad=selected_theta,
        normalized_pressure=selected_pressure,
        source=str(path.resolve()),
        case_id=case_id,
    )


def synthetic_observations(
    model: ParametricSoilKAN,
    parameters: np.ndarray,
    constants: PhysicalConstants,
    sensor_count: int,
    noise_fraction: float,
    seed: int,
) -> PipeObservations:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    theta = torch.linspace(
        0.0,
        2.0 * torch.pi,
        sensor_count + 1,
        device=device,
        dtype=dtype,
    )[:-1]
    base = torch.as_tensor(
        parameters[None, :], device=device, dtype=dtype
    )
    permeability = base[0, 2]
    burial = base[0, 3]
    with torch.no_grad():
        pressure = predict_pipe_pressure(
            model,
            base,
            theta,
            permeability,
            burial,
            constants,
        ).cpu().numpy()
    if noise_fraction > 0.0:
        generator = np.random.default_rng(seed)
        scale = noise_fraction * float(np.sqrt(np.mean(pressure**2)))
        pressure = np.maximum(
            pressure + generator.normal(0.0, scale, pressure.shape),
            0.0,
        )
    return PipeObservations(
        theta_rad=theta.cpu().numpy(),
        normalized_pressure=pressure,
        source="frozen-model synthetic observations",
        case_id="synthetic",
    )


def local_identifiability(
    model: ParametricSoilKAN,
    base_parameters: torch.Tensor,
    theta: torch.Tensor,
    true_permeability: float,
    true_burial: float,
    domain: ParameterDomain,
    constants: PhysicalConstants,
) -> dict[str, object]:
    log_bounds = np.log10(domain.permeability_m_per_s)
    fractions = np.asarray(
        [
            (
                math.log10(true_permeability) - log_bounds[0]
            )
            / (log_bounds[1] - log_bounds[0]),
            (
                true_burial
                - domain.burial_center_depth_over_diameter[0]
            )
            / (
                domain.burial_center_depth_over_diameter[1]
                - domain.burial_center_depth_over_diameter[0]
            ),
        ]
    )
    latent = initial_latent_from_fraction(
        fractions,
        device=theta.device,
        dtype=theta.dtype,
    )

    def response(value: torch.Tensor) -> torch.Tensor:
        permeability, burial = bounded_inverse_parameters(value, domain)
        return predict_pipe_pressure(
            model,
            base_parameters,
            theta,
            permeability,
            burial,
            constants,
        )

    jacobian = torch.autograd.functional.jacobian(response, latent)
    singular = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
    tolerance = np.finfo(np.float64).eps * max(jacobian.shape)
    nonzero = singular[singular > tolerance]
    condition = (
        float(nonzero.max() / nonzero.min())
        if nonzero.size >= 2
        else float("inf")
    )
    return {
        "singular_values": singular.tolist(),
        "condition_number": condition,
        "rank": int(nonzero.size),
        "parameters": ["log10_permeability", "burial_ratio"],
        "parameterization": "bounded latent logits",
    }


def invert_parameters(
    model: ParametricSoilKAN,
    base_parameters: np.ndarray,
    observations: PipeObservations,
    domain: ParameterDomain,
    constants: PhysicalConstants,
    options: InverseOptions,
) -> tuple[InverseEstimate, list[InverseEstimate], list[dict[str, float]]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    theta = torch.as_tensor(
        observations.theta_rad, device=device, dtype=dtype
    )
    target = torch.as_tensor(
        observations.normalized_pressure, device=device, dtype=dtype
    )
    base = torch.as_tensor(
        base_parameters[None, :], device=device, dtype=dtype
    )
    target_scale = target.square().mean().clamp_min(1.0e-12)
    generator = np.random.default_rng(options.seed)
    initial = generator.uniform(0.08, 0.92, size=(options.starts, 2))
    initial[0] = np.asarray([0.5, 0.5])
    estimates: list[InverseEstimate] = []
    best_history: list[dict[str, float]] = []

    for start_index, fraction in enumerate(initial):
        latent = initial_latent_from_fraction(
            fraction, device=device, dtype=dtype
        )
        optimizer = torch.optim.Adam(
            [latent], lr=options.learning_rate
        )
        history: list[dict[str, float]] = []
        started = time.perf_counter()

        def objective() -> tuple[torch.Tensor, torch.Tensor]:
            permeability, burial = bounded_inverse_parameters(
                latent, domain
            )
            prediction = predict_pipe_pressure(
                model,
                base,
                theta,
                permeability,
                burial,
                constants,
            )
            data_loss = (prediction - target).square().mean() / target_scale
            prior = options.prior_weight * latent.square().mean()
            return data_loss + prior, prediction

        for step in range(options.adam_steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective()
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 20 == 0:
                permeability, burial = bounded_inverse_parameters(
                    latent, domain
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
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss, _ = objective()
            loss.backward()
            return loss

        lbfgs.step(closure)
        loss, prediction = objective()
        permeability, burial = bounded_inverse_parameters(latent, domain)
        metrics = regression_metrics(
            target.detach().cpu().numpy(),
            prediction.detach().cpu().numpy(),
        )
        elapsed = time.perf_counter() - started
        estimate = InverseEstimate(
            start=start_index,
            permeability_m_per_s=float(permeability.detach().cpu()),
            burial_center_depth_over_diameter=float(
                burial.detach().cpu()
            ),
            loss=float(loss.detach().cpu()),
            relative_l2=metrics.relative_l2,
            rmse=metrics.rmse,
            r2=metrics.r2,
            elapsed_seconds=elapsed,
            iterations=options.adam_steps + options.lbfgs_steps,
        )
        estimates.append(estimate)
        if estimate.loss <= min(item.loss for item in estimates):
            permeability_value = estimate.permeability_m_per_s
            burial_value = estimate.burial_center_depth_over_diameter
            history.append(
                {
                    "iteration": float(estimate.iterations),
                    "loss": estimate.loss,
                    "permeability_m_per_s": permeability_value,
                    "burial_ratio": burial_value,
                }
            )
            best_history = history

    estimates.sort(key=lambda item: item.loss)
    return estimates[0], estimates, best_history


def _predict_estimate(
    model: ParametricSoilKAN,
    parameters: np.ndarray,
    theta: np.ndarray,
    estimate: InverseEstimate,
    constants: PhysicalConstants,
) -> np.ndarray:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    base = torch.as_tensor(parameters[None, :], device=device, dtype=dtype)
    query = torch.as_tensor(theta, device=device, dtype=dtype)
    permeability = torch.tensor(
        estimate.permeability_m_per_s, device=device, dtype=dtype
    )
    burial = torch.tensor(
        estimate.burial_center_depth_over_diameter,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        return predict_pipe_pressure(
            model,
            base,
            query,
            permeability,
            burial,
            constants,
        ).cpu().numpy()


def _write_estimates(path: Path, estimates: list[InverseEstimate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(estimates[0]).keys())
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(asdict(estimate))


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def _plot_result(
    path: Path,
    observations: PipeObservations,
    prediction: np.ndarray,
    history: list[dict[str, float]],
    estimate: InverseEstimate,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(
        observations.theta_rad / np.pi,
        observations.normalized_pressure,
        "o",
        color="#6A51A3",
        markerfacecolor="white",
        label="Observations",
    )
    axes[0].plot(
        observations.theta_rad / np.pi,
        prediction,
        "-",
        color="#0072B2",
        linewidth=1.7,
        label="Inverse prediction",
    )
    axes[0].set_xlabel(r"$\theta/\pi$")
    axes[0].set_ylabel(r"$|p|/p_0$")
    axes[0].grid(True, color="#C9D0D8", linewidth=0.45)
    axes[0].legend(frameon=False, fontsize=8)

    if history:
        iteration = [row["iteration"] for row in history]
        loss = [row["loss"] for row in history]
        axes[1].semilogy(
            iteration, loss, color="#D55E00", linewidth=1.6
        )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Normalised observation loss")
    axes[1].grid(True, color="#C9D0D8", linewidth=0.45)
    axes[1].text(
        0.04,
        0.06,
        (
            rf"$k={estimate.permeability_m_per_s:.3e}$ m/s"
            "\n"
            rf"$b/D={estimate.burial_center_depth_over_diameter:.3f}$"
            "\n"
            rf"$L_2={100.0 * estimate.relative_l2:.2f}\%$"
        ),
        transform=axes[1].transAxes,
        fontsize=8,
        va="bottom",
    )
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
    figure.tight_layout()
    figure.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def run_inverse_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    (
        model,
        _design,
        domain,
        constants,
        _training_options,
        _augmented,
        _history,
    ) = load_checkpoint(args.checkpoint, device)
    parameters = read_case_parameters(args.cases, args.case_id)
    options = InverseOptions(
        sensor_count=args.sensors,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        learning_rate=args.learning_rate,
        starts=args.starts,
        noise_fraction=args.noise,
        prior_weight=args.prior_weight,
        seed=args.seed,
    )
    if args.mode == "synthetic":
        observations = synthetic_observations(
            model,
            parameters,
            constants,
            options.sensor_count,
            options.noise_fraction,
            options.seed,
        )
        observations = PipeObservations(
            observations.theta_rad,
            observations.normalized_pressure,
            observations.source,
            args.case_id,
        )
    else:
        observations = read_reference_observations(
            args.reference, args.case_id, options.sensor_count
        )
    best, estimates, inverse_history = invert_parameters(
        model,
        parameters,
        observations,
        domain,
        constants,
        options,
    )
    prediction = _predict_estimate(
        model,
        parameters,
        observations.theta_rad,
        best,
        constants,
    )
    base = torch.as_tensor(
        parameters[None, :],
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    theta = torch.as_tensor(
        observations.theta_rad,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    identifiability = local_identifiability(
        model,
        base,
        theta,
        float(parameters[2]),
        float(parameters[3]),
        domain,
        constants,
    )
    truth = {
        "permeability_m_per_s": float(parameters[2]),
        "burial_center_depth_over_diameter": float(parameters[3]),
    }
    result = {
        "mode": args.mode,
        "case_id": args.case_id,
        "observation_source": observations.source,
        "reference_used_in_forward_training": False,
        "forward_weights_frozen": True,
        "optimised_parameters": [
            "permeability_m_per_s",
            "burial_center_depth_over_diameter",
        ],
        "known_parameters": {
            "wave_height_m": float(parameters[0]),
            "wave_period_s": float(parameters[1]),
            "xi_1": float(parameters[4]),
            "xi_2": float(parameters[5]),
            "xi_3": float(parameters[6]),
            "xi_4": float(parameters[7]),
        },
        "truth": truth,
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
        "identifiability_at_declared_truth": identifiability,
        "options": asdict(options),
        "inverse_cost": {
            "trainable_inverse_variables": 2,
            "frozen_network_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "multistart_total_seconds": sum(
                item.elapsed_seconds for item in estimates
            ),
            "mean_seconds_per_start": float(
                np.mean([item.elapsed_seconds for item in estimates])
            ),
        },
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "interpretation": (
            "Synthetic mode tests inverse self-consistency of the frozen "
            "surrogate. COMSOL mode tests external inverse accuracy and "
            "includes forward-model discrepancy."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inverse_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_estimates(args.output / "inverse_multistart.csv", estimates)
    _write_history(args.output / "inverse_history.csv", inverse_history)
    with (args.output / "inverse_profile.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("theta_rad", "observed_pressure", "inverse_prediction")
        )
        writer.writerows(
            zip(
                observations.theta_rad,
                observations.normalized_pressure,
                prediction,
                strict=True,
            )
        )
    _plot_result(
        args.output / "inverse_result.png",
        observations,
        prediction,
        inverse_history,
        best,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a conditional SeaKAN checkpoint and identify mean "
            "permeability and pipe burial from sparse wall-pressure data."
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
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "outputs/strict_multiseed_20260728/comsol_reference/"
            "comsol_pipe_reference.csv"
        ),
    )
    parser.add_argument("--case-id", default="blind_004")
    parser.add_argument(
        "--mode", choices=("synthetic", "comsol"), default="synthetic"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sensors", type=int, default=24)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--adam-steps", type=int, default=600)
    parser.add_argument("--lbfgs-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5.0e-2)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=202420)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/inverse/synthetic")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_inverse_experiment(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
