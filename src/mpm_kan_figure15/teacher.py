from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import solve_bvp

from .config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
)
from .model import SoilPipeKAN, surface_y


@dataclass(frozen=True)
class BiotBVPProfile:
    y: np.ndarray
    displacement_x: np.ndarray
    displacement_y: np.ndarray
    pressure: np.ndarray


def solve_biot_bvp(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    *,
    points: int = 401,
) -> BiotBVPProfile:
    """Solve the flat-bed 1-D harmonic Biot reference numerically.

    This BVP supplies an initialization only.  It contains no pipeline,
    trench, experiment, or digitized Figure 15 values.
    """

    coefficients = dimensionless_coefficients(
        wave, soil, geometry
    )
    kappa = coefficients["kappa"]
    mass_pressure = coefficients["mass_pressure"]
    mass_strain = coefficients["mass_strain"]
    navier = 1.0 / (1.0 - 2.0 * soil.poisson_ratio)
    lame = soil.lambda_over_g

    def equations(
        _: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        values = state[:6] + 1j * state[6:]
        u, du, w, dw, pressure, dp = values
        strain = 1j * kappa * u + dw
        ddu = (
            kappa**2 * u
            - navier * 1j * kappa * strain
            + 1j * kappa * pressure
        )
        ddw = (
            kappa**2 * w
            - navier * 1j * kappa * du
            + dp
        ) / (1.0 + navier)
        ddp = (
            kappa**2 * pressure
            - 1j * mass_pressure * pressure
            - 1j * mass_strain * strain
        )
        derivative = np.vstack((du, ddu, dw, ddw, dp, ddp))
        return np.vstack((derivative.real, derivative.imag))

    def boundary(
        bottom: np.ndarray, top: np.ndarray
    ) -> np.ndarray:
        lower = bottom[:6] + 1j * bottom[6:]
        upper = top[:6] + 1j * top[6:]
        u0, _, w0, _, _, dp0 = lower
        u1, du1, w1, dw1, pressure1, _ = upper
        constraints = np.asarray(
            [
                u0,
                w0,
                dp0,
                pressure1 - 1.0,
                du1 + 1j * kappa * w1,
                lame * 1j * kappa * u1
                + (2.0 + lame) * dw1,
            ]
        )
        return np.concatenate(
            (constraints.real, constraints.imag)
        )

    initial_y = np.linspace(0.0, 1.0, 161)
    initial = np.zeros((12, initial_y.size))
    initial[4] = 1.0
    solution = solve_bvp(
        equations,
        boundary,
        initial_y,
        initial,
        tol=1.0e-7,
        max_nodes=20000,
    )
    if not solution.success:
        raise RuntimeError(
            f"Biot BVP teacher failed: {solution.message}"
        )
    y = np.linspace(0.0, 1.0, points)
    evaluated = solution.sol(y)
    displacement_x = evaluated[0] + 1j * evaluated[6]
    displacement_y = evaluated[2] + 1j * evaluated[8]
    pressure = evaluated[4] + 1j * evaluated[10]
    return BiotBVPProfile(
        y=y,
        displacement_x=displacement_x,
        displacement_y=displacement_y,
        pressure=pressure,
    )


def field_teacher_targets(
    model: SoilPipeKAN,
    xy: torch.Tensor,
    profile: BiotBVPProfile,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map the flat-bed BVP profile to the current surface geometry."""

    x = xy[:, 0]
    local_top = surface_y(x, model.geometry)
    normalized_depth = (xy[:, 1] / local_top).clamp(
        0.0, 1.0
    )
    depth_numpy = normalized_depth.detach().cpu().numpy()
    def interpolate(values: np.ndarray) -> torch.Tensor:
        real = np.interp(
            depth_numpy, profile.y, values.real
        )
        imaginary = np.interp(
            depth_numpy, profile.y, values.imag
        )
        return torch.complex(
            torch.as_tensor(
                real, dtype=xy.dtype, device=xy.device
            ),
            torch.as_tensor(
                imaginary, dtype=xy.dtype, device=xy.device
            ),
        )

    forcing = model.interface_pressure(x).detach()
    return (
        interpolate(profile.displacement_x) * forcing,
        interpolate(profile.displacement_y) * forcing,
        interpolate(profile.pressure) * forcing,
    )
