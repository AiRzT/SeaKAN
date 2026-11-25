from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from .config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
)


@dataclass
class ReferenceResult:
    z: np.ndarray
    pressure: np.ndarray
    vertical_effective_stress: np.ndarray
    maximum_rms_residual: float
    mesh_nodes: int

    def profile(self) -> dict[str, np.ndarray]:
        return {
            "z": self.z,
            "pressure": self.pressure,
            "vertical_effective_stress": (
                self.vertical_effective_stress
            ),
        }


def solve_reference(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    *,
    n_depth: int = 1001,
    tolerance: float = 1.0e-9,
) -> ReferenceResult:
    """
    Independent complex-amplitude boundary-value solve.

    It uses the progressive-wave separation only for post-training
    verification and never supplies labels or losses to the KAN.
    """

    coefficients = dimensionless_coefficients(
        wave, soil, geometry
    )
    kappa = coefficients["kappa"]
    mass_pressure = coefficients["mass_pressure"]
    mass_strain = coefficients["mass_strain"]
    lam = soil.lambda_over_g

    def equations(
        _: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        u, du, w, dw, pressure, dp = state
        ddu = (
            (2.0 + lam) * kappa**2 * u
            - 1j * kappa * (lam + 1.0) * dw
            + 1j * kappa * pressure
        )
        ddw = (
            kappa**2 * w
            - 1j * kappa * (lam + 1.0) * du
            + dp
        ) / (2.0 + lam)
        ddp = (
            kappa**2 * pressure
            - 1j * mass_pressure * pressure
            - 1j * mass_strain * (1j * kappa * u + dw)
        )
        return np.vstack((du, ddu, dw, ddw, dp, ddp))

    def boundaries(
        bottom: np.ndarray,
        top: np.ndarray,
    ) -> np.ndarray:
        u0, _, w0, _, _, dp0 = bottom
        u1, du1, w1, dw1, pressure1, _ = top
        sigma_y = (
            lam * 1j * kappa * u1
            + (2.0 + lam) * dw1
        )
        tau = du1 + 1j * kappa * w1
        return np.asarray(
            (u0, w0, dp0, pressure1 - 1.0, sigma_y, tau)
        )

    initial_y = np.linspace(0.0, 1.0, 201)
    initial = np.zeros((6, initial_y.size), dtype=np.complex128)
    initial[4] = initial_y
    solution = solve_bvp(
        equations,
        boundaries,
        initial_y,
        initial,
        tol=tolerance,
        max_nodes=20000,
    )
    if not solution.success:
        raise RuntimeError(
            f"Independent BVP reference failed: {solution.message}"
        )
    y = np.linspace(0.0, 1.0, n_depth)
    u, _, _, dw, pressure, _ = solution.sol(y)
    sigma_y = (
        lam * 1j * kappa * u + (2.0 + lam) * dw
    )
    z = (
        geometry.bottom_z + y * geometry.seabed_thickness
    )
    return ReferenceResult(
        z=z,
        pressure=np.abs(pressure),
        vertical_effective_stress=np.abs(sigma_y),
        maximum_rms_residual=float(
            np.max(solution.rms_residuals)
        ),
        mesh_nodes=int(solution.x.size),
    )
