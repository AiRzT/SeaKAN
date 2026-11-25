from __future__ import annotations

import math

from .config import SeabedCase, SoilParameters, WaveParameters


FIGURE8_WAVE_HEIGHT = 3.5


def linear_wave_number(
    period: float, water_depth: float, gravity: float = 9.81
) -> float:
    """Solve omega^2 = g k tanh(k d) by safeguarded bisection."""

    if period <= 0.0 or water_depth <= 0.0 or gravity <= 0.0:
        raise ValueError("period, water_depth, and gravity must be positive")
    omega_squared = (2.0 * math.pi / period) ** 2

    def residual(wave_number: float) -> float:
        return (
            gravity
            * wave_number
            * math.tanh(wave_number * water_depth)
            - omega_squared
        )

    lower = 0.0
    upper = max(omega_squared / gravity, 1.0 / water_depth)
    while residual(upper) < 0.0:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def figure8_setup() -> tuple[WaveParameters, SoilParameters, SeabedCase]:
    """Table 4 parameters for Zhu et al. (2024) Figure 8."""

    period = 9.0
    water_depth = 5.2
    wave_number = linear_wave_number(period, water_depth)
    wave = WaveParameters(
        period=period,
        wavelength=2.0 * math.pi / wave_number,
        water_depth=water_depth,
        pressure_amplitude=1.0,
    )
    soil = SoilParameters(
        shear_modulus=1.27e7,
        poisson_ratio=0.3,
        porosity=0.425,
        thickness=1.8,
    )
    case = SeabedCase(
        key="figure8",
        label="Liu et al. (2015a) test",
        permeability=1.8e-4,
        saturation=0.996,
        color="#1f77b4",
    )
    return wave, soil, case
