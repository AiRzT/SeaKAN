from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class WaveParameters:
    """Wave parameters from Zhu et al. (2024), Table 6."""

    height: float = 0.50
    period: float = 6.00
    water_depth: float = 10.00
    rho_w: float = 1000.0
    gravity: float = 9.81

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period

    @property
    def gamma_w(self) -> float:
        return self.rho_w * self.gravity

    @property
    def wave_number(self) -> float:
        omega_squared = self.omega**2
        return brentq(
            lambda value: (
                self.gravity
                * value
                * math.tanh(value * self.water_depth)
                - omega_squared
            ),
            1.0e-8,
            5.0,
        )

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.wave_number

    @property
    def mudline_pressure_amplitude(self) -> float:
        return (
            self.rho_w
            * self.gravity
            * self.height
            / (2.0 * math.cosh(self.wave_number * self.water_depth))
        )


@dataclass(frozen=True)
class SoilParameters:
    """Homogeneous seabed parameters from Table 6."""

    porosity: float = 0.40
    permeability: float = 0.01
    saturation: float = 0.98
    density: float = 1500.0
    shear_modulus: float = 1.0e7
    poisson_ratio: float = 0.30
    water_bulk_modulus: float = 1.95e9

    @property
    def lame_lambda(self) -> float:
        return (
            2.0
            * self.shear_modulus
            * self.poisson_ratio
            / (1.0 - 2.0 * self.poisson_ratio)
        )

    @property
    def lambda_over_g(self) -> float:
        return self.lame_lambda / self.shear_modulus

    def pore_water_compressibility(self, wave: WaveParameters) -> float:
        hydrostatic_mudline_pressure = (
            wave.gamma_w * wave.water_depth
        )
        return (
            1.0 / self.water_bulk_modulus
            + (1.0 - self.saturation) / hydrostatic_mudline_pressure
        )


@dataclass(frozen=True)
class Geometry:
    """One-wavelength strip of the 25 m homogeneous seabed."""

    seabed_thickness: float = 25.0
    mudline_z: float = -10.0
    evaluation_x: float = 50.0

    @property
    def bottom_z(self) -> float:
        return self.mudline_z - self.seabed_thickness

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def dimensionless_coefficients(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
) -> dict[str, float]:
    hydraulic = soil.permeability / wave.gamma_w
    storage = soil.porosity * soil.pore_water_compressibility(wave)
    thickness = geometry.seabed_thickness
    return {
        "kappa": wave.wave_number * thickness,
        "mass_pressure": (
            wave.omega * storage * thickness**2 / hydraulic
        ),
        "mass_strain": (
            wave.omega
            * thickness**2
            / (hydraulic * soil.shear_modulus)
        ),
        "hydraulic_m2_per_pa_s": hydraulic,
        "storage_per_pa": storage,
        "pressure_scale_pa": wave.mudline_pressure_amplitude,
        "wavelength_m": wave.wavelength,
    }


def figure13_setup() -> tuple[WaveParameters, SoilParameters, Geometry]:
    return WaveParameters(), SoilParameters(), Geometry()
