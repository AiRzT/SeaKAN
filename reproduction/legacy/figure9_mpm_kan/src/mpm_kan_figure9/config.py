from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WaveParameters:
    """Wave-flume parameters from Zhu et al. (2024), Table 5."""

    height: float = 0.0524
    period: float = 0.9
    wavelength: float = 1.25
    water_depth: float = 0.533
    rho_w: float = 1000.0
    gravity: float = 9.81

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period

    @property
    def wave_number(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def gamma_w(self) -> float:
        return self.rho_w * self.gravity

    @property
    def hydrostatic_bed_pressure(self) -> float:
        return self.gamma_w * self.water_depth

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
    """Homogeneous sediment parameters from Table 5."""

    shear_modulus: float = 6.4e5
    poisson_ratio: float = 0.33
    porosity: float = 0.42
    permeability: float = 0.0011
    saturation: float = 0.95
    water_bulk_modulus: float = 1.95e9

    @property
    def lame_lambda(self) -> float:
        return 2.0 * self.shear_modulus * self.poisson_ratio / (
            1.0 - 2.0 * self.poisson_ratio
        )

    @property
    def lambda_over_g(self) -> float:
        return self.lame_lambda / self.shear_modulus

    def pore_water_compressibility(self, wave: WaveParameters) -> float:
        return 1.0 / self.water_bulk_modulus + (
            1.0 - self.saturation
        ) / wave.hydrostatic_bed_pressure


@dataclass(frozen=True)
class Geometry:
    """Trench and pipeline geometry in Figure 9."""

    trench_height: float = 0.826
    trench_width: float = 4.57
    pipe_diameter: float = 0.168
    pipe_center_depth: float = 0.167

    @property
    def pipe_radius(self) -> float:
        return 0.5 * self.pipe_diameter

    @property
    def half_width_over_height(self) -> float:
        return 0.5 * self.trench_width / self.trench_height

    @property
    def radius_over_height(self) -> float:
        return self.pipe_radius / self.trench_height

    @property
    def center_y(self) -> float:
        # Dimensionless Y: 0 at the rigid bottom and 1 at the mudline.
        return 1.0 - self.pipe_center_depth / self.trench_height

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def dimensionless_coefficients(
    wave: WaveParameters, soil: SoilParameters, geometry: Geometry
) -> dict[str, float]:
    hydraulic = soil.permeability / wave.gamma_w
    storage = soil.porosity * soil.pore_water_compressibility(wave)
    height = geometry.trench_height
    return {
        "kappa": wave.wave_number * height,
        "mass_pressure": wave.omega * storage * height**2 / hydraulic,
        "mass_strain": (
            wave.omega * height**2 / (hydraulic * soil.shear_modulus)
        ),
        "hydraulic_m2_per_pa_s": hydraulic,
        "storage_per_pa": storage,
    }


def figure9_setup() -> tuple[WaveParameters, SoilParameters, Geometry]:
    return WaveParameters(), SoilParameters(), Geometry()
