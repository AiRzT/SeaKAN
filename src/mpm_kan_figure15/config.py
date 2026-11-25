from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class WaveParameters:
    height: float
    period: float
    water_depth: float = 0.40
    rho_w: float = 1000.0
    gravity: float = 9.81

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period

    @property
    def wave_number(self) -> float:
        return brentq(
            lambda value: (
                self.gravity
                * value
                * math.tanh(value * self.water_depth)
                - self.omega**2
            ),
            1.0e-8,
            100.0,
        )

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.wave_number

    @property
    def gamma_w(self) -> float:
        return self.rho_w * self.gravity

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
    porosity: float = 0.396
    permeability: float = 3.56e-5
    saturation: float = 0.999
    density: float = 1500.0
    shear_modulus: float = 1.0e7
    poisson_ratio: float = 0.32
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

    def pore_water_compressibility(
        self, wave: WaveParameters
    ) -> float:
        hydrostatic_pressure = wave.gamma_w * wave.water_depth
        return (
            1.0 / self.water_bulk_modulus
            + (1.0 - self.saturation) / hydrostatic_pressure
        )


@dataclass(frozen=True)
class Geometry:
    case_name: str
    trench_depth: float
    backfill_depth: float
    seabed_thickness: float = 0.58
    basin_width: float = 2.0
    pipe_diameter: float = 0.10
    trench_bottom_half_width: float = 0.10
    trench_horizontal_to_vertical: float = 2.0

    @property
    def pipe_radius(self) -> float:
        return 0.5 * self.pipe_diameter

    @property
    def pipe_center_z(self) -> float:
        return -self.trench_depth + self.pipe_radius

    @property
    def pipe_center_y(self) -> float:
        return (
            self.pipe_center_z + self.seabed_thickness
        ) / self.seabed_thickness

    @property
    def radius_over_height(self) -> float:
        return self.pipe_radius / self.seabed_thickness

    @property
    def half_width_over_height(self) -> float:
        return (
            0.5 * self.basin_width / self.seabed_thickness
        )

    @property
    def water_top_y(self) -> float:
        return 1.0 + 0.40 / self.seabed_thickness

    @property
    def embedded_full_circle(self) -> bool:
        return self.case_name == "test10"

    @property
    def backfill_surface_z(self) -> float:
        return -self.trench_depth + self.backfill_depth

    @property
    def fill_intersection_x(self) -> float:
        return (
            self.trench_bottom_half_width
            + self.trench_horizontal_to_vertical
            * self.backfill_depth
        )

    @property
    def trench_edge_x(self) -> float:
        return (
            self.trench_bottom_half_width
            + self.trench_horizontal_to_vertical
            * self.trench_depth
        )

    def surface_z_numpy(
        self, x_m: np.ndarray
    ) -> np.ndarray:
        if self.embedded_full_circle:
            return np.zeros_like(x_m, dtype=float)
        absolute = np.abs(np.asarray(x_m, dtype=float))
        slope_surface = (
            -self.trench_depth
            + (
                absolute - self.trench_bottom_half_width
            )
            / self.trench_horizontal_to_vertical
        )
        return np.where(
            absolute <= self.fill_intersection_x,
            self.backfill_surface_z,
            np.where(
                absolute < self.trench_edge_x,
                slope_surface,
                0.0,
            ),
        )

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def dimensionless_coefficients(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
) -> dict[str, float]:
    hydraulic = soil.permeability / wave.gamma_w
    storage = soil.porosity * soil.pore_water_compressibility(
        wave
    )
    height = geometry.seabed_thickness
    return {
        "kappa": wave.wave_number * height,
        "mass_pressure": (
            wave.omega * storage * height**2 / hydraulic
        ),
        "mass_strain": (
            wave.omega
            * height**2
            / (hydraulic * soil.shear_modulus)
        ),
        "hydraulic_m2_per_pa_s": hydraulic,
        "storage_per_pa": storage,
        "pressure_scale_pa": wave.mudline_pressure_amplitude,
        "wavelength_m": wave.wavelength,
    }


def figure15_setups() -> dict[
    str, tuple[WaveParameters, SoilParameters, Geometry]
]:
    soil = SoilParameters()
    return {
        "test10": (
            WaveParameters(height=0.14, period=1.40),
            soil,
            Geometry(
                case_name="test10",
                trench_depth=0.15,
                backfill_depth=0.15,
            ),
        ),
        "test49": (
            WaveParameters(height=0.12, period=1.60),
            soil,
            Geometry(
                case_name="test49",
                trench_depth=0.20,
                backfill_depth=0.05,
            ),
        ),
    }
