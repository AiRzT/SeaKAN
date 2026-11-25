from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WaveParameters:
    """Table 3 wave loading."""

    period: float = 15.0
    wavelength: float = 311.53
    water_depth: float = 70.0
    rho_w: float = 1000.0
    gravity: float = 9.81
    pressure_amplitude: float = 1.0

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
        return self.rho_w * self.gravity * self.water_depth


@dataclass(frozen=True)
class SoilParameters:
    """Table 3 homogeneous poroelastic seabed properties."""

    shear_modulus: float = 1.0e7
    poisson_ratio: float = 1.0 / 3.0
    porosity: float = 0.3
    thickness: float = 25.0
    water_bulk_modulus: float = 1.95e9

    @property
    def lame_lambda(self) -> float:
        return 2.0 * self.shear_modulus * self.poisson_ratio / (
            1.0 - 2.0 * self.poisson_ratio
        )

    @property
    def lambda_over_g(self) -> float:
        return self.lame_lambda / self.shear_modulus


@dataclass(frozen=True)
class Fig7Case:
    key: str
    label: str
    permeability: float
    saturation: float
    color: str

    def pore_water_compressibility(
        self, wave: WaveParameters, soil: SoilParameters
    ) -> float:
        return 1.0 / soil.water_bulk_modulus + (
            1.0 - self.saturation
        ) / wave.hydrostatic_bed_pressure

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fig7_cases() -> dict[str, Fig7Case]:
    return {
        "saturated_fine": Fig7Case(
            "saturated_fine", "Saturated fine", 1.0e-4, 1.0, "#d62728"
        ),
        "saturated_coarse": Fig7Case(
            "saturated_coarse", "Saturated coarse", 1.0e-2, 1.0, "#1f77b4"
        ),
        "unsaturated_fine": Fig7Case(
            "unsaturated_fine", "Unsaturated fine", 1.0e-4, 0.932, "#d49a1f"
        ),
        "unsaturated_coarse": Fig7Case(
            "unsaturated_coarse", "Unsaturated coarse", 1.0e-2, 0.932, "#6f4bb2"
        ),
    }

