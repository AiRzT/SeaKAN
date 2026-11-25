from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PhysicalConstants:
    water_depth_m: float = 0.40
    seabed_thickness_m: float = 0.58
    basin_width_m: float = 2.0
    pipe_diameter_m: float = 0.10
    rho_w: float = 1000.0
    gravity: float = 9.81
    porosity: float = 0.396
    saturation: float = 0.999
    shear_modulus_pa: float = 1.0e7
    poisson_ratio: float = 0.32
    water_bulk_modulus_pa: float = 1.95e9

    @property
    def half_width(self) -> float:
        return (
            0.5 * self.basin_width_m / self.seabed_thickness_m
        )

    @property
    def radius(self) -> float:
        return (
            0.5 * self.pipe_diameter_m / self.seabed_thickness_m
        )

    @property
    def lambda_over_g(self) -> float:
        return 2.0 * self.poisson_ratio / (
            1.0 - 2.0 * self.poisson_ratio
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterDomain:
    wave_height_m: tuple[float, float] = (0.08, 0.16)
    wave_period_s: tuple[float, float] = (1.20, 1.80)
    permeability_m_per_s: tuple[float, float] = (
        1.0e-5,
        8.0e-5,
    )
    burial_center_depth_over_diameter: tuple[float, float] = (
        0.75,
        1.50,
    )
    random_coefficient: tuple[float, float] = (-1.75, 1.75)
    permeability_cov: float = 0.35
    random_modes: int = 4

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


PARAMETER_NAMES = (
    "wave_height_m",
    "wave_period_s",
    "permeability_m_per_s",
    "burial_center_depth_over_diameter",
    "xi_1",
    "xi_2",
    "xi_3",
    "xi_4",
)


def solve_wave_number(
    period_s: np.ndarray,
    constants: PhysicalConstants,
) -> np.ndarray:
    period = np.asarray(period_s, dtype=np.float64)
    omega = 2.0 * np.pi / period
    value = np.maximum(
        omega**2 / constants.gravity, 1.0e-6
    )
    for _ in range(40):
        kh = value * constants.water_depth_m
        tangent = np.tanh(kh)
        function = (
            constants.gravity * value * tangent - omega**2
        )
        derivative = constants.gravity * (
            tangent
            + value
            * constants.water_depth_m
            * (1.0 - tangent**2)
        )
        value = np.maximum(value - function / derivative, 1.0e-8)
    return value


def derived_case_columns(
    parameters: np.ndarray,
    constants: PhysicalConstants,
) -> dict[str, np.ndarray]:
    height = parameters[:, 0]
    period = parameters[:, 1]
    permeability = parameters[:, 2]
    burial = parameters[:, 3]
    omega = 2.0 * np.pi / period
    wave_number = solve_wave_number(period, constants)
    gamma_w = constants.rho_w * constants.gravity
    compressibility = (
        1.0 / constants.water_bulk_modulus_pa
        + (1.0 - constants.saturation)
        / (gamma_w * constants.water_depth_m)
    )
    storage = constants.porosity * compressibility
    hydraulic = permeability / gamma_w
    seabed = constants.seabed_thickness_m
    pressure_scale = (
        gamma_w
        * height
        / (2.0 * np.cosh(
            wave_number * constants.water_depth_m
        ))
    )
    return {
        "omega": omega,
        "wave_number_m_inv": wave_number,
        "kappa": wave_number * seabed,
        "mass_pressure": (
            omega * storage * seabed**2 / hydraulic
        ),
        "mass_strain": (
            omega
            * seabed**2
            / (hydraulic * constants.shear_modulus_pa)
        ),
        "pressure_scale_pa": pressure_scale,
        "pipe_center_y": (
            1.0
            - burial
            * constants.pipe_diameter_m
            / constants.seabed_thickness_m
        ),
    }


def normalize_parameters(
    parameters: torch.Tensor,
    domain: ParameterDomain,
) -> torch.Tensor:
    bounds = torch.as_tensor(
        [
            domain.wave_height_m,
            domain.wave_period_s,
            (
                math.log10(domain.permeability_m_per_s[0]),
                math.log10(domain.permeability_m_per_s[1]),
            ),
            domain.burial_center_depth_over_diameter,
            *(
                [domain.random_coefficient]
                * domain.random_modes
            ),
        ],
        dtype=parameters.dtype,
        device=parameters.device,
    )
    transformed = torch.cat(
        (
            parameters[:, :2],
            torch.log10(parameters[:, 2:3]),
            parameters[:, 3:],
        ),
        dim=1,
    )
    return 2.0 * (
        transformed - bounds[:, 0]
    ) / (bounds[:, 1] - bounds[:, 0]) - 1.0
