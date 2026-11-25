"""MPM-KAN physics-informed solver for Zhu et al. (2024) Figures 7 and 8."""

from .config import Fig7Case, SoilParameters, WaveParameters, fig7_cases
from .figure8 import figure8_setup, linear_wave_number
from .model import HarmonicKAN
from .mpm import MaterialPointGrid1D
from .physics import MPMKANBiotProblem

__all__ = [
    "Fig7Case",
    "HarmonicKAN",
    "MPMKANBiotProblem",
    "MaterialPointGrid1D",
    "SoilParameters",
    "WaveParameters",
    "fig7_cases",
    "figure8_setup",
    "linear_wave_number",
]
