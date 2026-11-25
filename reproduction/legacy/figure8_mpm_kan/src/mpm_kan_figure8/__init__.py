"""MPM-KAN physics-informed reproduction of Zhu et al. (2024) Figure 8."""

from .config import SeabedCase, SoilParameters, WaveParameters
from .figure8 import figure8_setup, linear_wave_number
from .model import HarmonicKAN
from .mpm import MaterialPointGrid1D
from .physics import MPMKANBiotProblem

__version__ = "1.0.0"

__all__ = [
    "HarmonicKAN",
    "MPMKANBiotProblem",
    "MaterialPointGrid1D",
    "SeabedCase",
    "SoilParameters",
    "WaveParameters",
    "figure8_setup",
    "linear_wave_number",
]
