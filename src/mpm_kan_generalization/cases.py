from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import (
    PhysicalConstants,
    derived_case_columns,
)


@dataclass
class CaseTable:
    parameters: torch.Tensor
    kappa: torch.Tensor
    mass_pressure: torch.Tensor
    mass_strain: torch.Tensor
    pressure_scale_pa: torch.Tensor
    pipe_center_y: torch.Tensor

    @classmethod
    def from_numpy(
        cls,
        parameters: np.ndarray,
        constants: PhysicalConstants,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CaseTable:
        derived = derived_case_columns(parameters, constants)
        tensor = lambda value: torch.as_tensor(
            value, device=device, dtype=dtype
        )
        return cls(
            parameters=tensor(parameters),
            kappa=tensor(derived["kappa"]),
            mass_pressure=tensor(derived["mass_pressure"]),
            mass_strain=tensor(derived["mass_strain"]),
            pressure_scale_pa=tensor(
                derived["pressure_scale_pa"]
            ),
            pipe_center_y=tensor(
                derived["pipe_center_y"]
            ),
        )

    @property
    def n_cases(self) -> int:
        return int(self.parameters.shape[0])

    def to(
        self, device: torch.device, dtype: torch.dtype
    ) -> CaseTable:
        return CaseTable(
            **{
                name: getattr(self, name).to(
                    device=device, dtype=dtype
                )
                for name in (
                    "parameters",
                    "kappa",
                    "mass_pressure",
                    "mass_strain",
                    "pressure_scale_pa",
                    "pipe_center_y",
                )
            }
        )
