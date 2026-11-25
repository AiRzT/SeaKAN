from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kan import KAN


@dataclass
class HarmonicFields:
    u: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor


class HarmonicKAN(nn.Module):
    """
    KAN approximation for dimensionless harmonic amplitudes.

    Input y is zero at the rigid seabed bottom and one at the mudline.
    The transforms enforce U(0)=W(0)=0 and P(1)=1+0i exactly.
    """

    def __init__(
        self,
        width: int = 18,
        hidden_layers: int = 2,
        n_basis: int = 10,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        widths = [1] + [width] * hidden_layers + [6]
        self.kan = KAN(widths, n_basis=n_basis, dtype=dtype)

    def forward(self, y: torch.Tensor) -> HarmonicFields:
        raw = self.kan(2.0 * y - 1.0)
        u = y[:, 0] * torch.complex(raw[:, 0], raw[:, 1])
        w = y[:, 0] * torch.complex(raw[:, 2], raw[:, 3])
        pressure_correction = torch.complex(raw[:, 4], raw[:, 5])
        # A linear distance factor enforces P(1)=1 without imposing an
        # unphysical zero mudline pressure gradient.
        p = torch.ones_like(pressure_correction) + (1.0 - y[:, 0]) * pressure_correction
        return HarmonicFields(u=u, w=w, p=p)

    def spline_roughness(self) -> torch.Tensor:
        return self.kan.spline_roughness()
