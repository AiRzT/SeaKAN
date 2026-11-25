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


class DepthHarmonicKAN(nn.Module):
    """
    Progressive-wave harmonic Biot fields with depth-dependent KAN amplitudes.

    X=x/h and Y=(z-z_bottom)/h. The traveling phase exp(i*kappa*X)
    supplies the exact horizontal periodicity. Hard transforms impose the
    mudline pressure and rigid bottom displacement.
    """

    def __init__(
        self,
        kappa: float,
        *,
        width: int = 20,
        hidden_layers: int = 2,
        n_basis: int = 11,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.kappa = kappa
        self.kan = KAN(
            [1] + [width] * hidden_layers + [6],
            n_basis=n_basis,
            dtype=dtype,
        )

    def forward(self, xy: torch.Tensor) -> HarmonicFields:
        x = xy[:, 0]
        y = xy[:, 1]
        raw = self.kan((2.0 * y - 1.0)[:, None])
        phase = torch.exp(
            torch.complex(
                torch.zeros_like(x),
                self.kappa * x,
            )
        )
        u = y * phase * torch.complex(raw[:, 0], raw[:, 1])
        w = y * phase * torch.complex(raw[:, 2], raw[:, 3])
        correction = torch.complex(raw[:, 4], raw[:, 5])
        p = phase * (1.0 + (1.0 - y) * correction)
        return HarmonicFields(u=u, w=w, p=p)

    def roughness(self) -> torch.Tensor:
        return self.kan.roughness()
