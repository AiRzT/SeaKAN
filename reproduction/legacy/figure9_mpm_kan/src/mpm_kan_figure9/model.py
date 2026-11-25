from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import Geometry
from .kan import KAN


@dataclass
class HarmonicFields:
    u: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor


class PipeHarmonicKAN(nn.Module):
    """
    Complex harmonic displacement/pressure fields over the 2-D trench.

    X=x/b and Y=(z+b)/b. Hard transforms impose the traveling mudline
    pressure and zero displacement on the rigid bottom, sides and pipe.
    """

    def __init__(
        self,
        geometry: Geometry,
        kappa: float,
        width: int = 18,
        hidden_layers: int = 2,
        n_basis: int = 9,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.geometry = geometry
        self.kappa = kappa
        self.kan = KAN(
            [2] + [width] * hidden_layers + [6],
            n_basis=n_basis,
            dtype=dtype,
        )

    def forward(self, xy: torch.Tensor) -> HarmonicFields:
        x = xy[:, 0]
        y = xy[:, 1]
        half_width = self.geometry.half_width_over_height
        normalized = torch.stack((x / half_width, 2.0 * y - 1.0), dim=1)
        raw = self.kan(normalized)

        radial_level = (
            x.square()
            + (y - self.geometry.center_y).square()
            - self.geometry.radius_over_height**2
        )
        pipe_factor = 1.0 - torch.exp(
            -torch.clamp(radial_level, min=0.0)
            / (self.geometry.radius_over_height**2)
        )
        outer_factor = y * (
            1.0 - (x / self.geometry.half_width_over_height).square()
        )
        displacement_factor = outer_factor * pipe_factor
        top_wave = torch.exp(
            torch.complex(torch.zeros_like(x), self.kappa * x)
        )
        u = displacement_factor * torch.complex(raw[:, 0], raw[:, 1])
        w = displacement_factor * torch.complex(raw[:, 2], raw[:, 3])
        correction = torch.complex(raw[:, 4], raw[:, 5])
        p = top_wave + (1.0 - y) * correction
        return HarmonicFields(u=u, w=w, p=p)

    def roughness(self) -> torch.Tensor:
        return self.kan.roughness()


class TopGapLocalizedKAN(nn.Module):
    """
    Zero-initialized residual KAN localized to the pipe-top gap.

    The released adaptive strong-form model supplies the global solution.
    A compact geometric envelope confines new capacity to the upper pipe and
    nearby mudline, while the same hard pressure/displacement factors preserve
    the original essential boundary conditions.
    """

    def __init__(
        self,
        base: PipeHarmonicKAN,
        geometry: Geometry,
        *,
        width: int = 10,
        n_basis: int = 9,
        dtype: torch.dtype = torch.float32,
        freeze_base: bool = True,
        x_support_radii: float = 0.60,
        y_support_center_gap: float = 1.40,
    ):
        super().__init__()
        self.base = base
        self.geometry = geometry
        self.x_support_radii = x_support_radii
        self.y_support_center_gap = y_support_center_gap
        self.local_kan = KAN(
            [2, width, 6],
            n_basis=n_basis,
            dtype=dtype,
        )
        output = self.local_kan.layers[-1]
        nn.init.zeros_(output.base_weight)
        nn.init.zeros_(output.spline_weight)
        nn.init.zeros_(output.bias)
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    def forward(self, xy: torch.Tensor) -> HarmonicFields:
        base = self.base(xy)
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        local_coordinates = self.local_coordinates(xy)
        raw = self.local_kan(local_coordinates)
        envelope = self.envelope(xy)
        radial_level = (
            x.square()
            + (y - self.geometry.center_y).square()
            - radius**2
        )
        pipe_factor = 1.0 - torch.exp(
            -torch.clamp(radial_level, min=0.0) / radius**2
        )
        outer_factor = y * (
            1.0 - (x / self.geometry.half_width_over_height).square()
        )
        displacement_factor = envelope * outer_factor * pipe_factor
        pressure_factor = self.pressure_factor(xy)
        u = base.u + displacement_factor * torch.complex(
            raw[:, 0], raw[:, 1]
        )
        w = base.w + displacement_factor * torch.complex(
            raw[:, 2], raw[:, 3]
        )
        p = base.p + pressure_factor * torch.complex(
            raw[:, 4], raw[:, 5]
        )
        return HarmonicFields(u=u, w=w, p=p)

    def roughness(self) -> torch.Tensor:
        return self.local_kan.roughness()

    def local_coordinates(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        vertical_gap = 1.0 - self.geometry.center_y
        dx = x
        dy = y - self.geometry.center_y
        radial = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0e-12)
        cosine = dx / radial
        sine = dy / radial
        outer_radius = vertical_gap / sine.clamp_min(0.15)
        gap = (outer_radius - radius).clamp_min(0.02)
        normal = ((radial - radius) / gap).clamp(0.0, 1.0)
        return torch.stack(
            (
                cosine / self.x_support_radii,
                2.0 * normal.square() - 1.0,
            ),
            dim=1,
        )

    def envelope(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        vertical_gap = 1.0 - self.geometry.center_y
        return torch.exp(
            -(x / (self.x_support_radii * radius)).pow(4)
            - (
                (1.0 - y)
                / (self.y_support_center_gap * vertical_gap)
            ).pow(4)
        )

    def pressure_factor(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Pipe-normal Hermite factor: h(0)=1, h'(0)=0, h(1)=h'(1)=0.

        The angular terms depend only on theta, so the added pressure has
        exactly zero normal derivative at the circular pipe and vanishes at
        the mudline along every upper radial ray.
        """

        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        vertical_gap = 1.0 - self.geometry.center_y
        dx = x
        dy = y - self.geometry.center_y
        radial = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0e-12)
        cosine = dx / radial
        sine = dy / radial
        outer_radius = vertical_gap / sine.clamp_min(0.15)
        gap = (outer_radius - radius).clamp_min(0.02)
        normal = ((radial - radius) / gap).clamp(0.0, 1.0)
        hermite = 1.0 - 3.0 * normal.square() + 2.0 * normal.pow(3)
        angular = torch.exp(
            -(cosine / self.x_support_radii).pow(4)
        ) * (0.5 * (1.0 + sine)).clamp(0.0, 1.0).pow(6)
        return angular * hermite
