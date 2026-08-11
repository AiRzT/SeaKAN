from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import (
    ParameterDomain,
    PhysicalConstants,
    normalize_parameters,
)
from .kan import KAN


@dataclass
class SoilFields:
    u: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor


class DenseMLP(nn.Module):
    """Parameter-budget-matched dense baseline for KAN ablation."""

    def __init__(
        self,
        widths: list[int],
        *,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    widths[index],
                    widths[index + 1],
                    dtype=dtype,
                )
                for index in range(len(widths) - 1)
            ]
        )
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.65)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def roughness(self) -> torch.Tensor:
        # MLPs have no spline curvature. AdamW supplies the matched
        # parameter regularisation used by both backbones.
        return self.layers[0].weight.sum() * 0.0


class ParametricSoilKAN(nn.Module):
    """One KAN for every parameter combination in the fixed domain."""

    def __init__(
        self,
        domain: ParameterDomain,
        constants: PhysicalConstants,
        *,
        width: int = 24,
        hidden_layers: int = 2,
        n_basis: int = 7,
        local_width: int = 16,
        local_support: float = 0.24,
        backbone: str = "kan",
        condition_random_field: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if backbone not in {"kan", "mlp"}:
            raise ValueError(
                f"backbone must be 'kan' or 'mlp', got {backbone!r}"
            )
        self.domain = domain
        self.constants = constants
        self.backbone = backbone
        self.condition_random_field = bool(condition_random_field)
        self.input_dimensions = 2 + 4 + domain.random_modes
        if backbone == "kan":
            self.global_kan = KAN(
                [self.input_dimensions]
                + [width] * hidden_layers
                + [6],
                n_basis=n_basis,
                dtype=dtype,
            )
            self.local_kan = KAN(
                [
                    self.input_dimensions,
                    local_width,
                    local_width,
                    6,
                ],
                n_basis=n_basis,
                dtype=dtype,
            )
        else:
            # For n_basis=8 this 3.3x width gives 32,161 MLP
            # parameters versus 31,836 KAN parameters (<1.1% gap).
            multiplier = 3.3
            dense_width = max(2, int(round(width * multiplier)))
            dense_local_width = max(
                2, int(round(local_width * multiplier))
            )
            self.global_kan = DenseMLP(
                [self.input_dimensions]
                + [dense_width] * hidden_layers
                + [6],
                dtype=dtype,
            )
            self.local_kan = DenseMLP(
                [
                    self.input_dimensions,
                    dense_local_width,
                    dense_local_width,
                    6,
                ],
                dtype=dtype,
            )
        output = self.local_kan.layers[-1]
        if isinstance(output, nn.Linear):
            nn.init.zeros_(output.weight)
        else:
            nn.init.zeros_(output.base_weight)
            nn.init.zeros_(output.spline_weight)
        nn.init.zeros_(output.bias)
        self.local_support = float(local_support)

    def pipe_neumann_coordinates(
        self,
        xy: torch.Tensor,
        pipe_center_y: torch.Tensor,
        *,
        support: float = 0.08,
    ) -> torch.Tensor:
        """Quintic radial map with zero wall derivative."""

        dx = xy[:, 0]
        dy = xy[:, 1] - pipe_center_y
        radial = torch.sqrt(
            dx.square() + dy.square()
        ).clamp_min(1.0e-12)
        radius = self.constants.radius
        distance = (radial - radius).clamp_min(0.0)
        coordinate = (distance / support).clamp(0.0, 1.0)
        mapped_coordinate = (
            6.0 * coordinate.pow(3)
            - 8.0 * coordinate.pow(4)
            + 3.0 * coordinate.pow(5)
        )
        mapped_distance = torch.where(
            distance < support,
            support * mapped_coordinate,
            distance,
        )
        scale = (radius + mapped_distance) / radial
        mapped = torch.stack(
            (
                scale * dx,
                pipe_center_y + scale * dy,
            ),
            dim=1,
        )
        top_distance = (1.0 - xy[:, 1]).clamp_min(0.0)
        blend = top_distance.square() / (
            top_distance.square()
            + distance.square()
            + torch.finfo(xy.dtype).eps
        )
        return xy + blend[:, None] * (mapped - xy)

    def features(
        self,
        xy: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = torch.stack(
            (
                xy[:, 0] / self.constants.half_width,
                2.0 * xy[:, 1] - 1.0,
            ),
            dim=1,
        )
        normalized = normalize_parameters(
            parameters, self.domain
        )
        if not self.condition_random_field:
            normalized = normalized.clone()
            normalized[:, -self.domain.random_modes :] = 0.0
        return torch.cat((coordinates, normalized), dim=1)

    def forward(
        self,
        xy: torch.Tensor,
        parameters: torch.Tensor,
        kappa: torch.Tensor,
        pipe_center_y: torch.Tensor,
    ) -> SoilFields:
        feature_xy = self.pipe_neumann_coordinates(
            xy, pipe_center_y
        )
        features = self.features(feature_xy, parameters)
        raw = self.global_kan(features)
        radius = self.constants.radius
        distance = torch.sqrt(
            feature_xy[:, 0].square()
            + (
                feature_xy[:, 1] - pipe_center_y
            ).square()
        )
        local_weight = (
            1.0
            - (distance - radius).clamp_min(0.0)
            / self.local_support
        ).clamp(0.0, 1.0).pow(3)
        raw = raw + local_weight[:, None] * self.local_kan(
            features
        )

        pipe_level = (
            xy[:, 0].square()
            + (xy[:, 1] - pipe_center_y).square()
            - radius**2
        ).clamp_min(0.0)
        pipe_factor = 1.0 - torch.exp(
            -pipe_level / radius**2
        )
        side_factor = (
            1.0
            - (xy[:, 0] / self.constants.half_width).square()
        ).clamp_min(0.0)
        displacement_factor = xy[:, 1] * side_factor * pipe_factor
        u = displacement_factor * torch.complex(
            raw[:, 0], raw[:, 1]
        )
        w = displacement_factor * torch.complex(
            raw[:, 2], raw[:, 3]
        )
        interface = torch.exp(
            torch.complex(
                torch.zeros_like(kappa),
                kappa * feature_xy[:, 0],
            )
        )
        depth = (1.0 - feature_xy[:, 1]).clamp_min(0.0)
        correction = torch.complex(raw[:, 4], raw[:, 5])
        p = interface + depth * correction
        return SoilFields(u=u, w=w, p=p)

    def roughness(self) -> torch.Tensor:
        return (
            self.global_kan.roughness()
            + self.local_kan.roughness()
        )


class ComsolCalibratedSoilKAN(nn.Module):
    """Frozen physics model plus a small conditional pressure corrector.

    The corrector uses the same pipe-normal coordinate map as the base
    SeaKAN and is multiplied by the distance from the seabed surface.  It
    therefore leaves the prescribed interface pressure unchanged while the
    displacement fields continue to come entirely from the physics model.
    """

    def __init__(
        self,
        base: ParametricSoilKAN,
        *,
        width: int = 24,
        hidden_layers: int = 2,
        n_basis: int = 7,
        condition_wave_height: bool = True,
        condition_random_field: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base = base
        self.domain = base.domain
        self.constants = base.constants
        self.input_dimensions = base.input_dimensions
        self.condition_wave_height = bool(condition_wave_height)
        self.condition_random_field = bool(condition_random_field)
        self.pressure_correction = KAN(
            [self.input_dimensions]
            + [width] * hidden_layers
            + [2],
            n_basis=n_basis,
            dtype=dtype,
        )
        output = self.pressure_correction.layers[-1]
        nn.init.zeros_(output.base_weight)
        nn.init.zeros_(output.spline_weight)
        nn.init.zeros_(output.bias)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        xy: torch.Tensor,
        parameters: torch.Tensor,
        kappa: torch.Tensor,
        pipe_center_y: torch.Tensor,
    ) -> SoilFields:
        base = self.base(xy, parameters, kappa, pipe_center_y)
        feature_xy = self.base.pipe_neumann_coordinates(
            xy, pipe_center_y
        )
        features = self.base.features(feature_xy, parameters)
        if not self.condition_wave_height or not self.condition_random_field:
            features = features.clone()
        if not self.condition_wave_height:
            features[:, 2] = 0.0
        if not self.condition_random_field:
            features[:, -self.domain.random_modes :] = 0.0
        raw = self.pressure_correction(features)
        depth = (1.0 - feature_xy[:, 1]).clamp_min(0.0)
        delta_p = depth * torch.complex(raw[:, 0], raw[:, 1])
        return SoilFields(u=base.u, w=base.w, p=base.p + delta_p)

    def roughness(self) -> torch.Tensor:
        return self.pressure_correction.roughness()
