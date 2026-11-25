from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from .config import Geometry, WaveParameters
from .kan import KAN

if TYPE_CHECKING:
    from .rans import HarmonicRANSKAN


def surface_y(
    x: torch.Tensor, geometry: Geometry
) -> torch.Tensor:
    if geometry.embedded_full_circle:
        return torch.ones_like(x)
    absolute_m = x.abs() * geometry.seabed_thickness
    slope_z = (
        -geometry.trench_depth
        + (
            absolute_m - geometry.trench_bottom_half_width
        )
        / geometry.trench_horizontal_to_vertical
    )
    z = torch.where(
        absolute_m <= geometry.fill_intersection_x,
        torch.full_like(
            x, geometry.backfill_surface_z
        ),
        torch.where(
            absolute_m < geometry.trench_edge_x,
            slope_z,
            torch.zeros_like(x),
        ),
    )
    return (z + geometry.seabed_thickness) / (
        geometry.seabed_thickness
    )


def surface_slope(
    x: torch.Tensor, geometry: Geometry
) -> torch.Tensor:
    if geometry.embedded_full_circle:
        return torch.zeros_like(x)
    absolute_m = x.abs() * geometry.seabed_thickness
    on_slope = (
        (absolute_m > geometry.fill_intersection_x)
        & (absolute_m < geometry.trench_edge_x)
    )
    signed = torch.sign(x)
    return torch.where(
        on_slope,
        signed / geometry.trench_horizontal_to_vertical,
        torch.zeros_like(x),
    )


@dataclass
class WaterFields:
    q: torch.Tensor


@dataclass
class SoilFields:
    u: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor


def pipe_neumann_coordinates(
    xy: torch.Tensor,
    geometry: Geometry,
    *,
    support: float,
    lower_only: bool = False,
) -> torch.Tensor:
    """Map the pipe-normal coordinate with zero wall derivative.

    The quintic map is the identity, including its first two
    derivatives, at the outer edge of the support.  At the pipe it has
    zero first and second radial derivatives.  A field evaluated at the
    mapped coordinates therefore satisfies the rigid-pipe Neumann
    condition without an independently trainable wall value.
    """

    dx = xy[:, 0]
    dy = xy[:, 1] - geometry.pipe_center_y
    radial = torch.sqrt(
        dx.square() + dy.square()
    ).clamp_min(1.0e-12)
    radius = geometry.radius_over_height
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
    if lower_only:
        # Regularize the two mixed-boundary contact corners while
        # retaining an exact map on the open embedded arc.
        lower_coordinate = (
            -dy / (0.01 * radial)
        ).clamp(0.0, 1.0)
        lower = (
            3.0 * lower_coordinate.square()
            - 2.0 * lower_coordinate.pow(3)
        )
        mapped_distance = (
            lower * mapped_distance
            + (1.0 - lower) * distance
        )
    mapped_radial = radius + mapped_distance
    scale = mapped_radial / radial
    return torch.stack(
        (
            scale * dx,
            geometry.pipe_center_y + scale * dy,
        ),
        dim=1,
    )


class WaterPressureKAN(nn.Module):
    """Complex harmonic potential-pressure field in the water domain."""

    def __init__(
        self,
        wave: WaveParameters,
        geometry: Geometry,
        *,
        width: int = 16,
        hidden_layers: int = 2,
        n_basis: int = 9,
        pipe_support: float = 0.16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.wave = wave
        self.geometry = geometry
        self.kappa = wave.wave_number * geometry.seabed_thickness
        if pipe_support <= 0.0:
            raise ValueError("pipe_support must be positive")
        self.pipe_support = float(pipe_support)
        self.kan = KAN(
            [2] + [width] * hidden_layers + [2],
            n_basis=n_basis,
            dtype=dtype,
        )
        output = self.kan.layers[-1]
        nn.init.zeros_(output.base_weight)
        nn.init.zeros_(output.spline_weight)
        nn.init.zeros_(output.bias)

    def incident(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        phase = torch.exp(
            torch.complex(
                torch.zeros_like(x), self.kappa * x
            )
        )
        vertical = torch.cosh(self.kappa * (y - 1.0))
        return phase * vertical

    def forward(self, xy: torch.Tensor) -> WaterFields:
        support = self.pipe_support
        effective = pipe_neumann_coordinates(
            xy,
            self.geometry,
            support=support,
        )
        half = self.geometry.half_width_over_height
        vertical_center = 0.5 * (
            self.geometry.water_top_y
            + self.geometry.pipe_center_y
        )
        vertical_half = 0.5 * (
            self.geometry.water_top_y
            - self.geometry.pipe_center_y
        )
        normalized = torch.stack(
            (
                effective[:, 0] / half,
                (
                    effective[:, 1] - vertical_center
                )
                / vertical_half,
            ),
            dim=1,
        )
        raw = self.kan(normalized)
        correction = torch.complex(raw[:, 0], raw[:, 1])
        side_factor = (
            1.0 - (effective[:, 0] / half).square()
        ).clamp_min(0.0)
        return WaterFields(
            q=self.incident(xy)
            + self._incident_pipe_lift(xy, support)
            + side_factor * correction
        )

    def roughness(self) -> torch.Tensor:
        return self.kan.roughness()

    def _incident_pipe_lift(
        self,
        xy: torch.Tensor,
        support: float,
    ) -> torch.Tensor:
        """Compact lift cancelling incident-wave pipe-normal flux."""

        dx = xy[:, 0]
        dy = xy[:, 1] - self.geometry.pipe_center_y
        radial = torch.sqrt(
            dx.square() + dy.square()
        ).clamp_min(1.0e-12)
        normal_x = dx / radial
        normal_y = dy / radial
        radius = self.geometry.radius_over_height
        wall_x = radius * normal_x
        wall_y = (
            self.geometry.pipe_center_y
            + radius * normal_y
        )
        phase = torch.exp(
            torch.complex(
                torch.zeros_like(wall_x),
                self.kappa * wall_x,
            )
        )
        wall_q = phase * torch.cosh(
            self.kappa * (wall_y - 1.0)
        )
        imaginary = torch.as_tensor(
            1j,
            dtype=wall_q.dtype,
            device=xy.device,
        )
        gradient_x = imaginary * self.kappa * wall_q
        gradient_y = (
            phase
            * self.kappa
            * torch.sinh(self.kappa * (wall_y - 1.0))
        )
        wall_normal_gradient = (
            gradient_x * normal_x
            + gradient_y * normal_y
        )
        distance = radial - radius
        coordinate = (distance / support).clamp(0.0, 1.0)
        compact_profile = torch.where(
            distance < support,
            -distance * (1.0 - coordinate).pow(3),
            torch.zeros_like(distance),
        )
        return compact_profile * wall_normal_gradient


class ContactEnrichedWaterKAN(nn.Module):
    """Potential interface field with trainable contact-corner KANs.

    The base potential field is preserved outside two compact supports.
    Support dimensions are the horizontal and vertical projections of
    one RANS Stokes boundary-layer arc on the pipe.  The enrichment is
    optimized with water-physics and complex-pressure continuity losses;
    it is part of the raw network output, not a plotting correction.
    """

    def __init__(
        self,
        base_model: WaterPressureKAN,
        *,
        contact_angle: float,
        width: int = 8,
        n_basis: int = 7,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not 0.0 < contact_angle < torch.pi / 2.0:
            raise ValueError(
                "contact_angle must lie between 0 and pi/2"
            )
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.wave = base_model.wave
        self.geometry = base_model.geometry
        self.pipe_support = base_model.pipe_support
        self.contact_angle = float(contact_angle)
        radius = self.geometry.radius_over_height
        self.horizontal_support = float(
            radius * (1.0 - torch.cos(
                torch.tensor(self.contact_angle)
            ))
        )
        self.vertical_support = float(
            radius * torch.sin(torch.tensor(self.contact_angle))
        )
        self.contact_kans = nn.ModuleList(
            [
                KAN(
                    [2, width, width, 2],
                    n_basis=n_basis,
                    dtype=dtype,
                )
                for _ in range(2)
            ]
        )
        for network in self.contact_kans:
            output = network.layers[-1]
            nn.init.zeros_(output.base_weight)
            nn.init.zeros_(output.spline_weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _compact_weight(coordinate: torch.Tensor) -> torch.Tensor:
        value = coordinate.abs().clamp(0.0, 1.0)
        smoothstep = (
            10.0 * value.pow(3)
            - 15.0 * value.pow(4)
            + 6.0 * value.pow(5)
        )
        return torch.where(
            coordinate.abs() < 1.0,
            1.0 - smoothstep,
            torch.zeros_like(coordinate),
        )

    def incident(self, xy: torch.Tensor) -> torch.Tensor:
        return self.base_model.incident(xy)

    def forward(self, xy: torch.Tensor) -> WaterFields:
        pressure = self.base_model(xy).q
        radius = self.geometry.radius_over_height
        center_y = self.geometry.pipe_center_y
        for center_x, network in zip(
            (-radius, radius), self.contact_kans, strict=True
        ):
            local_x = (
                xy[:, 0] - center_x
            ) / self.horizontal_support
            local_y = (
                xy[:, 1] - center_y
            ) / self.vertical_support
            normalized = torch.stack(
                (
                    local_x.clamp(-1.0, 1.0),
                    local_y.clamp(-1.0, 1.0),
                ),
                dim=1,
            )
            raw = network(normalized)
            correction = torch.complex(
                raw[:, 0], raw[:, 1]
            )
            weight = (
                self._compact_weight(local_x)
                * self._compact_weight(local_y)
            )
            pressure = pressure + weight * correction
        return WaterFields(q=pressure)

    def roughness(self) -> torch.Tensor:
        return self.base_model.roughness() + sum(
            (
                network.roughness()
                for network in self.contact_kans
            ),
            start=torch.zeros(
                (),
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            ),
        )


class SoilPipeKAN(nn.Module):
    """Strong-form Biot fields in the trenched seabed."""

    def __init__(
        self,
        wave: WaveParameters,
        geometry: Geometry,
        water_model: (
            WaterPressureKAN
            | ContactEnrichedWaterKAN
            | HarmonicRANSKAN
            | None
        ) = None,
        *,
        width: int = 18,
        hidden_layers: int = 2,
        n_basis: int = 9,
        local_width: int = 12,
        local_basis: int = 9,
        local_support: float = 0.30,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.wave = wave
        self.geometry = geometry
        self.kappa = wave.wave_number * geometry.seabed_thickness
        self.water_model = water_model
        if water_model is not None:
            for parameter in water_model.parameters():
                parameter.requires_grad_(False)
        self.kan = KAN(
            [2] + [width] * hidden_layers + [6],
            n_basis=n_basis,
            dtype=dtype,
        )
        if local_support <= 0.0:
            raise ValueError("local_support must be positive")
        self.local_support = float(local_support)
        self.local_kan = KAN(
            [2, local_width, local_width, 6],
            n_basis=local_basis,
            dtype=dtype,
        )
        local_output = self.local_kan.layers[-1]
        nn.init.zeros_(local_output.base_weight)
        nn.init.zeros_(local_output.spline_weight)
        nn.init.zeros_(local_output.bias)

    def interface_pressure(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        top = surface_y(x, self.geometry)
        xy = torch.stack((x, top), dim=1)
        if self.water_model is None:
            return torch.exp(
                torch.complex(
                    torch.zeros_like(x), self.kappa * x
                )
            )
        return self.water_model(xy).q

    def forward(self, xy: torch.Tensor) -> SoilFields:
        x = xy[:, 0]
        y = xy[:, 1]
        half = self.geometry.half_width_over_height
        feature_xy = pipe_neumann_coordinates(
            xy,
            self.geometry,
            support=0.08,
            lower_only=not self.geometry.embedded_full_circle,
        )
        feature_x = feature_xy[:, 0]
        feature_y = feature_xy[:, 1]
        normalized = torch.stack(
            (
                feature_x / half,
                2.0 * feature_y - 1.0,
            ),
            dim=1,
        )
        raw = self.kan(normalized)
        radius = self.geometry.radius_over_height
        local_scale = radius + self.local_support
        local_normalized = torch.stack(
            (
                feature_x / local_scale,
                (
                    feature_y - self.geometry.pipe_center_y
                )
                / local_scale,
            ),
            dim=1,
        )
        local_distance = (
            torch.sqrt(
                feature_x.square()
                + (
                    feature_y
                    - self.geometry.pipe_center_y
                ).square()
            )
            - radius
        ).clamp_min(0.0)
        local_weight = (
            1.0 - local_distance / self.local_support
        ).clamp(0.0, 1.0).pow(3)
        raw = raw + local_weight[:, None] * self.local_kan(
            local_normalized
        )
        radial_level = (
            x.square()
            + (y - self.geometry.pipe_center_y).square()
            - radius**2
        )
        pipe_factor = 1.0 - torch.exp(
            -torch.clamp(radial_level, min=0.0)
            / radius**2
        )
        outer_factor = y * (
            1.0 - (x / half).square()
        ).clamp_min(0.0)
        displacement_factor = outer_factor * pipe_factor
        u = displacement_factor * torch.complex(
            raw[:, 0], raw[:, 1]
        )
        w = displacement_factor * torch.complex(
            raw[:, 2], raw[:, 3]
        )
        top = surface_y(feature_x, self.geometry)
        correction = torch.complex(
            raw[:, 4], raw[:, 5]
        )
        depth = (top - feature_y).clamp_min(0.0)
        interface_pressure = self.interface_pressure(feature_x)
        if isinstance(
            self.water_model, ContactEnrichedWaterKAN
        ):
            top_xy = torch.stack((feature_x, top), dim=1)
            base_pressure = self.water_model.base_model(
                top_xy
            ).q
            enrichment = interface_pressure - base_pressure
            enrichment_decay = torch.exp(
                -depth / self.water_model.vertical_support
            )
            interface_extension = (
                base_pressure
                + enrichment_decay * enrichment
            )
        else:
            interface_extension = interface_pressure
        p = interface_extension + depth * correction
        return SoilFields(
            u=u,
            w=w,
            p=p,
        )

    def roughness(self) -> torch.Tensor:
        return (
            self.kan.roughness()
            + self.local_kan.roughness()
        )
