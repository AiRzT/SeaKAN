from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import Geometry, WaveParameters
from .kan import KAN
from .model import WaterFields, surface_slope, surface_y
from .mpm import MaterialPointCloud
from .physics import complex_gradient, squared_magnitude


@dataclass
class HarmonicRANSFields(WaterFields):
    u: torch.Tensor
    v: torch.Tensor


class HarmonicRANSKAN(nn.Module):
    """First-harmonic incompressible RANS field with an eddy-viscosity closure.

    Pressure and velocity are solved together.  Unlike the potential-pressure
    model, pressure has no hard zero-normal-gradient condition at solid walls;
    the wall pressure is determined by the momentum equations and no-slip
    velocity conditions.
    """

    def __init__(
        self,
        wave: WaveParameters,
        geometry: Geometry,
        *,
        effective_kinematic_viscosity: float = 5.0e-4,
        width: int = 24,
        hidden_layers: int = 3,
        n_basis: int = 9,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if effective_kinematic_viscosity <= 0.0:
            raise ValueError(
                "effective_kinematic_viscosity must be positive"
            )
        self.wave = wave
        self.geometry = geometry
        self.kappa = wave.wave_number * geometry.seabed_thickness
        self.viscous_number = (
            effective_kinematic_viscosity
            / (
                wave.omega
                * geometry.seabed_thickness**2
            )
        )
        self.boundary_layer_thickness = (
            2.0 * self.viscous_number
        ) ** 0.5
        self.kan = KAN(
            [2] + [width] * hidden_layers + [6],
            n_basis=n_basis,
            dtype=dtype,
        )
        output = self.kan.layers[-1]
        nn.init.zeros_(output.base_weight)
        nn.init.zeros_(output.spline_weight)
        nn.init.zeros_(output.bias)

    def _normalized(self, xy: torch.Tensor) -> torch.Tensor:
        half = self.geometry.half_width_over_height
        vertical_center = 0.5 * (
            self.geometry.water_top_y + self.geometry.pipe_center_y
        )
        vertical_half = 0.5 * (
            self.geometry.water_top_y - self.geometry.pipe_center_y
        )
        return torch.stack(
            (
                xy[:, 0] / half,
                (xy[:, 1] - vertical_center) / vertical_half,
            ),
            dim=1,
        )

    def incident(self, xy: torch.Tensor) -> HarmonicRANSFields:
        x = xy[:, 0]
        y = xy[:, 1]
        phase = torch.exp(
            torch.complex(torch.zeros_like(x), self.kappa * x)
        )
        q = phase * torch.cosh(self.kappa * (y - 1.0))
        gradient_x = 1j * self.kappa * q
        gradient_y = (
            phase
            * self.kappa
            * torch.sinh(self.kappa * (y - 1.0))
        )
        # With exp(-i omega t), the inviscid momentum equation gives
        # velocity = -i * grad(q) in the chosen nondimensionalization.
        return HarmonicRANSFields(
            q=q,
            u=-1j * gradient_x,
            v=-1j * gradient_y,
        )

    def forward(self, xy: torch.Tensor) -> HarmonicRANSFields:
        raw = self.kan(self._normalized(xy))
        correction = [
            torch.complex(raw[:, index], raw[:, index + 1])
            for index in range(0, 6, 2)
        ]
        incident = self.incident(xy)
        half = self.geometry.half_width_over_height
        side_factor = (
            1.0 - (xy[:, 0] / half).square()
        ).clamp_min(0.0)
        bed_distance = (
            xy[:, 1] - surface_y(xy[:, 0], self.geometry)
        ).clamp_min(0.0)
        radial_distance = (
            torch.sqrt(
                xy[:, 0].square()
                + (
                    xy[:, 1] - self.geometry.pipe_center_y
                ).square()
            )
            - self.geometry.radius_over_height
        ).clamp_min(0.0)
        bed_factor = 1.0 - torch.exp(
            -bed_distance / self.boundary_layer_thickness
        )
        pipe_factor = 1.0 - torch.exp(
            -radial_distance / self.boundary_layer_thickness
        )
        no_slip_factor = bed_factor * pipe_factor
        return HarmonicRANSFields(
            q=incident.q + side_factor * correction[0],
            u=no_slip_factor
            * (incident.u + side_factor * correction[1]),
            v=no_slip_factor
            * (incident.v + side_factor * correction[2]),
        )

    def roughness(self) -> torch.Tensor:
        return self.kan.roughness()


@dataclass
class HarmonicRANSLoss:
    total: torch.Tensor
    particle_momentum: torch.Tensor
    particle_continuity: torch.Tensor
    particle_pressure_laplace: torch.Tensor
    p2g: torch.Tensor
    free_surface: torch.Tensor
    seabed_no_slip: torch.Tensor
    pipe_no_slip: torch.Tensor
    wall_momentum: torch.Tensor
    far_field: torch.Tensor
    roughness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "particle_momentum",
                "particle_continuity",
                "particle_pressure_laplace",
                "p2g",
                "free_surface",
                "seabed_no_slip",
                "pipe_no_slip",
                "wall_momentum",
                "far_field",
                "roughness",
            )
        }


class HarmonicRANSMPMProblem:
    """Strong-form harmonic RANS residual sampled by an MPM particle cloud."""

    def __init__(
        self,
        wave: WaveParameters,
        geometry: Geometry,
        cloud: MaterialPointCloud,
        *,
        effective_kinematic_viscosity: float = 5.0e-4,
        momentum_weight: float = 1.0,
        continuity_weight: float = 8.0,
        pressure_laplace_weight: float = 0.2,
        p2g_weight: float = 0.25,
        free_surface_weight: float = 80.0,
        seabed_weight: float = 140.0,
        pipe_weight: float = 220.0,
        wall_momentum_weight: float = 0.5,
        far_field_weight: float = 1200.0,
        roughness_weight: float = 1.0e-4,
        boundary_points: int = 160,
    ):
        if effective_kinematic_viscosity <= 0.0:
            raise ValueError("effective_kinematic_viscosity must be positive")
        self.wave = wave
        self.geometry = geometry
        self.cloud = cloud
        self.viscous_number = (
            effective_kinematic_viscosity
            / (
                wave.omega
                * geometry.seabed_thickness**2
            )
        )
        self.momentum_weight = momentum_weight
        self.continuity_weight = continuity_weight
        self.pressure_laplace_weight = pressure_laplace_weight
        self.p2g_weight = p2g_weight
        self.free_surface_weight = free_surface_weight
        self.seabed_weight = seabed_weight
        self.pipe_weight = pipe_weight
        self.wall_momentum_weight = wall_momentum_weight
        self.far_field_weight = far_field_weight
        self.roughness_weight = roughness_weight
        self.boundary_points = boundary_points

    def _residuals(
        self,
        model: HarmonicRANSKAN,
        xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = model(xy)
        pressure_gradient = complex_gradient(fields.q, xy)
        velocity_u_gradient = complex_gradient(fields.u, xy)
        velocity_v_gradient = complex_gradient(fields.v, xy)
        velocity_u_gradient_x = complex_gradient(
            velocity_u_gradient[:, 0], xy
        )
        velocity_u_gradient_y = complex_gradient(
            velocity_u_gradient[:, 1], xy
        )
        velocity_v_gradient_x = complex_gradient(
            velocity_v_gradient[:, 0], xy
        )
        velocity_v_gradient_y = complex_gradient(
            velocity_v_gradient[:, 1], xy
        )
        pressure_gradient_x = complex_gradient(
            pressure_gradient[:, 0], xy
        )
        pressure_gradient_y = complex_gradient(
            pressure_gradient[:, 1], xy
        )
        laplace_u = (
            velocity_u_gradient_x[:, 0]
            + velocity_u_gradient_y[:, 1]
        )
        laplace_v = (
            velocity_v_gradient_x[:, 0]
            + velocity_v_gradient_y[:, 1]
        )
        momentum_u = (
            -1j * fields.u
            + pressure_gradient[:, 0]
            - self.viscous_number * laplace_u
        )
        momentum_v = (
            -1j * fields.v
            + pressure_gradient[:, 1]
            - self.viscous_number * laplace_v
        )
        continuity = (
            velocity_u_gradient[:, 0]
            + velocity_v_gradient[:, 1]
        )
        pressure_laplace = (
            pressure_gradient_x[:, 0]
            + pressure_gradient_y[:, 1]
        )
        return momentum_u, momentum_v, continuity, pressure_laplace

    def loss(
        self,
        model: HarmonicRANSKAN,
        indices: torch.Tensor,
    ) -> HarmonicRANSLoss:
        xy = (
            self.cloud.xy[indices]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        momentum_u, momentum_v, continuity, pressure_laplace = (
            self._residuals(model, xy)
        )
        if len(self.cloud.counts) == 3:
            level_weights = (0.45, 0.25, 0.30)
        else:
            level_weights = tuple(
                1.0 / len(self.cloud.counts)
                for _ in self.cloud.counts
            )
        particle_momentum = self.cloud.weighted_level_mean(
            squared_magnitude(momentum_u)
            + squared_magnitude(momentum_v),
            indices,
            level_weights,
        )
        particle_continuity = self.cloud.weighted_level_mean(
            squared_magnitude(continuity),
            indices,
            level_weights,
        )
        particle_pressure_laplace = self.cloud.weighted_level_mean(
            squared_magnitude(pressure_laplace),
            indices,
            level_weights,
        )
        p2g = (
            squared_magnitude(
                self.cloud.p2g_residual(momentum_u, indices)
            ).mean()
            + squared_magnitude(
                self.cloud.p2g_residual(momentum_v, indices)
            ).mean()
            + squared_magnitude(
                self.cloud.p2g_residual(continuity, indices)
            ).mean()
        )
        (
            free_surface,
            seabed_no_slip,
            pipe_no_slip,
            wall_momentum,
            far_field,
        ) = self._boundary_losses(model, xy.dtype, xy.device)
        roughness = model.roughness()
        total = (
            self.momentum_weight * particle_momentum
            + self.continuity_weight * particle_continuity
            + self.pressure_laplace_weight
            * particle_pressure_laplace
            + self.p2g_weight * p2g
            + self.free_surface_weight * free_surface
            + self.seabed_weight * seabed_no_slip
            + self.pipe_weight * pipe_no_slip
            + self.wall_momentum_weight * wall_momentum
            + self.far_field_weight * far_field
            + self.roughness_weight * roughness
        )
        return HarmonicRANSLoss(
            total=total,
            particle_momentum=particle_momentum,
            particle_continuity=particle_continuity,
            particle_pressure_laplace=particle_pressure_laplace,
            p2g=p2g,
            free_surface=free_surface,
            seabed_no_slip=seabed_no_slip,
            pipe_no_slip=pipe_no_slip,
            wall_momentum=wall_momentum,
            far_field=far_field,
            roughness=roughness,
        )

    def _boundary_losses(
        self,
        model: HarmonicRANSKAN,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        n = self.boundary_points
        half = self.geometry.half_width_over_height
        x = torch.linspace(-half, half, n, dtype=dtype, device=device)

        free = torch.stack(
            (
                x,
                torch.full_like(x, self.geometry.water_top_y),
            ),
            dim=1,
        ).requires_grad_(True)
        free_fields = model(free)
        free_u_gradient = complex_gradient(free_fields.u, free)
        free_v_gradient = complex_gradient(free_fields.v, free)
        robin_coefficient = (
            self.wave.omega**2
            * self.geometry.seabed_thickness
            / self.wave.gravity
        )
        kinematic = (
            free_fields.v + 1j * robin_coefficient * free_fields.q
        )
        tangential_stress = (
            free_u_gradient[:, 1] + free_v_gradient[:, 0]
        )
        free_surface = (
            squared_magnitude(kinematic).mean()
            + 0.05 * squared_magnitude(tangential_stress).mean()
        )

        bed_x = x
        bed_y = surface_y(bed_x, self.geometry)
        pipe_shadow = (
            bed_x.abs()
            <= self.geometry.radius_over_height * 1.02
        )
        bed = torch.stack(
            (bed_x[~pipe_shadow], bed_y[~pipe_shadow]), dim=1
        ).requires_grad_(True)
        bed_fields = model(bed)
        seabed_no_slip = (
            squared_magnitude(bed_fields.u)
            + squared_magnitude(bed_fields.v)
        ).mean()

        theta = torch.linspace(
            0.0, torch.pi, n, dtype=dtype, device=device
        )
        radius = self.geometry.radius_over_height
        pipe = torch.stack(
            (
                radius * torch.cos(theta),
                self.geometry.pipe_center_y
                + radius * torch.sin(theta),
            ),
            dim=1,
        ).requires_grad_(True)
        pipe_fields = model(pipe)
        pipe_no_slip = (
            squared_magnitude(pipe_fields.u)
            + squared_magnitude(pipe_fields.v)
        ).mean()

        wall_momentum_values: list[torch.Tensor] = []
        for wall in (bed, pipe):
            momentum_u, momentum_v, continuity, _ = self._residuals(
                model, wall
            )
            wall_momentum_values.append(
                (
                    squared_magnitude(momentum_u)
                    + squared_magnitude(momentum_v)
                    + squared_magnitude(continuity)
                ).mean()
            )
        wall_momentum = torch.stack(wall_momentum_values).mean()

        far_x_value = (
            self.geometry.trench_edge_x
            / self.geometry.seabed_thickness
        )
        far_y = torch.linspace(
            min(
                self.geometry.water_top_y,
                1.0 + 4.0 * model.boundary_layer_thickness,
            ),
            self.geometry.water_top_y,
            max(48, n // 2),
            dtype=dtype,
            device=device,
        )
        far = torch.cat(
            (
                torch.stack(
                    (
                        -far_x_value * torch.ones_like(far_y),
                        far_y,
                    ),
                    dim=1,
                ),
                torch.stack(
                    (
                        far_x_value * torch.ones_like(far_y),
                        far_y,
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        fields = model(far)
        incident = model.incident(far)
        far_field = (
            squared_magnitude(fields.q - incident.q)
            + squared_magnitude(fields.u - incident.u)
            + squared_magnitude(fields.v - incident.v)
        ).mean()
        return (
            free_surface,
            seabed_no_slip,
            pipe_no_slip,
            wall_momentum,
            far_field,
        )
