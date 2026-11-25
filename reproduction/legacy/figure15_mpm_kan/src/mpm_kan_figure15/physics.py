from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
)
from .model import (
    SoilPipeKAN,
    WaterPressureKAN,
    surface_slope,
    surface_y,
)
from .mpm import MaterialPointCloud


def complex_gradient(
    value: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    real = torch.autograd.grad(
        value.real.sum(),
        coordinates,
        create_graph=True,
        retain_graph=True,
    )[0]
    imaginary = torch.autograd.grad(
        value.imag.sum(),
        coordinates,
        create_graph=True,
        retain_graph=True,
    )[0]
    return torch.complex(real, imaginary)


def squared_magnitude(value: torch.Tensor) -> torch.Tensor:
    return value.real.square() + value.imag.square()


@dataclass
class WaterLoss:
    total: torch.Tensor
    particle_laplace: torch.Tensor
    p2g_laplace: torch.Tensor
    free_surface: torch.Tensor
    seabed_flux: torch.Tensor
    pipe_flux: torch.Tensor
    far_field: torch.Tensor
    roughness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "particle_laplace",
                "p2g_laplace",
                "free_surface",
                "seabed_flux",
                "pipe_flux",
                "far_field",
                "roughness",
            )
        }


@dataclass
class AugmentedBoundaryState:
    lambda_surface: float = 0.0
    lambda_outer: float = 0.0
    lambda_pipe: float = 0.0
    rho_surface: float = 40.0
    rho_outer: float = 50.0
    rho_pipe: float = 80.0

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "lambda_surface",
                "lambda_outer",
                "lambda_pipe",
                "rho_surface",
                "rho_outer",
                "rho_pipe",
            )
        }

    def update(
        self,
        constraints: dict[str, float],
        previous: dict[str, float] | None = None,
        *,
        progress_ratio: float = 0.90,
        growth: float = 1.5,
        maximum: float = 1000.0,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in ("surface", "outer", "pipe"):
            value = max(0.0, float(constraints[name]))
            result[name] = value
            rho_name = f"rho_{name}"
            lambda_name = f"lambda_{name}"
            rho = float(getattr(self, rho_name))
            setattr(
                self,
                lambda_name,
                max(
                    0.0,
                    float(getattr(self, lambda_name))
                    + rho * value,
                ),
            )
            if (
                previous is not None
                and value
                > progress_ratio * float(previous[name])
            ):
                setattr(
                    self,
                    rho_name,
                    min(maximum, growth * rho),
                )
        return result


@dataclass
class SoilLoss:
    total: torch.Tensor
    particle_momentum: torch.Tensor
    particle_fluid: torch.Tensor
    p2g_momentum: torch.Tensor
    p2g_fluid: torch.Tensor
    surface_traction: torch.Tensor
    outer_flux: torch.Tensor
    pipe_flux: torch.Tensor
    surface_constraint: torch.Tensor
    outer_constraint: torch.Tensor
    pipe_constraint: torch.Tensor
    boundary_augmented: torch.Tensor
    roughness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "particle_momentum",
                "particle_fluid",
                "p2g_momentum",
                "p2g_fluid",
                "surface_traction",
                "outer_flux",
                "pipe_flux",
                "surface_constraint",
                "outer_constraint",
                "pipe_constraint",
                "boundary_augmented",
                "roughness",
            )
        }


class WaterMPMProblem:
    def __init__(
        self,
        wave: WaveParameters,
        geometry: Geometry,
        cloud: MaterialPointCloud,
        *,
        p2g_weight: float = 0.25,
        free_surface_weight: float = 80.0,
        seabed_weight: float = 180.0,
        pipe_weight: float = 240.0,
        far_field_weight: float = 2500.0,
        annulus_residual_weight: float = 0.38,
        roughness_weight: float = 1.0e-4,
        boundary_points: int = 160,
    ):
        self.wave = wave
        self.geometry = geometry
        self.cloud = cloud
        self.p2g_weight = p2g_weight
        self.free_surface_weight = free_surface_weight
        self.seabed_weight = seabed_weight
        self.pipe_weight = pipe_weight
        self.far_field_weight = far_field_weight
        if not 0.0 < annulus_residual_weight < 1.0:
            raise ValueError(
                "annulus_residual_weight must lie between 0 and 1"
            )
        self.annulus_residual_weight = (
            annulus_residual_weight
        )
        self.roughness_weight = roughness_weight
        self.boundary_points = boundary_points

    def loss(
        self,
        model: WaterPressureKAN,
        indices: torch.Tensor,
    ) -> WaterLoss:
        xy = (
            self.cloud.xy[indices]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        q = model(xy).q
        gradient = complex_gradient(q, xy)
        gradient_x = complex_gradient(gradient[:, 0], xy)
        gradient_y = complex_gradient(gradient[:, 1], xy)
        residual = gradient_x[:, 0] + gradient_y[:, 1]
        level_weights = (
            (
                1.0 - self.annulus_residual_weight,
                self.annulus_residual_weight,
            )
            if len(self.cloud.counts) == 2
            else tuple(
                1.0 / len(self.cloud.counts)
                for _ in self.cloud.counts
            )
        )
        particle = self.cloud.weighted_level_mean(
            squared_magnitude(residual),
            indices,
            level_weights,
        )
        p2g = squared_magnitude(
            self.cloud.p2g_residual(residual, indices)
        ).mean()
        free_surface, seabed_flux, pipe_flux, far_field = (
            self._boundary_losses(
                model, xy.dtype, xy.device
            )
        )
        roughness = model.roughness()
        total = (
            particle
            + self.p2g_weight * p2g
            + self.free_surface_weight * free_surface
            + self.seabed_weight * seabed_flux
            + self.pipe_weight * pipe_flux
            + self.far_field_weight * far_field
            + self.roughness_weight * roughness
        )
        return WaterLoss(
            total=total,
            particle_laplace=particle,
            p2g_laplace=p2g,
            free_surface=free_surface,
            seabed_flux=seabed_flux,
            pipe_flux=pipe_flux,
            far_field=far_field,
            roughness=roughness,
        )

    def _boundary_losses(
        self,
        model: WaterPressureKAN,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        n = self.boundary_points
        half = self.geometry.half_width_over_height
        x = torch.linspace(
            -half, half, n, dtype=dtype, device=device
        )
        free = torch.stack(
            (
                x,
                torch.full_like(
                    x, self.geometry.water_top_y
                ),
            ),
            dim=1,
        ).requires_grad_(True)
        free_q = model(free).q
        free_gradient = complex_gradient(free_q, free)
        robin_coefficient = (
            self.wave.omega**2
            * self.geometry.seabed_thickness
            / self.wave.gravity
        )
        free_surface = squared_magnitude(
            free_gradient[:, 1] - robin_coefficient * free_q
        ).mean()

        bed_x = x
        bed_y = surface_y(bed_x, self.geometry)
        if not self.geometry.embedded_full_circle:
            pipe_shadow = (
                bed_x.abs()
                <= self.geometry.radius_over_height * 1.02
            )
            bed_x = bed_x[~pipe_shadow]
            bed_y = bed_y[~pipe_shadow]
        bed = torch.stack((bed_x, bed_y), dim=1).requires_grad_(
            True
        )
        bed_gradient = complex_gradient(model(bed).q, bed)
        slope = surface_slope(bed[:, 0], self.geometry)
        normal_x = -slope
        normal_y = torch.ones_like(slope)
        normal_scale = torch.sqrt(
            normal_x.square() + normal_y.square()
        )
        normal_x = normal_x / normal_scale
        normal_y = normal_y / normal_scale
        seabed_flux = squared_magnitude(
            bed_gradient[:, 0] * normal_x
            + bed_gradient[:, 1] * normal_y
        ).mean()

        theta = torch.linspace(
            0.0,
            torch.pi,
            n,
            dtype=dtype,
            device=device,
        )
        normal_x = torch.cos(theta)
        normal_y = torch.sin(theta)
        radius = self.geometry.radius_over_height
        pipe = torch.stack(
            (
                radius * normal_x,
                self.geometry.pipe_center_y
                + radius * normal_y,
            ),
            dim=1,
        ).requires_grad_(True)
        pipe_gradient = complex_gradient(model(pipe).q, pipe)
        pipe_flux = squared_magnitude(
            pipe_gradient[:, 0] * normal_x
            + pipe_gradient[:, 1] * normal_y
        ).mean()

        # Match the incident travelling wave in the flat-bed regions
        # outside the reconstructed trench.  These internal far-field
        # sections represent the wave-maker/outlet forcing of the
        # laboratory flume and prevent a low-amplitude boundary-layer
        # escape between the two remote side boundaries.
        far_x_value = (
            self.geometry.trench_edge_x
            / self.geometry.seabed_thickness
        )
        far_y = torch.linspace(
            1.0,
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
        far_field = squared_magnitude(
            model(far).q - model.incident(far)
        ).mean()
        return free_surface, seabed_flux, pipe_flux, far_field


class SoilMPMProblem:
    def __init__(
        self,
        wave: WaveParameters,
        soil: SoilParameters,
        geometry: Geometry,
        cloud: MaterialPointCloud,
        *,
        momentum_weight: float = 60.0,
        fluid_weight: float = 1000.0,
        p2g_weight: float = 0.25,
        roughness_weight: float = 1.0e-4,
        boundary_points: int = 160,
    ):
        self.wave = wave
        self.soil = soil
        self.geometry = geometry
        self.cloud = cloud
        self.momentum_weight = momentum_weight
        self.fluid_weight = fluid_weight
        self.p2g_weight = p2g_weight
        self.roughness_weight = roughness_weight
        self.boundary_points = boundary_points
        coefficients = dimensionless_coefficients(
            wave, soil, geometry
        )
        self.mass_pressure = coefficients["mass_pressure"]
        self.mass_strain = coefficients["mass_strain"]
        self.mass_scale = max(
            1.0,
            abs(self.mass_pressure) + abs(self.mass_strain),
        )

    def loss(
        self,
        model: SoilPipeKAN,
        indices: torch.Tensor,
        augmented: AugmentedBoundaryState,
    ) -> SoilLoss:
        xy = (
            self.cloud.xy[indices]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        fields = model(xy)
        grad_u = complex_gradient(fields.u, xy)
        grad_w = complex_gradient(fields.w, xy)
        grad_p = complex_gradient(fields.p, xy)
        eps_x = grad_u[:, 0]
        eps_y = grad_w[:, 1]
        eps_v = eps_x + eps_y
        lam = self.soil.lambda_over_g
        sigma_x = (2.0 + lam) * eps_x + lam * eps_y
        sigma_y = lam * eps_x + (2.0 + lam) * eps_y
        tau = grad_u[:, 1] + grad_w[:, 0]
        grad_sigma_x = complex_gradient(sigma_x, xy)
        grad_sigma_y = complex_gradient(sigma_y, xy)
        grad_tau = complex_gradient(tau, xy)
        grad_px = complex_gradient(grad_p[:, 0], xy)
        grad_py = complex_gradient(grad_p[:, 1], xy)
        imaginary = torch.as_tensor(
            1j, dtype=fields.p.dtype, device=xy.device
        )
        residual_u = (
            grad_sigma_x[:, 0]
            + grad_tau[:, 1]
            - grad_p[:, 0]
        )
        residual_w = (
            grad_tau[:, 0]
            + grad_sigma_y[:, 1]
            - grad_p[:, 1]
        )
        residual_p = (
            grad_px[:, 0]
            + grad_py[:, 1]
            + imaginary * self.mass_pressure * fields.p
            + imaginary * self.mass_strain * eps_v
        ) / self.mass_scale
        level_weights = (0.25, 0.30, 0.45)
        particle_momentum = self.cloud.weighted_level_mean(
            squared_magnitude(residual_u)
            + squared_magnitude(residual_w),
            indices,
            level_weights,
        )
        particle_fluid = self.cloud.weighted_level_mean(
            squared_magnitude(residual_p),
            indices,
            level_weights,
        )
        p2g_u = self.cloud.p2g_residual(
            residual_u, indices
        )
        p2g_w = self.cloud.p2g_residual(
            residual_w, indices
        )
        p2g_p = self.cloud.p2g_residual(
            residual_p, indices
        )
        p2g_momentum = (
            squared_magnitude(p2g_u).mean()
            + squared_magnitude(p2g_w).mean()
        )
        p2g_fluid = squared_magnitude(p2g_p).mean()
        surface_traction, outer_flux, pipe_flux = (
            self._boundary_losses(
                model, xy.dtype, xy.device
            )
        )
        epsilon = torch.finfo(xy.dtype).eps
        surface_constraint = torch.sqrt(
            surface_traction + epsilon
        )
        outer_constraint = torch.sqrt(
            outer_flux + epsilon
        )
        pipe_constraint = torch.sqrt(pipe_flux + epsilon)
        boundary_augmented = (
            augmented.lambda_surface * surface_constraint
            + 0.5
            * augmented.rho_surface
            * surface_constraint.square()
            + augmented.lambda_outer * outer_constraint
            + 0.5
            * augmented.rho_outer
            * outer_constraint.square()
            + augmented.lambda_pipe * pipe_constraint
            + 0.5
            * augmented.rho_pipe
            * pipe_constraint.square()
        )
        roughness = model.roughness()
        total = (
            self.momentum_weight * particle_momentum
            + self.fluid_weight * particle_fluid
            + self.p2g_weight
            * (
                self.momentum_weight * p2g_momentum
                + self.fluid_weight * p2g_fluid
            )
            + boundary_augmented
            + self.roughness_weight * roughness
        )
        return SoilLoss(
            total=total,
            particle_momentum=particle_momentum,
            particle_fluid=particle_fluid,
            p2g_momentum=p2g_momentum,
            p2g_fluid=p2g_fluid,
            surface_traction=surface_traction,
            outer_flux=outer_flux,
            pipe_flux=pipe_flux,
            surface_constraint=surface_constraint,
            outer_constraint=outer_constraint,
            pipe_constraint=pipe_constraint,
            boundary_augmented=boundary_augmented,
            roughness=roughness,
        )

    def _boundary_losses(
        self,
        model: SoilPipeKAN,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.boundary_points
        half = self.geometry.half_width_over_height
        x = torch.linspace(
            -half, half, n, dtype=dtype, device=device
        )
        top_y = surface_y(x, self.geometry)
        if not self.geometry.embedded_full_circle:
            keep = (
                x.abs()
                > self.geometry.radius_over_height * 1.02
            )
            x = x[keep]
            top_y = top_y[keep]
        top = torch.stack((x, top_y), dim=1).requires_grad_(
            True
        )
        fields = model(top)
        grad_u = complex_gradient(fields.u, top)
        grad_w = complex_gradient(fields.w, top)
        lam = self.soil.lambda_over_g
        sigma_x = (
            (2.0 + lam) * grad_u[:, 0]
            + lam * grad_w[:, 1]
        )
        sigma_y = (
            lam * grad_u[:, 0]
            + (2.0 + lam) * grad_w[:, 1]
        )
        tau = grad_u[:, 1] + grad_w[:, 0]
        slope = surface_slope(top[:, 0], self.geometry)
        normal_x = -slope
        normal_y = torch.ones_like(slope)
        normal_scale = torch.sqrt(
            normal_x.square() + normal_y.square()
        )
        normal_x = normal_x / normal_scale
        normal_y = normal_y / normal_scale
        traction_x = sigma_x * normal_x + tau * normal_y
        traction_y = tau * normal_x + sigma_y * normal_y
        surface_traction = (
            squared_magnitude(traction_x).mean()
            + squared_magnitude(traction_y).mean()
        )

        bottom_x = torch.linspace(
            -half, half, n, dtype=dtype, device=device
        )
        bottom = torch.stack(
            (bottom_x, torch.zeros_like(bottom_x)), dim=1
        )
        side_y = torch.linspace(
            0.0, 1.0, max(32, n // 2), dtype=dtype, device=device
        )
        left = torch.stack(
            (-half * torch.ones_like(side_y), side_y), dim=1
        )
        right = torch.stack(
            (half * torch.ones_like(side_y), side_y), dim=1
        )
        outer = torch.cat(
            (bottom, left, right), dim=0
        ).requires_grad_(True)
        outer_gradient = complex_gradient(
            model(outer).p, outer
        )
        bottom_n = bottom.shape[0]
        side_n = left.shape[0]
        outer_flux = (
            squared_magnitude(
                outer_gradient[:bottom_n, 1]
            ).mean()
            + squared_magnitude(
                outer_gradient[
                    bottom_n : bottom_n + side_n, 0
                ]
            ).mean()
            + squared_magnitude(
                outer_gradient[bottom_n + side_n :, 0]
            ).mean()
        )

        theta_start = (
            0.0
            if self.geometry.embedded_full_circle
            else torch.pi
        )
        if self.geometry.embedded_full_circle:
            theta = torch.linspace(
                theta_start,
                2.0 * torch.pi,
                n,
                dtype=dtype,
                device=device,
            )
        else:
            # The two pipe/seabed contact corners carry mixed boundary
            # conditions and are singular measure-zero endpoints.
            theta = torch.linspace(
                theta_start,
                2.0 * torch.pi,
                n + 2,
                dtype=dtype,
                device=device,
            )[1:-1]
        normal_x = torch.cos(theta)
        normal_y = torch.sin(theta)
        radius = self.geometry.radius_over_height
        pipe = torch.stack(
            (
                radius * normal_x,
                self.geometry.pipe_center_y
                + radius * normal_y,
            ),
            dim=1,
        ).requires_grad_(True)
        pipe_gradient = complex_gradient(model(pipe).p, pipe)
        pipe_flux = squared_magnitude(
            pipe_gradient[:, 0] * normal_x
            + pipe_gradient[:, 1] * normal_y
        ).mean()
        return surface_traction, outer_flux, pipe_flux

    def evaluate_pipe(
        self,
        model: SoilPipeKAN,
        water_model: WaterPressureKAN | None,
        *,
        n_theta: int = 721,
    ) -> dict[str, torch.Tensor]:
        parameter = next(model.parameters())
        theta = torch.linspace(
            0.0,
            2.0 * torch.pi,
            n_theta,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        radius = self.geometry.radius_over_height
        pipe = torch.stack(
            (
                radius * torch.cos(theta),
                self.geometry.pipe_center_y
                + radius * torch.sin(theta),
            ),
            dim=1,
        )
        pore = model(pipe).p.abs()
        if self.geometry.embedded_full_circle:
            hydrodynamic = torch.full_like(
                pore, torch.nan
            )
            combined = pore
        else:
            if water_model is None:
                raise ValueError(
                    "Test 49 requires a trained water model"
                )
            hydrodynamic = water_model(pipe).q.abs()
            upper = theta <= torch.pi
            combined = torch.where(
                upper, hydrodynamic, pore
            )
            pore = torch.where(
                upper, torch.full_like(pore, torch.nan), pore
            )
            hydrodynamic = torch.where(
                upper,
                hydrodynamic,
                torch.full_like(hydrodynamic, torch.nan),
            )
        return {
            "theta": theta.detach(),
            "combined_pressure": combined.detach(),
            "pore_pressure": pore.detach(),
            "hydrodynamic_pressure": hydrodynamic.detach(),
        }
