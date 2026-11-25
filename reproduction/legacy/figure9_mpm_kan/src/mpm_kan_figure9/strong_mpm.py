from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_mpm import AdaptiveMaterialPointCloud
from .config import Geometry, SoilParameters, WaveParameters, dimensionless_coefficients
from .model import PipeHarmonicKAN
from .physics import complex_gradient, squared_magnitude


@dataclass
class AugmentedBoundaryState:
    """Scalar augmented-Lagrangian state for the three boundary groups."""

    lambda_top: float = 0.0
    lambda_outer: float = 0.0
    lambda_pipe: float = 0.0
    rho_top: float = 20.0
    rho_outer: float = 20.0
    rho_pipe: float = 40.0

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "lambda_top",
                "lambda_outer",
                "lambda_pipe",
                "rho_top",
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
        penalty_growth: float = 1.5,
        maximum_penalty: float = 5000.0,
    ) -> None:
        for name in ("top", "outer", "pipe"):
            constraint = max(0.0, float(constraints[name]))
            lambda_name = f"lambda_{name}"
            rho_name = f"rho_{name}"
            rho = float(getattr(self, rho_name))
            setattr(
                self,
                lambda_name,
                max(0.0, float(getattr(self, lambda_name)) + rho * constraint),
            )
            if (
                previous is not None
                and constraint > progress_ratio * float(previous[name])
            ):
                setattr(
                    self,
                    rho_name,
                    min(maximum_penalty, penalty_growth * rho),
                )


@dataclass
class StrongLossBreakdown:
    total: torch.Tensor
    particle_momentum: torch.Tensor
    particle_fluid: torch.Tensor
    p2g_momentum: torch.Tensor
    p2g_fluid: torch.Tensor
    top_traction: torch.Tensor
    outer_flux: torch.Tensor
    pipe_flux: torch.Tensor
    top_constraint: torch.Tensor
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
                "top_traction",
                "outer_flux",
                "pipe_flux",
                "top_constraint",
                "outer_constraint",
                "pipe_constraint",
                "boundary_augmented",
                "roughness",
            )
        }


class AdaptiveStrongMPMProblem:
    """Strong-form Biot residual evaluated on adaptive MPM particles."""

    def __init__(
        self,
        wave: WaveParameters,
        soil: SoilParameters,
        geometry: Geometry,
        cloud: AdaptiveMaterialPointCloud,
        *,
        level_weights: tuple[float, ...] = (0.20, 0.32, 0.48),
        momentum_weight: float = 10.0,
        fluid_weight: float = 800.0,
        p2g_weight: float = 0.20,
        boundary_weight: float = 80.0,
        pipe_flux_weight: float = 160.0,
        roughness_weight: float = 1.0e-3,
        boundary_points: int = 160,
    ):
        self.wave = wave
        self.soil = soil
        self.geometry = geometry
        self.cloud = cloud
        self.level_weights = level_weights
        self.momentum_weight = momentum_weight
        self.fluid_weight = fluid_weight
        self.p2g_weight = p2g_weight
        self.boundary_weight = boundary_weight
        self.pipe_flux_weight = pipe_flux_weight
        self.roughness_weight = roughness_weight
        self.boundary_points = boundary_points
        coefficients = dimensionless_coefficients(wave, soil, geometry)
        self.mass_pressure = coefficients["mass_pressure"]
        self.mass_strain = coefficients["mass_strain"]
        self.mass_scale = max(
            1.0, abs(self.mass_pressure) + abs(self.mass_strain)
        )

    def loss(
        self,
        model: PipeHarmonicKAN,
        indices: torch.Tensor,
        augmented: AugmentedBoundaryState | None = None,
    ) -> StrongLossBreakdown:
        xy = self.cloud.xy[indices].detach().clone().requires_grad_(True)
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
        residual_u = grad_sigma_x[:, 0] + grad_tau[:, 1] - grad_p[:, 0]
        residual_w = grad_tau[:, 0] + grad_sigma_y[:, 1] - grad_p[:, 1]
        residual_p = (
            grad_px[:, 0]
            + grad_py[:, 1]
            + imaginary * self.mass_pressure * fields.p
            + imaginary * self.mass_strain * eps_v
        ) / self.mass_scale

        particle_momentum = self.cloud.weighted_level_mean(
            squared_magnitude(residual_u) + squared_magnitude(residual_w),
            indices,
            self.level_weights,
        )
        particle_fluid = self.cloud.weighted_level_mean(
            squared_magnitude(residual_p), indices, self.level_weights
        )
        nodal_u, _ = self.cloud.p2g_residual(residual_u, indices)
        nodal_w, _ = self.cloud.p2g_residual(residual_w, indices)
        nodal_p, _ = self.cloud.p2g_residual(residual_p, indices)
        p2g_momentum = (
            squared_magnitude(nodal_u).mean()
            + squared_magnitude(nodal_w).mean()
        )
        p2g_fluid = squared_magnitude(nodal_p).mean()

        top_traction, outer_flux, pipe_flux = self._boundary_losses(
            model, xy.dtype, xy.device
        )
        epsilon = torch.finfo(xy.dtype).eps
        top_constraint = torch.sqrt(top_traction + epsilon)
        outer_constraint = torch.sqrt(outer_flux + epsilon)
        pipe_constraint = torch.sqrt(pipe_flux + epsilon)
        if augmented is None:
            boundary_augmented = (
                self.boundary_weight * (top_traction + outer_flux)
                + self.pipe_flux_weight * pipe_flux
            )
        else:
            boundary_augmented = (
                augmented.lambda_top * top_constraint
                + 0.5 * augmented.rho_top * top_constraint.square()
                + augmented.lambda_outer * outer_constraint
                + 0.5 * augmented.rho_outer * outer_constraint.square()
                + augmented.lambda_pipe * pipe_constraint
                + 0.5 * augmented.rho_pipe * pipe_constraint.square()
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
        return StrongLossBreakdown(
            total=total,
            particle_momentum=particle_momentum,
            particle_fluid=particle_fluid,
            p2g_momentum=p2g_momentum,
            p2g_fluid=p2g_fluid,
            top_traction=top_traction,
            outer_flux=outer_flux,
            pipe_flux=pipe_flux,
            top_constraint=top_constraint,
            outer_constraint=outer_constraint,
            pipe_constraint=pipe_constraint,
            boundary_augmented=boundary_augmented,
            roughness=roughness,
        )

    def _boundary_losses(
        self,
        model: PipeHarmonicKAN,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.boundary_points
        half = self.geometry.half_width_over_height
        x = torch.linspace(-half, half, n, dtype=dtype, device=device)
        top = torch.stack((x, torch.ones_like(x)), dim=1).requires_grad_(True)
        top_fields = model(top)
        top_grad_u = complex_gradient(top_fields.u, top)
        top_grad_w = complex_gradient(top_fields.w, top)
        lam = self.soil.lambda_over_g
        sigma_y = lam * top_grad_u[:, 0] + (2.0 + lam) * top_grad_w[:, 1]
        tau = top_grad_u[:, 1] + top_grad_w[:, 0]
        top_traction = (
            squared_magnitude(sigma_y).mean()
            + squared_magnitude(tau).mean()
        )

        bottom = torch.stack((x, torch.zeros_like(x)), dim=1)
        y = torch.linspace(
            0.0, 1.0, max(40, n // 2) + 2, dtype=dtype, device=device
        )[1:-1]
        left = torch.stack((-half * torch.ones_like(y), y), dim=1)
        right = torch.stack((half * torch.ones_like(y), y), dim=1)
        outer = torch.cat((bottom, left, right), dim=0).requires_grad_(True)
        outer_pressure = model(outer).p
        outer_gradient = complex_gradient(outer_pressure, outer)
        bottom_n = x.numel()
        side_n = y.numel()
        outer_flux = (
            squared_magnitude(outer_gradient[:bottom_n, 1]).mean()
            + squared_magnitude(
                outer_gradient[bottom_n : bottom_n + side_n, 0]
            ).mean()
            + squared_magnitude(
                outer_gradient[bottom_n + side_n :, 0]
            ).mean()
        )

        theta = torch.linspace(
            0.0, 2.0 * torch.pi, n + 1, dtype=dtype, device=device
        )[:-1]
        normal_x = torch.cos(theta)
        normal_y = torch.sin(theta)
        radius = self.geometry.radius_over_height
        pipe = torch.stack(
            (
                radius * normal_x,
                self.geometry.center_y + radius * normal_y,
            ),
            dim=1,
        ).requires_grad_(True)
        pipe_gradient = complex_gradient(model(pipe).p, pipe)
        pipe_flux = squared_magnitude(
            pipe_gradient[:, 0] * normal_x + pipe_gradient[:, 1] * normal_y
        ).mean()
        return top_traction, outer_flux, pipe_flux
