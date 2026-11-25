from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
)
from .model import DepthHarmonicKAN
from .mpm import MaterialPointCloud2D


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
class AugmentedBoundaryState:
    lambda_top: float = 0.0
    lambda_bottom: float = 0.0
    rho_top: float = 160.0
    rho_bottom: float = 320.0

    def as_dict(self) -> dict[str, float]:
        return {
            "lambda_top": float(self.lambda_top),
            "lambda_bottom": float(self.lambda_bottom),
            "rho_top": float(self.rho_top),
            "rho_bottom": float(self.rho_bottom),
        }

    def update(
        self,
        top: float,
        bottom: float,
        previous: tuple[float, float] | None = None,
        *,
        progress_ratio: float = 0.90,
        growth: float = 1.5,
        maximum_penalty: float = 5000.0,
    ) -> tuple[float, float]:
        top = max(0.0, float(top))
        bottom = max(0.0, float(bottom))
        self.lambda_top = max(
            0.0, self.lambda_top + self.rho_top * top
        )
        self.lambda_bottom = max(
            0.0, self.lambda_bottom + self.rho_bottom * bottom
        )
        if previous is not None:
            if top > progress_ratio * previous[0]:
                self.rho_top = min(
                    maximum_penalty, growth * self.rho_top
                )
            if bottom > progress_ratio * previous[1]:
                self.rho_bottom = min(
                    maximum_penalty, growth * self.rho_bottom
                )
        return top, bottom


@dataclass
class LossBreakdown:
    total: torch.Tensor
    particle_momentum: torch.Tensor
    particle_fluid: torch.Tensor
    p2g_momentum: torch.Tensor
    p2g_fluid: torch.Tensor
    top_traction: torch.Tensor
    bottom_flux: torch.Tensor
    top_constraint: torch.Tensor
    bottom_constraint: torch.Tensor
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
                "bottom_flux",
                "top_constraint",
                "bottom_constraint",
                "boundary_augmented",
                "roughness",
            )
        }


class Figure13MPMKANProblem:
    """Strong-form 2-D harmonic Biot problem on MPM particles."""

    def __init__(
        self,
        wave: WaveParameters,
        soil: SoilParameters,
        geometry: Geometry,
        cloud: MaterialPointCloud2D,
        *,
        momentum_weight: float = 10.0,
        fluid_weight: float = 1000.0,
        p2g_weight: float = 0.25,
        roughness_weight: float = 1.0e-4,
        boundary_points: int = 128,
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
        model: DepthHarmonicKAN,
        indices: torch.Tensor,
        augmented: AugmentedBoundaryState,
    ) -> LossBreakdown:
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
            1j,
            dtype=fields.p.dtype,
            device=xy.device,
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

        momentum_density = (
            squared_magnitude(residual_u)
            + squared_magnitude(residual_w)
        )
        fluid_density = squared_magnitude(residual_p)
        particle_momentum = self.cloud.weighted_mean(
            momentum_density, indices
        )
        particle_fluid = self.cloud.weighted_mean(
            fluid_density, indices
        )
        nodal_u = self.cloud.p2g_residual(residual_u, indices)
        nodal_w = self.cloud.p2g_residual(residual_w, indices)
        nodal_p = self.cloud.p2g_residual(residual_p, indices)
        p2g_momentum = (
            squared_magnitude(nodal_u).mean()
            + squared_magnitude(nodal_w).mean()
        )
        p2g_fluid = squared_magnitude(nodal_p).mean()

        top_traction, bottom_flux = self._boundary_losses(
            model, xy.dtype, xy.device
        )
        epsilon = torch.finfo(xy.dtype).eps
        top_constraint = torch.sqrt(top_traction + epsilon)
        bottom_constraint = torch.sqrt(bottom_flux + epsilon)
        boundary_augmented = (
            augmented.lambda_top * top_constraint
            + 0.5 * augmented.rho_top * top_constraint.square()
            + augmented.lambda_bottom * bottom_constraint
            + 0.5
            * augmented.rho_bottom
            * bottom_constraint.square()
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
        return LossBreakdown(
            total=total,
            particle_momentum=particle_momentum,
            particle_fluid=particle_fluid,
            p2g_momentum=p2g_momentum,
            p2g_fluid=p2g_fluid,
            top_traction=top_traction,
            bottom_flux=bottom_flux,
            top_constraint=top_constraint,
            bottom_constraint=bottom_constraint,
            boundary_augmented=boundary_augmented,
            roughness=roughness,
        )

    def _boundary_losses(
        self,
        model: DepthHarmonicKAN,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(
            0.0,
            self.cloud.length_x,
            self.boundary_points,
            dtype=dtype,
            device=device,
        )
        top = torch.stack(
            (x, torch.ones_like(x)), dim=1
        ).requires_grad_(True)
        top_fields = model(top)
        top_grad_u = complex_gradient(top_fields.u, top)
        top_grad_w = complex_gradient(top_fields.w, top)
        lam = self.soil.lambda_over_g
        sigma_y = (
            lam * top_grad_u[:, 0]
            + (2.0 + lam) * top_grad_w[:, 1]
        )
        tau = top_grad_u[:, 1] + top_grad_w[:, 0]
        top_traction = (
            squared_magnitude(sigma_y).mean()
            + squared_magnitude(tau).mean()
        )

        bottom = torch.stack(
            (x, torch.zeros_like(x)), dim=1
        ).requires_grad_(True)
        bottom_gradient = complex_gradient(model(bottom).p, bottom)
        bottom_flux = squared_magnitude(
            bottom_gradient[:, 1]
        ).mean()
        return top_traction, bottom_flux

    def evaluate_profile(
        self,
        model: DepthHarmonicKAN,
        *,
        n_depth: int = 501,
    ) -> dict[str, torch.Tensor]:
        parameter = next(model.parameters())
        y = torch.linspace(
            0.0,
            1.0,
            n_depth,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        x_value = self.geometry.evaluation_x / (
            self.geometry.seabed_thickness
        )
        x = torch.full_like(y, x_value)
        xy = torch.stack((x, y), dim=1).requires_grad_(True)
        fields = model(xy)
        grad_u = complex_gradient(fields.u, xy)
        grad_w = complex_gradient(fields.w, xy)
        lam = self.soil.lambda_over_g
        sigma_y = (
            lam * grad_u[:, 0]
            + (2.0 + lam) * grad_w[:, 1]
        )
        z = (
            self.geometry.bottom_z
            + y * self.geometry.seabed_thickness
        )
        return {
            "z": z.detach(),
            "pressure": fields.p.abs().detach(),
            "vertical_effective_stress": sigma_y.abs().detach(),
        }
