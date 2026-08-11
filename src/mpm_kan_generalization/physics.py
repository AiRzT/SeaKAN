from __future__ import annotations

from dataclasses import dataclass

import torch

from .cases import CaseTable
from .config import ParameterDomain, PhysicalConstants
from .model import ParametricSoilKAN
from .mpm import ParametricCloud
from .random_field import lognormal_permeability_ratio


def squared_magnitude(value: torch.Tensor) -> torch.Tensor:
    return value.real.square() + value.imag.square()


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


@dataclass
class AugmentedState:
    lambda_surface: float = 0.0
    lambda_outer: float = 0.0
    lambda_pipe: float = 0.0
    rho_surface: float = 20.0
    rho_outer: float = 30.0
    rho_pipe: float = 50.0

    def update(
        self,
        constraints: dict[str, float],
        *,
        growth: float = 1.25,
        maximum: float = 500.0,
    ) -> None:
        for name in ("surface", "outer", "pipe"):
            value = float(constraints[name])
            rho_name = f"rho_{name}"
            lambda_name = f"lambda_{name}"
            rho = float(getattr(self, rho_name))
            setattr(
                self,
                lambda_name,
                float(getattr(self, lambda_name))
                + rho * value,
            )
            setattr(
                self, rho_name, min(maximum, growth * rho)
            )

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


@dataclass
class GeneralizationLoss:
    total: torch.Tensor
    particle_momentum: torch.Tensor
    particle_fluid: torch.Tensor
    p2g_momentum: torch.Tensor
    p2g_fluid: torch.Tensor
    surface_traction: torch.Tensor
    outer_boundary: torch.Tensor
    pipe_flux: torch.Tensor
    surface_constraint: torch.Tensor
    outer_constraint: torch.Tensor
    pipe_constraint: torch.Tensor
    interface_pressure_error: torch.Tensor
    roughness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in self.__dataclass_fields__
        }


class ParametricBiotMPMProblem:
    def __init__(
        self,
        table: CaseTable,
        cloud: ParametricCloud,
        domain: ParameterDomain,
        constants: PhysicalConstants,
        *,
        momentum_weight: float = 25.0,
        fluid_weight: float = 500.0,
        particle_weight: float = 1.0,
        p2g_weight: float = 0.15,
        roughness_weight: float = 1.0e-5,
        boundary_points: int = 28,
        side_boundary_mode: str = "bloch",
    ):
        self.table = table
        self.cloud = cloud
        self.domain = domain
        self.constants = constants
        self.momentum_weight = float(momentum_weight)
        self.fluid_weight = float(fluid_weight)
        self.particle_weight = float(particle_weight)
        self.p2g_weight = float(p2g_weight)
        self.roughness_weight = float(roughness_weight)
        self.boundary_points = int(boundary_points)
        if side_boundary_mode not in {"bloch", "incident_fixed"}:
            raise ValueError(
                "side_boundary_mode must be 'bloch' or 'incident_fixed'"
            )
        self.side_boundary_mode = side_boundary_mode

    def _case_values(
        self, case_index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.table.parameters[case_index],
            self.table.kappa[case_index],
            self.table.pipe_center_y[case_index],
            self.table.mass_pressure[case_index],
            self.table.mass_strain[case_index],
        )

    def loss(
        self,
        model: ParametricSoilKAN,
        indices: torch.Tensor,
        case_ids: torch.Tensor,
        augmented: AugmentedState,
    ) -> GeneralizationLoss:
        xy = (
            self.cloud.xy[indices]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        point_cases = self.cloud.case_index[indices]
        (
            parameters,
            kappa,
            center,
            mass_pressure,
            mass_strain,
        ) = self._case_values(point_cases)
        fields = model(xy, parameters, kappa, center)
        grad_u = complex_gradient(fields.u, xy)
        grad_w = complex_gradient(fields.w, xy)
        grad_p = complex_gradient(fields.p, xy)
        eps_x = grad_u[:, 0]
        eps_y = grad_w[:, 1]
        eps_v = eps_x + eps_y
        lam = self.constants.lambda_over_g
        sigma_x = (2.0 + lam) * eps_x + lam * eps_y
        sigma_y = lam * eps_x + (2.0 + lam) * eps_y
        tau = grad_u[:, 1] + grad_w[:, 0]
        grad_sigma_x = complex_gradient(sigma_x, xy)
        grad_sigma_y = complex_gradient(sigma_y, xy)
        grad_tau = complex_gradient(tau, xy)
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

        permeability_ratio = lognormal_permeability_ratio(
            xy, parameters, self.domain, self.constants
        )
        flux_x = permeability_ratio * grad_p[:, 0]
        flux_y = permeability_ratio * grad_p[:, 1]
        divergence = (
            complex_gradient(flux_x, xy)[:, 0]
            + complex_gradient(flux_y, xy)[:, 1]
        )
        imaginary = torch.as_tensor(
            1j, dtype=fields.p.dtype, device=xy.device
        )
        mass_scale = (
            mass_pressure.abs() + mass_strain.abs()
        ).clamp_min(1.0)
        residual_p = (
            divergence
            + imaginary * mass_pressure * fields.p
            + imaginary * mass_strain * eps_v
        ) / mass_scale

        particle_momentum = self.cloud.weighted_level_mean(
            squared_magnitude(residual_u)
            + squared_magnitude(residual_w),
            indices,
        )
        particle_fluid = self.cloud.weighted_level_mean(
            squared_magnitude(residual_p), indices
        )
        p2g_u = self.cloud.p2g(residual_u, indices)
        p2g_w = self.cloud.p2g(residual_w, indices)
        p2g_p = self.cloud.p2g(residual_p, indices)
        p2g_momentum = (
            squared_magnitude(p2g_u).mean()
            + squared_magnitude(p2g_w).mean()
        )
        p2g_fluid = squared_magnitude(p2g_p).mean()
        (
            surface_traction,
            outer_boundary,
            pipe_flux,
            interface_pressure_error,
        ) = self.boundary_losses(model, case_ids)
        epsilon = torch.finfo(xy.dtype).eps
        surface_constraint = torch.sqrt(
            surface_traction + epsilon
        )
        outer_constraint = torch.sqrt(
            outer_boundary + epsilon
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
            self.particle_weight
            * (
                self.momentum_weight * particle_momentum
                + self.fluid_weight * particle_fluid
            )
            + self.p2g_weight
            * (
                self.momentum_weight * p2g_momentum
                + self.fluid_weight * p2g_fluid
            )
            + boundary_augmented
            + self.roughness_weight * roughness
        )
        return GeneralizationLoss(
            total=total,
            particle_momentum=particle_momentum,
            particle_fluid=particle_fluid,
            p2g_momentum=p2g_momentum,
            p2g_fluid=p2g_fluid,
            surface_traction=surface_traction,
            outer_boundary=outer_boundary,
            pipe_flux=pipe_flux,
            surface_constraint=surface_constraint,
            outer_constraint=outer_constraint,
            pipe_constraint=pipe_constraint,
            interface_pressure_error=interface_pressure_error,
            roughness=roughness,
        )

    def _repeat_boundary(
        self,
        template: torch.Tensor,
        case_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = template.shape[0]
        xy = template.repeat((case_ids.numel(), 1))
        repeated_cases = case_ids.repeat_interleave(count)
        return xy, repeated_cases

    def boundary_losses(
        self,
        model: ParametricSoilKAN,
        case_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        dtype = self.table.parameters.dtype
        device = self.table.parameters.device
        n = self.boundary_points
        half = self.constants.half_width

        x = torch.linspace(
            -half, half, n, device=device, dtype=dtype
        )
        top_template = torch.stack(
            (x, torch.ones_like(x)), dim=1
        )
        top, top_cases = self._repeat_boundary(
            top_template, case_ids
        )
        top = top.requires_grad_(True)
        parameters, kappa, center, _, _ = self._case_values(
            top_cases
        )
        top_fields = model(top, parameters, kappa, center)
        grad_u = complex_gradient(top_fields.u, top)
        grad_w = complex_gradient(top_fields.w, top)
        lam = self.constants.lambda_over_g
        sigma_y = lam * grad_u[:, 0] + (
            2.0 + lam
        ) * grad_w[:, 1]
        tau = grad_u[:, 1] + grad_w[:, 0]
        surface_traction = (
            squared_magnitude(sigma_y).mean()
            + squared_magnitude(tau).mean()
        )
        expected = torch.exp(
            torch.complex(
                torch.zeros_like(kappa), kappa * top[:, 0]
            )
        )
        interface_pressure_error = squared_magnitude(
            top_fields.p - expected
        ).mean()

        side_y = torch.linspace(
            0.0, 1.0, max(12, n // 2),
            device=device,
            dtype=dtype,
        )
        bottom = torch.stack(
            (x, torch.zeros_like(x)), dim=1
        )
        left = torch.stack(
            (-half * torch.ones_like(side_y), side_y), dim=1
        )
        right = torch.stack(
            (half * torch.ones_like(side_y), side_y), dim=1
        )
        bottom_points, bottom_cases = self._repeat_boundary(
            bottom, case_ids
        )
        bottom_points = bottom_points.requires_grad_(True)
        parameters, kappa, center, _, _ = self._case_values(
            bottom_cases
        )
        bottom_gradient = complex_gradient(
            model(
                bottom_points, parameters, kappa, center
            ).p,
            bottom_points,
        )
        bottom_flux = squared_magnitude(
            bottom_gradient[:, 1]
        ).mean()

        left_points, side_cases = self._repeat_boundary(
            left, case_ids
        )
        right_points, _ = self._repeat_boundary(
            right, case_ids
        )
        parameters, kappa, center, _, _ = self._case_values(
            side_cases
        )
        left_fields = model(
            left_points, parameters, kappa, center
        )
        right_fields = model(
            right_points, parameters, kappa, center
        )
        if self.side_boundary_mode == "bloch":
            phase = torch.exp(
                torch.complex(
                    torch.zeros_like(kappa),
                    2.0 * kappa * half,
                )
            )
            side_boundary = (
                squared_magnitude(
                    right_fields.p - phase * left_fields.p
                ).mean()
                + squared_magnitude(
                    right_fields.u - phase * left_fields.u
                ).mean()
                + squared_magnitude(
                    right_fields.w - phase * left_fields.w
                ).mean()
            )
        else:
            left_expected = torch.exp(
                torch.complex(
                    torch.zeros_like(kappa),
                    kappa * left_points[:, 0],
                )
            )
            right_expected = torch.exp(
                torch.complex(
                    torch.zeros_like(kappa),
                    kappa * right_points[:, 0],
                )
            )
            side_boundary = (
                squared_magnitude(
                    left_fields.p - left_expected
                ).mean()
                + squared_magnitude(
                    right_fields.p - right_expected
                ).mean()
                + squared_magnitude(left_fields.u).mean()
                + squared_magnitude(left_fields.w).mean()
                + squared_magnitude(right_fields.u).mean()
                + squared_magnitude(right_fields.w).mean()
            )
        outer_boundary = bottom_flux + side_boundary

        theta = torch.linspace(
            0.0,
            2.0 * torch.pi,
            n + 1,
            device=device,
            dtype=dtype,
        )[:-1]
        pipe_case = case_ids.repeat_interleave(theta.numel())
        repeated_theta = theta.repeat(case_ids.numel())
        pipe_center = self.table.pipe_center_y[pipe_case]
        normal_x = torch.cos(repeated_theta)
        normal_y = torch.sin(repeated_theta)
        pipe = torch.stack(
            (
                self.constants.radius * normal_x,
                pipe_center
                + self.constants.radius * normal_y,
            ),
            dim=1,
        ).requires_grad_(True)
        parameters, kappa, center, _, _ = self._case_values(
            pipe_case
        )
        pipe_gradient = complex_gradient(
            model(pipe, parameters, kappa, center).p, pipe
        )
        pipe_flux = squared_magnitude(
            pipe_gradient[:, 0] * normal_x
            + pipe_gradient[:, 1] * normal_y
        ).mean()
        return (
            surface_traction,
            outer_boundary,
            pipe_flux,
            interface_pressure_error,
        )
