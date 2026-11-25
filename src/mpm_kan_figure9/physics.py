from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Geometry, SoilParameters, WaveParameters, dimensionless_coefficients
from .model import HarmonicFields, PipeHarmonicKAN
from .mpm import MaterialPointGrid2D


def complex_gradient(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    real = torch.autograd.grad(
        value.real.sum(), xy, create_graph=True, retain_graph=True
    )[0]
    imag = torch.autograd.grad(
        value.imag.sum(), xy, create_graph=True, retain_graph=True
    )[0]
    return torch.complex(real, imag)


def squared_magnitude(value: torch.Tensor) -> torch.Tensor:
    return value.real.square() + value.imag.square()


@dataclass
class FieldResponse:
    fields: HarmonicFields
    grad_u: torch.Tensor
    grad_w: torch.Tensor
    grad_p: torch.Tensor
    eps_v: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    tau_xy: torch.Tensor


@dataclass
class LossBreakdown:
    total: torch.Tensor
    momentum: torch.Tensor
    fluid: torch.Tensor
    strong: torch.Tensor
    top_traction: torch.Tensor
    outer_flux: torch.Tensor
    pipe_flux: torch.Tensor
    roughness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "momentum",
                "fluid",
                "strong",
                "top_traction",
                "outer_flux",
                "pipe_flux",
                "roughness",
            )
        }


class MPMKANPipeProblem:
    """Frequency-domain 2-D Biot problem with MPM weak residual assembly."""

    def __init__(
        self,
        wave: WaveParameters,
        soil: SoilParameters,
        geometry: Geometry,
        grid: MaterialPointGrid2D,
        *,
        momentum_weight: float = 1.0,
        fluid_weight: float = 20.0,
        boundary_weight: float = 80.0,
        pipe_flux_weight: float = 120.0,
        strong_weight: float = 0.02,
        roughness_weight: float = 1.0e-6,
        strong_stride: int = 3,
        boundary_points: int = 128,
    ):
        self.wave = wave
        self.soil = soil
        self.geometry = geometry
        self.grid = grid
        self.momentum_weight = momentum_weight
        self.fluid_weight = fluid_weight
        self.boundary_weight = boundary_weight
        self.pipe_flux_weight = pipe_flux_weight
        self.strong_weight = strong_weight
        self.roughness_weight = roughness_weight
        self.strong_stride = max(1, strong_stride)
        self.boundary_points = max(24, boundary_points)
        coefficients = dimensionless_coefficients(wave, soil, geometry)
        self.kappa = coefficients["kappa"]
        self.mass_pressure = coefficients["mass_pressure"]
        self.mass_strain = coefficients["mass_strain"]
        self.mass_scale = max(1.0, abs(self.mass_pressure) + abs(self.mass_strain))

    def response(
        self, model: PipeHarmonicKAN, xy: torch.Tensor
    ) -> FieldResponse:
        fields = model(xy)
        grad_u = complex_gradient(fields.u, xy)
        grad_w = complex_gradient(fields.w, xy)
        grad_p = complex_gradient(fields.p, xy)
        eps_x = grad_u[:, 0]
        eps_y = grad_w[:, 1]
        gamma_xy = grad_u[:, 1] + grad_w[:, 0]
        eps_v = eps_x + eps_y
        lam = self.soil.lambda_over_g
        sigma_x = (2.0 + lam) * eps_x + lam * eps_y
        sigma_y = lam * eps_x + (2.0 + lam) * eps_y
        tau_xy = gamma_xy
        return FieldResponse(
            fields=fields,
            grad_u=grad_u,
            grad_w=grad_w,
            grad_p=grad_p,
            eps_v=eps_v,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            tau_xy=tau_xy,
        )

    def loss(self, model: PipeHarmonicKAN) -> LossBreakdown:
        xy = self.grid.xy.detach().clone().requires_grad_(True)
        response = self.response(model, xy)
        i = torch.as_tensor(1j, dtype=response.fields.p.dtype, device=xy.device)

        # MPM weak equilibrium for total stress sigma' - p I.
        r_u = -self.grid.p2g(
            response.sigma_x - response.fields.p, derivative="x"
        ) - self.grid.p2g(response.tau_xy, derivative="y")
        r_w = -self.grid.p2g(
            response.tau_xy, derivative="x"
        ) - self.grid.p2g(
            response.sigma_y - response.fields.p, derivative="y"
        )
        top_pressure = model(self.grid.top_xy).p
        r_w = r_w + self.grid.top_p2g(-top_pressure)

        volume_scale = self.grid.nodal_volume.clamp_min(1.0e-9)
        r_u = r_u[self.grid.momentum_nodes] / volume_scale[self.grid.momentum_nodes]
        r_w = r_w[self.grid.momentum_nodes] / volume_scale[self.grid.momentum_nodes]
        momentum = squared_magnitude(r_u).mean() + squared_magnitude(r_w).mean()

        # Weak fluid conservation. Zero flux is the natural condition at the
        # rigid walls and impermeable pipe; top pressure is essential.
        r_p = (
            -self.grid.p2g(response.grad_p[:, 0], derivative="x")
            - self.grid.p2g(response.grad_p[:, 1], derivative="y")
            + i * self.mass_pressure * self.grid.p2g(response.fields.p)
            + i * self.mass_strain * self.grid.p2g(response.eps_v)
        )
        r_p = (
            r_p[self.grid.fluid_nodes]
            / volume_scale[self.grid.fluid_nodes]
            / self.mass_scale
        )
        fluid = squared_magnitude(r_p).mean()

        strong = (
            self._strong_residual(model, xy.device, xy.dtype)
            if self.strong_weight > 0.0
            else torch.zeros((), dtype=xy.dtype, device=xy.device)
        )
        top_traction, outer_flux, pipe_flux = self._boundary_residuals(
            model, xy.device, xy.dtype
        )
        roughness = model.roughness()
        total = (
            self.momentum_weight * momentum
            + self.fluid_weight * fluid
            + self.strong_weight * strong
            + self.boundary_weight * (top_traction + outer_flux)
            + self.pipe_flux_weight * pipe_flux
            + self.roughness_weight * roughness
        )
        return LossBreakdown(
            total=total,
            momentum=momentum,
            fluid=fluid,
            strong=strong,
            top_traction=top_traction,
            outer_flux=outer_flux,
            pipe_flux=pipe_flux,
            roughness=roughness,
        )

    def _strong_residual(
        self, model: PipeHarmonicKAN, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        xy = self.grid.xy[:: self.strong_stride].detach().clone()
        xy = xy.to(device=device, dtype=dtype).requires_grad_(True)
        response = self.response(model, xy)
        grad_sigma_x = complex_gradient(response.sigma_x, xy)
        grad_sigma_y = complex_gradient(response.sigma_y, xy)
        grad_tau = complex_gradient(response.tau_xy, xy)
        grad_px = complex_gradient(response.grad_p[:, 0], xy)
        grad_py = complex_gradient(response.grad_p[:, 1], xy)
        i = torch.as_tensor(1j, dtype=response.fields.p.dtype, device=device)
        r_u = grad_sigma_x[:, 0] + grad_tau[:, 1] - response.grad_p[:, 0]
        r_w = grad_tau[:, 0] + grad_sigma_y[:, 1] - response.grad_p[:, 1]
        r_p = (
            grad_px[:, 0]
            + grad_py[:, 1]
            + i * self.mass_pressure * response.fields.p
            + i * self.mass_strain * response.eps_v
        ) / self.mass_scale
        return (
            squared_magnitude(r_u).mean()
            + squared_magnitude(r_w).mean()
            + self.fluid_weight * squared_magnitude(r_p).mean()
        )

    def _boundary_residuals(
        self, model: PipeHarmonicKAN, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.boundary_points
        half = self.geometry.half_width_over_height
        x = torch.linspace(-half, half, n, dtype=dtype, device=device)
        top = torch.stack((x, torch.ones_like(x)), dim=1).requires_grad_(True)
        top_response = self.response(model, top)
        top_traction = squared_magnitude(top_response.sigma_y).mean() + (
            squared_magnitude(top_response.tau_xy).mean()
        )

        bottom = torch.stack((x, torch.zeros_like(x)), dim=1)
        y = torch.linspace(
            0.0, 1.0, max(32, n // 2) + 2, dtype=dtype, device=device
        )[1:-1]
        left = torch.stack((-half * torch.ones_like(y), y), dim=1)
        right = torch.stack((half * torch.ones_like(y), y), dim=1)
        outer = torch.cat((bottom, left, right), dim=0).requires_grad_(True)
        outer_response = self.response(model, outer)
        bottom_n = n
        side_n = y.numel()
        bottom_flux = outer_response.grad_p[:bottom_n, 1]
        left_flux = outer_response.grad_p[bottom_n : bottom_n + side_n, 0]
        right_flux = outer_response.grad_p[bottom_n + side_n :, 0]
        outer_flux = (
            squared_magnitude(bottom_flux).mean()
            + squared_magnitude(left_flux).mean()
            + squared_magnitude(right_flux).mean()
        )

        theta = torch.linspace(
            0.0,
            2.0 * torch.pi,
            n + 1,
            dtype=dtype,
            device=device,
        )[:-1]
        nx = torch.cos(theta)
        ny = torch.sin(theta)
        radius = self.geometry.radius_over_height
        pipe = torch.stack(
            (
                radius * nx,
                self.geometry.center_y + radius * ny,
            ),
            dim=1,
        ).requires_grad_(True)
        pipe_response = self.response(model, pipe)
        normal_flux = pipe_response.grad_p[:, 0] * nx + pipe_response.grad_p[:, 1] * ny
        pipe_flux = squared_magnitude(normal_flux).mean()
        return top_traction, outer_flux, pipe_flux

    def evaluate_pipe(
        self, model: PipeHarmonicKAN, n_theta: int = 721
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
        xy = torch.stack(
            (
                radius * torch.cos(theta),
                self.geometry.center_y + radius * torch.sin(theta),
            ),
            dim=1,
        )
        pressure = model(xy).p
        return {
            "theta": theta.detach(),
            "normalized_pressure": torch.abs(pressure).detach(),
            "pressure_real": pressure.real.detach(),
            "pressure_imag": pressure.imag.detach(),
        }

    def evaluate_field(
        self, model: PipeHarmonicKAN, nx: int = 241, ny: int = 81
    ) -> dict[str, torch.Tensor]:
        parameter = next(model.parameters())
        half = self.geometry.half_width_over_height
        x = torch.linspace(
            -half, half, nx, dtype=parameter.dtype, device=parameter.device
        )
        y = torch.linspace(
            0.0, 1.0, ny, dtype=parameter.dtype, device=parameter.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        p = torch.abs(model(xy).p).reshape(ny, nx)
        inside = (
            xx.square() + (yy - self.geometry.center_y).square()
            <= self.geometry.radius_over_height**2
        )
        p = p.masked_fill(inside, torch.nan)
        return {"x": x.detach(), "y": y.detach(), "pressure": p.detach()}
