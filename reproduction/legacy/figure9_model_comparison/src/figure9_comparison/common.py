from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mpm_kan_figure9.config import (
    Geometry,
    SoilParameters,
    WaveParameters,
    dimensionless_coefficients,
)


@dataclass(frozen=True)
class PhysicsConstants:
    half_width: float
    center_y: float
    radius: float
    kappa: float
    lam: float
    mass_pressure: float
    mass_strain: float
    mass_scale: float


def physics_constants(
    wave: WaveParameters, soil: SoilParameters, geometry: Geometry
) -> PhysicsConstants:
    coefficients = dimensionless_coefficients(wave, soil, geometry)
    return PhysicsConstants(
        half_width=geometry.half_width_over_height,
        center_y=geometry.center_y,
        radius=geometry.radius_over_height,
        kappa=coefficients["kappa"],
        lam=soil.lambda_over_g,
        mass_pressure=coefficients["mass_pressure"],
        mass_strain=coefficients["mass_strain"],
        mass_scale=max(
            1.0,
            abs(coefficients["mass_pressure"])
            + abs(coefficients["mass_strain"]),
        ),
    )


class TorchMLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [2] + [width] * depth + [6]
        for index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if index < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


class DeepXDEFNN(nn.Module):
    """DeepXDE's PyTorch FNN exposed through the common physics loss."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        import deepxde as dde

        self.net = dde.nn.FNN(
            [2] + [width] * depth + [6], "tanh", "Glorot normal"
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def normalized_coordinates(xy: torch.Tensor, constants: PhysicsConstants) -> torch.Tensor:
    return torch.stack(
        (xy[:, 0] / constants.half_width, 2.0 * xy[:, 1] - 1.0), dim=1
    )


def hard_fields(
    network: nn.Module, xy: torch.Tensor, constants: PhysicsConstants
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = network(normalized_coordinates(xy, constants))
    x, y = xy[:, 0], xy[:, 1]
    radial_level = x.square() + (y - constants.center_y).square() - constants.radius**2
    pipe_factor = 1.0 - torch.exp(
        -torch.clamp(radial_level, min=0.0) / constants.radius**2
    )
    outer_factor = y * (1.0 - (x / constants.half_width).square())
    displacement_factor = outer_factor * pipe_factor
    phase = constants.kappa * x
    top_real, top_imag = torch.cos(phase), torch.sin(phase)
    u = torch.complex(
        displacement_factor * raw[:, 0], displacement_factor * raw[:, 1]
    )
    w = torch.complex(
        displacement_factor * raw[:, 2], displacement_factor * raw[:, 3]
    )
    p = torch.complex(
        top_real + (1.0 - y) * raw[:, 4],
        top_imag + (1.0 - y) * raw[:, 5],
    )
    return u, w, p


def complex_gradient(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    real = torch.autograd.grad(
        value.real.sum(), xy, create_graph=True, retain_graph=True
    )[0]
    imag = torch.autograd.grad(
        value.imag.sum(), xy, create_graph=True, retain_graph=True
    )[0]
    return torch.complex(real, imag)


def abs2(value: torch.Tensor) -> torch.Tensor:
    return value.real.square() + value.imag.square()


def strong_residuals(
    network: nn.Module, xy: torch.Tensor, constants: PhysicsConstants
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, w, p = hard_fields(network, xy, constants)
    grad_u = complex_gradient(u, xy)
    grad_w = complex_gradient(w, xy)
    grad_p = complex_gradient(p, xy)
    eps_x, eps_y = grad_u[:, 0], grad_w[:, 1]
    eps_v = eps_x + eps_y
    sigma_x = (2.0 + constants.lam) * eps_x + constants.lam * eps_y
    sigma_y = constants.lam * eps_x + (2.0 + constants.lam) * eps_y
    tau = grad_u[:, 1] + grad_w[:, 0]
    grad_sigma_x = complex_gradient(sigma_x, xy)
    grad_sigma_y = complex_gradient(sigma_y, xy)
    grad_tau = complex_gradient(tau, xy)
    grad_px = complex_gradient(grad_p[:, 0], xy)
    grad_py = complex_gradient(grad_p[:, 1], xy)
    r_u = grad_sigma_x[:, 0] + grad_tau[:, 1] - grad_p[:, 0]
    r_w = grad_tau[:, 0] + grad_sigma_y[:, 1] - grad_p[:, 1]
    imaginary = torch.as_tensor(1j, dtype=p.dtype, device=p.device)
    r_p = (
        grad_px[:, 0]
        + grad_py[:, 1]
        + imaginary * constants.mass_pressure * p
        + imaginary * constants.mass_strain * eps_v
    ) / constants.mass_scale
    return r_u, r_w, r_p


def sample_interior(
    n: int,
    constants: PhysicsConstants,
    device: torch.device,
    dtype: torch.dtype,
    *,
    generator: torch.Generator | None = None,
    x_bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    left, right = x_bounds or (-constants.half_width, constants.half_width)
    pieces: list[torch.Tensor] = []
    total = 0
    while total < n:
        batch = max(n - total, 256)
        unit = torch.rand(
            (batch, 2), device=device, dtype=dtype, generator=generator
        )
        x = left + (right - left) * unit[:, 0]
        y = unit[:, 1]
        keep = (
            x.square() + (y - constants.center_y).square()
            > (1.003 * constants.radius) ** 2
        )
        accepted = torch.stack((x[keep], y[keep]), dim=1)
        pieces.append(accepted)
        total += accepted.shape[0]
    return torch.cat(pieces, dim=0)[:n]


def boundary_points(
    n: int,
    constants: PhysicsConstants,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(
        -constants.half_width, constants.half_width, n, device=device, dtype=dtype
    )
    top = torch.stack((x, torch.ones_like(x)), dim=1)
    bottom = torch.stack((x, torch.zeros_like(x)), dim=1)
    y = torch.linspace(0.0, 1.0, n // 2 + 2, device=device, dtype=dtype)[1:-1]
    sides = torch.cat(
        (
            torch.stack((-constants.half_width * torch.ones_like(y), y), dim=1),
            torch.stack((constants.half_width * torch.ones_like(y), y), dim=1),
        ),
        dim=0,
    )
    theta = torch.linspace(
        0.0, 2.0 * torch.pi, n + 1, device=device, dtype=dtype
    )[:-1]
    pipe = torch.stack(
        (
            constants.radius * torch.cos(theta),
            constants.center_y + constants.radius * torch.sin(theta),
        ),
        dim=1,
    )
    return top, bottom, sides, pipe


def boundary_loss(
    network: nn.Module, constants: PhysicsConstants, n: int = 96
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = next(network.parameters())
    top, bottom, sides, pipe = boundary_points(
        n, constants, parameter.device, parameter.dtype
    )
    top.requires_grad_(True)
    u, w, _ = hard_fields(network, top, constants)
    grad_u, grad_w = complex_gradient(u, top), complex_gradient(w, top)
    eps_x, eps_y = grad_u[:, 0], grad_w[:, 1]
    sigma_y = constants.lam * eps_x + (2.0 + constants.lam) * eps_y
    tau = grad_u[:, 1] + grad_w[:, 0]
    top_traction = abs2(sigma_y).mean() + abs2(tau).mean()

    outer = torch.cat((bottom, sides), dim=0).requires_grad_(True)
    _, _, p_outer = hard_fields(network, outer, constants)
    grad_outer = complex_gradient(p_outer, outer)
    n_bottom = bottom.shape[0]
    n_side = sides.shape[0] // 2
    outer_flux = (
        abs2(grad_outer[:n_bottom, 1]).mean()
        + abs2(grad_outer[n_bottom : n_bottom + n_side, 0]).mean()
        + abs2(grad_outer[n_bottom + n_side :, 0]).mean()
    )

    pipe.requires_grad_(True)
    _, _, p_pipe = hard_fields(network, pipe, constants)
    grad_pipe = complex_gradient(p_pipe, pipe)
    theta = torch.linspace(
        0.0, 2.0 * torch.pi, n + 1, device=parameter.device, dtype=parameter.dtype
    )[:-1]
    normal_flux = grad_pipe[:, 0] * torch.cos(theta) + grad_pipe[:, 1] * torch.sin(theta)
    pipe_flux = abs2(normal_flux).mean()
    total = 50.0 * top_traction + 50.0 * outer_flux + 100.0 * pipe_flux
    return total, {
        "top_traction": top_traction,
        "outer_flux": outer_flux,
        "pipe_flux": pipe_flux,
    }


def pipe_profile(
    network: nn.Module, constants: PhysicsConstants, n_theta: int = 721
) -> tuple[np.ndarray, np.ndarray]:
    parameter = next(network.parameters())
    theta = torch.linspace(
        0.0,
        2.0 * torch.pi,
        n_theta,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    xy = torch.stack(
        (
            constants.radius * torch.cos(theta),
            constants.center_y + constants.radius * torch.sin(theta),
        ),
        dim=1,
    )
    with torch.no_grad():
        _, _, p = hard_fields(network, xy, constants)
    return theta.cpu().numpy(), torch.abs(p).cpu().numpy()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
