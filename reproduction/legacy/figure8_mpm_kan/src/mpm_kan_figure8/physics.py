from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SeabedCase, SoilParameters, WaveParameters
from .model import HarmonicFields, HarmonicKAN
from .mpm import MaterialPointGrid1D


def complex_gradient(value: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    real = torch.autograd.grad(
        value.real.sum(), coordinate, create_graph=True, retain_graph=True
    )[0]
    imag = torch.autograd.grad(
        value.imag.sum(), coordinate, create_graph=True, retain_graph=True
    )[0]
    return torch.complex(real[:, 0], imag[:, 0])


def squared_magnitude(value: torch.Tensor) -> torch.Tensor:
    return value.real.square() + value.imag.square()


@dataclass
class ParticleResponse:
    fields: HarmonicFields
    du_dy: torch.Tensor
    dw_dy: torch.Tensor
    dp_dy: torch.Tensor
    eps_v: torch.Tensor
    sigma_x: torch.Tensor
    sigma_z: torch.Tensor
    tau_xz: torch.Tensor


@dataclass
class LossBreakdown:
    total: torch.Tensor
    momentum: torch.Tensor
    fluid: torch.Tensor
    strong: torch.Tensor
    boundary: torch.Tensor
    roughness: torch.Tensor
    top_sigma_z: torch.Tensor
    top_tau_xz: torch.Tensor
    bottom_flux: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "momentum",
                "fluid",
                "strong",
                "boundary",
                "roughness",
                "top_sigma_z",
                "top_tau_xz",
                "bottom_flux",
            )
        }


class MPMKANBiotProblem:
    """Dimensionless harmonic Biot problem with MPM weak residual assembly."""

    def __init__(
        self,
        wave: WaveParameters,
        soil: SoilParameters,
        case: SeabedCase,
        grid: MaterialPointGrid1D,
        momentum_weight: float = 1.0,
        boundary_weight: float = 25.0,
        fluid_weight: float = 500.0,
        strong_weight: float = 0.05,
        roughness_weight: float = 1.0e-7,
    ):
        self.wave = wave
        self.soil = soil
        self.case = case
        self.grid = grid
        self.momentum_weight = momentum_weight
        self.boundary_weight = boundary_weight
        self.fluid_weight = fluid_weight
        self.strong_weight = strong_weight
        self.roughness_weight = roughness_weight

        self.kh = wave.wave_number * soil.thickness
        hydraulic = case.permeability / wave.gamma_w
        storage = soil.porosity * case.pore_water_compressibility(wave, soil)
        self.mass_pressure = wave.omega * storage * soil.thickness**2 / hydraulic
        self.mass_strain = (
            wave.omega * soil.thickness**2 / (hydraulic * soil.shear_modulus)
        )
        self.mass_scale = max(1.0, abs(self.mass_pressure) + abs(self.mass_strain))

    def particle_response(
        self, model: HarmonicKAN, y: torch.Tensor
    ) -> ParticleResponse:
        fields = model(y)
        du_dy = complex_gradient(fields.u, y)
        dw_dy = complex_gradient(fields.w, y)
        dp_dy = complex_gradient(fields.p, y)

        i = torch.as_tensor(1j, dtype=fields.u.dtype, device=fields.u.device)
        eps_x = i * self.kh * fields.u
        eps_z = dw_dy
        gamma_xz = du_dy + i * self.kh * fields.w
        eps_v = eps_x + eps_z

        lam = self.soil.lambda_over_g
        sigma_x = (2.0 + lam) * eps_x + lam * eps_z
        sigma_z = lam * eps_x + (2.0 + lam) * eps_z
        tau_xz = gamma_xz
        return ParticleResponse(
            fields=fields,
            du_dy=du_dy,
            dw_dy=dw_dy,
            dp_dy=dp_dy,
            eps_v=eps_v,
            sigma_x=sigma_x,
            sigma_z=sigma_z,
            tau_xz=tau_xz,
        )

    def loss(self, model: HarmonicKAN) -> LossBreakdown:
        y = self.grid.y.detach().clone().requires_grad_(True)
        response = self.particle_response(model, y)
        i = torch.as_tensor(1j, dtype=response.fields.u.dtype, device=y.device)

        # Weak equilibrium:
        #   int N i*kh*(sigma_x-p) dy - int N_y*tau dy = 0
        #   int N i*kh*tau dy - int N_y*sigma_z dy - int N*p_y dy = 0
        #
        # Only effective stress is integrated by parts in the vertical
        # equation. Keeping the pressure gradient in particle form preserves
        # the mudline condition sigma'_z=0, p=P0. Integrating (sigma'_z-p)
        # together would incorrectly replace it by a zero total-traction
        # boundary.
        r_u = self.grid.p2g(
            i * self.kh * (response.sigma_x - response.fields.p)
        ) - self.grid.p2g(response.tau_xz, use_gradient=True)
        r_w = (
            self.grid.p2g(i * self.kh * response.tau_xz)
            - self.grid.p2g(response.sigma_z, use_gradient=True)
            - self.grid.p2g(response.dp_dy)
        )

        # Bottom displacement is essential, so the bottom test node is removed.
        nodal_scale = self.grid.nodal_volume.clamp_min(1.0e-12)
        r_u = r_u[1:] / nodal_scale[1:]
        r_w = r_w[1:] / nodal_scale[1:]
        momentum = torch.mean(squared_magnitude(r_u)) + torch.mean(
            squared_magnitude(r_w)
        )

        # Weak fluid conservation:
        # -int N_y p_y - kh^2 int N p + i*a int N p + i*b int N eps_v = 0
        r_p = (
            -self.grid.p2g(response.dp_dy, use_gradient=True)
            - self.kh**2 * self.grid.p2g(response.fields.p)
            + i * self.mass_pressure * self.grid.p2g(response.fields.p)
            + i * self.mass_strain * self.grid.p2g(response.eps_v)
        )
        # Mudline pressure is essential, so the top pressure test node is removed.
        r_p = r_p[:-1] / nodal_scale[:-1] / self.mass_scale
        fluid = torch.mean(squared_magnitude(r_p))

        # Particle strong-residual stabilization suppresses weak-form
        # hourglass/high-frequency modes. The MPM nodal weak residual remains
        # the primary loss; this term is intentionally down-weighted.
        dtau_dy = complex_gradient(response.tau_xz, y)
        dsigma_z_dy = complex_gradient(response.sigma_z, y)
        d2p_dy2 = complex_gradient(response.dp_dy, y)
        strong_u = i * self.kh * (
            response.sigma_x - response.fields.p
        ) + dtau_dy
        strong_w = (
            i * self.kh * response.tau_xz
            + dsigma_z_dy
            - response.dp_dy
        )
        strong_p = (
            d2p_dy2
            - self.kh**2 * response.fields.p
            + i * self.mass_pressure * response.fields.p
            + i * self.mass_strain * response.eps_v
        ) / self.mass_scale
        strong = (
            self.momentum_weight
            * (
                torch.mean(squared_magnitude(strong_u))
                + torch.mean(squared_magnitude(strong_w))
            )
            + self.fluid_weight * torch.mean(squared_magnitude(strong_p))
        )

        boundary_y = torch.tensor(
            [[0.0], [1.0]], dtype=y.dtype, device=y.device, requires_grad=True
        )
        boundary_response = self.particle_response(model, boundary_y)
        bottom_flux = boundary_response.dp_dy[0]
        top_sigma_z = boundary_response.sigma_z[1]
        top_tau_xz = boundary_response.tau_xz[1]
        boundary = (
            squared_magnitude(bottom_flux)
            + squared_magnitude(top_sigma_z)
            + squared_magnitude(top_tau_xz)
        )

        roughness = model.spline_roughness()
        total = (
            self.momentum_weight * momentum
            + self.fluid_weight * fluid
            + self.strong_weight * strong
            + self.boundary_weight * boundary
            + self.roughness_weight * roughness
        )
        return LossBreakdown(
            total=total,
            momentum=momentum,
            fluid=fluid,
            strong=strong,
            boundary=boundary,
            roughness=roughness,
            top_sigma_z=squared_magnitude(top_sigma_z),
            top_tau_xz=squared_magnitude(top_tau_xz),
            bottom_flux=squared_magnitude(bottom_flux),
        )

    def evaluate(
        self, model: HarmonicKAN, n_points: int = 401
    ) -> dict[str, torch.Tensor]:
        parameter = next(model.parameters())
        y = torch.linspace(
            0.0,
            1.0,
            n_points,
            dtype=parameter.dtype,
            device=parameter.device,
        )[:, None].requires_grad_(True)
        response = self.particle_response(model, y)
        return {
            "z_over_h": (y[:, 0] - 1.0).detach(),
            "pore_pressure": torch.abs(response.fields.p).detach(),
            "sigma_x_eff": torch.abs(response.sigma_x).detach(),
            "sigma_z_eff": torch.abs(response.sigma_z).detach(),
            "tau_xz": torch.abs(response.tau_xz).detach(),
        }
