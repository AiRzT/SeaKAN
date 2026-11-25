from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_CSV = REPO_ROOT / "data" / "benchmarks" / "figure9.csv"

from mpm_kan_figure9.adaptive_mpm import AdaptiveMaterialPointCloud, _p2g_connectivity
from mpm_kan_figure9.config import Geometry, SoilParameters, WaveParameters, dimensionless_coefficients, figure9_setup
from mpm_kan_figure9.kan import CubicBSplineBasis, KAN, KANLayer
from mpm_kan_figure9.model import HarmonicFields
from mpm_kan_figure9.mpm import MaterialPointGrid2D
from mpm_kan_figure9.physics import MPMKANPipeProblem
from mpm_kan_figure9.strong_mpm import AdaptiveStrongMPMProblem, AugmentedBoundaryState, StrongLossBreakdown
from figure9_comparison.common import (
    PhysicsConstants,
    abs2,
    boundary_loss,
    parameter_count,
    physics_constants,
    pipe_profile,
    sample_interior,
    strong_residuals,
)
from figure9_comparison.evaluation import error_metrics, load_benchmark, periodic_interpolate

# Paths configured relative to REPO_ROOT above

_POLY_CACHE: dict[tuple[int, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def install_fast_cubic_bspline_patch(basis_cls: type) -> None:
    """Replace iterative Cox-de Boor torch.where recursion with exact Horner cubic polynomials."""
    orig_forward = basis_cls.forward

    def _build_coeffs_f64(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_spans = self.n_basis - 3
        s_nodes = torch.tensor([0.1, 0.35, 0.65, 0.9], dtype=torch.float64, device="cpu")
        vand = torch.stack([torch.ones_like(s_nodes), s_nodes, s_nodes.square(), s_nodes.pow(3)], dim=1)
        span_ids = torch.arange(n_spans, dtype=torch.float64, device="cpu").unsqueeze(1)
        u_nodes = (span_ids + s_nodes.unsqueeze(0)) / float(n_spans)
        x_nodes = 2.0 * u_nodes - 1.0
        with torch.no_grad():
            cpu_self = copy.deepcopy(self).to(device="cpu", dtype=torch.float64)
            y_vals = orig_forward(cpu_self, x_nodes)
            coeffs = torch.linalg.solve(vand.unsqueeze(0).expand(n_spans, -1, -1), y_vals)
        return coeffs[:, 0, :], coeffs[:, 1, :], coeffs[:, 2, :], coeffs[:, 3, :]

    def fast_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.degree != 3:
            return orig_forward(self, x)
        key = (self.n_basis, x.dtype, x.device)
        cached = _POLY_CACHE.get(key)
        if cached is None:
            c0_64, c1_64, c2_64, c3_64 = _build_coeffs_f64(self)
            cached = (
                c0_64.to(dtype=x.dtype, device=x.device).contiguous(),
                c1_64.to(dtype=x.dtype, device=x.device).contiguous(),
                c2_64.to(dtype=x.dtype, device=x.device).contiguous(),
                c3_64.to(dtype=x.dtype, device=x.device).contiguous(),
            )
            _POLY_CACHE[key] = cached
        c0, c1, c2, c3 = cached
        n_spans = self.n_basis - 3
        u = ((x + 1.0) * 0.5).clamp(0.0, 1.0 - 1.0e-7)
        u_scaled = u * float(n_spans)
        span = u_scaled.to(torch.long).clamp(0, n_spans - 1)
        s = (u_scaled - span.to(x.dtype)).unsqueeze(-1)
        return ((c3[span] * s + c2[span]) * s + c1[span]) * s + c0[span]

    test_b = basis_cls(n_basis=9, degree=3)
    tx = torch.linspace(-0.999, 0.999, 513, dtype=torch.float64, requires_grad=True)
    w_rand = torch.sin(torch.arange(1, 10, dtype=torch.float64))
    y_ref = (orig_forward(test_b, tx) * w_rand).sum(dim=-1)
    g1_ref = torch.autograd.grad(y_ref.sum(), tx, create_graph=True)[0]
    g2_ref = torch.autograd.grad(g1_ref.sum(), tx)[0]

    tx2 = tx.detach().clone().requires_grad_(True)
    y_fast = (fast_forward(test_b, tx2) * w_rand).sum(dim=-1)
    g1_fast = torch.autograd.grad(y_fast.sum(), tx2, create_graph=True)[0]
    g2_fast = torch.autograd.grad(g1_fast.sum(), tx2)[0]
    err_max = max(
        float((y_ref - y_fast).detach().abs().max()),
        float((g1_ref - g1_fast).detach().abs().max()),
        float((g2_ref - g2_fast).detach().abs().max()),
    )
    assert err_max < 1.0e-12, f"Horner B-spline verification failed: {err_max}"
    basis_cls.forward = fast_forward


def install_fast_kan_layer_patch(layer_cls: type) -> None:
    """Fuse base SiLU + spline contraction into a single cuBLAS F.linear GEMM."""
    def fast_layer_forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        feats = torch.cat((F.silu(bounded), self.basis(bounded).flatten(1)), dim=1)
        weights = torch.cat((self.base_weight, self.spline_weight.flatten(1)), dim=1)
        return F.linear(feats, weights, self.bias)

    layer_cls.forward = fast_layer_forward


install_fast_cubic_bspline_patch(CubicBSplineBasis)
install_fast_kan_layer_patch(KANLayer)


def _batched_complex_grads_3(
    u: torch.Tensor, w: torch.Tensor, p: torch.Tensor, xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute complex spatial gradients of (u, w, p) in 1 vmap backward pass instead of 6."""
    s6 = torch.stack((u.real.sum(), u.imag.sum(), w.real.sum(), w.imag.sum(), p.real.sum(), p.imag.sum()))
    eye6 = torch.eye(6, dtype=xy.dtype, device=xy.device)
    g6 = torch.autograd.grad(s6, xy, eye6, is_grads_batched=True, create_graph=True, retain_graph=True)[0]
    gu = torch.complex(g6[0], g6[1])
    gw = torch.complex(g6[2], g6[3])
    gp = torch.complex(g6[4], g6[5])
    return gu, gw, gp


def _batched_second_grads_5(
    sx: torch.Tensor, sy: torch.Tensor, tau: torch.Tensor, gpx: torch.Tensor, gpy: torch.Tensor, xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all 5 complex second-order spatial derivatives in 1 vmap backward pass instead of 10."""
    s10 = torch.stack((
        sx.real.sum(), sx.imag.sum(),
        sy.real.sum(), sy.imag.sum(),
        tau.real.sum(), tau.imag.sum(),
        gpx.real.sum(), gpx.imag.sum(),
        gpy.real.sum(), gpy.imag.sum(),
    ))
    eye10 = torch.eye(10, dtype=xy.dtype, device=xy.device)
    g10 = torch.autograd.grad(s10, xy, eye10, is_grads_batched=True, create_graph=True, retain_graph=True)[0]
    return (
        torch.complex(g10[0], g10[1]),
        torch.complex(g10[2], g10[3]),
        torch.complex(g10[4], g10[5]),
        torch.complex(g10[6], g10[7]),
        torch.complex(g10[8], g10[9]),
    )


def install_fast_problem_patches() -> None:
    """Fuse interior + boundary evaluations and replace 24 autograd passes with 1-2 batched passes."""
    from mpm_kan_figure9.physics import LossBreakdown, squared_magnitude

    orig_weak_loss = MPMKANPipeProblem.loss
    orig_strong_loss = AdaptiveStrongMPMProblem.loss

    def fast_weak_loss(self: MPMKANPipeProblem, model: nn.Module) -> LossBreakdown:
        device = self.grid.xy.device
        dtype = self.grid.xy.dtype
        n_int = self.grid.xy.shape[0]
        n_bc = self.boundary_points
        half = self.geometry.half_width_over_height
        x_b = torch.linspace(-half, half, n_bc, dtype=dtype, device=device)
        top = torch.stack((x_b, torch.ones_like(x_b)), dim=1)
        bottom = torch.stack((x_b, torch.zeros_like(x_b)), dim=1)
        y_b = torch.linspace(0.0, 1.0, max(32, n_bc // 2) + 2, dtype=dtype, device=device)[1:-1]
        side_n = y_b.numel()
        outer = torch.cat((bottom, torch.stack((-half * torch.ones_like(y_b), y_b), dim=1), torch.stack((half * torch.ones_like(y_b), y_b), dim=1)), dim=0)
        n_out = outer.shape[0]
        theta = torch.linspace(0.0, 2.0 * torch.pi, n_bc + 1, dtype=dtype, device=device)[:-1]
        nx, ny = torch.cos(theta), torch.sin(theta)
        pipe = torch.stack((self.geometry.radius_over_height * nx, self.geometry.center_y + self.geometry.radius_over_height * ny), dim=1)

        xy_cat = torch.cat((self.grid.xy.detach(), top, outer, pipe), dim=0).requires_grad_(True)
        fields_cat = model(xy_cat)
        gu_cat, gw_cat, gp_cat = _batched_complex_grads_3(fields_cat.u, fields_cat.w, fields_cat.p, xy_cat)

        lam = self.soil.lambda_over_g
        u_int, p_int = fields_cat.u[:n_int], fields_cat.p[:n_int]
        gu, gw, gp = gu_cat[:n_int], gw_cat[:n_int], gp_cat[:n_int]
        ex, ey = gu[:, 0], gw[:, 1]
        ev = ex + ey
        sx = (2.0 + lam) * ex + lam * ey
        sy = lam * ex + (2.0 + lam) * ey
        txy = gu[:, 1] + gw[:, 0]

        imag_unit = torch.as_tensor(1j, dtype=p_int.dtype, device=device)
        r_u = -self.grid.p2g(sx - p_int, derivative="x") - self.grid.p2g(txy, derivative="y")
        r_w = -self.grid.p2g(txy, derivative="x") - self.grid.p2g(sy - p_int, derivative="y")
        top_p = model(self.grid.top_xy).p
        r_w = r_w + self.grid.top_p2g(-top_p)

        vol_scale = self.grid.nodal_volume.clamp_min(1.0e-9)
        r_u = r_u[self.grid.momentum_nodes] / vol_scale[self.grid.momentum_nodes]
        r_w = r_w[self.grid.momentum_nodes] / vol_scale[self.grid.momentum_nodes]
        momentum = squared_magnitude(r_u).mean() + squared_magnitude(r_w).mean()

        r_p = (
            -self.grid.p2g(gp[:, 0], derivative="x")
            -self.grid.p2g(gp[:, 1], derivative="y")
            + imag_unit * self.mass_pressure * self.grid.p2g(p_int)
            + imag_unit * self.mass_strain * self.grid.p2g(ev)
        )
        r_p = r_p[self.grid.fluid_nodes] / vol_scale[self.grid.fluid_nodes] / self.mass_scale
        fluid = squared_magnitude(r_p).mean()

        s_top = n_int
        s_out = s_top + n_bc
        s_pipe = s_out + n_out
        gu_top, gw_top = gu_cat[s_top:s_out], gw_cat[s_top:s_out]
        sy_top = lam * gu_top[:, 0] + (2.0 + lam) * gw_top[:, 1]
        txy_top = gu_top[:, 1] + gw_top[:, 0]
        top_traction = squared_magnitude(sy_top).mean() + squared_magnitude(txy_top).mean()

        gp_out = gp_cat[s_out:s_pipe]
        outer_flux = (
            squared_magnitude(gp_out[:n_bc, 1]).mean()
            + squared_magnitude(gp_out[n_bc : n_bc + side_n, 0]).mean()
            + squared_magnitude(gp_out[n_bc + side_n :, 0]).mean()
        )
        gp_pipe = gp_cat[s_pipe:]
        pipe_flux = squared_magnitude(gp_pipe[:, 0] * nx + gp_pipe[:, 1] * ny).mean()

        if self.strong_weight > 0.0:
            xy_s = self.grid.xy[:: self.strong_stride].detach().clone().requires_grad_(True)
            fs = model(xy_s)
            gus, gws, gps = _batched_complex_grads_3(fs.u, fs.w, fs.p, xy_s)
            exs, eys = gus[:, 0], gws[:, 1]
            evs = exs + eys
            sxs = (2.0 + lam) * exs + lam * eys
            sys_ = lam * exs + (2.0 + lam) * eys
            txys = gus[:, 1] + gws[:, 0]
            gsx, gsy, gtau, gpx, gpy = _batched_second_grads_5(sxs, sys_, txys, gps[:, 0], gps[:, 1], xy_s)
            rus = gsx[:, 0] + gtau[:, 1] - gps[:, 0]
            rws = gtau[:, 0] + gsy[:, 1] - gps[:, 1]
            rps = (gpx[:, 0] + gpy[:, 1] + imag_unit * self.mass_pressure * fs.p + imag_unit * self.mass_strain * evs) / self.mass_scale
            strong = squared_magnitude(rus).mean() + squared_magnitude(rws).mean() + self.fluid_weight * squared_magnitude(rps).mean()
        else:
            strong = torch.zeros((), dtype=dtype, device=device)

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
            total=total, momentum=momentum, fluid=fluid, strong=strong,
            top_traction=top_traction, outer_flux=outer_flux, pipe_flux=pipe_flux, roughness=roughness,
        )

    def fast_strong_loss(
        self: AdaptiveStrongMPMProblem,
        model: nn.Module,
        indices: torch.Tensor,
        augmented: AugmentedBoundaryState | None = None,
    ) -> StrongLossBreakdown:
        xy = self.cloud.xy[indices].detach().clone().requires_grad_(True)
        dtype, device = xy.dtype, xy.device
        fields = model(xy)
        gu, gw, gp = _batched_complex_grads_3(fields.u, fields.w, fields.p, xy)
        ex, ey = gu[:, 0], gw[:, 1]
        ev = ex + ey
        lam = self.soil.lambda_over_g
        sx = (2.0 + lam) * ex + lam * ey
        sy = lam * ex + (2.0 + lam) * ey
        tau = gu[:, 1] + gw[:, 0]

        gsx, gsy, gtau, gpx, gpy = _batched_second_grads_5(sx, sy, tau, gp[:, 0], gp[:, 1], xy)
        imag_unit = torch.as_tensor(1j, dtype=fields.p.dtype, device=device)
        ru = gsx[:, 0] + gtau[:, 1] - gp[:, 0]
        rw = gtau[:, 0] + gsy[:, 1] - gp[:, 1]
        rp = (gpx[:, 0] + gpy[:, 1] + imag_unit * self.mass_pressure * fields.p + imag_unit * self.mass_strain * ev) / self.mass_scale

        particle_momentum = self.cloud.weighted_level_mean(squared_magnitude(ru) + squared_magnitude(rw), indices, self.level_weights)
        particle_fluid = self.cloud.weighted_level_mean(squared_magnitude(rp), indices, self.level_weights)

        if self.p2g_weight > 0.0:
            nodal_u, _ = self.cloud.p2g_residual(ru, indices)
            nodal_w, _ = self.cloud.p2g_residual(rw, indices)
            nodal_p, _ = self.cloud.p2g_residual(rp, indices)
            p2g_momentum = squared_magnitude(nodal_u).mean() + squared_magnitude(nodal_w).mean()
            p2g_fluid = squared_magnitude(nodal_p).mean()
        else:
            p2g_momentum = torch.zeros((), dtype=dtype, device=device)
            p2g_fluid = torch.zeros((), dtype=dtype, device=device)

        n = self.boundary_points
        half = self.geometry.half_width_over_height
        x_b = torch.linspace(-half, half, n, dtype=dtype, device=device)
        top = torch.stack((x_b, torch.ones_like(x_b)), dim=1)
        bottom = torch.stack((x_b, torch.zeros_like(x_b)), dim=1)
        y_b = torch.linspace(0.0, 1.0, max(40, n // 2) + 2, dtype=dtype, device=device)[1:-1]
        side_n = y_b.numel()
        outer = torch.cat((bottom, torch.stack((-half * torch.ones_like(y_b), y_b), dim=1), torch.stack((half * torch.ones_like(y_b), y_b), dim=1)), dim=0)
        n_out = outer.shape[0]
        theta = torch.linspace(0.0, 2.0 * torch.pi, n + 1, dtype=dtype, device=device)[:-1]
        nx, ny = torch.cos(theta), torch.sin(theta)
        pipe = torch.stack((self.geometry.radius_over_height * nx, self.geometry.center_y + self.geometry.radius_over_height * ny), dim=1)

        bc_cat = torch.cat((top, outer, pipe), dim=0).requires_grad_(True)
        fbc = model(bc_cat)
        gubc, gwbc, gpbc = _batched_complex_grads_3(fbc.u, fbc.w, fbc.p, bc_cat)
        sy_top = lam * gubc[:n, 0] + (2.0 + lam) * gwbc[:n, 1]
        tau_top = gubc[:n, 1] + gwbc[:n, 0]
        top_traction = squared_magnitude(sy_top).mean() + squared_magnitude(tau_top).mean()

        gp_out = gpbc[n : n + n_out]
        outer_flux = (
            squared_magnitude(gp_out[:n, 1]).mean()
            + squared_magnitude(gp_out[n : n + side_n, 0]).mean()
            + squared_magnitude(gp_out[n + side_n :, 0]).mean()
        )
        gp_pipe = gpbc[n + n_out :]
        pipe_flux = squared_magnitude(gp_pipe[:, 0] * nx + gp_pipe[:, 1] * ny).mean()

        eps = torch.finfo(dtype).eps
        top_c = torch.sqrt(top_traction + eps)
        out_c = torch.sqrt(outer_flux + eps)
        pipe_c = torch.sqrt(pipe_flux + eps)
        if augmented is None:
            b_aug = self.boundary_weight * (top_traction + outer_flux) + self.pipe_flux_weight * pipe_flux
        else:
            b_aug = (
                augmented.lambda_top * top_c + 0.5 * augmented.rho_top * top_c.square()
                + augmented.lambda_outer * out_c + 0.5 * augmented.rho_outer * out_c.square()
                + augmented.lambda_pipe * pipe_c + 0.5 * augmented.rho_pipe * pipe_c.square()
            )
        roughness = model.roughness()
        total = (
            self.momentum_weight * particle_momentum
            + self.fluid_weight * particle_fluid
            + self.p2g_weight * (self.momentum_weight * p2g_momentum + self.fluid_weight * p2g_fluid)
            + b_aug
            + self.roughness_weight * roughness
        )
        return StrongLossBreakdown(
            total=total, particle_momentum=particle_momentum, particle_fluid=particle_fluid,
            p2g_momentum=p2g_momentum, p2g_fluid=p2g_fluid, top_traction=top_traction,
            outer_flux=outer_flux, pipe_flux=pipe_flux, top_constraint=top_c,
            outer_constraint=out_c, pipe_constraint=pipe_c, boundary_augmented=b_aug, roughness=roughness,
        )

    # Numerical equivalence check in float64
    w_t, s_t, g_t = figure9_setup()
    c_t = dimensionless_coefficients(w_t, s_t, g_t)
    from mpm_kan_figure9.model import PipeHarmonicKAN
    m_t = PipeHarmonicKAN(g_t, c_t["kappa"], width=8, hidden_layers=1, n_basis=6, dtype=torch.float64)
    gr_t = MaterialPointGrid2D.create(16, 12, g_t, torch.device("cpu"), torch.float64)
    pr_t = MPMKANPipeProblem(w_t, s_t, g_t, gr_t, strong_weight=0.5, boundary_points=24)
    ref_w = orig_weak_loss(pr_t, m_t).total.detach()
    fast_w = fast_weak_loss(pr_t, m_t).total.detach()
    assert float((ref_w - fast_w).abs()) < 1.0e-11, f"Weak loss patch mismatch: {float((ref_w - fast_w).abs())}"

    cl_t = AdaptiveMaterialPointCloud.create(g_t, torch.device("cpu"), torch.float64, base_x=24, base_y=12, middle_radial=2, middle_theta=16, fine_radial=2, fine_theta=16, gap_normal=0, gap_theta=0, p2g_x=24, p2g_y=12)
    ps_t = AdaptiveStrongMPMProblem(w_t, s_t, g_t, cl_t, p2g_weight=0.3, boundary_points=24)
    idx_t = torch.arange(cl_t.n_particles, dtype=torch.long)
    ref_s = orig_strong_loss(ps_t, m_t, idx_t).total.detach()
    fast_s = fast_strong_loss(ps_t, m_t, idx_t).total.detach()
    assert float((ref_s - fast_s).abs()) < 1.0e-11, f"Strong loss patch mismatch: {float((ref_s - fast_s).abs())}"

    MPMKANPipeProblem.loss = fast_weak_loss
    AdaptiveStrongMPMProblem.loss = fast_strong_loss


install_fast_problem_patches()


class MatchedMLP(nn.Module):
    """Tanh MLP matched to KAN parameter count."""

    def __init__(self, widths: list[int], *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1], dtype=dtype) for i in range(len(widths) - 1)]
        )
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=0.8)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def roughness(self) -> torch.Tensor:
        return self.layers[0].weight.sum() * 0.0


class PipeHarmonicBackbone(nn.Module):
    """Global harmonic field network supporting KAN or parameter-matched MLP."""

    def __init__(
        self,
        geometry: Geometry,
        kappa: float,
        *,
        backbone: str = "kan",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.geometry = geometry
        self.kappa = kappa
        self.backbone_type = backbone
        if backbone == "kan":
            # 4,830 trainable parameters
            self.net: nn.Module = KAN([2, 18, 18, 6], n_basis=9, dtype=dtype)
        elif backbone == "mlp":
            # 2*65+65 + 65*65+65 + 65*6+6 = 4,881 trainable parameters
            self.net = MatchedMLP([2, 65, 65, 6], dtype=dtype)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, xy: torch.Tensor) -> HarmonicFields:
        x = xy[:, 0]
        y = xy[:, 1]
        half_width = self.geometry.half_width_over_height
        normalized = torch.stack((x / half_width, 2.0 * y - 1.0), dim=1)
        raw = self.net(normalized)

        radial_level = (
            x.square()
            + (y - self.geometry.center_y).square()
            - self.geometry.radius_over_height**2
        )
        pipe_factor = 1.0 - torch.exp(
            -torch.clamp(radial_level, min=0.0) / (self.geometry.radius_over_height**2)
        )
        outer_factor = y * (1.0 - (x / self.geometry.half_width_over_height).square())
        displacement_factor = outer_factor * pipe_factor
        top_wave = torch.exp(torch.complex(torch.zeros_like(x), self.kappa * x))
        u = displacement_factor * torch.complex(raw[:, 0], raw[:, 1])
        w = displacement_factor * torch.complex(raw[:, 2], raw[:, 3])
        correction = torch.complex(raw[:, 4], raw[:, 5])
        p = top_wave + (1.0 - y) * correction
        return HarmonicFields(u=u, w=w, p=p)

    def roughness(self) -> torch.Tensor:
        return self.net.roughness()


class TopGapLocalizedBackbone(nn.Module):
    """Pipe-local Hermite normal-zero-derivative residual branch (KAN or matched MLP)."""

    def __init__(
        self,
        base: PipeHarmonicBackbone,
        geometry: Geometry,
        *,
        backbone: str = "kan",
        dtype: torch.dtype = torch.float32,
        freeze_base: bool = True,
        x_support_radii: float = 0.60,
        y_support_center_gap: float = 1.40,
    ):
        super().__init__()
        self.base = base
        self.geometry = geometry
        self.x_support_radii = x_support_radii
        self.y_support_center_gap = y_support_center_gap
        if backbone == "kan":
            # 773 parameters -> total 5,603
            self.local_net: nn.Module = KAN([2, 10, 6], n_basis=9, dtype=dtype)
            out = self.local_net.layers[-1]
            nn.init.zeros_(out.base_weight)
            nn.init.zeros_(out.spline_weight)
            nn.init.zeros_(out.bias)
        elif backbone == "mlp":
            # 2*85+85 + 85*6+6 = 771 parameters -> total 5,652
            self.local_net = MatchedMLP([2, 85, 6], dtype=dtype)
            out = self.local_net.layers[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    def forward(self, xy: torch.Tensor) -> HarmonicFields:
        base = self.base(xy)
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        local_coords = self.local_coordinates(xy)
        raw = self.local_net(local_coords)
        envelope = self.envelope(xy)
        radial_level = x.square() + (y - self.geometry.center_y).square() - radius**2
        pipe_factor = 1.0 - torch.exp(-torch.clamp(radial_level, min=0.0) / radius**2)
        outer_factor = y * (1.0 - (x / self.geometry.half_width_over_height).square())
        displacement_factor = envelope * outer_factor * pipe_factor
        pressure_factor = self.pressure_factor(xy)
        u = base.u + displacement_factor * torch.complex(raw[:, 0], raw[:, 1])
        w = base.w + displacement_factor * torch.complex(raw[:, 2], raw[:, 3])
        p = base.p + pressure_factor * torch.complex(raw[:, 4], raw[:, 5])
        return HarmonicFields(u=u, w=w, p=p)

    def roughness(self) -> torch.Tensor:
        return self.local_net.roughness()

    def local_coordinates(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        vertical_gap = 1.0 - self.geometry.center_y
        dx = x
        dy = y - self.geometry.center_y
        radial = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0e-12)
        cosine = dx / radial
        sine = dy / radial
        outer_radius = vertical_gap / sine.clamp_min(0.15)
        gap = (outer_radius - radius).clamp_min(0.02)
        normal = ((radial - radius) / gap).clamp(0.0, 1.0)
        return torch.stack((cosine / self.x_support_radii, 2.0 * normal.square() - 1.0), dim=1)

    def envelope(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        vertical_gap = 1.0 - self.geometry.center_y
        return torch.exp(
            -(x / (self.x_support_radii * radius)).pow(4)
            - ((1.0 - y) / (self.y_support_center_gap * vertical_gap)).pow(4)
        )

    def pressure_factor(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        radius = self.geometry.radius_over_height
        vertical_gap = 1.0 - self.geometry.center_y
        dx = x
        dy = y - self.geometry.center_y
        radial = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0e-12)
        cosine = dx / radial
        sine = dy / radial
        outer_radius = vertical_gap / sine.clamp_min(0.15)
        gap = (outer_radius - radius).clamp_min(0.02)
        normal = ((radial - radius) / gap).clamp(0.0, 1.0)
        hermite = 1.0 - 3.0 * normal.square() + 2.0 * normal.pow(3)
        angular = torch.exp(-(cosine / self.x_support_radii).pow(4)) * (
            (0.5 * (1.0 + sine)).clamp(0.0, 1.0).pow(6)
        )
        return angular * hermite


def create_uniform_cloud(
    geometry: Geometry,
    device: torch.device,
    dtype: torch.dtype,
    *,
    nx: int = 108,
    ny: int = 40,
    p2g_x: int = 128,
    p2g_y: int = 48,
) -> AdaptiveMaterialPointCloud:
    """Create a uniform Cartesian particle cloud matched in count (~4,295 vs 4,302)."""
    half = geometry.half_width_over_height
    radius = geometry.radius_over_height
    center_y = geometry.center_y
    dx = 2.0 * half / nx
    dy = 1.0 / ny
    ix = torch.arange(nx, device=device, dtype=dtype)
    iy = torch.arange(ny, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(-half + (ix + 0.5) * dx, (iy + 0.5) * dy, indexing="xy")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    outside = xy[:, 0].square() + (xy[:, 1] - center_y).square() > (1.003 * radius) ** 2
    xy = xy[outside]
    volume = torch.full((xy.shape[0],), dx * dy, dtype=dtype, device=device)
    level = torch.zeros((xy.shape[0],), dtype=torch.long, device=device)
    node_ids, shape, node_xy = _p2g_connectivity(xy, half, p2g_x, p2g_y)
    return AdaptiveMaterialPointCloud(
        xy=xy,
        volume=volume,
        level=level,
        node_ids=node_ids,
        shape=shape,
        n_nodes=(p2g_x + 1) * (p2g_y + 1),
        node_xy=node_xy,
        counts=(xy.shape[0],),
        level_names=("uniform",),
    )


def run_standard_kinn(
    constants: PhysicsConstants,
    benchmark: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    seed: int = 202409,
    steps: int = 1200,
    batch_size: int = 1536,
    boundary_pts: int = 96,
    lbfgs_stage1: int = 400,
    lbfgs_stage2: int = 120,
) -> dict[str, object]:
    """Train Standard KINN (Wang et al., 2025) under the exact pointwise PINN protocol."""
    print("\n=== Running Standard KINN (KINN-PINN) under exact pointwise protocol ===", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # [2, 25, 25, 25, 6] with n_basis=8 -> 13,131 parameters (vs 13,062 for DeepXDE/gPINN)
    model = KAN([2, 25, 25, 25, 6], n_basis=8, dtype=torch.float32).to(device)
    n_params = parameter_count(model)
    print(f"Standard KINN trainable parameters: {n_params}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5.0e-5)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    started = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        xy = sample_interior(batch_size, constants, device, torch.float32, generator=generator)
        xy = xy.detach().requires_grad_(True)
        residuals = strong_residuals(model, xy, constants)
        physics = abs2(residuals[0]).mean() + abs2(residuals[1]).mean() + 20.0 * abs2(residuals[2]).mean()
        bc, _ = boundary_loss(model, constants, boundary_pts)
        total = physics + bc + 1.0e-6 * model.roughness()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 200 == 0 or step == steps:
            print(
                f"Standard KINN Adam {step:4d}/{steps}: loss={float(total.detach()):.3e}, pde={float(physics.detach()):.3e}, bc={float(bc.detach()):.3e}",
                flush=True,
            )

    adam_elapsed = time.perf_counter() - started
    torch.manual_seed(seed + 991)
    xy_lbfgs_base = sample_interior(3072, constants, device, torch.float64).detach()

    lbfgs_started = time.perf_counter()
    total_evals = 0
    for stage_dtype, max_it in ((torch.float32, lbfgs_stage1), (torch.float64, lbfgs_stage2)):
        if max_it <= 0:
            continue
        model = model.to(dtype=stage_dtype)
        xy_stage = xy_lbfgs_base.to(dtype=stage_dtype)
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=max_it,
            max_eval=int(max_it * 1.25),
            tolerance_grad=1.0e-8,
            tolerance_change=1.0e-10,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal total_evals
            opt_lbfgs.zero_grad(set_to_none=True)
            xy_req = xy_stage.detach().requires_grad_(True)
            res = strong_residuals(model, xy_req, constants)
            phys = abs2(res[0]).mean() + abs2(res[1]).mean() + 20.0 * abs2(res[2]).mean()
            bc_val, _ = boundary_loss(model, constants, boundary_pts)
            tot = phys + bc_val + 1.0e-6 * model.roughness()
            tot.backward()
            total_evals += 1
            if total_evals == 1 or total_evals % 50 == 0:
                print(
                    f"Standard KINN LBFGS ({stage_dtype}) eval {total_evals:4d}: loss={float(tot.detach()):.3e}, pde={float(phys.detach()):.3e}, bc={float(bc_val.detach()):.3e}",
                    flush=True,
                )
            return tot

        opt_lbfgs.step(closure)

    lbfgs_elapsed = time.perf_counter() - lbfgs_started
    total_elapsed = adam_elapsed + lbfgs_elapsed
    theta, pressure = pipe_profile(model, constants, n_theta=721)

    kinn_dir = OUT_DIR / "standard_kinn"
    kinn_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        kinn_dir / "profile.csv",
        np.column_stack((theta, pressure)),
        delimiter=",",
        header="theta_rad,normalized_pressure",
        comments="",
    )

    metrics_out: dict[str, object] = {}
    for ref_name, (ref_theta, ref_val) in benchmark.items():
        pred = periodic_interpolate(theta, pressure, ref_theta)
        metrics_out[ref_name] = error_metrics(pred, ref_val)

    summary = {
        "method": "standard_kinn",
        "method_label": "Standard KINN",
        "parameters": n_params,
        "seed": seed,
        "adam_steps": steps,
        "lbfgs_steps_total": lbfgs_stage1 + lbfgs_stage2,
        "lbfgs_evaluations": total_evals,
        "elapsed_seconds": total_elapsed,
        "metrics": metrics_out,
    }
    (kinn_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Standard KINN metrics:", json.dumps(metrics_out, indent=2), flush=True)
    return summary


def evaluate_seakan_model(
    model: nn.Module,
    geometry: Geometry,
    eval_problem: AdaptiveStrongMPMProblem,
    eval_indices: torch.Tensor,
    benchmark: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate physics residual score and circumferential benchmark metrics."""
    loss = eval_problem.loss(model, eval_indices, None)
    d = loss.detached()
    # Invariant strong-form physics score (independent of P2G or AL multipliers)
    physics_score = (
        10.0 * d["particle_momentum"]
        + 1000.0 * d["particle_fluid"]
        + 100.0 * d["top_traction"]
        + 100.0 * d["outer_flux"]
        + 300.0 * d["pipe_flux"]
    )

    theta = np.linspace(0.0, 2.0 * math.pi, 721)
    t_tensor = torch.as_tensor(theta, dtype=torch.float64, device=device)
    radius = geometry.radius_over_height
    xy = torch.stack(
        (radius * torch.cos(t_tensor), geometry.center_y + radius * torch.sin(t_tensor)),
        dim=1,
    )
    with torch.no_grad():
        pred_p = model(xy).p.abs().detach().cpu().numpy()

    metrics_by_ref: dict[str, object] = {}
    for ref_name, (ref_theta, ref_val) in benchmark.items():
        interp_pred = periodic_interpolate(theta, pred_p, ref_theta)
        metrics_by_ref[ref_name] = error_metrics(interp_pred, ref_val)

    exp_theta, exp_val = benchmark["experimental"]
    crown_idx = int(np.argmin(np.abs(exp_theta - 0.5 * math.pi)))
    crown_pred = float(periodic_interpolate(theta, pred_p, np.asarray([exp_theta[crown_idx]]))[0])
    crown_abs_err = abs(crown_pred - float(exp_val[crown_idx]))

    return (
        {
            "physics_score": float(physics_score),
            "particle_momentum": d["particle_momentum"],
            "particle_fluid": d["particle_fluid"],
            "top_traction": d["top_traction"],
            "outer_flux": d["outer_flux"],
            "pipe_flux": d["pipe_flux"],
            "crown_abs_error": crown_abs_err,
            "experimental": metrics_by_ref["experimental"],
            "zhu_numerical": metrics_by_ref["zhu_numerical"],
        },
        theta,
        pred_p,
    )


def _run_al_stage(
    model: nn.Module,
    problem32: AdaptiveStrongMPMProblem,
    problem64: AdaptiveStrongMPMProblem,
    cloud32: AdaptiveMaterialPointCloud,
    cloud64: AdaptiveMaterialPointCloud,
    batch_per_level: tuple[int, ...],
    *,
    fp32_steps: int,
    fp32_lr: float,
    fp64_adam_steps: int,
    fp64_lr: float,
    fp64_lbfgs_steps: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    generator = torch.Generator(device=device).manual_seed(seed)
    augmented = AugmentedBoundaryState(rho_top=160.0, rho_outer=160.0, rho_pipe=320.0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt32 = torch.optim.Adam(trainable, lr=fp32_lr)
    prev_c: dict[str, float] | None = None

    for step in range(1, fp32_steps + 1):
        indices = cloud32.sample_indices(batch_per_level, generator)
        opt32.zero_grad(set_to_none=True)
        loss = problem32.loss(model, indices, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=12.0)
        opt32.step()
        if step % 40 == 0:
            cur_c = {
                "top": float(loss.top_constraint.detach().cpu()),
                "outer": float(loss.outer_constraint.detach().cpu()),
                "pipe": float(loss.pipe_constraint.detach().cpu()),
            }
            augmented.update(cur_c, prev_c, progress_ratio=0.90, penalty_growth=1.5, maximum_penalty=5000.0)
            prev_c = cur_c

    model = model.to(dtype=torch.float64)
    full_idx = torch.arange(cloud64.n_particles, device=device, dtype=torch.long)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt64 = torch.optim.Adam(trainable, lr=fp64_lr)
    for step in range(1, fp64_adam_steps + 1):
        opt64.zero_grad(set_to_none=True)
        loss = problem64.loss(model, full_idx, augmented)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        opt64.step()
        if step % 10 == 0:
            cur_c = {
                "top": float(loss.top_constraint.detach().cpu()),
                "outer": float(loss.outer_constraint.detach().cpu()),
                "pipe": float(loss.pipe_constraint.detach().cpu()),
            }
            augmented.update(cur_c, prev_c, progress_ratio=0.90, penalty_growth=1.5, maximum_penalty=5000.0)
            prev_c = cur_c

    if fp64_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            trainable,
            lr=0.35,
            max_iter=fp64_lbfgs_steps,
            max_eval=fp64_lbfgs_steps * 2,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            val = problem64.loss(model, full_idx, augmented)
            val.total.backward()
            return val.total

        lbfgs.step(closure)
    return model


def run_ablation_variant(
    variant: str,
    seed: int,
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    benchmark: dict[str, tuple[np.ndarray, np.ndarray]],
    eval_cloud64: AdaptiveMaterialPointCloud,
    device: torch.device,
) -> dict[str, object]:
    """Train one converged buried-pipe ablation variant for one random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()

    coeffs = dimensionless_coefficients(wave, soil, geometry)
    backbone = "mlp" if variant == "mlp_backbone" else "kan"
    use_weak = variant != "no_weak_precond"
    use_refinement = variant != "no_crown_refinement"
    use_local_branch = variant != "no_local_branch"
    p2g_w_stage2 = 0.30 if variant == "with_strong_p2g" else 0.0
    p2g_w_stage3 = 0.25 if variant == "with_strong_p2g" else 0.0
    rough_w_stage1 = 0.01 if backbone == "kan" else 0.0
    rough_w_later = 0.001 if backbone == "kan" else 0.0

    base_model = PipeHarmonicBackbone(geometry, coeffs["kappa"], backbone=backbone, dtype=torch.float32).to(device)

    # Build clouds
    if use_refinement:
        cloud3_32 = AdaptiveMaterialPointCloud.create(
            geometry, device, torch.float32, base_x=48, base_y=18,
            middle_radial=8, middle_theta=128, fine_radial=7, fine_theta=192,
            gap_normal=0, gap_theta=0, p2g_x=96, p2g_y=36,
        )
        cloud4_32 = AdaptiveMaterialPointCloud.create(
            geometry, device, torch.float32, base_x=48, base_y=18,
            middle_radial=8, middle_theta=128, fine_radial=7, fine_theta=192,
            gap_normal=10, gap_theta=128, p2g_x=128, p2g_y=48,
        )
        cloud4_64 = AdaptiveMaterialPointCloud.create(
            geometry, device, torch.float64, base_x=48, base_y=18,
            middle_radial=8, middle_theta=128, fine_radial=7, fine_theta=192,
            gap_normal=10, gap_theta=128, p2g_x=128, p2g_y=48,
        )
        lvl_w3 = (0.20, 0.32, 0.48)
        batch3 = (384, 640, 896)
        lvl_w4 = (0.12, 0.18, 0.25, 0.45)
        batch4 = (320, 480, 704, 1024)
    else:
        cloud3_32 = create_uniform_cloud(geometry, device, torch.float32, nx=108, ny=40)
        cloud4_32 = cloud3_32
        cloud4_64 = create_uniform_cloud(geometry, device, torch.float64, nx=108, ny=40)
        lvl_w3 = (1.0,)
        batch3 = (1920,)
        lvl_w4 = (1.0,)
        batch4 = (2528,)

    # Stage 1: Homotopy 1st-order conservative MPM weak-form pre-conditioning + hybrid transition
    if use_weak:
        s1_cache = OUT_DIR / "stage1_cache_v5" / f"stage1_{backbone}_seed_{seed}.pt"
        s1_cache.parent.mkdir(parents=True, exist_ok=True)
        if s1_cache.is_file():
            base_model.load_state_dict(torch.load(s1_cache, map_location=device, weights_only=True))
        else:
            grid_coarse = MaterialPointGrid2D.create(48, 20, geometry, device=device, dtype=torch.float32)
            grid_fine = MaterialPointGrid2D.create(64, 24, geometry, device=device, dtype=torch.float32)
            homotopy = [
                (grid_coarse, 1.0, 20.0, 0.0, 1.0e-6 if backbone == "kan" else 0.0, 1.5e-3, 800),
                (grid_coarse, 1.0, 20.0, 0.02, 1.0e-6 if backbone == "kan" else 0.0, 6.0e-4, 200),
                (grid_coarse, 10.0, 500.0, 0.0, 1.0e-3 if backbone == "kan" else 0.0, 8.0e-4, 600),
                (grid_coarse, 10.0, 2000.0, 0.0, 1.0e-2 if backbone == "kan" else 0.0, 5.0e-4, 500),
                (grid_fine, 10.0, 2000.0, 0.0, 1.0e-2 if backbone == "kan" else 0.0, 3.0e-4, 400),
                (grid_fine, 10.0, 2000.0, 0.5, 1.0e-2 if backbone == "kan" else 0.0, 2.0e-4, 300),
            ]
            for gr, mw, fw, sw, rw, lr_s, n_st in homotopy:
                prob_s = MPMKANPipeProblem(
                    wave, soil, geometry, gr,
                    momentum_weight=mw, fluid_weight=fw,
                    boundary_weight=80.0, pipe_flux_weight=120.0,
                    strong_weight=sw, roughness_weight=rw,
                    strong_stride=8, boundary_points=128,
                )
                opt_s = torch.optim.Adam(base_model.parameters(), lr=lr_s)
                for _ in range(n_st):
                    opt_s.zero_grad(set_to_none=True)
                    ls = prob_s.loss(base_model)
                    ls.total.backward()
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=20.0)
                    opt_s.step()

            # Float64 L-BFGS polish + 220-step float32 fine transition matching released figure9.pt
            m64 = base_model.to(dtype=torch.float64)
            gf64 = MaterialPointGrid2D.create(64, 24, geometry, device=device, dtype=torch.float64)
            pr64 = MPMKANPipeProblem(
                wave, soil, geometry, gf64,
                momentum_weight=10.0, fluid_weight=2000.0,
                boundary_weight=80.0, pipe_flux_weight=120.0,
                strong_weight=0.5, roughness_weight=1.0e-2 if backbone == "kan" else 0.0,
                strong_stride=8, boundary_points=128,
            )
            lb64 = torch.optim.LBFGS(
                m64.parameters(), lr=0.5, max_iter=30, max_eval=60,
                tolerance_grad=1.0e-10, tolerance_change=1.0e-12, history_size=50, line_search_fn="strong_wolfe",
            )

            def _clos64() -> torch.Tensor:
                lb64.zero_grad(set_to_none=True)
                v = pr64.loss(m64)
                v.total.backward()
                return v.total

            lb64.step(_clos64)
            base_model = m64.to(dtype=torch.float32)

            prob_fin = MPMKANPipeProblem(
                wave, soil, geometry, grid_fine,
                momentum_weight=20.0, fluid_weight=2000.0,
                boundary_weight=80.0, pipe_flux_weight=120.0,
                strong_weight=1.0, roughness_weight=1.0e-2 if backbone == "kan" else 0.0,
                strong_stride=8, boundary_points=128,
            )
            opt_fin = torch.optim.Adam(base_model.parameters(), lr=1.0e-4)
            for _ in range(220):
                opt_fin.zero_grad(set_to_none=True)
                lf = prob_fin.loss(base_model)
                lf.total.backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=20.0)
                opt_fin.step()

            torch.save(base_model.state_dict(), s1_cache)
    else:
        # Exact step- and homotopy-matched strong-form warmup without 1st-order MPM weak integration
        gen1 = torch.Generator(device=device).manual_seed(seed + 11)
        for mw, fw, lr_warm, n_warm in (
            (1.0, 20.0, 1.5e-3, 1000),
            (10.0, 500.0, 8.0e-4, 600),
            (10.0, 1600.0, 4.0e-4, 900),
            (20.0, 2000.0, 1.5e-4, 520),
        ):
            warm_prob = AdaptiveStrongMPMProblem(
                wave, soil, geometry, cloud3_32,
                level_weights=lvl_w3, momentum_weight=mw, fluid_weight=fw,
                p2g_weight=p2g_w_stage2, boundary_weight=100.0, pipe_flux_weight=800.0,
                roughness_weight=rough_w_later, boundary_points=160,
            )
            opt1 = torch.optim.Adam(base_model.parameters(), lr=lr_warm)
            for _ in range(n_warm):
                idx = cloud3_32.sample_indices(batch3, gen1)
                opt1.zero_grad(set_to_none=True)
                loss1 = warm_prob.loss(base_model, idx)
                loss1.total.backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=15.0)
                opt1.step()

    # Stage 2: Exact 3-phase adaptive strong-form refinement matching adaptive_training_history.json
    can_cache_s2 = use_weak and use_refinement and (p2g_w_stage2 == 0.0)
    s2_cache = OUT_DIR / "stage2_cache_v5" / f"stage2_{backbone}_seed_{seed}.pt"
    if can_cache_s2 and s2_cache.is_file():
        base_model.load_state_dict(torch.load(s2_cache, map_location=device, weights_only=True))
    else:
        gen2 = torch.Generator(device=device).manual_seed(seed + 22)
        for fw2, bw2, pf2, lr2, n2 in (
            (800.0, 80.0, 160.0, 1.0e-4, 500),
            (1200.0, 90.0, 400.0, 5.0e-5, 400),
            (1600.0, 100.0, 800.0, 2.0e-5, 250),
        ):
            prob2 = AdaptiveStrongMPMProblem(
                wave, soil, geometry, cloud3_32,
                level_weights=lvl_w3, momentum_weight=10.0, fluid_weight=fw2,
                p2g_weight=p2g_w_stage2, boundary_weight=bw2, pipe_flux_weight=pf2,
                roughness_weight=rough_w_later, boundary_points=160,
            )
            opt2 = torch.optim.Adam(base_model.parameters(), lr=lr2)
            for _ in range(n2):
                idx = cloud3_32.sample_indices(batch3, gen2)
                opt2.zero_grad(set_to_none=True)
                loss2 = prob2.loss(base_model, idx)
                loss2.total.backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=15.0)
                opt2.step()
        if can_cache_s2:
            s2_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(base_model.state_dict(), s2_cache)

    prob4_32 = AdaptiveStrongMPMProblem(
        wave, soil, geometry, cloud4_32,
        level_weights=lvl_w4, momentum_weight=10.0, fluid_weight=1000.0,
        p2g_weight=p2g_w_stage3, boundary_weight=0.0, pipe_flux_weight=0.0,
        roughness_weight=rough_w_later, boundary_points=192,
    )
    prob4_64 = AdaptiveStrongMPMProblem(
        wave, soil, geometry, cloud4_64,
        level_weights=lvl_w4, momentum_weight=10.0, fluid_weight=1000.0,
        p2g_weight=p2g_w_stage3, boundary_weight=0.0, pipe_flux_weight=0.0,
        roughness_weight=rough_w_later, boundary_points=192,
    )

    if not use_local_branch:
        # Balanced 4-level weights so global-only network refines crown without sacrificing lower pipe arc
        lvl_g = (0.22, 0.26, 0.30, 0.22) if use_refinement else (1.0,)
        probg_32 = AdaptiveStrongMPMProblem(
            wave, soil, geometry, cloud4_32,
            level_weights=lvl_g, momentum_weight=10.0, fluid_weight=1000.0,
            p2g_weight=p2g_w_stage3, boundary_weight=0.0, pipe_flux_weight=0.0,
            roughness_weight=rough_w_later, boundary_points=192,
        )
        probg_64 = AdaptiveStrongMPMProblem(
            wave, soil, geometry, cloud4_64,
            level_weights=lvl_g, momentum_weight=10.0, fluid_weight=1000.0,
            p2g_weight=p2g_w_stage3, boundary_weight=0.0, pipe_flux_weight=0.0,
            roughness_weight=rough_w_later, boundary_points=192,
        )
        final_model: nn.Module = _run_al_stage(
            base_model, probg_32, probg_64, cloud4_32, cloud4_64, batch4,
            fp32_steps=80, fp32_lr=1.2e-5,
            fp64_adam_steps=12, fp64_lr=3.0e-6,
            fp64_lbfgs_steps=6, seed=seed + 33, device=device,
        )
        total_params = sum(p.numel() for p in final_model.parameters())
    else:
        # Stage 3: Global physics teacher on 4-level cloud
        teacher_model = copy.deepcopy(base_model)
        teacher_model = _run_al_stage(
            teacher_model, prob4_32, prob4_64, cloud4_32, cloud4_64, batch4,
            fp32_steps=280, fp32_lr=2.5e-5,
            fp64_adam_steps=30, fp64_lr=3.0e-6,
            fp64_lbfgs_steps=12, seed=seed + 33, device=device,
        )
        teacher_32 = teacher_model.to(dtype=torch.float32).eval()

        # Stage 4: Local Hermite normal-zero-derivative branch initialization + AL float64 finish
        student = TopGapLocalizedBackbone(
            base_model.to(dtype=torch.float32),
            geometry,
            backbone=backbone,
            dtype=torch.float32,
            freeze_base=True,
        ).to(device)
        total_params = sum(p.numel() for p in student.parameters())

        # Physics-teacher distillation in the local pipe-top zone
        if use_refinement:
            gap_xy = cloud4_32.xy[cloud4_32.level == len(cloud4_32.counts) - 1]
        else:
            # Top-gap geometric subset of the uniform cloud
            mask_top = (cloud4_32.xy[:, 0].abs() <= 0.8 * geometry.radius_over_height) & (
                cloud4_32.xy[:, 1] >= geometry.center_y
            )
            gap_xy = cloud4_32.xy[mask_top]
        theta_arc = torch.linspace(0.25 * math.pi, 0.75 * math.pi, 257, dtype=torch.float32, device=device)[:-1]
        pipe_xy = torch.stack(
            (
                geometry.radius_over_height * torch.cos(theta_arc),
                geometry.center_y + geometry.radius_over_height * torch.sin(theta_arc),
            ),
            dim=1,
        )
        dist_xy = torch.cat((gap_xy, pipe_xy), dim=0)
        with torch.no_grad():
            b_f = student.base(dist_xy)
            t_f = teacher_32(dist_xy)
            env = student.envelope(dist_xy)
            tgt_u = b_f.u + env * (t_f.u - b_f.u)
            tgt_w = b_f.w + env * (t_f.w - b_f.w)
            tgt_p = b_f.p + env * (t_f.p - b_f.p)

        opt_dist = torch.optim.Adam(student.local_net.parameters(), lr=1.0e-3)
        for _ in range(800):
            opt_dist.zero_grad(set_to_none=True)
            pred = student(dist_xy)
            d_loss = (
                (pred.p - tgt_p).abs().square().mean()
                + 0.10 * (pred.u - tgt_u).abs().square().mean()
                + 0.10 * (pred.w - tgt_w).abs().square().mean()
            )
            d_loss.backward()
            opt_dist.step()
        del teacher_32, teacher_model

        final_model = _run_al_stage(
            student, prob4_32, prob4_64, cloud4_32, cloud4_64, batch4,
            fp32_steps=60, fp32_lr=2.5e-5,
            fp64_adam_steps=10, fp64_lr=3.0e-6,
            fp64_lbfgs_steps=4, seed=seed + 44, device=device,
        )

    elapsed = time.perf_counter() - started
    eval_prob = AdaptiveStrongMPMProblem(
        wave, soil, geometry, eval_cloud64,
        level_weights=(0.12, 0.18, 0.25, 0.45),
        momentum_weight=10.0, fluid_weight=1000.0,
        p2g_weight=0.0, boundary_weight=0.0, pipe_flux_weight=0.0,
        roughness_weight=0.0, boundary_points=192,
    )
    eval_idx = torch.arange(eval_cloud64.n_particles, device=device, dtype=torch.long)
    eval_res, theta, pred_p = evaluate_seakan_model(
        final_model, geometry, eval_prob, eval_idx, benchmark, device
    )

    var_dir = OUT_DIR / "ablation_v5" / variant / f"seed_{seed}"
    var_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        var_dir / "profile.csv",
        np.column_stack((theta, pred_p)),
        delimiter=",",
        header="theta_rad,normalized_pressure",
        comments="",
    )
    record = {
        "variant": variant,
        "seed": seed,
        "parameters": total_params,
        "elapsed_seconds": elapsed,
        **eval_res,
    }
    (var_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"[{variant} | seed={seed}] time={elapsed:.1f}s | phys={eval_res['physics_score']:.2f} | "
        f"Exp L2={100*eval_res['experimental']['relative_l2']:.2f}% (R2={eval_res['experimental']['r2']:.4f}) | "
        f"Zhu L2={100*eval_res['zhu_numerical']['relative_l2']:.2f}% (R2={eval_res['zhu_numerical']['r2']:.4f})",
        flush=True,
    )
    return record


def _ablation_worker(variant: str, seed: int) -> dict[str, object]:
    cached_json = OUT_DIR / "ablation_v5" / variant / f"seed_{seed}" / "metrics.json"
    if cached_json.is_file():
        rec = json.loads(cached_json.read_text(encoding="utf-8"))
        print(f"[CACHE HIT] {variant} | seed={seed} -> Exp L2={100*rec['experimental']['relative_l2']:.2f}%", flush=True)
        return rec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wave, soil, geometry = figure9_setup()
    benchmark = load_benchmark(BENCHMARK_CSV)
    eval_cloud64 = AdaptiveMaterialPointCloud.create(
        geometry, device, torch.float64, base_x=48, base_y=18,
        middle_radial=8, middle_theta=128, fine_radial=7, fine_theta=192,
        gap_normal=10, gap_theta=128, p2g_x=128, p2g_y=48,
    )
    return run_ablation_variant(variant, seed, wave, soil, geometry, benchmark, eval_cloud64, device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    wave, soil, geometry = figure9_setup()
    constants = physics_constants(wave, soil, geometry)
    benchmark = load_benchmark(BENCHMARK_CSV)

    # 1. Standard KINN baseline (Table 4 / Fig. 7), with checkpoint resume
    kinn_cache = OUT_DIR / "standard_kinn" / "summary.json"
    if kinn_cache.is_file() and (OUT_DIR / "standard_kinn" / "profile.csv").is_file():
        kinn_summary = json.loads(kinn_cache.read_text(encoding="utf-8"))
        print("Loaded cached Standard KINN summary.", flush=True)
    else:
        kinn_summary = run_standard_kinn(constants, benchmark, device, seed=202409)

    # 2. Converged 3-seed component ablation on buried-pipe benchmark (replaces Table 8)
    variants = [
        "full_seakan",
        "with_strong_p2g",
        "no_local_branch",
        "no_crown_refinement",
        "no_weak_precond",
        "mlp_backbone",
    ]
    seeds = [202409, 202410, 202411]
    all_records: list[dict[str, object]] = []

    # Phase A: run full_seakan + mlp_backbone first to populate shared Stage-1 weak-form caches race-free
    phase_a = [(v, s) for v in ("full_seakan", "mlp_backbone") for s in seeds]
    phase_b = [(v, s) for v in ("with_strong_p2g", "no_local_branch", "no_crown_refinement", "no_weak_precond") for s in seeds]
    with ProcessPoolExecutor(max_workers=3) as pool:
        print(f"\n--- Phase A: {len(phase_a)} primary runs (3 parallel GPU workers) ---", flush=True)
        for fut in as_completed({pool.submit(_ablation_worker, v, s): (v, s) for v, s in phase_a}):
            all_records.append(fut.result())
        print(f"\n--- Phase B: {len(phase_b)} ablated runs reusing Stage-1 cache (3 parallel GPU workers) ---", flush=True)
        for fut in as_completed({pool.submit(_ablation_worker, v, s): (v, s) for v, s in phase_b}):
            all_records.append(fut.result())

    # Aggregate 3-seed summary
    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        sub = [r for r in all_records if r["variant"] == variant]
        phys = np.array([r["physics_score"] for r in sub], dtype=float)
        exp_l2 = np.array([100.0 * r["experimental"]["relative_l2"] for r in sub], dtype=float)
        exp_r2 = np.array([r["experimental"]["r2"] for r in sub], dtype=float)
        zhu_l2 = np.array([100.0 * r["zhu_numerical"]["relative_l2"] for r in sub], dtype=float)
        zhu_r2 = np.array([r["zhu_numerical"]["r2"] for r in sub], dtype=float)
        crown_err = np.array([r["crown_abs_error"] for r in sub], dtype=float)
        times = np.array([r["elapsed_seconds"] for r in sub], dtype=float)
        summary_rows.append(
            {
                "variant": variant,
                "parameters": int(sub[0]["parameters"]),
                "physics_score_mean": float(phys.mean()),
                "physics_score_std": float(phys.std(ddof=1)),
                "exp_l2_pct_mean": float(exp_l2.mean()),
                "exp_l2_pct_std": float(exp_l2.std(ddof=1)),
                "exp_r2_mean": float(exp_r2.mean()),
                "exp_r2_std": float(exp_r2.std(ddof=1)),
                "zhu_l2_pct_mean": float(zhu_l2.mean()),
                "zhu_l2_pct_std": float(zhu_l2.std(ddof=1)),
                "zhu_r2_mean": float(zhu_r2.mean()),
                "zhu_r2_std": float(zhu_r2.std(ddof=1)),
                "crown_err_mean": float(crown_err.mean()),
                "crown_err_std": float(crown_err.std(ddof=1)),
                "elapsed_s_mean": float(times.mean()),
            }
        )

    out_bundle = {
        "standard_kinn": kinn_summary,
        "ablation_per_seed": all_records,
        "ablation_summary": summary_rows,
    }
    (OUT_DIR / "exp1_complete_results.json").write_text(
        json.dumps(out_bundle, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    print("\n=== EXP-1 COMPLETE SUMMARY ===", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
