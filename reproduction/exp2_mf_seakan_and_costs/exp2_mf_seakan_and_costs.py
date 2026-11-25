from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))

from mpm_kan_generalization.config import ParameterDomain, PhysicalConstants, solve_wave_number
from mpm_kan_generalization.kan import CubicBSplineBasis, KAN, KANLayer

_POLY_CACHE_EXP2: dict[tuple[int, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def install_fast_cubic_bspline_patch(basis_cls: type) -> None:
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
        cached = _POLY_CACHE_EXP2.get(key)
        if cached is None:
            c0_64, c1_64, c2_64, c3_64 = _build_coeffs_f64(self)
            cached = (
                c0_64.to(dtype=x.dtype, device=x.device).contiguous(),
                c1_64.to(dtype=x.dtype, device=x.device).contiguous(),
                c2_64.to(dtype=x.dtype, device=x.device).contiguous(),
                c3_64.to(dtype=x.dtype, device=x.device).contiguous(),
            )
            _POLY_CACHE_EXP2[key] = cached
        c0, c1, c2, c3 = cached
        n_spans = self.n_basis - 3
        u = ((x + 1.0) * 0.5).clamp(0.0, 1.0 - 1.0e-7)
        u_scaled = u * float(n_spans)
        span = u_scaled.to(torch.long).clamp(0, n_spans - 1)
        s = (u_scaled - span.to(x.dtype)).unsqueeze(-1)
        return ((c3[span] * s + c2[span]) * s + c1[span]) * s + c0[span]

    basis_cls.forward = fast_forward


def install_fast_kan_layer_patch(layer_cls: type) -> None:
    def fast_layer_forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        feats = torch.cat((F.silu(bounded), self.basis(bounded).flatten(1)), dim=1)
        weights = torch.cat((self.base_weight, self.spline_weight.flatten(1)), dim=1)
        return F.linear(feats, weights, self.bias)

    layer_cls.forward = fast_layer_forward


install_fast_cubic_bspline_patch(CubicBSplineBasis)
install_fast_kan_layer_patch(KANLayer)

EXP2_DIR = Path(__file__).resolve().parent
SUPP_ROOT = EXP2_DIR.parent
COMSOL_RUN = Path(os.environ.get("SEAKAN_COMSOL_RUN", SUPP_ROOT / "comsol_clean_suite"))
CLEAN_COMSOL_DIR = COMSOL_RUN / "outputs"
CLEAN_TIMING_CSV = COMSOL_RUN / "comsol_timings_and_mesh.csv"

MANIFEST_CSV = SUPP_ROOT / "comsol_clean_suite" / "calibration_cases.csv"
OUT_DIR = Path(os.environ.get("SEAKAN_OUTPUT_DIR", EXP2_DIR / "runs"))


def batched_complex_grads_3(
    u: torch.Tensor, w: torch.Tensor, p: torch.Tensor, xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s6 = torch.stack((u.real.sum(), u.imag.sum(), w.real.sum(), w.imag.sum(), p.real.sum(), p.imag.sum()))
    eye6 = torch.eye(6, dtype=xy.dtype, device=xy.device)
    g6 = torch.autograd.grad(s6, xy, eye6, is_grads_batched=True, create_graph=True, retain_graph=True)[0]
    return torch.complex(g6[0], g6[1]), torch.complex(g6[2], g6[3]), torch.complex(g6[4], g6[5])


def complex_grad_1(val: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    s2 = torch.stack((val.real.sum(), val.imag.sum()))
    eye2 = torch.eye(2, dtype=xy.dtype, device=xy.device)
    g2 = torch.autograd.grad(s2, xy, eye2, is_grads_batched=True, create_graph=False)[0]
    return torch.complex(g2[0], g2[1])


def sq_abs(val: torch.Tensor) -> torch.Tensor:
    return val.real.square() + val.imag.square()


class CleanParametricSeaKAN(nn.Module):
    """
    Deterministic parametric physics-informed KAN over (x*, y*, T, log10(k), b/D).
    Embeds:
      - Quintic C2 pipe-normal coordinate flattening (exact dp/dn = 0 at pipe wall)
      - Boundary-layer stretched spatial coordinates & exponential trial factors
      - Exact surface & remote-lateral wave pressure p* = exp(i * kappa * x*)
      - Exact rigid no-slip displacements u* = w* = 0 at bottom, remote sides, and pipe
      - Global + moving pipe-local KAN branches
    """

    def __init__(
        self,
        constants: PhysicalConstants,
        domain: ParameterDomain,
        *,
        width: int = 28,
        hidden_layers: int = 2,
        local_width: int = 18,
        n_basis: int = 8,
        local_support: float = 0.28,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.constants = constants
        self.domain = domain
        self.local_support = local_support
        # Strictly 5 physical inputs: (x_stretched, y_stretched, T_norm, log10k_norm, burial_norm)
        self.global_kan = KAN([5] + [width] * hidden_layers + [6], n_basis=n_basis, dtype=dtype)
        self.local_kan = KAN([5, local_width, local_width, 6], n_basis=n_basis, dtype=dtype)
        out = self.local_kan.layers[-1]
        nn.init.zeros_(out.base_weight)
        nn.init.zeros_(out.spline_weight)
        nn.init.zeros_(out.bias)

    def pipe_neumann_map(
        self, xy: torch.Tensor, pipe_center_y: torch.Tensor, support: float = 0.085
    ) -> torch.Tensor:
        dx = xy[:, 0]
        dy = xy[:, 1] - pipe_center_y
        radial = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0e-12)
        radius = self.constants.radius
        dist = (radial - radius).clamp_min(0.0)
        s = (dist / support).clamp(0.0, 1.0)
        mapped_s = 6.0 * s.pow(3) - 8.0 * s.pow(4) + 3.0 * s.pow(5)
        mapped_dist = torch.where(dist < support, support * mapped_s, dist)
        scale = (radius + mapped_dist) / radial
        mapped = torch.stack((scale * dx, pipe_center_y + scale * dy), dim=1)
        top_dist = (1.0 - xy[:, 1]).clamp_min(0.0)
        side_dist = (self.constants.half_width - xy[:, 0].abs()).clamp_min(0.0)
        outer_dist2 = top_dist.square() * side_dist.square()
        blend = outer_dist2 / (outer_dist2 + dist.pow(4) + torch.finfo(xy.dtype).eps)
        return xy + blend[:, None] * (mapped - xy)

    def encode_features(self, feat_xy: torch.Tensor, params_tkb: torch.Tensor) -> torch.Tensor:
        x_raw = (feat_xy[:, 0] / self.constants.half_width).clamp(-1.0, 1.0)
        # Smooth boundary-layer knot concentration near mudline (y=1) and side boundaries (|x|=half_width)
        x_norm = torch.sign(x_raw) * (1.0 - (1.0 - x_raw.abs() + 1.0e-6).pow(0.70))
        y_depth = (1.0 - feat_xy[:, 1]).clamp(0.0, 1.0)
        y_norm = 1.0 - 2.0 * (y_depth + 1.0e-6).pow(0.65)

        t_low, t_high = self.domain.wave_period_s
        k_low, k_high = math.log10(self.domain.permeability_m_per_s[0]), math.log10(
            self.domain.permeability_m_per_s[1]
        )
        b_low, b_high = self.domain.burial_center_depth_over_diameter
        t_norm = 2.0 * (params_tkb[:, 0] - t_low) / (t_high - t_low) - 1.0
        k_norm = 2.0 * (torch.log10(params_tkb[:, 1]) - k_low) / (k_high - k_low) - 1.0
        b_norm = 2.0 * (params_tkb[:, 2] - b_low) / (b_high - b_low) - 1.0
        return torch.stack((x_norm, y_norm, t_norm, k_norm, b_norm), dim=1)

    def forward(
        self, xy: torch.Tensor, params_tkb: torch.Tensor, kappa: torch.Tensor, pipe_center_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_xy = self.pipe_neumann_map(xy, pipe_center_y)
        feats = self.encode_features(feat_xy, params_tkb)
        raw = self.global_kan(feats)
        radius = self.constants.radius
        dist = torch.sqrt(feat_xy[:, 0].square() + (feat_xy[:, 1] - pipe_center_y).square())
        local_w = (1.0 - (dist - radius).clamp_min(0.0) / self.local_support).clamp(0.0, 1.0).pow(3)
        raw = raw + local_w[:, None] * self.local_kan(feats)

        half_w = self.constants.half_width
        pipe_lvl = (xy[:, 0].square() + (xy[:, 1] - pipe_center_y).square() - radius**2).clamp_min(0.0)
        pipe_fac = 1.0 - torch.exp(-pipe_lvl / radius**2)
        side_fac_xy = 1.0 - torch.exp(-(half_w - xy[:, 0].abs()).clamp_min(0.0) / 0.10)
        bot_fac_xy = 1.0 - torch.exp(-xy[:, 1].clamp_min(0.0) / 0.10)
        disp_fac = bot_fac_xy * side_fac_xy * pipe_fac
        u = disp_fac * torch.complex(raw[:, 0], raw[:, 1])
        w = disp_fac * torch.complex(raw[:, 2], raw[:, 3])

        inc_wave = torch.exp(torch.complex(torch.zeros_like(kappa), kappa * feat_xy[:, 0]))
        depth_fac = 1.0 - torch.exp(-(1.0 - feat_xy[:, 1]).clamp_min(0.0) / 0.08)
        side_fac_p = 1.0 - torch.exp(-(half_w - feat_xy[:, 0].abs()).clamp_min(0.0) / 0.08)
        p = inc_wave + depth_fac * side_fac_p * torch.complex(raw[:, 4], raw[:, 5])
        return u, w, p

    def roughness(self) -> torch.Tensor:
        return self.global_kan.roughness() + self.local_kan.roughness()


class BatchedParametricMPMWeakGrid:
    """Vectorized 1st-order conservative MPM weak-form assembly across moving pipe geometries."""

    def __init__(self, nx: int, ny: int, constants: PhysicalConstants, device: torch.device, dtype: torch.dtype):
        self.nx = nx
        self.ny = ny
        self.constants = constants
        self.device = device
        self.dtype = dtype
        half = constants.half_width
        self.dx = 2.0 * half / nx
        self.dy = 1.0 / ny

        ix = torch.arange(nx, device=device)
        iy = torch.arange(ny, device=device)
        cx, cy = torch.meshgrid(ix, iy, indexing="xy")
        px = -half + (cx.reshape(-1).to(dtype) + 0.5) * self.dx
        py = (cy.reshape(-1).to(dtype) + 0.5) * self.dy
        self.cell_xy = torch.stack((px, py), dim=1)  # [P, 2]
        self.n_cells = self.cell_xy.shape[0]

        npx = nx + 1
        n00 = cy.reshape(-1) * npx + cx.reshape(-1)
        self.node_ids = torch.stack((n00, n00 + 1, n00 + npx, n00 + npx + 1), dim=1)  # [P, 4]
        self.n_nodes = (nx + 1) * (ny + 1)

        self.shape = torch.full((1, self.n_cells, 4), 0.25, dtype=dtype, device=device)
        self.grad_x = torch.tensor(
            [-0.5 / self.dx, 0.5 / self.dx, -0.5 / self.dx, 0.5 / self.dx], dtype=dtype, device=device
        ).view(1, 1, 4).expand(1, self.n_cells, 4)
        self.grad_y = torch.tensor(
            [-0.5 / self.dy, -0.5 / self.dy, 0.5 / self.dy, 0.5 / self.dy], dtype=dtype, device=device
        ).view(1, 1, 4).expand(1, self.n_cells, 4)

        nx_lin = torch.linspace(-half, half, nx + 1, dtype=dtype, device=device)
        ny_lin = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype, device=device)
        nxx, nyy = torch.meshgrid(nx_lin, ny_lin, indexing="xy")
        self.node_xy = torch.stack((nxx.reshape(-1), nyy.reshape(-1)), dim=1)  # [N_nodes, 2]

        self.is_outer_disp = (self.node_xy[:, 1] <= 1.0e-8) | (self.node_xy[:, 0].abs() >= half - 1.0e-8)
        self.is_outer_fluid = (self.node_xy[:, 1] >= 1.0 - 1.0e-8) | (self.node_xy[:, 0].abs() >= half - 1.0e-8)
        self.pipe_tol = 0.51 * max(self.dx, self.dy)

        xb = torch.linspace(-half + 0.02, half - 0.02, 64, dtype=dtype, device=device)
        self.bot_xy = torch.stack((xb, torch.zeros_like(xb)), dim=1)

    def _p2g_batch(self, val: torch.Tensor, vol: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        # val: [B, P] (real or complex), vol: [B, P] (real), basis: [1, P, 4] (real)
        B = val.shape[0]
        contrib = (vol.unsqueeze(-1) * basis).to(val.dtype) * val.unsqueeze(-1)  # [B, P, 4]
        idx = self.node_ids.reshape(1, -1).expand(B, -1)  # [B, 4*P]
        out = torch.zeros((B, self.n_nodes), dtype=val.dtype, device=val.device)
        return out.scatter_add(1, idx, contrib.reshape(B, -1))

    def batch_loss(
        self,
        model: CleanParametricSeaKAN,
        tkb_batch: torch.Tensor,
        kap_batch: torch.Tensor,
        cy_batch: torch.Tensor,
        mp_batch: torch.Tensor,
        me_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = tkb_batch.shape[0]
        P = self.n_cells
        Nb = self.bot_xy.shape[0]
        radius = self.constants.radius
        lam = self.constants.lambda_over_g

        # Active mask per case [B, P]
        dx_p = self.cell_xy[:, 0].unsqueeze(0)  # [1, P]
        dy_p = self.cell_xy[:, 1].unsqueeze(0) - cy_batch.unsqueeze(1)  # [B, P]
        outside = (dx_p.square() + dy_p.square()) > (radius**2)
        vol = torch.where(outside, torch.full_like(dy_p, self.dx * self.dy), torch.zeros_like(dy_p))  # [B, P]
        nodal_vol = self._p2g_batch(torch.ones_like(vol), vol, self.shape).clamp_min(1.0e-9)  # [B, N_nodes]
        active = nodal_vol > 1.0e-8

        dist_nodes = torch.sqrt(
            self.node_xy[:, 0].unsqueeze(0).square()
            + (self.node_xy[:, 1].unsqueeze(0) - cy_batch.unsqueeze(1)).square()
        )
        near_pipe = dist_nodes <= (radius + self.pipe_tol)
        mom_mask = active & (~self.is_outer_disp.unsqueeze(0)) & (~near_pipe)
        flu_mask = active & (~self.is_outer_fluid.unsqueeze(0))

        # Concatenate [B*P] interior + [B*Nb] bottom points for a single batched autograd call
        xy_int = self.cell_xy.unsqueeze(0).expand(B, P, 2).reshape(B * P, 2)
        xy_bot = self.bot_xy.unsqueeze(0).expand(B, Nb, 2).reshape(B * Nb, 2)
        xy_all = torch.cat((xy_int, xy_bot), dim=0).detach().requires_grad_(True)

        tkb_all = torch.cat(
            (tkb_batch.unsqueeze(1).expand(B, P, 3).reshape(B * P, 3), tkb_batch.unsqueeze(1).expand(B, Nb, 3).reshape(B * Nb, 3)),
            dim=0,
        )
        kap_all = torch.cat((kap_batch.unsqueeze(1).expand(B, P).reshape(B * P), kap_batch.unsqueeze(1).expand(B, Nb).reshape(B * Nb)), dim=0)
        cy_all = torch.cat((cy_batch.unsqueeze(1).expand(B, P).reshape(B * P), cy_batch.unsqueeze(1).expand(B, Nb).reshape(B * Nb)), dim=0)

        u_all, w_all, p_all = model(xy_all, tkb_all, kap_all, cy_all)
        gu_all, gw_all, gp_all = batched_complex_grads_3(u_all, w_all, p_all, xy_all)

        n_int = B * P
        p_i = p_all[:n_int].view(B, P)
        gu = gu_all[:n_int].view(B, P, 2)
        gw = gw_all[:n_int].view(B, P, 2)
        gp = gp_all[:n_int].view(B, P, 2)

        ex, ey = gu[:, :, 0], gw[:, :, 1]
        ev = ex + ey
        sx = (2.0 + lam) * ex + lam * ey
        sy = lam * ex + (2.0 + lam) * ey
        txy = gu[:, :, 1] + gw[:, :, 0]

        ru = (-self._p2g_batch(sx - p_i, vol, self.grad_x) - self._p2g_batch(txy, vol, self.grad_y)) / nodal_vol
        rw = (-self._p2g_batch(txy, vol, self.grad_x) - self._p2g_batch(sy - p_i, vol, self.grad_y)) / nodal_vol

        # COMSOL Frequency Domain uses exp(+i * omega * t) -> -1j * (mp * p + me * ev)
        m_scale = (mp_batch.abs() + me_batch.abs()).clamp_min(1.0).unsqueeze(1)
        rp = (
            -self._p2g_batch(gp[:, :, 0], vol, self.grad_x)
            - self._p2g_batch(gp[:, :, 1], vol, self.grad_y)
            - 1j * mp_batch.unsqueeze(1) * self._p2g_batch(p_i, vol, self.shape)
            - 1j * me_batch.unsqueeze(1) * self._p2g_batch(ev, vol, self.shape)
        ) / (nodal_vol * m_scale)

        mom_loss = (sq_abs(ru)[mom_mask].mean() + sq_abs(rw)[mom_mask].mean())
        flu_loss = sq_abs(rp)[flu_mask].mean()
        gp_bot = gp_all[n_int:]
        bot_loss = sq_abs(gp_bot[:, 1]).mean()
        return mom_loss, flu_loss, bot_loss


class PressureCorrectorKAN(nn.Module):
    """
    Either Multi-Fidelity (base != None, frozen physics trunk + residual KAN)
    or Pure Data-Driven KAN (base == None, incident wave + hard-BC KAN).
    """

    def __init__(
        self,
        constants: PhysicalConstants,
        domain: ParameterDomain,
        base: CleanParametricSeaKAN | None,
        *,
        width: int = 24,
        hidden_layers: int = 2,
        n_basis: int = 7,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.constants = constants
        self.domain = domain
        self.base = base
        if self.base is not None:
            for p in self.base.parameters():
                p.requires_grad_(False)
        self.helper = CleanParametricSeaKAN(constants, domain, width=8, hidden_layers=1, local_width=8, dtype=dtype)
        for p in self.helper.parameters():
            p.requires_grad_(False)
        self.corrector = KAN([5] + [width] * hidden_layers + [2], n_basis=n_basis, dtype=dtype)
        if self.base is not None:
            out = self.corrector.layers[-1]
            nn.init.zeros_(out.base_weight)
            nn.init.zeros_(out.spline_weight)
            nn.init.zeros_(out.bias)

    def forward_pressure(
        self, xy: torch.Tensor, params_tkb: torch.Tensor, kappa: torch.Tensor, pipe_center_y: torch.Tensor
    ) -> torch.Tensor:
        mapper = self.base if self.base is not None else self.helper
        feat_xy = mapper.pipe_neumann_map(xy, pipe_center_y)
        feats = mapper.encode_features(feat_xy, params_tkb)
        raw = self.corrector(feats)
        half_w = self.constants.half_width
        depth_fac = 1.0 - torch.exp(-(1.0 - feat_xy[:, 1]).clamp_min(0.0) / 0.08)
        side_fac_p = 1.0 - torch.exp(-(half_w - feat_xy[:, 0].abs()).clamp_min(0.0) / 0.08)
        delta_p = depth_fac * side_fac_p * torch.complex(raw[:, 0], raw[:, 1])
        if self.base is not None:
            _, _, base_p = self.base(xy, params_tkb, kappa, pipe_center_y)
            return base_p + delta_p
        inc_wave = torch.exp(torch.complex(torch.zeros_like(kappa), kappa * feat_xy[:, 0]))
        return inc_wave + delta_p

    def roughness(self) -> torch.Tensor:
        return self.corrector.roughness()


def load_all_comsol_cases(constants: PhysicalConstants) -> dict[str, list[dict[str, object]]]:
    """Load the clean sealed-bottom COMSOL 6.4 suite (32 train, 10 val, 2 blind, 6 prospective, 2 heterogeneous)."""
    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "blind_sealed": [],
        "prospective": [],
        "heterogeneous_stress": [],
    }
    with MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row["case_id"]
            sp = row["split"]
            T = float(row["wave_period_s"])
            H = float(row["wave_height_m"])
            k = float(row["permeability_m_per_s"])
            bD = float(row["burial_center_depth_over_diameter"])
            wn = float(solve_wave_number(np.array([T]), constants)[0])
            csv_p = CLEAN_COMSOL_DIR / f"{cid}_complex_domain.csv"
            arr = np.loadtxt(csv_p, delimiter=",", comments="%")
            xy = arr[:, 2:4] / constants.seabed_thickness_m
            target = arr[:, 4] + 1j * arr[:, 5]
            item = {
                "case_id": cid,
                "H": H,
                "T": T,
                "k": k,
                "bD": bD,
                "kappa": wn * constants.seabed_thickness_m,
                "pipe_center_y": 1.0 - bD * constants.pipe_diameter_m / constants.seabed_thickness_m,
                "xy": torch.from_numpy(xy.astype(np.float32)),
                "target": torch.from_numpy(target.astype(np.complex64)),
            }
            if sp == "train":
                splits["train"].append(item)
            elif sp == "validation":
                splits["validation"].append(item)
            else:
                splits["blind_sealed"].append(item)

    prospective_specs = [
        ("B6", "sealed_002", 0.1100, 1.45, 5.5e-5, 1.15),
        ("B7", "sealed_003", 0.1450, 1.60, 6.5e-5, 1.35),
        ("B8", "sealed_004", 0.0950, 1.32, 4.5e-5, 1.25),
        ("B9", "sealed_005", 0.1300, 1.72, 3.8e-5, 1.40),
        ("B10", "sealed_006", 0.1550, 1.25, 1.2e-5, 0.80),
        ("B11", "sealed_007", 0.0850, 1.78, 1.6e-5, 0.88),
    ]
    for label, file_stem, H, T, k, bD in prospective_specs:
        wn = float(solve_wave_number(np.array([T]), constants)[0])
        csv_p = CLEAN_COMSOL_DIR / f"{file_stem}_complex_domain.csv"
        arr = np.loadtxt(csv_p, delimiter=",", comments="%")
        xy = arr[:, 2:4] / constants.seabed_thickness_m
        target = arr[:, 4] + 1j * arr[:, 5]
        splits["prospective"].append(
            {
                "case_id": label,
                "file_id": file_stem,
                "H": H,
                "T": T,
                "k": k,
                "bD": bD,
                "kappa": wn * constants.seabed_thickness_m,
                "pipe_center_y": 1.0 - bD * constants.pipe_diameter_m / constants.seabed_thickness_m,
                "xy": torch.from_numpy(xy.astype(np.float32)),
                "target": torch.from_numpy(target.astype(np.complex64)),
            }
        )

    for label, file_stem, H, T, k, bD in (
        ("B10_het", "sealed_006_het", 0.1550, 1.25, 1.2e-5, 0.80),
        ("B11_het", "sealed_007_het", 0.0850, 1.78, 1.6e-5, 0.88),
    ):
        wn = float(solve_wave_number(np.array([T]), constants)[0])
        csv_p = CLEAN_COMSOL_DIR / f"{file_stem}_complex_domain.csv"
        arr = np.loadtxt(csv_p, delimiter=",", comments="%")
        xy = arr[:, 2:4] / constants.seabed_thickness_m
        target = arr[:, 4] + 1j * arr[:, 5]
        splits["heterogeneous_stress"].append(
            {
                "case_id": label,
                "file_id": file_stem,
                "H": H,
                "T": T,
                "k": k,
                "bD": bD,
                "kappa": wn * constants.seabed_thickness_m,
                "pipe_center_y": 1.0 - bD * constants.pipe_diameter_m / constants.seabed_thickness_m,
                "xy": torch.from_numpy(xy.astype(np.float32)),
                "target": torch.from_numpy(target.astype(np.complex64)),
            }
        )
    return splits


def evaluate_field_list(
    predict_fn,
    cases: list[dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    rows = []
    with torch.no_grad():
        for c in cases:
            xy = c["xy"].to(device=device, dtype=torch.float32)
            target = c["target"].to(device=device, dtype=torch.complex64)
            n = xy.shape[0]
            tkb = torch.tensor([[c["T"], c["k"], c["bD"]]], device=device, dtype=torch.float32).expand(n, -1)
            kap = torch.full((n,), float(c["kappa"]), device=device, dtype=torch.float32)
            cy = torch.full((n,), float(c["pipe_center_y"]), device=device, dtype=torch.float32)
            pred = predict_fn(xy, tkb, kap, cy)

            res_c = pred - target
            res_a = pred.abs() - target.abs()
            c_l2 = float(torch.linalg.vector_norm(res_c) / torch.linalg.vector_norm(target).clamp_min(1e-12))
            a_l2 = float(torch.linalg.vector_norm(res_a) / torch.linalg.vector_norm(target.abs()).clamp_min(1e-12))
            a_mse = float(res_a.square().mean())
            a_sst = float((target.abs() - target.abs().mean()).square().sum().clamp_min(1e-12))
            a_r2 = float(1.0 - res_a.square().sum().item() / a_sst)
            c_sst = float(sq_abs(target - target.mean()).sum().clamp_min(1e-12))
            c_r2 = float(1.0 - sq_abs(res_c).sum().item() / c_sst)
            rows.append(
                {
                    "case_id": c["case_id"],
                    "H": c["H"],
                    "T": c["T"],
                    "k": c["k"],
                    "bD": c["bD"],
                    "complex_l2": c_l2,
                    "amplitude_l2": a_l2,
                    "amplitude_mse": a_mse,
                    "amplitude_rmse": math.sqrt(a_mse),
                    "amplitude_r2": a_r2,
                    "complex_r2": c_r2,
                }
            )
    return {
        "mean_complex_l2": float(np.mean([r["complex_l2"] for r in rows])),
        "mean_amplitude_l2": float(np.mean([r["amplitude_l2"] for r in rows])),
        "mean_amplitude_mse": float(np.mean([r["amplitude_mse"] for r in rows])),
        "mean_amplitude_rmse": float(np.mean([r["amplitude_rmse"] for r in rows])),
        "mean_amplitude_r2": float(np.mean([r["amplitude_r2"] for r in rows])),
        "min_amplitude_r2": float(np.min([r["amplitude_r2"] for r in rows])),
        "mean_complex_r2": float(np.mean([r["complex_r2"] for r in rows])),
        "cases": rows,
    }


def train_clean_physics_trunk(
    constants: PhysicalConstants,
    domain: ParameterDomain,
    val_cases: list[dict[str, object]],
    device: torch.device,
    *,
    cases_per_step: int = 8,
    seed: int = 202416,
) -> tuple[CleanParametricSeaKAN, float, list[dict[str, float]]]:
    """Train the pure-physics parametric SeaKAN trunk via 1st-order conservative MPM weak-form assembly (ZERO COMSOL labels!)."""
    print(f"\n=== Training Clean 1st-Order MPM Weak-Form Parametric SeaKAN Trunk (seed={seed}) ===", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = CleanParametricSeaKAN(constants, domain, dtype=torch.float32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Clean Physics Trunk parameters: {n_params}", flush=True)

    gamma_w = constants.rho_w * constants.gravity
    comp = 1.0 / constants.water_bulk_modulus_pa + (1.0 - constants.saturation) / (
        gamma_w * constants.water_depth_m
    )
    storage = constants.porosity * comp
    h_bed = constants.seabed_thickness_m

    grid_coarse = BatchedParametricMPMWeakGrid(64, 32, constants, device, torch.float32)
    grid_fine = BatchedParametricMPMWeakGrid(96, 48, constants, device, torch.float32)

    rng = np.random.default_rng(seed + 13)
    t_low, t_high = domain.wave_period_s
    lk_low, lk_high = math.log10(domain.permeability_m_per_s[0]), math.log10(domain.permeability_m_per_s[1])
    b_low, b_high = domain.burial_center_depth_over_diameter

    stages = [
        (grid_coarse, 1.0, 25.0, 2.0e-3, 450),
        (grid_coarse, 8.0, 300.0, 1.0e-3, 450),
        (grid_fine, 15.0, 1500.0, 5.0e-4, 650),
        (grid_fine, 20.0, 2500.0, 1.5e-4, 450),
    ]

    started = time.perf_counter()
    history: list[dict[str, float]] = []
    global_step = 0

    for grid, mw, fw, lr_s, n_steps in stages:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_s, weight_decay=1.0e-6)
        for _ in range(n_steps):
            global_step += 1
            model.train()
            opt.zero_grad(set_to_none=True)

            # Sample continuous physics-only collocation tuples across ParameterDomain (ZERO COMSOL inputs!)
            T_np = rng.uniform(t_low, t_high, size=cases_per_step)
            k_np = 10.0 ** rng.uniform(lk_low, lk_high, size=cases_per_step)
            bD_np = rng.uniform(b_low, b_high, size=cases_per_step)
            wn_np = solve_wave_number(T_np, constants)
            omega_np = 2.0 * np.pi / T_np
            hyd_np = k_np / gamma_w
            mp_np = omega_np * storage * (h_bed**2) / hyd_np
            me_np = omega_np * (h_bed**2) / (hyd_np * constants.shear_modulus_pa)
            cy_np = 1.0 - bD_np * constants.pipe_diameter_m / h_bed
            kap_np = wn_np * h_bed

            tkb_b = torch.from_numpy(np.column_stack((T_np, k_np, bD_np)).astype(np.float32)).to(device)
            kap_b = torch.from_numpy(kap_np.astype(np.float32)).to(device)
            cy_b = torch.from_numpy(cy_np.astype(np.float32)).to(device)
            mp_b = torch.from_numpy(mp_np.astype(np.float32)).to(device)
            me_b = torch.from_numpy(me_np.astype(np.float32)).to(device)

            mom_l, flu_l, bot_l = grid.batch_loss(model, tkb_b, kap_b, cy_b, mp_b, me_b)
            total = mw * mom_l + fw * flu_l + 40.0 * bot_l + 1.0e-5 * model.roughness()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()

            if global_step == 1 or global_step % 400 == 0 or global_step == 2000:
                row = {
                    "step": global_step,
                    "total": float(total.detach().cpu()),
                    "mom": float(mom_l.detach().cpu()),
                    "flu": float(flu_l.detach().cpu()),
                    "bot": float(bot_l.detach().cpu()),
                }
                history.append(row)
                print(
                    f"Physics Trunk {global_step:4d}/2000: mom={row['mom']:.3e}, flu={row['flu']:.3e}, bot={row['bot']:.3e}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    model.eval()
    val_eval = evaluate_field_list(
        lambda xy, tkb, kap, cy: model(xy, tkb, kap, cy)[2], val_cases, device
    )
    print(
        f"Physics-Only Trunk finished in {elapsed:.1f}s | Val Amplitude L2={100*val_eval['mean_amplitude_l2']:.2f}%, "
        f"Val Amplitude R2={val_eval['mean_amplitude_r2']:.4f} (min R2={val_eval['min_amplitude_r2']:.4f}), "
        f"Val Complex L2={100*val_eval['mean_complex_l2']:.2f}%, Val Complex R2={val_eval['mean_complex_r2']:.4f}",
        flush=True,
    )
    return model, elapsed, history


def train_surrogate_variant(
    constants: PhysicalConstants,
    domain: ParameterDomain,
    base_trunk: CleanParametricSeaKAN | None,
    train_subset: list[dict[str, object]],
    val_cases: list[dict[str, object]],
    device: torch.device,
    *,
    steps: int = 800,
    batch_per_case: int = 192,
    seed: int = 202438,
) -> tuple[PressureCorrectorKAN, float]:
    """Train either MF-SeaKAN (with physics trunk) or Pure-Data KAN (base_trunk=None)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = PressureCorrectorKAN(constants, domain, base_trunk, width=24, hidden_layers=2, n_basis=7).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2.0e-3, weight_decay=1.0e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1.0e-4)
    rng = np.random.default_rng(seed)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for c in train_subset:
            n_pts = c["xy"].shape[0]
            idx = rng.choice(n_pts, size=min(batch_per_case, n_pts), replace=False)
            xy = c["xy"][idx].to(device=device, dtype=torch.float32)
            tgt = c["target"][idx].to(device=device, dtype=torch.complex64)
            m = xy.shape[0]
            tkb = torch.tensor([[c["T"], c["k"], c["bD"]]], device=device, dtype=torch.float32).expand(m, -1)
            kap = torch.full((m,), float(c["kappa"]), device=device, dtype=torch.float32)
            cy = torch.full((n_pts if False else m,), float(c["pipe_center_y"]), device=device, dtype=torch.float32)
            pred = model.forward_pressure(xy, tkb, kap, cy)
            energy = float(tgt.abs().square().mean().clamp_min(1e-6))
            losses.append(sq_abs(pred - tgt).mean() / energy)

        loss = torch.stack(losses).mean() + 2.0e-6 * model.roughness()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        opt.step()
        sched.step()

        if step % 40 == 0 or step == steps:
            model.eval()
            val_m = evaluate_field_list(model.forward_pressure, val_cases, device)
            if val_m["mean_complex_l2"] < best_val:
                best_val = val_m["mean_complex_l2"]
                best_state = copy.deepcopy(model.state_dict())

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


def select_fisher_sensors_clean(
    model: PressureCorrectorKAN,
    T: float,
    bD: float,
    kappa: float,
    pipe_center_y: float,
    constants: PhysicalConstants,
    domain: ParameterDomain,
    device: torch.device,
    count: int = 64,
) -> np.ndarray:
    xs = np.linspace(-1.6, 1.6, 97)
    ys = np.linspace(0.04, 0.96, 57)
    clearance = constants.radius + 0.025
    cands = np.array(
        [(x, y) for y in ys for x in xs if x**2 + (y - pipe_center_y) ** 2 > clearance**2],
        dtype=np.float32,
    )
    xy = torch.from_numpy(cands).to(device=device)
    n = xy.shape[0]
    kap = torch.full((n,), kappa, device=device, dtype=torch.float32)
    cy = torch.full((n,), pipe_center_y, device=device, dtype=torch.float32)
    low, high = math.log10(domain.permeability_m_per_s[0]), math.log10(domain.permeability_m_per_s[1])

    sens_list = []
    with torch.no_grad():
        for frac in (0.15, 0.50, 0.85):
            k_plus = 10.0 ** (low + (high - low) * min(0.99, frac + 0.02))
            k_minus = 10.0 ** (low + (high - low) * max(0.01, frac - 0.02))
            tkb_p = torch.tensor([[T, k_plus, bD]], device=device, dtype=torch.float32).expand(n, -1)
            tkb_m = torch.tensor([[T, k_minus, bD]], device=device, dtype=torch.float32).expand(n, -1)
            dp = (model.forward_pressure(xy, tkb_p, kap, cy) - model.forward_pressure(xy, tkb_m, kap, cy)) / 0.04
            sens_list.append(dp.abs().square().cpu().numpy())

    score = np.mean(sens_list, axis=0)
    order = np.argsort(score)[::-1]
    chosen: list[int] = []
    for idx in order:
        if all(np.linalg.norm(cands[idx] - cands[p]) >= 0.075 for p in chosen):
            chosen.append(int(idx))
            if len(chosen) == count:
                break
    return cands[chosen]


def invert_comsol_case_clean(
    model: PressureCorrectorKAN,
    case: dict[str, object],
    constants: PhysicalConstants,
    domain: ParameterDomain,
    device: torch.device,
    *,
    noise_rel: float = 0.0,
    noise_seed: int = 0,
) -> dict[str, object]:
    started = time.perf_counter()
    T, bD = float(case["T"]), float(case["bD"])
    k_true = float(case["k"])
    kappa = float(case["kappa"])
    cy = float(case["pipe_center_y"])

    sensor_xy_np = select_fisher_sensors_clean(model, T, bD, kappa, cy, constants, domain, device, count=64)
    mesh_xy = case["xy"].numpy()
    mesh_tgt = case["target"].numpy()
    interp_re = LinearNDInterpolator(mesh_xy, mesh_tgt.real)(sensor_xy_np)
    interp_im = LinearNDInterpolator(mesh_xy, mesh_tgt.imag)(sensor_xy_np)
    clean_obs = (interp_re + 1j * interp_im).astype(np.complex64)
    if noise_rel > 0.0:
        nrng = np.random.default_rng(noise_seed)
        rms = float(np.sqrt(np.mean(np.abs(clean_obs) ** 2)))
        sigma = noise_rel * rms / math.sqrt(2.0)
        clean_obs = (
            clean_obs
            + (nrng.normal(0.0, sigma, size=clean_obs.shape) + 1j * nrng.normal(0.0, sigma, size=clean_obs.shape)).astype(np.complex64)
        )
    obs = torch.from_numpy(clean_obs).to(device)
    sensor_xy = torch.from_numpy(sensor_xy_np).to(device=device, dtype=torch.float32)

    low, high = math.log10(domain.permeability_m_per_s[0]), math.log10(domain.permeability_m_per_s[1])
    n_s = sensor_xy.shape[0]
    kap_t = torch.full((n_s,), kappa, device=device, dtype=torch.float32)
    cy_t = torch.full((n_s,), cy, device=device, dtype=torch.float32)

    best_loss = float("inf")
    best_k = None
    for frac in np.linspace(0.08, 0.92, 8):
        z = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac)), device=device, dtype=torch.float32))
        opt = torch.optim.Adam([z], lr=0.08)
        for _ in range(120):
            opt.zero_grad(set_to_none=True)
            k_val = 10.0 ** (low + (high - low) * torch.sigmoid(z))
            tkb = torch.stack(
                (
                    torch.full((n_s,), T, device=device, dtype=torch.float32),
                    k_val.expand(n_s),
                    torch.full((n_s,), bD, device=device, dtype=torch.float32),
                ),
                dim=1,
            )
            pred = model.forward_pressure(sensor_xy, tkb, kap_t, cy_t)
            loss = sq_abs(pred - obs).mean()
            loss.backward()
            opt.step()

        lbfgs = torch.optim.LBFGS([z], lr=0.5, max_iter=25, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            kv = 10.0 ** (low + (high - low) * torch.sigmoid(z))
            tkb_l = torch.stack(
                (
                    torch.full((n_s,), T, device=device, dtype=torch.float32),
                    kv.expand(n_s),
                    torch.full((n_s,), bD, device=device, dtype=torch.float32),
                ),
                dim=1,
            )
            pr = model.forward_pressure(sensor_xy, tkb_l, kap_t, cy_t)
            lv = sq_abs(pr - obs).mean()
            lv.backward()
            return lv

        lbfgs.step(closure)
        with torch.no_grad():
            kv_f = float((10.0 ** (low + (high - low) * torch.sigmoid(z))).item())
            tkb_f = torch.tensor([[T, kv_f, bD]], device=device, dtype=torch.float32).expand(n_s, -1)
            lf = float(sq_abs(model.forward_pressure(sensor_xy, tkb_f, kap_t, cy_t) - obs).mean().item())
            if lf < best_loss:
                best_loss = lf
                best_k = kv_f

    elapsed = time.perf_counter() - started
    if best_k is None:
        raise RuntimeError(f"No finite inversion loss for {case['case_id']}")
    fake_case = dict(case)
    fake_case["k"] = best_k
    recon = evaluate_field_list(model.forward_pressure, [fake_case], device)["cases"][0]
    rel_err_pct = 100.0 * abs(best_k - k_true) / k_true
    return {
        "case_id": case["case_id"],
        "H": case["H"],
        "T": T,
        "bD": bD,
        "k_true": k_true,
        "k_inv": best_k,
        "E_k_pct": rel_err_pct,
        "recon_mse": recon["amplitude_mse"],
        "recon_rmse": recon["amplitude_rmse"],
        "recon_r2": recon["amplitude_r2"],
        "recon_complex_l2_pct": 100.0 * recon["complex_l2"],
        "elapsed_seconds": elapsed,
        "sensors_xy": sensor_xy_np.tolist(),
    }


def generate_design_chart(mf_model, constants, device):
    # 40 x 40 = 1,600 (k, b/D) Ocean Engineering Design Chart at T = 1.50 s, H = 0.14 m
    H_design, T_design = 0.14, 1.50
    wn_d = float(solve_wave_number(np.array([T_design]), constants)[0])
    kap_d = wn_d * constants.seabed_thickness_m
    gamma_w = constants.rho_w * constants.gravity
    p0_pa = gamma_w * H_design / (2.0 * math.cosh(wn_d * constants.water_depth_m))

    k_grid = np.logspace(-5.0, math.log10(8.0e-5), 40)
    bD_grid = np.linspace(0.75, 1.50, 40)
    theta_ring = np.linspace(0.0, 2.0 * math.pi, 181)[:-1]
    dtheta = 2.0 * math.pi / 180.0
    th_t = torch.from_numpy(theta_ring.astype(np.float32)).to(device)
    nz_t = torch.sin(th_t)

    C_gamma_map = np.zeros((40, 40), dtype=float)
    C_uplift_map = np.zeros((40, 40), dtype=float)
    i_max_map = np.zeros((40, 40), dtype=float)

    k_tensor = torch.from_numpy(k_grid.astype(np.float32)).to(device)
    n_k = k_tensor.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    sweep_t0 = time.perf_counter()
    for ib, bD_val in enumerate(bD_grid):
        cy_val = 1.0 - float(bD_val) * constants.pipe_diameter_m / constants.seabed_thickness_m
        wall_xy = torch.stack(
            (constants.radius * torch.cos(th_t), cy_val + constants.radius * torch.sin(th_t)), dim=1
        )
        gap_thick = max(0.015, 0.995 - cy_val)
        r_vals = torch.linspace(constants.radius * 1.02, gap_thick, 12, device=device)
        ang_vals = torch.linspace(0.25 * math.pi, 0.75 * math.pi, 25, device=device)
        rr, aa = torch.meshgrid(r_vals, ang_vals, indexing="ij")
        crown_xy = torch.stack((rr.reshape(-1) * torch.cos(aa.reshape(-1)), cy_val + rr.reshape(-1) * torch.sin(aa.reshape(-1))), dim=1)
        crown_xy[:, 1] = crown_xy[:, 1].clamp(max=0.998)

        with torch.no_grad():
            n_w = wall_xy.shape[0]
            w_xy_b = wall_xy.unsqueeze(0).expand(n_k, n_w, 2).reshape(n_k * n_w, 2)
            tkb_w = (
                torch.stack(
                    (
                        torch.full((n_k,), T_design, device=device, dtype=torch.float32),
                        k_tensor,
                        torch.full((n_k,), float(bD_val), device=device, dtype=torch.float32),
                    ),
                    dim=1,
                )
                .unsqueeze(1)
                .expand(n_k, n_w, 3)
                .reshape(n_k * n_w, 3)
            )
            kap_w = torch.full((n_k * n_w,), kap_d, device=device, dtype=torch.float32)
            cy_w = torch.full((n_k * n_w,), cy_val, device=device, dtype=torch.float32)
            p_wall = mf_model.forward_pressure(w_xy_b, tkb_w, kap_w, cy_w).view(n_k, n_w)
            C_gamma_map[ib, :] = p_wall.abs().max(dim=1).values.cpu().numpy()
            uplift_vec = 0.5 * torch.abs(torch.sum(p_wall * nz_t.unsqueeze(0), dim=1) * dtheta)
            C_uplift_map[ib, :] = uplift_vec.cpu().numpy()

        n_c = crown_xy.shape[0]
        c_xy_b = crown_xy.unsqueeze(0).expand(n_k, n_c, 2).reshape(n_k * n_c, 2).detach().clone().requires_grad_(True)
        tkb_c = (
            torch.stack(
                (
                    torch.full((n_k,), T_design, device=device, dtype=torch.float32),
                    k_tensor,
                    torch.full((n_k,), float(bD_val), device=device, dtype=torch.float32),
                ),
                dim=1,
            )
            .unsqueeze(1)
            .expand(n_k, n_c, 3)
            .reshape(n_k * n_c, 3)
        )
        kap_c = torch.full((n_k * n_c,), kap_d, device=device, dtype=torch.float32)
        cy_c = torch.full((n_k * n_c,), cy_val, device=device, dtype=torch.float32)
        p_crown = mf_model.forward_pressure(c_xy_b, tkb_c, kap_c, cy_c)
        gp_c = complex_grad_1(p_crown, c_xy_b)
        grad_norm_star = torch.sqrt(sq_abs(gp_c[:, 0]) + sq_abs(gp_c[:, 1])).view(n_k, n_c).max(dim=1).values
        i_max_map[ib, :] = ((p0_pa / (gamma_w * constants.seabed_thickness_m)) * grad_norm_star).detach().cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    sweep_elapsed_s = time.perf_counter() - sweep_t0
    return sweep_elapsed_s, dict(
        k_grid=k_grid,
        bD_grid=bD_grid,
        C_gamma=C_gamma_map,
        C_uplift=C_uplift_map,
        i_max=i_max_map,
        H_design=H_design,
        T_design=T_design,
        p0_pa=p0_pa,
    )


def compute_comsol_timings_and_design_chart(
    mf_model: PressureCorrectorKAN,
    constants: PhysicalConstants,
    domain: ParameterDomain,
    device: torch.device,
) -> dict[str, object]:
    """
    1. Summarize live-measured COMSOL 6.4 Coarse / Medium / Fine / Ultrafine mesh DOFs, accuracy, and runtimes.
    2. Compute 40 x 40 = 1,600 (k, b/D) ocean-engineering design screening grid (C_Gamma, C_U, i_max)
       and measure exact MF-SeaKAN GPU & single-thread CPU wall-clock query times.
    """
    tdf = pd.read_csv(CLEAN_TIMING_CSV)
    tdf = tdf[tdf["tag"] != "warmup"]
    tier_stats: dict[str, dict[str, float]] = {}
    for tier, grp in tdf.groupby("tier"):
        tier_stats[str(tier)] = {
            "num_elements_mean": float(grp["elems"].mean()),
            "dofs_mean": float(grp["dofs"].mean()),
            "mesh_ms_mean": float(grp["mesh_ms"].mean()),
            "solve_ms_mean": float(grp["solve_ms"].mean()),
            "total_s_mean": float(grp["total_ms"].mean() / 1000.0),
            "total_s_sd": float(grp["total_ms"].std(ddof=1) / 1000.0) if len(grp) > 1 else 0.0,
        }

    # Mesh discretization accuracy on validation_000..009
    uf = np.loadtxt(CLEAN_COMSOL_DIR / "validation_000_ultrafine_domain.csv", delimiter=",", comments="%")
    fn0 = np.loadtxt(CLEAN_COMSOL_DIR / "validation_000_complex_domain.csv", delimiter=",", comments="%")
    interp_uf = LinearNDInterpolator(uf[:, 2:4], uf[:, 4] + 1j * uf[:, 5])(fn0[:, 2:4])
    ok = ~np.isnan(interp_uf.real)
    f0_c = fn0[ok, 4] + 1j * fn0[ok, 5]
    tier_stats["fine"]["mesh_convergence_vs_ultrafine_complex_l2_pct"] = float(
        100.0 * np.linalg.norm(f0_c - interp_uf[ok]) / np.linalg.norm(interp_uf[ok])
    )

    for tier in ("coarse", "medium"):
        cl2s, al2s = [], []
        for i in range(10):
            c_arr = np.loadtxt(CLEAN_COMSOL_DIR / f"validation_{i:03d}_{tier}_domain.csv", delimiter=",", comments="%")
            f_arr = np.loadtxt(CLEAN_COMSOL_DIR / f"validation_{i:03d}_complex_domain.csv", delimiter=",", comments="%")
            pred = LinearNDInterpolator(c_arr[:, 2:4], c_arr[:, 4] + 1j * c_arr[:, 5])(f_arr[:, 2:4])
            m = ~np.isnan(pred.real)
            ref = f_arr[m, 4] + 1j * f_arr[m, 5]
            cl2s.append(100.0 * np.linalg.norm(pred[m] - ref) / np.linalg.norm(ref))
            al2s.append(100.0 * np.linalg.norm(np.abs(pred[m]) - np.abs(ref)) / np.linalg.norm(np.abs(ref)))
        tier_stats[tier]["amp_l2_pct_vs_fine_mean"] = float(np.mean(al2s))
        tier_stats[tier]["complex_l2_pct_vs_fine_mean"] = float(np.mean(cl2s))

    # Measure MF-SeaKAN matched-mesh GPU & CPU query latency
    sample_csv = np.loadtxt(CLEAN_COMSOL_DIR / "validation_000_complex_domain.csv", delimiter=",", comments="%")
    mesh_xy = torch.from_numpy((sample_csv[:, 2:4] / constants.seabed_thickness_m).astype(np.float32)).to(device)
    n_mesh = mesh_xy.shape[0]
    tkb_1 = torch.tensor([[1.50, 4.0e-5, 1.10]], device=device, dtype=torch.float32).expand(n_mesh, -1)
    wn_15 = float(solve_wave_number(np.array([1.50]), constants)[0])
    kap_1 = torch.full((n_mesh,), wn_15 * constants.seabed_thickness_m, device=device, dtype=torch.float32)
    cy_1 = torch.full(
        (n_mesh,), 1.0 - 1.10 * constants.pipe_diameter_m / constants.seabed_thickness_m, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        for _ in range(5):
            _ = mf_model.forward_pressure(mesh_xy, tkb_1, kap_1, cy_1)
    torch.cuda.synchronize()
    gpu_mesh_times = []
    with torch.no_grad():
        for _ in range(30):
            s0 = time.perf_counter()
            _ = mf_model.forward_pressure(mesh_xy, tkb_1, kap_1, cy_1)
            torch.cuda.synchronize()
            gpu_mesh_times.append(time.perf_counter() - s0)

    cpu_model = copy.deepcopy(mf_model).to("cpu").eval()
    mesh_xy_cpu, tkb_cpu, kap_cpu, cy_cpu = mesh_xy.cpu(), tkb_1.cpu(), kap_1.cpu(), cy_1.cpu()
    with torch.no_grad():
        _ = cpu_model.forward_pressure(mesh_xy_cpu, tkb_cpu, kap_cpu, cy_cpu)
        cpu_times = []
        for _ in range(10):
            s0 = time.perf_counter()
            _ = cpu_model.forward_pressure(mesh_xy_cpu, tkb_cpu, kap_cpu, cy_cpu)
            cpu_times.append(time.perf_counter() - s0)

    sweep_elapsed_s, chart = generate_design_chart(mf_model, constants, device)
    np.savez(OUT_DIR / "design_chart_1600.npz", **chart)

    comsol_fine_s = tier_stats["fine"]["total_s_mean"]
    comsol_medium_s = tier_stats["medium"]["total_s_mean"]
    return {
        "comsol_mesh_tiers": tier_stats,
        "comsol_fine_elements_mean": tier_stats["fine"]["num_elements_mean"],
        "comsol_fine_dofs_mean": tier_stats["fine"]["dofs_mean"],
        "comsol_fine_per_case_wall_s_mean": comsol_fine_s,
        "comsol_fine_per_case_wall_s_sd": tier_stats["fine"]["total_s_sd"],
        "comsol_medium_matched_per_case_wall_s_mean": comsol_medium_s,
        "mf_seakan_matched_mesh_nodes": int(n_mesh),
        "mf_seakan_matched_mesh_gpu_ms_mean": 1000.0 * float(np.mean(gpu_mesh_times)),
        "mf_seakan_matched_mesh_gpu_ms_sd": 1000.0 * float(np.std(gpu_mesh_times, ddof=1)),
        "mf_seakan_matched_mesh_cpu_ms_mean": 1000.0 * float(np.mean(cpu_times)),
        "design_sweep_1600_cases_wall_s": sweep_elapsed_s,
        "comsol_fine_1600_cases_wall_s": 1600.0 * comsol_fine_s,
        "comsol_medium_1600_cases_wall_s": 1600.0 * comsol_medium_s,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Full training and GPU timings require CUDA. Use verify_tier2.py for CPU verification.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EXP-2 (Clean Sealed-Bottom Suite) running on {device}", flush=True)
    constants = PhysicalConstants()
    domain = ParameterDomain()
    splits = load_all_comsol_cases(constants)
    all_test_cases = splits["blind_sealed"] + splits["prospective"]
    het_cases = splits["heterogeneous_stress"]
    print(
        f"Loaded clean COMSOL 6.4 cases: train={len(splits['train'])}, val={len(splits['validation'])}, "
        f"sealed_blind={len(splits['blind_sealed'])}, prospective={len(splits['prospective'])}, het={len(het_cases)}",
        flush=True,
    )

    # 1. Train Clean 1st-Order MPM Weak-Form Physics-Only Parametric Trunk (3 seeds, ZERO COMSOL labels!)
    trunk_seeds = [202416, 202417, 202418]
    trunk_records = []
    best_trunk: CleanParametricSeaKAN | None = None
    best_trunk_val_l2 = float("inf")
    trunk_train_time = 0.0

    for sd in trunk_seeds:
        ckpt_pt = OUT_DIR / f"clean_trunk_seed_{sd}.pt"
        ckpt_meta = OUT_DIR / f"clean_trunk_seed_{sd}.json"
        if ckpt_pt.is_file() and ckpt_meta.is_file():
            tr_model = CleanParametricSeaKAN(constants, domain, dtype=torch.float32).to(device)
            tr_model.load_state_dict(torch.load(ckpt_pt, map_location=device, weights_only=True))
            tr_model.eval()
            tr_time = float(json.loads(ckpt_meta.read_text(encoding="utf-8"))["elapsed_s"])
            print(f"[CACHE HIT] Loaded Clean Physics Trunk seed={sd} ({tr_time:.1f}s)", flush=True)
        else:
            tr_model, tr_time, _ = train_clean_physics_trunk(
                constants, domain, splits["validation"], device, seed=sd
            )
            torch.save(tr_model.state_dict(), ckpt_pt)
            ckpt_meta.write_text(json.dumps({"seed": sd, "elapsed_s": tr_time}), encoding="utf-8")
        val_m = evaluate_field_list(lambda xy, tkb, kap, cy, m=tr_model: m(xy, tkb, kap, cy)[2], splits["validation"], device)
        tst_m = evaluate_field_list(lambda xy, tkb, kap, cy, m=tr_model: m(xy, tkb, kap, cy)[2], all_test_cases, device)
        trunk_records.append(
            {
                "seed": sd,
                "elapsed_s": tr_time,
                "val_amp_l2_pct": 100.0 * val_m["mean_amplitude_l2"],
                "val_amp_r2": val_m["mean_amplitude_r2"],
                "val_min_amp_r2": val_m["min_amplitude_r2"],
                "val_complex_l2_pct": 100.0 * val_m["mean_complex_l2"],
                "val_complex_r2": val_m["mean_complex_r2"],
                "test8_amp_l2_pct": 100.0 * tst_m["mean_amplitude_l2"],
                "test8_amp_r2": tst_m["mean_amplitude_r2"],
                "test8_min_amp_r2": tst_m["min_amplitude_r2"],
                "test8_complex_l2_pct": 100.0 * tst_m["mean_complex_l2"],
                "test8_complex_r2": tst_m["mean_complex_r2"],
            }
        )
        if val_m["mean_complex_l2"] < best_trunk_val_l2:
            best_trunk_val_l2 = val_m["mean_complex_l2"]
            best_trunk = tr_model
            trunk_train_time = tr_time

    assert best_trunk is not None

    # 2. Sample-Efficiency Study: Pure-Data KAN vs Multi-Fidelity SeaKAN across N_cal in {4, 8, 16, 32}
    print("\n=== Running Multi-Fidelity vs Pure-Data KAN Sample-Efficiency Study ===", flush=True)
    sample_curve_rows = []
    selected_mf_model: PressureCorrectorKAN | None = None
    mf32_cal_time = 0.0

    for n_cal in (4, 8, 16, 32):
        subset = splits["train"][:n_cal]
        for mode_name, use_trunk in (("pure_data_kan", False), ("mf_seakan", True)):
            seed_vals_val_l2, seed_vals_tst_l2, seed_vals_tst_r2, seed_vals_tst_cl2, cal_times = [], [], [], [], []
            for sd in (202438, 202439, 202440):
                m_sur, c_time = train_surrogate_variant(
                    constants,
                    domain,
                    best_trunk if use_trunk else None,
                    subset,
                    splits["validation"],
                    device,
                    steps=800,
                    seed=sd,
                )
                vm = evaluate_field_list(m_sur.forward_pressure, splits["validation"], device)
                tm = evaluate_field_list(m_sur.forward_pressure, all_test_cases, device)
                seed_vals_val_l2.append(100.0 * vm["mean_amplitude_l2"])
                seed_vals_tst_l2.append(100.0 * tm["mean_amplitude_l2"])
                seed_vals_tst_r2.append(tm["mean_amplitude_r2"])
                seed_vals_tst_cl2.append(100.0 * tm["mean_complex_l2"])
                cal_times.append(c_time)
                if use_trunk and n_cal == 32 and sd == 202438:
                    selected_mf_model = m_sur
                    mf32_cal_time = c_time
                    torch.save(m_sur.state_dict(), OUT_DIR / "selected_mf32_model.pt")

            row = {
                "mode": mode_name,
                "n_cal": n_cal,
                "val_amp_l2_pct_mean": float(np.mean(seed_vals_val_l2)),
                "val_amp_l2_pct_std": float(np.std(seed_vals_val_l2, ddof=1)),
                "test8_amp_l2_pct_mean": float(np.mean(seed_vals_tst_l2)),
                "test8_amp_l2_pct_std": float(np.std(seed_vals_tst_l2, ddof=1)),
                "test8_complex_l2_pct_mean": float(np.mean(seed_vals_tst_cl2)),
                "test8_complex_l2_pct_std": float(np.std(seed_vals_tst_cl2, ddof=1)),
                "test8_amp_r2_mean": float(np.mean(seed_vals_tst_r2)),
                "test8_amp_r2_std": float(np.std(seed_vals_tst_r2, ddof=1)),
                "cal_time_s_mean": float(np.mean(cal_times)),
            }
            sample_curve_rows.append(row)
            print(
                f"[{mode_name} | N_cal={n_cal:2d}] Test8 Amp L2={row['test8_amp_l2_pct_mean']:.2f}±{row['test8_amp_l2_pct_std']:.2f}%, "
                f"Complex L2={row['test8_complex_l2_pct_mean']:.2f}±{row['test8_complex_l2_pct_std']:.2f}%, "
                f"R2={row['test8_amp_r2_mean']:.4f}",
                flush=True,
            )

    assert selected_mf_model is not None

    trunk_val_detail = evaluate_field_list(
        lambda xy, tkb, kap, cy: best_trunk(xy, tkb, kap, cy)[2], splits["validation"], device
    )
    mf_val_detail = evaluate_field_list(selected_mf_model.forward_pressure, splits["validation"], device)
    mf_test_detail = evaluate_field_list(selected_mf_model.forward_pressure, all_test_cases, device)
    trunk_test_detail = evaluate_field_list(
        lambda xy, tkb, kap, cy: best_trunk(xy, tkb, kap, cy)[2], all_test_cases, device
    )
    mf_het_detail = evaluate_field_list(selected_mf_model.forward_pressure, het_cases, device)

    # 3. Inverse Permeability Identification on all 8 homogeneous held-out cases + 2 heterogeneous stress cases
    print("\n=== Running Permeability Inversion on Held-Out & Stress-Test Cases ===", flush=True)
    inverse_comsol_rows = []
    for c in all_test_cases + het_cases:
        inv_r = invert_comsol_case_clean(selected_mf_model, c, constants, domain, device)
        inverse_comsol_rows.append(inv_r)
        print(
            f"Inverse {inv_r['case_id']:8s}: k_true={inv_r['k_true']:.2e}, k_inv={inv_r['k_inv']:.2e}, "
            f"E_k={inv_r['E_k_pct']:.2f}%, recon R2={inv_r['recon_r2']:.4f}, time={inv_r['elapsed_seconds']:.2f}s",
            flush=True,
        )

    # 3b. Sensor Noise Repeatability Study on blind_000 & blind_001 (0%, 1%, 2%, 5% Gaussian complex noise, 5 repeats)
    noise_study_rows = []
    for c in splits["blind_sealed"]:
        for n_lvl in (0.0, 0.01, 0.02, 0.05):
            errs = []
            for rep in range(1 if n_lvl == 0.0 else 5):
                r_n = invert_comsol_case_clean(
                    selected_mf_model, c, constants, domain, device, noise_rel=n_lvl, noise_seed=8000 + 100 * int(n_lvl * 100) + rep
                )
                errs.append(r_n["E_k_pct"])
            noise_study_rows.append(
                {
                    "case_id": c["case_id"],
                    "noise_pct": 100.0 * n_lvl,
                    "E_k_pct_mean": float(np.mean(errs)),
                    "E_k_pct_std": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
                }
            )
            print(f"Noise {c['case_id']} ({100*n_lvl:.0f}%): E_k={np.mean(errs):.2f}%", flush=True)

    # 4. External Laboratory Verification on Turcotte et al. (1984)
    from refine_turcotte_lab_inversion import reproduce_lab_inversion
    turcotte_inv = reproduce_lab_inversion(OUT_DIR, device)

    # 5. COMSOL 6.4 Direct Solver Benchmark & 1,600-Case Ocean Engineering Design Chart
    cost_and_design = compute_comsol_timings_and_design_chart(selected_mf_model, constants, domain, device)
    cost_and_design["clean_trunk_train_s"] = trunk_train_time
    cost_and_design["mf32_calibration_s"] = mf32_cal_time
    cost_and_design["total_offline_s_including_32_comsol"] = (
        32.0 * cost_and_design["comsol_fine_per_case_wall_s_mean"] + trunk_train_time + mf32_cal_time
    )
    t_c_fine = cost_and_design["comsol_fine_per_case_wall_s_mean"]
    t_c_med = cost_and_design["comsol_medium_matched_per_case_wall_s_mean"]
    t_s_gpu = cost_and_design["mf_seakan_matched_mesh_gpu_ms_mean"] / 1000.0
    cost_and_design["forward_breakeven_vs_fine_comsol_gpu"] = (
        cost_and_design["total_offline_s_including_32_comsol"] / (t_c_fine - t_s_gpu)
    )
    cost_and_design["forward_breakeven_vs_matched_medium_comsol_gpu"] = (
        cost_and_design["total_offline_s_including_32_comsol"] / (t_c_med - t_s_gpu)
    )
    cost_and_design["trunk_only_breakeven_vs_coarse_comsol_gpu"] = (
        trunk_train_time / (cost_and_design["comsol_mesh_tiers"]["coarse"]["total_s_mean"] - t_s_gpu)
    )

    bundle = {
        "physics_trunk_3seed": trunk_records,
        "physics_trunk_validation_10cases": trunk_val_detail,
        "sample_efficiency_curve": sample_curve_rows,
        "mf32_validation_10cases": mf_val_detail,
        "mf32_test_8cases": mf_test_detail,
        "mf32_het_stress_2cases": mf_het_detail,
        "physics_trunk_test_8cases": trunk_test_detail,
        "inverse_comsol_cases": inverse_comsol_rows,
        "inverse_noise_study": noise_study_rows,
        "inverse_turcotte_lab": turcotte_inv,
        "cost_and_design_summary": cost_and_design,
    }
    (OUT_DIR / "exp2_complete_results.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    pd.DataFrame(sample_curve_rows).to_csv(OUT_DIR / "sample_efficiency_curve.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "sensors_xy"} for r in inverse_comsol_rows]).to_csv(
        OUT_DIR / "inverse_comsol_cases.csv", index=False
    )
    print("\n=== EXP-2 ALL FINISHED SUCCESSFULLY ===", flush=True)
    print(json.dumps(cost_and_design, indent=2), flush=True)


if __name__ == "__main__":
    main()
