from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12.0,
        "axes.labelsize": 13.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.0,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SUPP_ROOT = Path(__file__).resolve().parent
EXP1_DIR = SUPP_ROOT / "exp1_kinn_and_ablation"
EXP2_DIR = SUPP_ROOT / "exp2_mf_seakan_and_costs"
sys.path.insert(0, str(EXP2_DIR))

CODE_ROOT = SUPP_ROOT / "legacy"
OUT_DIR = SUPP_ROOT / "reproduced_figures"

sys.path.insert(0, str(CODE_ROOT / "figure9_mpm_kan" / "src"))
sys.path.insert(0, str(CODE_ROOT / "figure15_mpm_kan" / "src"))

from mpm_kan_figure9.data import load_figure9_data
from mpm_kan_figure15.data import load_figure15_data


def _save_pdf(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT_DIR / filename, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {filename}", flush=True)


# ---------------------------------------------------------------------------
# Figure 1: Physical Domain Schematic (larger, crisp fonts)
# ---------------------------------------------------------------------------
def make_figure1_domain_schematic() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-3.75, 2.50)
    ax.set_aspect("equal")
    ax.axis("off")

    L_half, h_vis, d_vis = 4.35, 2.75, 1.55
    ax.add_patch(mpatches.Rectangle((-L_half, 0.0), 2 * L_half, d_vis, facecolor="#DBEAFE", edgecolor="none", zorder=1))
    ax.add_patch(mpatches.Rectangle((-L_half, -h_vis), 2 * L_half, h_vis, facecolor="#F5E6CA", edgecolor="none", zorder=1))
    ax.add_patch(mpatches.Rectangle((-L_half, -h_vis - 0.22), 2 * L_half, 0.22, facecolor="#E2E8F0", edgecolor="#334155", hatch="////", lw=0.9, zorder=2))

    xs = np.linspace(-L_half, L_half, 300)
    eta = d_vis + 0.17 * np.cos(1.85 * xs)
    ax.fill_between(xs, d_vis, eta, color="#DBEAFE", zorder=1)
    ax.plot(xs, eta, color="#0284C7", lw=2.0, zorder=4)
    ax.plot([-L_half, L_half], [d_vis, d_vis], color="#475569", ls="--", lw=1.0, zorder=3)
    ax.text(L_half - 0.08, d_vis + 0.22, "Still-water level ($z = d$)", ha="right", va="bottom", fontsize=11.0, style="italic", color="#1E293B")

    ax.annotate("", xy=(1.8, 2.15), xytext=(-1.8, 2.15),
                arrowprops=dict(arrowstyle="-|>,head_length=0.65,head_width=0.34", lw=1.7, color="#1D4ED8"))
    ax.text(0.0, 2.22, "Wave propagation ($+x$, $T$, $H$, $L_{\\mathrm{w}}$)", ha="center", va="bottom", fontsize=11.8, fontweight="bold", color="#1E3A8A")

    ax.plot([-L_half, L_half], [0.0, 0.0], color="#1D4ED8", lw=1.8, zorder=4)
    ax.plot([-L_half, L_half], [-h_vis, -h_vis], color="#0F172A", lw=1.7, zorder=4)
    ax.plot([-L_half, -L_half], [-h_vis, d_vis], color="#334155", ls="--", lw=1.2, zorder=4)
    ax.plot([L_half, L_half], [-h_vis, d_vis], color="#334155", ls="--", lw=1.2, zorder=4)

    ax.annotate("", xy=(0.95, 0.0), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#0F172A"), zorder=6)
    ax.annotate("", xy=(0.0, 0.72), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#0F172A"), zorder=6)
    ax.text(0.98, -0.18, "$x$", fontsize=11.8, fontweight="bold")
    ax.text(0.12, 0.65, "$z$", fontsize=11.8, fontweight="bold")
    ax.text(-L_half - 0.14, 0.0, "$z=0$\n($\\Gamma_{\\mathrm{s}}$)", ha="right", va="center", fontsize=11.0, fontweight="bold", color="#1E3A8A")
    ax.text(-L_half - 0.14, -h_vis, "$z=-h$\n($\\Gamma_{\\mathrm{b}}$)", ha="right", va="center", fontsize=11.0, fontweight="bold", color="#0F172A")

    b_vis, r_vis = 1.08, 0.52
    ax.add_patch(mpatches.Circle((0.0, -b_vis), r_vis, facecolor="#CBD5E1", edgecolor="#0F172A", lw=1.8, zorder=5))
    ax.add_patch(mpatches.Circle((0.0, -b_vis), 0.74 * r_vis, facecolor="#F8FAFC", edgecolor="#475569", lw=1.1, zorder=5))
    ax.scatter([0.0], [-b_vis], s=20, color="#0F172A", zorder=6)

    ax.annotate("", xy=(r_vis, -b_vis), xytext=(-r_vis, -b_vis), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#0F172A"), zorder=6)
    ax.text(0.0, -b_vis - 0.17, "$D$", ha="center", va="top", fontsize=11.5, fontweight="bold", zorder=6)

    ax.plot([0.0, 1.38], [-b_vis, -b_vis], color="#475569", ls=":", lw=1.0, zorder=4)
    ax.annotate("", xy=(1.25, 0.0), xytext=(1.25, -b_vis), arrowprops=dict(arrowstyle="<->", lw=1.35, color="#DC2626"), zorder=6)
    ax.text(1.36, -0.52 * b_vis, "$b$ (center)", ha="left", va="center", fontsize=11.0, fontweight="bold", color="#991B1B")

    ax.plot([0.0, -1.25], [-b_vis + r_vis, -b_vis + r_vis], color="#475569", ls=":", lw=1.0, zorder=4)
    ax.annotate("", xy=(-1.12, 0.0), xytext=(-1.12, -b_vis + r_vis), arrowprops=dict(arrowstyle="<->", lw=1.25, color="#0F766E"), zorder=6)
    ax.text(-1.24, -0.5 * (b_vis - r_vis), "$e_{\\mathrm{c}}=b-D/2$\n(cover)", ha="right", va="center", fontsize=10.4, fontweight="bold", color="#115E59")

    ax.annotate("", xy=(3.55, d_vis), xytext=(3.55, 0.0), arrowprops=dict(arrowstyle="<->", lw=1.25, color="#0F172A"), zorder=5)
    ax.text(3.68, 0.5 * d_vis, "$d$", ha="left", va="center", fontsize=12.0, fontweight="bold")

    ax.annotate("", xy=(3.55, 0.0), xytext=(3.55, -h_vis), arrowprops=dict(arrowstyle="<->", lw=1.25, color="#0F172A"), zorder=5)
    ax.text(3.68, -0.5 * h_vis, "$h$", ha="left", va="center", fontsize=12.0, fontweight="bold")

    ax.annotate("", xy=(L_half, -3.40), xytext=(-L_half, -3.40), arrowprops=dict(arrowstyle="<->", lw=1.3, color="#0F172A"), zorder=5)
    ax.text(0.0, -3.48, "Computational width $L$ ($x \\in [-L/2,\\ L/2]$)", ha="center", va="top", fontsize=11.2, fontweight="bold")

    ax.text(-2.35, 0.78, "Mudline wave loading\n$\\widehat{p}(x,0)=p_0 e^{\\mathrm{i}\\kappa x},\\ \\widehat{\\sigma}_z'=\\widehat{\\tau}_{xz}=0$",
            ha="center", va="center", fontsize=10.5, color="#1E3A8A",
            bbox=dict(facecolor="#EFF6FF", edgecolor="#93C5FD", boxstyle="round,pad=0.3", lw=0.9))
    ax.text(-2.72, -1.96, "Poroelastic seabed $\\Omega_{\\mathrm{s}}$\n$(G,\\ \\nu,\\ n,\\ S_r,\\ k)$",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#451A03")
    ax.annotate("Rigid pipe $\\Gamma_{\\mathrm{p}}$\n$(\\widehat{u}=\\widehat{w}=0,\\ \\nabla\\widehat{p}\\cdot\\boldsymbol{n}=0)$",
                xy=(r_vis * 0.72, -b_vis - r_vis * 0.70), xytext=(1.32, -2.10),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#1E293B"),
                fontsize=10.4, color="#0F172A",
                bbox=dict(facecolor="#FFFFFF", edgecolor="#CBD5E1", boxstyle="round,pad=0.3", lw=0.9, alpha=0.95))
    ax.text(0.0, -h_vis - 0.34, "Rigid impermeable base $\\Gamma_{\\mathrm{b}}$: $\\widehat{u}=\\widehat{w}=0,\\ \\partial\\widehat{p}/\\partial z=0$",
            ha="center", va="top", fontsize=10.8, color="#0F172A")
    ax.text(-L_half - 0.16, -1.35, "$\\Gamma_{\\mathrm{L}}$", ha="right", va="center", fontsize=11.5, fontweight="bold", color="#334155")
    ax.text(L_half + 0.16, -1.35, "$\\Gamma_{\\mathrm{R}}$", ha="left", va="center", fontsize=11.5, fontweight="bold", color="#334155")

    _save_pdf(fig, "fig1_domain_clean.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Visual Iconic Two-Tier Architecture Diagram (Inspired by User Ref Fig. 2)
# ---------------------------------------------------------------------------
def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    ax.set_xlim(0, 138)
    ax.set_ylim(0, 74)
    ax.axis("off")

    # Subtle graph-paper grid background
    for gx in np.arange(0, 139, 2.5):
        ax.plot([gx, gx], [0, 74], color="#E5E7EB", lw=0.45, zorder=0)
    for gy in np.arange(0, 75, 2.5):
        ax.plot([0, 138], [gy, gy], color="#E5E7EB", lw=0.45, zorder=0)

    checked_pairs: list[tuple[str, mpatches.FancyBboxPatch, mpl.text.Text]] = []

    def card(x, y, w, h, title="", fc="#F4F5F7", ec="#334155", lw=1.2, title_y_off=1.75, title_fs=10.6, title_c="#0F172A"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.85",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        )
        ax.add_patch(rect)
        if title:
            txt = ax.text(x + w / 2, y + h - title_y_off, title, ha="center", va="center",
                          fontsize=title_fs, fontweight="bold", color=title_c, zorder=5)
            checked_pairs.append((f"card:{title}", rect, txt))
        return rect

    def pill(x, y, w, h, text, fc="#FDE68A", ec="#334155", fs=9.0, tc="#0F172A", lw=0.95):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.14,rounding_size=0.45",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=4,
        )
        ax.add_patch(rect)
        txt = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                      fontsize=fs, color=tc, fontweight="bold", linespacing=1.18, zorder=5)
        checked_pairs.append((f"pill:{text.splitlines()[0]}", rect, txt))
        return rect

    def tensor_col(x, y, cw, ch, n_cells, labels=None, fc="#FDBA74", ec="#334155", header="", hdr_fs=9.8):
        for i in range(n_cells):
            yy = y + (n_cells - 1 - i) * ch
            ax.add_patch(mpatches.Rectangle((x, yy), cw, ch, facecolor=fc, edgecolor=ec, lw=0.95, zorder=4))
            if labels and i < len(labels):
                ax.text(x + cw / 2, yy + ch / 2, labels[i], ha="center", va="center", fontsize=9.0, color="#0F172A", zorder=5)
        if header:
            ax.text(x + cw / 2, y + n_cells * ch + 0.95, header, ha="center", va="bottom",
                    fontsize=hdr_fs, fontweight="bold", color="#0F172A", zorder=5)

    def draw_kan_net(cx, cy, w, h, layer_sizes, node_fc="#2563EB", edge_c="#334155", spline_wave=True):
        """Draw a visual KAN graph with nodes + curved B-spline wave edges."""
        n_layers = len(layer_sizes)
        xs = np.linspace(cx - w / 2, cx + w / 2, n_layers)
        coords = []
        for li, n_n in enumerate(layer_sizes):
            ys = np.linspace(cy - h / 2, cy + h / 2, n_n) if n_n > 1 else [cy]
            coords.append([(xs[li], yv) for yv in ys])
        for li in range(n_layers - 1):
            for (x0, y0) in coords[li]:
                for (x1, y1) in coords[li + 1]:
                    if spline_wave:
                        t = np.linspace(0, 1, 18)
                        xm = (1 - t) * x0 + t * x1
                        ym = (1 - t) * y0 + t * y1 + 0.36 * np.sin(2.0 * np.pi * t)
                        ax.plot(xm, ym, color=edge_c, lw=0.75, alpha=0.82, zorder=3)
                    else:
                        ax.plot([x0, x1], [y0, y1], color=edge_c, lw=0.75, alpha=0.82, zorder=3)
        r_node = 0.82
        for layer in coords:
            for (xn, yn) in layer:
                ax.add_patch(mpatches.Circle((xn, yn), r_node, facecolor=node_fc, edgecolor="#0F172A", lw=0.9, zorder=5))

    def op_circle(cx, cy, sym, fc="#FFFFFF", ec="#0F172A", r=1.55, fs=12.5):
        ax.add_patch(mpatches.Circle((cx, cy), r, facecolor=fc, edgecolor=ec, lw=1.25, zorder=5))
        ax.text(cx, cy, sym, ha="center", va="center", fontsize=fs, fontweight="bold", color=ec, zorder=6)

    def arrow(x0, y0, x1, y1, color="#1E293B", lw=1.35):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.30", color=color, lw=lw), zorder=6)

    def poly_arrow(pts, color="#1E293B", lw=1.35):
        for i in range(len(pts) - 2):
            ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=color, lw=lw, zorder=6)
        arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw)

    # =================== TIER 1 BANNER (TOP HALF: y = 39.2..72.6) ===================
    t1_bg = mpatches.FancyBboxPatch((1.0, 39.2), 136.0, 33.4, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.3, ls="--", alpha=0.55, zorder=1)
    ax.add_patch(t1_bg)
    ax.text(2.4, 70.9, "(a) Single-Scenario Physics-Only Solver (SeaKAN, Zero Reference Field Data)",
            fontsize=11.4, fontweight="bold", color="#1D4ED8", va="center", zorder=5)

    # 1. Input Tensor + Hard-BC Coordinate Map
    card(2.2, 40.7, 23.0, 28.2, "Input & Hard-BC Map", fc="#FFFFFF", ec="#2563EB", title_fs=10.4, title_c="#1E3A8A")
    tensor_col(3.3, 50.8, 4.3, 2.8, 2, [r"$x^*$", r"$y^*$"], fc="#FDBA74", header="Coords", hdr_fs=9.4)
    arrow(7.8, 53.6, 10.4, 53.6)
    ax.add_patch(mpatches.Rectangle((10.8, 47.7), 13.0, 11.4, facecolor="#FEF3C7", edgecolor="#334155", lw=0.95, zorder=3))
    ax.plot([10.8, 23.8], [59.1, 59.1], color="#1D4ED8", lw=2.0, zorder=4)
    ax.add_patch(mpatches.Circle((17.3, 53.7), 2.1, facecolor="#CBD5E1", edgecolor="#0F172A", lw=1.2, zorder=4))
    ax.add_patch(mpatches.Arc((17.3, 53.7), 6.4, 6.4, theta1=30, theta2=150, color="#DC2626", lw=1.6, ls="--", zorder=5))
    ax.text(17.3, 60.5, r"$\widehat{p}=p_0 e^{\mathrm{i}\kappa x}$", ha="center", fontsize=8.8, color="#1D4ED8", zorder=5)
    ax.text(17.3, 49.0, r"$\nabla\widehat{p}\cdot\boldsymbol{n}=0$", ha="center", fontsize=8.6, fontweight="bold", color="#991B1B", zorder=5)
    pill(3.3, 41.8, 20.8, 4.0, r"Quintic $\widetilde{\boldsymbol{x}}^*$ & polar $(\eta_\theta,\eta_n)$", fc="#E0F2FE", ec="#0284C7", fs=8.2)

    # 2. Dual-Branch 2D KAN (Global KAN + Crown-Local Hermite KAN)
    card(27.5, 40.7, 35.8, 28.2, "Dual-Branch 2D KAN Architecture", fc="#FFFFFF", ec="#0284C7", title_fs=10.6, title_c="#0369A1")
    # Upper sub-box: Global KAN
    card(29.0, 54.4, 26.2, 11.2, r"Global KAN $\mathcal{K}_{\mathrm{glob}}$ (B-spline)", fc="#F8FAFC", ec="#475569", lw=1.0, title_y_off=1.55, title_fs=9.0)
    draw_kan_net(42.1, 58.6, 17.8, 5.1, [2, 4, 4, 3], node_fc="#2563EB", edge_c="#1E3A8A", spline_wave=True)
    # Lower sub-box: Local Hermite KAN
    card(29.0, 41.9, 26.2, 11.2, r"Crown Hermite KAN $\Delta\mathcal{K}_{\mathrm{loc}}$", fc="#FFF7ED", ec="#C2410C", lw=1.0, title_y_off=1.55, title_fs=9.0, title_c="#9A3412")
    draw_kan_net(37.6, 46.1, 11.6, 4.9, [2, 3, 2], node_fc="#EA580C", edge_c="#9A3412", spline_wave=True)
    hx = np.linspace(0, 1, 30)
    hy = 1 - 3 * hx**2 + 2 * hx**3
    ax.plot(46.2 + 6.4 * hx, 43.5 + 4.4 * hy, color="#DC2626", lw=1.8, zorder=5)
    ax.text(49.4, 49.0, r"$h_{\mathrm{H}}(s_n)$", ha="center", fontsize=8.6, color="#DC2626", fontweight="bold", zorder=5)
    # Merge '+' circle
    op_circle(59.3, 53.6, "+", fc="#FEF9C3", ec="#0F172A", r=1.65)
    arrow(55.4, 59.6, 58.0, 54.9)
    arrow(55.4, 47.4, 58.0, 52.3)
    poly_arrow([(25.4, 59.6), (29.0, 59.6)])
    poly_arrow([(25.4, 47.4), (29.0, 47.4)])

    # 3. Output Phasor Tensor
    tensor_col(65.6, 48.0, 4.6, 3.6, 3, [r"$\widehat{u}$", r"$\widehat{w}$", r"$\widehat{p}$"], fc="#BBF7D0", header="Fields", hdr_fs=9.8)
    arrow(61.1, 53.6, 65.4, 53.6)

    # 4. Weak-to-Strong Physics Curriculum
    card(72.6, 40.7, 37.8, 28.2, "Two-Stage Weak-to-Strong Biot Physics", fc="#FFFFFF", ec="#0D9488", title_fs=10.6, title_c="#115E59")
    arrow(70.4, 53.6, 72.6, 53.6)

    # Stage A mini-card
    c_sa = card(74.1, 54.4, 34.8, 11.2, "Stage A: 1st-Order MPM Weak Form", fc="#F0FDFA", ec="#0D9488", lw=1.0, title_y_off=1.55, title_fs=9.1, title_c="#0F766E")
    for gx in np.linspace(75.8, 84.8, 4):
        ax.plot([gx, gx], [55.7, 61.3], color="#64748B", lw=0.8, zorder=3)
    for gy in np.linspace(55.7, 61.3, 3):
        ax.plot([75.8, 84.8], [gy, gy], color="#64748B", lw=0.8, zorder=3)
    ax.scatter([77.3, 80.3, 83.3, 77.3, 80.3, 83.3], [57.1, 57.1, 57.1, 59.9, 59.9, 59.9], s=16, color="#0D9488", zorder=5)
    t_sa1 = ax.text(97.0, 59.4, r"$\mathcal{R}^{\mathrm{weak}}_{u,w,p}$ (1st deriv.)", ha="center", va="center", fontsize=9.2, color="#0F172A", zorder=5)
    t_sa2 = ax.text(97.0, 56.6, r"Natural $\nabla p\cdot\boldsymbol{n}=0$", ha="center", va="center", fontsize=8.9, color="#115E59", zorder=5)
    checked_pairs.extend([("sa_t1", c_sa, t_sa1), ("sa_t2", c_sa, t_sa2)])

    # Stage B mini-card
    c_sb = card(74.1, 41.9, 34.8, 11.2, "Stage B: 4-Level Strong + AL", fc="#F8FAFC", ec="#475569", lw=1.0, title_y_off=1.55, title_fs=9.1, title_c="#0F172A")
    ax.add_patch(mpatches.Circle((80.3, 46.0), 1.35, facecolor="#CBD5E1", edgecolor="#0F172A", lw=1.0, zorder=4))
    for rr, cc in [(2.05, "#DC2626"), (2.85, "#F59E0B"), (3.60, "#3B82F6")]:
        ax.add_patch(mpatches.Circle((80.3, 46.0), rr, facecolor="none", edgecolor=cc, lw=1.0, ls=":", zorder=4))
    pill(86.4, 46.9, 6.4, 2.8, r"$w_m$", fc="#FCE7F3", ec="#BE185D", fs=9.2)
    pill(93.6, 46.9, 6.4, 2.8, r"$w_f$", fc="#FCE7F3", ec="#BE185D", fs=9.2)
    pill(100.8, 46.9, 6.4, 2.8, r"AL", fc="#FCE7F3", ec="#BE185D", fs=9.0)
    t_sb = ax.text(96.8, 43.9, r"4,302 pts + L-BFGS (64-bit)", ha="center", va="center", fontsize=8.7, color="#334155", zorder=5)
    checked_pairs.append(("sb_t", c_sb, t_sb))
    arrow(91.5, 54.4, 91.5, 53.2, color="#0F766E", lw=1.2)

    # 5. Tier 1 Output Card
    c_bv = card(112.8, 40.7, 23.0, 28.2, "Benchmark Verification", fc="#F0FDF4", ec="#16A34A", title_fs=10.4, title_c="#166534")
    arrow(110.6, 53.6, 112.8, 53.6, color="#16A34A")
    ax.add_patch(mpatches.Circle((124.3, 59.3), 4.5, facecolor="#FFFFFF", edgecolor="#334155", lw=0.95, zorder=3))
    th_m = np.linspace(0, 2 * np.pi, 60)
    rr_m = 2.0 + 1.35 * np.sin(th_m)
    ax.plot(124.3 + rr_m * np.cos(th_m), 59.3 + rr_m * np.sin(th_m), color="#1D4ED8", lw=1.6, zorder=4)
    ax.scatter(124.3 + rr_m[::8] * np.cos(th_m[::8]), 59.3 + rr_m[::8] * np.sin(th_m[::8]), s=14, facecolor="#FFFFFF", edgecolor="#0F172A", lw=0.9, zorder=5)
    t_bv = ax.text(124.3, 50.3, "Flat, Buried & Trenched\nPipes (Zero FEM Data)", ha="center", va="center", fontsize=9.0, color="#14532D", fontweight="bold", zorder=5)
    checked_pairs.append(("bv_t", c_bv, t_bv))
    pill(114.2, 42.0, 20.2, 4.9, "Buried pipe: 5.41%\n($R^2=0.9894$ vs. Exp.)", fc="#DCFCE7", ec="#16A34A", fs=8.5, tc="#14532D")

    # =================== TIER 2 BANNER (BOTTOM HALF: y = 1.4..35.4) ===================
    t2_bg = mpatches.FancyBboxPatch((1.0, 1.4), 136.0, 34.0, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    facecolor="#FFFBEB", edgecolor="#D97706", lw=1.3, ls="--", alpha=0.58, zorder=1)
    ax.add_patch(t2_bg)
    ax.text(2.4, 33.7, "(b) Multi-Fidelity Parametric Surrogate (MF-SeaKAN) & Offshore Pipeline Screening",
            fontsize=11.4, fontweight="bold", color="#B45309", va="center", zorder=5)

    # Vertical shared-formulation link cleanly stopping at Tier 2 dashed boundary
    arrow(44.5, 39.2, 44.5, 35.6, color="#475569", lw=1.6)
    ax.text(46.0, 37.4, "Shared Biot BVP & pipe-normal map", fontsize=9.3, fontweight="bold", color="#334155", va="center",
            bbox=dict(facecolor="#FFFFFF", edgecolor="#CBD5E1", boxstyle="round,pad=0.22", lw=0.8), zorder=6)

    # 6. Parametric Input Tensor
    c_in2 = card(2.2, 2.9, 23.0, 28.4, "Observable Inputs", fc="#FFFFFF", ec="#D97706", title_fs=10.6, title_c="#92400E")
    tensor_col(3.5, 7.6, 5.2, 3.4, 5, [r"$x^*$", r"$y^*$", r"$T$", r"$\log_{10}k$", r"$b/D$"], fc="#FDBA74", header=r"$\boldsymbol{s}\in\mathbb{R}^5$", hdr_fs=10.0)
    t_r1 = ax.text(17.1, 22.3, r"$T\in[1.2,1.8]\ \mathrm{s}$", ha="center", fontsize=9.0, color="#1E293B", zorder=5)
    t_r2 = ax.text(17.1, 18.1, r"$k\in[1,8]\!\times\!10^{-5}$", ha="center", fontsize=9.0, color="#1E293B", zorder=5)
    t_r3 = ax.text(17.1, 13.9, r"$b/D\in[0.75,1.50]$", ha="center", fontsize=9.0, color="#1E293B", zorder=5)
    checked_pairs.extend([("r1", c_in2, t_r1), ("r2", c_in2, t_r2), ("r3", c_in2, t_r3)])
    pill(10.0, 5.2, 14.0, 5.0, "Scale-invariant\nin wave height $H$", fc="#FEF3C7", ec="#D97706", fs=8.2, tc="#78350F")

    # 7. Parallel Frozen Physics Trunk + Residual MF Corrector
    c_mf = card(27.5, 2.9, 55.6, 28.4, "Multi-Fidelity Surrogate Architecture", fc="#FFFFFF", ec="#D97706", title_fs=10.6, title_c="#92400E")
    # Upper branch: Frozen Physics-Only Trunk
    card(29.0, 17.2, 36.4, 11.2, r"Frozen Physics Trunk ($N_{\mathrm{cal}}=0$)", fc="#EFF6FF", ec="#2563EB", lw=1.1, title_y_off=1.55, title_fs=9.3, title_c="#1D4ED8")
    draw_kan_net(38.4, 21.5, 13.4, 5.0, [3, 4, 4, 2], node_fc="#2563EB", edge_c="#1E40AF", spline_wave=True)
    pill(48.0, 18.5, 16.0, 5.5, "MPM Weak PDE\n$R^2=0.964$ (Frozen)", fc="#DBEAFE", ec="#2563EB", fs=8.3, tc="#1E3A8A")

    # Lower branch: Data-Assisted Residual Corrector
    card(29.0, 4.3, 36.4, 11.2, r"Residual KAN Corrector $\Delta\widetilde{p}_{\mathrm{corr}}$", fc="#FFF7ED", ec="#EA580C", lw=1.1, title_y_off=1.55, title_fs=9.3, title_c="#C2410C")
    draw_kan_net(38.4, 8.6, 13.4, 5.0, [3, 4, 4, 2], node_fc="#EA580C", edge_c="#9A3412", spline_wave=True)
    pill(48.0, 5.6, 16.0, 5.5, "Calibrated on 32\nCOMSOL Fields", fc="#FFEDD5", ec="#EA580C", fs=8.3, tc="#9A3412")

    # BC Mask multiplication 'x' and Sum '+'
    op_circle(70.2, 9.9, "×", fc="#E0F2FE", ec="#0284C7", r=1.55)
    t_bc = ax.text(70.2, 5.4, r"$\phi_{\mathrm{s}}\phi_{\mathrm{lat}}$" + "\n(exact BCs)", ha="center", va="center", fontsize=8.3, color="#0369A1", fontweight="bold", zorder=5)
    checked_pairs.append(("bc_lbl", c_mf, t_bc))
    op_circle(78.2, 16.5, "+", fc="#FEF9C3", ec="#0F172A", r=1.65)
    poly_arrow([(25.4, 22.8), (29.0, 22.8)])
    poly_arrow([(25.4, 9.9), (29.0, 9.9)])
    arrow(65.6, 9.9, 68.5, 9.9, color="#C2410C")
    arrow(65.6, 22.8, 76.7, 17.7, color="#1D4ED8")
    arrow(71.7, 10.7, 76.8, 15.3, color="#C2410C")

    # 8. MF Output Tensor
    tensor_col(85.3, 11.1, 4.8, 3.6, 3, [r"$\widehat{p}_{\mathrm{MF}}$", r"$\nabla\widehat{p}$", r"$C_U$"], fc="#BAE6FD", header="MF Out", hdr_fs=9.8)
    arrow(80.0, 16.5, 85.1, 16.5)

    # 9. Ocean Engineering Screening & Inversion Output Card
    card(92.4, 2.9, 43.4, 28.4, "Engineering Screening & Inversion", fc="#FEF2F2", ec="#DC2626", title_fs=10.4, title_c="#991B1B")
    arrow(90.3, 16.5, 92.4, 16.5, color="#DC2626")

    # (i) 1,600-case Design Chart mini contour
    c_m1 = card(93.8, 16.2, 19.6, 12.1, r"1,600-Case $(k,b/D)$", fc="#FFFFFF", ec="#DC2626", lw=0.95, title_y_off=1.50, title_fs=8.9, title_c="#991B1B")
    ax.add_patch(mpatches.Rectangle((95.0, 17.4), 7.2, 7.2, facecolor="#FDE68A", edgecolor="#334155", lw=0.8, zorder=3))
    ax.add_patch(mpatches.Polygon([[95.0, 17.4], [99.0, 17.4], [95.0, 22.7]], facecolor="#EF4444", alpha=0.65, zorder=4))
    ax.plot([95.0, 99.8], [23.1, 17.4], color="#1E3A8A", ls="--", lw=1.2, zorder=5)
    t_m1 = ax.text(107.7, 21.1, "$C_U$ & $i_{\\max}$\n1.08 s online", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0F172A", zorder=5)
    checked_pairs.append(("m1_t", c_m1, t_m1))

    # (ii) 64-Sensor Fisher Inversion mini curve
    c_m2 = card(114.8, 16.2, 19.6, 12.1, "64-Sensor $k$ Inversion", fc="#FFFFFF", ec="#1D4ED8", lw=0.95, title_y_off=1.50, title_fs=8.9, title_c="#1E3A8A")
    ix = np.linspace(0, 1, 25)
    iy = (ix - 0.52) ** 2
    ax.plot(116.0 + 7.2 * ix, 18.0 + 24.0 * iy, color="#1D4ED8", lw=1.6, zorder=4)
    ax.plot([119.7, 119.7], [17.6, 24.2], color="#DC2626", ls="--", lw=1.1, zorder=5)
    t_m2 = ax.text(128.7, 21.1, "0.84–5.10%\n12.8 s/case", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0F172A", zorder=5)
    checked_pairs.append(("m2_t", c_m2, t_m2))

    # (iii) Bottom pills: Accuracy + Break-even
    pill(93.8, 4.4, 19.6, 10.0, "Held-out B1–B8:\n$L_{2,\\mathrm{rel}}=3.87\\%$\n$R^2=0.9950$", fc="#FFFFFF", ec="#16A34A", fs=8.9, tc="#14532D")
    pill(114.8, 4.4, 19.6, 10.0, "Break-even vs. FEM:\n$N^*\\!\\approx\\!195$–$266$\nCPU query: 18.7 ms", fc="#FFFFFF", ec="#B45309", fs=8.3, tc="#78350F")

    # Automated renderer bounding-box margin check: verify zero text touches or crosses any box border
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_pad_px = 3.0
    violations = []
    for name, patch, txt in checked_pairs:
        pb = patch.get_window_extent(renderer)
        tb = txt.get_window_extent(renderer)
        ml, mr, mb, mt = tb.x0 - pb.x0, pb.x1 - tb.x1, tb.y0 - pb.y0, pb.y1 - tb.y1
        if min(ml, mr, mb, mt) < min_pad_px:
            violations.append(f"{name}: margins(px) L={ml:.1f}, R={mr:.1f}, B={mb:.1f}, T={mt:.1f}")
    if violations:
        raise RuntimeError("Text overflow/tight-margin detected in architecture diagram:\n" + "\n".join(violations))

    _save_pdf(fig, "seakan_architecture_2d.pdf")


# ---------------------------------------------------------------------------
# Shared Helpers for Pipe Schematics & Polar Axes (Large Fonts, No Baked Titles)
# ---------------------------------------------------------------------------
def _draw_pipe_schematic_standalone(mode: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 4.15))
    ax.set_xlim(-1.60, 1.60)
    ax.set_ylim(-1.38, 1.24)
    ax.set_aspect("equal")
    ax.axis("off")

    if mode == "turcotte":
        ax.add_patch(mpatches.Rectangle((-1.52, 0.24), 3.04, 0.68, facecolor="#DBEAFE", edgecolor="none", zorder=1))
        ax.add_patch(mpatches.Rectangle((-1.52, -1.32), 3.04, 1.56, facecolor="#F5E6CA", edgecolor="none", zorder=1))
        ax.plot([-1.52, 1.52], [0.24, 0.24], color="#1D4ED8", lw=1.7, zorder=3)
        ax.text(-1.45, 0.31, "Mudline ($z=0$)", fontsize=11.0, color="#1E3A8A", fontweight="bold")
        yc, r_vis = -0.46, 0.36
    elif mode == "sun_test10":
        ax.add_patch(mpatches.Rectangle((-1.52, 0.28), 3.04, 0.64, facecolor="#DBEAFE", edgecolor="none", zorder=1))
        ax.add_patch(mpatches.Rectangle((-1.52, -1.32), 3.04, 1.60, facecolor="#F5E6CA", edgecolor="none", zorder=1))
        ax.plot([-1.46, -0.78, 0.78, 1.46], [0.28, -0.82, -0.82, 0.28], color="#78350F", ls="--", lw=1.1, zorder=2)
        ax.plot([-1.52, 1.52], [0.28, 0.28], color="#1D4ED8", lw=1.7, zorder=3)
        ax.text(-1.45, 0.35, "Backfill ($e=0.15\\ \\mathrm{m}$)", fontsize=10.8, color="#1E3A8A", fontweight="bold")
        yc, r_vis = -0.44, 0.36
    else:
        yc, r_vis = -0.44, 0.36
        ax.add_patch(mpatches.Rectangle((-1.52, -1.32), 3.04, 2.24, facecolor="#DBEAFE", edgecolor="none", zorder=0))
        tx = [-1.52, -1.52, -1.36, -0.80, 0.80, 1.36, 1.52, 1.52]
        ty = [-1.32, 0.28, 0.28, yc, yc, 0.28, 0.28, -1.32]
        ax.fill(tx, ty, facecolor="#F5E6CA", edgecolor="none", zorder=1)
        ax.plot([-1.52, -1.36, -0.80, 0.80, 1.36, 1.52], [0.28, 0.28, yc, yc, 0.28, 0.28], color="#92400E", lw=1.5, zorder=3)
        ax.add_patch(mpatches.Arc((0.0, yc), 2.30 * r_vis, 2.30 * r_vis, theta1=0, theta2=180, color="#0284C7", lw=3.0, zorder=5))
        ax.add_patch(mpatches.Arc((0.0, yc), 2.30 * r_vis, 2.30 * r_vis, theta1=180, theta2=360, color="#D97706", lw=3.0, zorder=5))
        ax.text(0.0, yc + r_vis + 0.16, "Exposed ($P$, water)", ha="center", fontsize=10.5, color="#0369A1", fontweight="bold")
        ax.text(0.0, yc - r_vis - 0.30, "Embedded ($p$, soil)", ha="center", fontsize=10.5, color="#92400E", fontweight="bold")

    wx = np.linspace(-1.46, 1.46, 120)
    wy = 0.84 + 0.08 * np.cos(2.6 * wx)
    ax.plot(wx, wy, color="#0284C7", lw=1.8, zorder=4)
    ax.annotate("", xy=(0.86, 1.07), xytext=(-0.86, 1.07),
                arrowprops=dict(arrowstyle="-|>,head_length=0.58,head_width=0.30", color="#0F172A", lw=1.5))
    ax.text(0.0, 1.12, "Wave ($+x$)", ha="center", va="bottom", fontsize=11.2, fontweight="bold")

    ax.add_patch(mpatches.Circle((0.0, yc), r_vis, facecolor="#CBD5E1", edgecolor="#1E293B", lw=1.8, zorder=4))
    ax.add_patch(mpatches.Circle((0.0, yc), 0.72 * r_vis, facecolor="#F8FAFC", edgecolor="#475569", lw=1.0, zorder=4))
    ax.text(0.0, yc, "Pipe", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1E293B", zorder=5)

    ax.plot([-r_vis * 1.40, r_vis * 1.40], [yc, yc], color="#334155", ls=":", lw=1.0, zorder=4)
    ax.plot([0.0, 0.0], [yc - r_vis * 1.40, yc + r_vis * 1.40], color="#334155", ls=":", lw=1.0, zorder=4)
    lbl_box = dict(facecolor="#FFFFFF", edgecolor="none", alpha=0.82, pad=0.8)
    y_lat = yc - (0.12 if mode == "sun_test49" else 0.0)
    ax.text(r_vis * 1.46, y_lat, r"$0$ (Lee)", ha="left", va="center", fontsize=10.4, fontweight="bold", color="#0F172A", bbox=lbl_box, zorder=6)
    ax.text(-r_vis * 1.46, y_lat, r"$\pi$ (Wave)", ha="right", va="center", fontsize=10.4, fontweight="bold", color="#0F172A", bbox=lbl_box, zorder=6)
    if mode != "sun_test49":
        ax.text(0.0, yc + r_vis * 1.48, r"$\pi/2$ (Crown)", ha="center", va="bottom", fontsize=10.6, fontweight="bold", color="#0F172A")
        ax.text(0.0, yc - r_vis * 1.48, r"$3\pi/2$ (Invert)", ha="center", va="top", fontsize=10.6, fontweight="bold", color="#0F172A")

    th_arc = mpatches.FancyArrowPatch(
        (0.52 * r_vis, yc + 0.05), (0.08, yc + 0.52 * r_vis),
        connectionstyle="arc3,rad=0.38", arrowstyle="-|>,head_length=3.5,head_width=2.2", color="#DC2626", lw=1.2, zorder=6,
    )
    ax.add_patch(th_arc)
    ax.text(0.42 * r_vis, yc + 0.39 * r_vis, r"$\theta$", fontsize=11.0, color="#DC2626", fontweight="bold", zorder=6)

    _save_pdf(fig, filename)


def _decorate_polar_axes(ax: plt.Axes, rmax: float = 1.05, is_test49: bool = False) -> None:
    ax.set_ylim(0.0, rmax)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10.8, color="#334155")
    ax.set_rlabel_position(30)
    angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    labels = [
        r"$0$" + "\n(Lee)",
        r"$\pi/4$",
        r"$\pi/2$ (Crown)",
        r"$3\pi/4$",
        r"$\pi$" + "\n(Wave)",
        r"$5\pi/4$",
        r"$3\pi/2$ (Invert)",
        r"$7\pi/4$",
    ]
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11.0)
    ax.tick_params(axis="x", pad=4)
    ax.grid(True, color="#CBD5E1", lw=0.75)
    if is_test49:
        th_up = np.linspace(0, math.pi, 100)
        th_dn = np.linspace(math.pi, 2 * math.pi, 100)
        ax.fill_between(th_up, 0.99, rmax, color="#38BDF8", alpha=0.32, zorder=1)
        ax.fill_between(th_dn, 0.99, rmax, color="#F59E0B", alpha=0.32, zorder=1)
        ax.plot([0, math.pi], [rmax, rmax], color="#92400E", lw=1.4, ls="--", zorder=2)


def _save_shared_legend(handles, labels, filename: str, *, ncol: int = 3, figsize=(9.2, 0.72), fontsize: float = 10.8) -> None:
    fig = plt.figure(figsize=figsize)
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=True, fontsize=fontsize, columnspacing=1.3, handlelength=2.2)
    _save_pdf(fig, filename)


# ---------------------------------------------------------------------------
# Figure 5 (Split into 2 subfigures + shared legend)
# ---------------------------------------------------------------------------
def make_figure5_split() -> None:
    _draw_pipe_schematic_standalone("turcotte", "fig5a_buried_geom.pdf")

    fig9_csv = CODE_ROOT / "figure9_mpm_kan" / "data" / "figure9.csv"
    prof_csv = CODE_ROOT / "figure9_mpm_kan" / "artifacts" / "top_gap_augmented" / "figure9_top_gap_augmented_profiles.csv"
    ext, _ = load_figure9_data(fig9_csv)
    seakan_df = pd.read_csv(prof_csv)
    seakan_df = seakan_df[seakan_df["source"] == "mpm_kan_pinn"]

    fig = plt.figure(figsize=(5.0, 4.45))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    _decorate_polar_axes(ax, rmax=1.02, is_test49=False)

    for key, label, ls, col, lw in [
        ("poroelasticity", "Cheng & Liu (1986)", "-.", "#64748B", 1.3),
        ("darcy", "Darcy solution", ":", "#94A3B8", 1.35),
        ("jeng_numerical", "Jeng (2001) FEM", "--", "#475569", 1.3),
        ("paper_this_study", "Zhu et al. (2024) FEM", "--", "#EA580C", 1.8),
    ]:
        th = np.append(ext[key]["theta"], ext[key]["theta"][0])
        rr = np.append(ext[key]["radius"], ext[key]["radius"][0])
        ax.plot(th, rr, linestyle=ls, color=col, lw=lw, label=label, zorder=3)

    ax.plot(seakan_df["theta_rad"], seakan_df["normalized_pressure"], color="#1D4ED8", lw=2.3, label="SeaKAN (physics-only)", zorder=4)
    ax.scatter(ext["experimental"]["theta"], ext["experimental"]["radius"], s=48, facecolor="#FFFFFF", edgecolor="#0F172A", lw=1.5, label="Turcotte et al. (1984) exp.", zorder=6)

    h, l = ax.get_legend_handles_labels()
    _save_pdf(fig, "fig5b_buried_polar.pdf")
    _save_shared_legend(h, l, "fig5_legend.pdf", ncol=3, figsize=(9.4, 0.75), fontsize=10.8)


# ---------------------------------------------------------------------------
# Figure 6 (Split into 4 subfigures + shared legend)
# ---------------------------------------------------------------------------
def make_figure6_split() -> None:
    _draw_pipe_schematic_standalone("sun_test10", "fig6a_t10_geom.pdf")
    _draw_pipe_schematic_standalone("sun_test49", "fig6b_t49_geom.pdf")

    t10_csv = CODE_ROOT / "figure15_mpm_kan" / "data" / "figure15a.csv"
    t49_csv = CODE_ROOT / "figure15_mpm_kan" / "data" / "figure15b.csv"
    prof_csv = CODE_ROOT / "figure15_mpm_kan" / "artifacts" / "figure15_profiles.csv"
    ext, _ = load_figure15_data(t10_csv, t49_csv)
    prof_df = pd.read_csv(prof_csv)

    # (c) Test 10 polar
    fig10 = plt.figure(figsize=(5.0, 4.45))
    ax10 = fig10.add_subplot(1, 1, 1, projection="polar")
    _decorate_polar_axes(ax10, rmax=1.06, is_test49=False)
    s10 = prof_df[(prof_df["case"] == "test10") & (prof_df["source"] == "mpm_kan_combined_pressure")]
    p10_num, p10_exp = ext["test10"]["pore_numerical"], ext["test10"]["experimental"]
    ax10.plot(np.append(p10_num["theta"], p10_num["theta"][0]), np.append(p10_num["radius"], p10_num["radius"][0]),
              color="#EA580C", ls="--", lw=1.9, label="Zhu et al. (2024) pore pressure ($p$)", zorder=3)
    ax10.plot(s10["theta_rad"], s10["normalized_pressure"], color="#1D4ED8", lw=2.3, label="SeaKAN prediction", zorder=4)
    ax10.scatter(p10_exp["theta"], p10_exp["radius"], s=48, facecolor="#FFFFFF", edgecolor="#0F172A", lw=1.5, label="Sun et al. (2019) exp.", zorder=6)
    h1, l1 = ax10.get_legend_handles_labels()
    _save_pdf(fig10, "fig6c_t10_polar.pdf")

    # (d) Test 49 polar
    fig49 = plt.figure(figsize=(5.0, 4.45))
    ax49 = fig49.add_subplot(1, 1, 1, projection="polar")
    _decorate_polar_axes(ax49, rmax=1.06, is_test49=True)
    s49 = prof_df[(prof_df["case"] == "test49") & (prof_df["source"] == "mpm_kan_combined_pressure")]
    w49_num, p49_num, p49_exp = ext["test49"]["hydrodynamic_numerical"], ext["test49"]["pore_numerical"], ext["test49"]["experimental"]
    ax49.plot(w49_num["theta"], w49_num["radius"], color="#0284C7", ls=":", lw=2.2, label="Zhu et al. (2024) hydrodynamic ($P$)", zorder=3)
    ax49.plot(p49_num["theta"], p49_num["radius"], color="#EA580C", ls="--", lw=1.9, zorder=3)
    ax49.plot(s49["theta_rad"], s49["normalized_pressure"], color="#1D4ED8", lw=2.3, zorder=4)
    ax49.scatter(p49_exp["theta"], p49_exp["radius"], s=48, facecolor="#FFFFFF", edgecolor="#0F172A", lw=1.5, zorder=6)
    h2, l2 = ax49.get_legend_handles_labels()
    _save_pdf(fig49, "fig6d_t49_polar.pdf")

    _save_shared_legend(h1 + h2, l1 + l2, "fig6_legend.pdf", ncol=2, figsize=(9.2, 0.72), fontsize=10.8)


# ---------------------------------------------------------------------------
# Figure 7 (Split into 2 subfigures + shared legend)
# ---------------------------------------------------------------------------
def make_figure7_split() -> None:
    comp_dir = CODE_ROOT / "figure9_model_comparison" / "outputs"
    prof_df = pd.read_csv(comp_dir / "profiles.csv")
    metrics_df = pd.read_csv(comp_dir / "metrics.csv")
    kinn_prof = pd.read_csv(EXP1_DIR / "standard_kinn" / "profile.csv")
    kinn_m = json.loads((EXP1_DIR / "exp1_complete_results.json").read_text(encoding="utf-8"))["standard_kinn"]["metrics"]
    ext, _ = load_figure9_data(CODE_ROOT / "figure9_mpm_kan" / "data" / "figure9.csv")

    methods_spec = [
        ("deepxde_pinn", "DeepXDE-PINN", "#64748B", (0, (4, 2)), 1.35),
        ("gpinn_rar", "gPINN-RAR", "#E11D48", (0, (5, 1, 1, 1)), 1.35),
        ("xpinn", "XPINN", "#9333EA", (0, (2, 1.2)), 1.4),
        ("standard_kinn", "KINN", "#0D9488", (0, (6, 1.5, 1.5, 1.5)), 1.55),
        ("pino", "PINO (26 FD fields)", "#D97706", (0, (6, 2)), 1.6),
        ("seakan", "SeaKAN (proposed)", "#1D4ED8", "solid", 2.3),
    ]

    fig_p = plt.figure(figsize=(5.0, 4.45))
    ax_p = fig_p.add_subplot(1, 1, 1, projection="polar")
    _decorate_polar_axes(ax_p, rmax=1.75, is_test49=False)
    ax_p.set_yticks([0.4, 0.8, 1.2, 1.6])
    ax_p.set_yticklabels(["0.4", "0.8", "1.2", "1.6"], fontsize=10.5)

    exp_l2_vals, zhu_l2_vals, bar_labels = [], [], []
    for key, label, col, ls, lw in methods_spec:
        if key == "standard_kinn":
            th, pr = kinn_prof["theta_rad"].to_numpy(), kinn_prof["normalized_pressure"].to_numpy()
            e_l2, z_l2 = 100.0 * kinn_m["experimental"]["relative_l2"], 100.0 * kinn_m["zhu_numerical"]["relative_l2"]
        else:
            sub = prof_df[prof_df["method"] == key]
            th, pr = sub["theta_rad"].to_numpy(), sub["normalized_pressure"].to_numpy()
            e_l2 = 100.0 * float(metrics_df[(metrics_df["method"] == key) & (metrics_df["reference"] == "experimental")]["relative_l2"].iloc[0])
            z_l2 = 100.0 * float(metrics_df[(metrics_df["method"] == key) & (metrics_df["reference"] == "zhu_numerical")]["relative_l2"].iloc[0])
        ax_p.plot(th, pr, color=col, linestyle=ls, lw=lw, label=label, zorder=5 if key == "seakan" else 3)
        bar_labels.append(label.replace(" (26 FD fields)", "*").replace(" (proposed)", ""))
        exp_l2_vals.append(e_l2)
        zhu_l2_vals.append(z_l2)

    ax_p.plot(np.append(ext["paper_this_study"]["theta"], ext["paper_this_study"]["theta"][0]),
              np.append(ext["paper_this_study"]["radius"], ext["paper_this_study"]["radius"][0]),
              color="#EA580C", ls="--", lw=1.65, label="Zhu et al. (2024) FEM", zorder=4)
    ax_p.scatter(ext["experimental"]["theta"], ext["experimental"]["radius"], s=46,
                 facecolor="#FFFFFF", edgecolor="#0F172A", lw=1.5, label="Turcotte et al. (1984) exp.", zorder=6)
    h, l = ax_p.get_legend_handles_labels()
    _save_pdf(fig_p, "fig7a_comp_polar.pdf")
    _save_shared_legend(h, l, "fig7_legend.pdf", ncol=4, figsize=(9.6, 0.78), fontsize=10.2)

    fig_b, ax_b = plt.subplots(figsize=(5.3, 4.35))
    y_pos = np.arange(len(bar_labels))
    h_bar = 0.37
    ax_b.barh(y_pos - h_bar / 2, exp_l2_vals, height=h_bar, color="#3B82F6", edgecolor="#1E3A8A", lw=0.7, label="vs. Experiment ($N=8$)")
    ax_b.barh(y_pos + h_bar / 2, zhu_l2_vals, height=h_bar, color="#F59E0B", edgecolor="#92400E", lw=0.7, label="vs. Zhu FEM ($N=163$)")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(bar_labels, fontsize=11.2)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"Relative $L_2$ metric $L_{2,\mathrm{rel}}$ (%)", fontsize=12.5)
    ax_b.grid(axis="x", color="#E2E8F0", lw=0.8)
    ax_b.legend(loc="lower right", fontsize=10.5, frameon=True)
    max_x = max(max(exp_l2_vals), max(zhu_l2_vals))
    ax_b.set_xlim(0, max_x * 1.24)
    for i, (ev, zv) in enumerate(zip(exp_l2_vals, zhu_l2_vals)):
        ax_b.text(ev + max_x * 0.015, i - h_bar / 2, f"{ev:.1f}%", va="center", fontsize=10.2, color="#1E3A8A", fontweight="bold")
        ax_b.text(zv + max_x * 0.015, i + h_bar / 2, f"{zv:.1f}%", va="center", fontsize=10.2, color="#92400E", fontweight="bold")
    fig_b.tight_layout()
    _save_pdf(fig_b, "fig7b_comp_bar.pdf")


# ---------------------------------------------------------------------------
# Figure 8 (Split into 4 standalone subfigures)
# ---------------------------------------------------------------------------
def make_figure8_split() -> None:
    import torch
    from matplotlib.tri import Triangulation
    from exp2_mf_seakan_and_costs import (
        CleanParametricSeaKAN,
        ParameterDomain,
        PhysicalConstants,
        PressureCorrectorKAN,
        load_all_comsol_cases,
    )

    constants, domain = PhysicalConstants(), ParameterDomain()
    splits = load_all_comsol_cases(constants)
    exp2 = json.loads((EXP2_DIR / "exp2_complete_results.json").read_text(encoding="utf-8"))

    best_sd = min(exp2["physics_trunk_3seed"], key=lambda r: r["val_complex_l2_pct"])["seed"]
    trunk = CleanParametricSeaKAN(constants, domain, dtype=torch.float32)
    trunk.load_state_dict(torch.load(EXP2_DIR / f"clean_trunk_seed_{best_sd}.pt", map_location="cpu", weights_only=True))
    trunk.eval()

    mf = PressureCorrectorKAN(constants, domain, trunk, width=24, hidden_layers=2, n_basis=7)
    mf.load_state_dict(torch.load(EXP2_DIR / "selected_mf32_model.pt", map_location="cpu", weights_only=True))
    mf.eval()

    c0 = splits["validation"][0]
    xy_t = c0["xy"]
    n = xy_t.shape[0]
    tkb_t = torch.tensor([[c0["T"], c0["k"], c0["bD"]]], dtype=torch.float32).expand(n, -1)
    kap_t = torch.full((n,), float(c0["kappa"]), dtype=torch.float32)
    cy_t = torch.full((n,), float(c0["pipe_center_y"]), dtype=torch.float32)
    with torch.no_grad():
        p_trunk = trunk(xy_t, tkb_t, kap_t, cy_t)[2].abs().numpy()
        p_mf = mf.forward_pressure(xy_t, tkb_t, kap_t, cy_t).abs().numpy()
    p_ref = c0["target"].abs().numpy()

    x_m = xy_t[:, 0].numpy() * constants.seabed_thickness_m
    y_m = xy_t[:, 1].numpy() * constants.seabed_thickness_m
    yc_m = float(c0["pipe_center_y"]) * constants.seabed_thickness_m
    r_m = 0.5 * constants.pipe_diameter_m

    tri = Triangulation(x_m, y_m)
    xc_tri = x_m[tri.triangles].mean(axis=1)
    yc_tri = y_m[tri.triangles].mean(axis=1)
    tri.set_mask((xc_tri**2 + (yc_tri - yc_m) ** 2) < (1.02 * r_m) ** 2)
    levels = np.linspace(0.0, 1.02, 18)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for vals, fname in [
        (p_ref, "fig8a_comsol_ref.pdf"),
        (p_trunk, "fig8b_physics_trunk.pdf"),
        (p_mf, "fig8c_mf_seakan.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 2.35))
        cf = ax.tricontourf(tri, np.clip(vals, 0.0, 1.02), levels=levels, cmap="viridis")
        ax.add_patch(mpatches.Circle((0.0, yc_m), r_m, facecolor="#F8FAFC", edgecolor="#0F172A", lw=1.4, zorder=5))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.0, 0.58)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (m)", fontsize=12.5)
        ax.set_ylabel("Elevation $y$ (m)", fontsize=12.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.6%", pad=0.10)
        cb = fig.colorbar(cf, cax=cax, ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
        cb.set_label(r"$|\widehat{p}|/p_0$", fontsize=12.0)
        cb.ax.tick_params(labelsize=10.5)
        fig.tight_layout()
        _save_pdf(fig, fname)

    val_tr = exp2["physics_trunk_validation_10cases"]["cases"]
    val_mf = exp2["mf32_validation_10cases"]["cases"]
    tst_tr = exp2["physics_trunk_test_8cases"]["cases"]
    tst_mf = exp2["mf32_test_8cases"]["cases"]
    x_idx = np.arange(len(val_mf) + len(tst_mf))
    c_labels = [f"V{i + 1}" for i in range(len(val_mf))] + [f"B{i + 1}" for i in range(len(tst_mf))]
    tr_l2 = [100.0 * r["amplitude_l2"] for r in val_tr + tst_tr]
    mf_l2 = [100.0 * r["amplitude_l2"] for r in val_mf + tst_mf]

    fig_d, ax_d = plt.subplots(figsize=(5.6, 2.55))
    w_b = 0.40
    ax_d.bar(x_idx - w_b / 2, tr_l2, width=w_b, color="#0D9488", edgecolor="#115E59", lw=0.6, label=r"Physics Trunk ($N_{\mathrm{cal}}=0$)")
    ax_d.bar(x_idx + w_b / 2, mf_l2, width=w_b, color="#1D4ED8", edgecolor="#1E3A8A", lw=0.6, label=r"MF-SeaKAN ($N_{\mathrm{cal}}=32$)")
    ax_d.axvline(len(val_mf) - 0.5, color="#64748B", ls="--", lw=1.1)
    ax_d.set_ylim(0.0, 18.8)
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels(c_labels, rotation=48, ha="right", fontsize=10.0)
    ax_d.set_ylabel(r"Amplitude $L_{2,\mathrm{rel}}$ (%)", fontsize=12.0)
    ax_d.grid(axis="y", color="#E2E8F0", lw=0.75)
    ax_d.legend(loc="upper right", ncol=2, fontsize=9.2, frameon=True, columnspacing=0.9)
    fig_d.tight_layout()
    _save_pdf(fig_d, "fig8d_casewise_l2.pdf")


# ---------------------------------------------------------------------------
# Figure 9 (Split into 2 standalone subfigures)
# ---------------------------------------------------------------------------
def make_figure9_split() -> None:
    exp2 = json.loads((EXP2_DIR / "exp2_complete_results.json").read_text(encoding="utf-8"))
    inv_rows = exp2["inverse_comsol_cases"]
    noise_rows = exp2["inverse_noise_study"]

    hom_rows = [r for r in inv_rows if not r["case_id"].endswith("_het")]
    het_rows = [r for r in inv_rows if r["case_id"].endswith("_het")]

    fig1, ax1 = plt.subplots(figsize=(5.1, 4.25))
    lims = [0.8, 8.5]
    ax1.plot(lims, lims, color="#0F172A", ls="--", lw=1.4, label="1:1 exact")
    ax1.fill_between(lims, [0.9 * v for v in lims], [1.1 * v for v in lims], color="#DBEAFE", alpha=0.6, label=r"$\pm 10\%$ band")
    ax1.scatter([r["k_true"] * 1e5 for r in hom_rows], [r["k_inv"] * 1e5 for r in hom_rows],
                s=64, color="#1D4ED8", edgecolor="#0F172A", lw=1.0, zorder=5, label="Homogeneous (B1–B8)")
    ax1.scatter([r["k_true"] * 1e5 for r in het_rows], [r["k_inv"] * 1e5 for r in het_rows],
                s=72, marker="^", color="#D97706", edgecolor="#0F172A", lw=1.0, zorder=6, label=r"Heterogeneous ($\mathrm{COV}_k=0.35$)")
    for idx_h, r in enumerate(hom_rows):
        lbl = f"B{idx_h + 1}"
        off = (-22, 7) if lbl == "B8" else ((6, -12) if lbl == "B1" else ((6, -10) if lbl == "B7" else (5, 4)))
        ax1.annotate(lbl, (r["k_true"] * 1e5, r["k_inv"] * 1e5), xytext=off, textcoords="offset points", fontsize=10.2, fontweight="bold")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(r"True $k$ ($10^{-5}\ \mathrm{m/s}$)", fontsize=12.8)
    ax1.set_ylabel(r"Inverted $\widehat{k}$ ($10^{-5}\ \mathrm{m/s}$)", fontsize=12.8)
    ax1.grid(True, color="#E2E8F0", lw=0.75)
    ax1.legend(loc="upper left", fontsize=10.0, frameon=True)
    fig1.tight_layout()
    _save_pdf(fig1, "fig9a_inv_parity.pdf")

    fig2, ax2 = plt.subplots(figsize=(5.1, 4.25))
    for cid, col, mk, lbl in [
        ("blind_000", "#1D4ED8", "o", r"B1 ($k=1.76\times 10^{-5}\ \mathrm{m/s}$)"),
        ("blind_001", "#EA580C", "s", r"B2 ($k=7.39\times 10^{-5}\ \mathrm{m/s}$)"),
    ]:
        sub = [r for r in noise_rows if r["case_id"] == cid]
        ax2.errorbar([r["noise_pct"] for r in sub], [r["E_k_pct_mean"] for r in sub], yerr=[r["E_k_pct_std"] for r in sub],
                     marker=mk, ms=6.8, lw=2.0, capsize=4, color=col, label=lbl)
    ax2.set_xticks([0, 1, 2, 5])
    ax2.set_xlabel("Additive Gaussian sensor noise (%)", fontsize=12.8)
    ax2.set_ylabel(r"Relative error $E_k$ (%)", fontsize=12.8)
    ax2.grid(True, color="#E2E8F0", lw=0.75)
    ax2.legend(loc="lower left", fontsize=10.2, frameon=True)
    fig2.tight_layout()
    _save_pdf(fig2, "fig9b_inv_noise.pdf")


# ---------------------------------------------------------------------------
# Figure 10 (Split into 2 standalone subfigures)
# ---------------------------------------------------------------------------
def make_figure10_split() -> None:
    exp1 = json.loads((EXP1_DIR / "exp1_complete_results.json").read_text(encoding="utf-8"))
    exp2 = json.loads((EXP2_DIR / "exp2_complete_results.json").read_text(encoding="utf-8"))

    ab_map = {r["variant"]: r for r in exp1["ablation_summary"]}
    order = [
        ("full_seakan", "Full SeaKAN"),
        ("with_strong_p2g", "+ Strong P2G"),
        ("no_local_branch", "w/o Local Hermite"),
        ("no_weak_precond", "w/o Weak pre-cond."),
        ("no_crown_refinement", "w/o Crown refine"),
        ("mlp_backbone", "Matched MLP"),
    ]

    fig1, ax1 = plt.subplots(figsize=(5.4, 4.3))
    y = np.arange(len(order))
    means_exp = [ab_map[k]["exp_l2_pct_mean"] for k, _ in order]
    stds_exp = [ab_map[k]["exp_l2_pct_std"] for k, _ in order]
    means_zhu = [ab_map[k]["zhu_l2_pct_mean"] for k, _ in order]
    stds_zhu = [ab_map[k]["zhu_l2_pct_std"] for k, _ in order]
    r2_exp = [ab_map[k]["exp_r2_mean"] for k, _ in order]

    h = 0.36
    ax1.barh(y - h / 2, means_exp, xerr=stds_exp, height=h, capsize=3.0, color="#2563EB", edgecolor="#1E3A8A", lw=0.7, label="vs. Exp. ($N=8$)")
    ax1.barh(y + h / 2, means_zhu, xerr=stds_zhu, height=h, capsize=3.0, color="#F59E0B", edgecolor="#92400E", lw=0.7, label="vs. Zhu FEM ($N=163$)")
    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, lbl in order], fontsize=11.0)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlim(6.0, 620.0)
    ax1.set_xticks([10, 20, 50, 100, 200])
    ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.set_xlabel(r"$L_{2,\mathrm{rel}}$ (%) [mean $\pm$ SD, 3 seeds]", fontsize=12.4)
    ax1.grid(axis="x", color="#E2E8F0", lw=0.75)
    for idx, (me, se, mz, sz, r2v) in enumerate(zip(means_exp, stds_exp, means_zhu, stds_zhu, r2_exp)):
        right_edge = max(me + se, mz + sz)
        ax1.text(right_edge * 1.12, idx, f"$R^2={r2v:.3f}$", va="center", fontsize=10.0, fontweight="bold", color="#0F172A")
    ax1.legend(loc="upper right", fontsize=10.0, frameon=True)
    fig1.tight_layout()
    _save_pdf(fig1, "fig10a_ablation.pdf")

    curve = exp2["sample_efficiency_curve"]
    trunk_3 = exp2["physics_trunk_3seed"]
    trunk_mean = float(np.mean([r["test8_amp_l2_pct"] for r in trunk_3]))
    trunk_r2 = float(np.mean([r["test8_amp_r2"] for r in trunk_3]))
    n_cals = [4, 8, 16, 32]
    pd_m = [next(r["test8_amp_l2_pct_mean"] for r in curve if r["mode"] == "pure_data_kan" and r["n_cal"] == n) for n in n_cals]
    pd_s = [next(r["test8_amp_l2_pct_std"] for r in curve if r["mode"] == "pure_data_kan" and r["n_cal"] == n) for n in n_cals]
    mf_m = [next(r["test8_amp_l2_pct_mean"] for r in curve if r["mode"] == "mf_seakan" and r["n_cal"] == n) for n in n_cals]
    mf_s = [next(r["test8_amp_l2_pct_std"] for r in curve if r["mode"] == "mf_seakan" and r["n_cal"] == n) for n in n_cals]

    fig2, ax2 = plt.subplots(figsize=(5.1, 4.3))
    ax2.axhline(trunk_mean, color="#0D9488", ls="--", lw=1.7, label=f"Physics trunk ($N_{{\\mathrm{{cal}}}}=0, R^2={trunk_r2:.3f}$)")
    ax2.scatter([0], [trunk_mean], color="#0D9488", marker="D", s=58, zorder=5)
    ax2.errorbar(n_cals, pd_m, yerr=pd_s, marker="s", ms=6.8, lw=1.9, capsize=3.5, color="#DC2626", label="Pure data-driven KAN")
    ax2.errorbar(n_cals, mf_m, yerr=mf_s, marker="o", ms=6.8, lw=2.1, capsize=3.5, color="#1D4ED8", label="MF-SeaKAN")
    ax2.set_xticks([0, 4, 8, 16, 32])
    ax2.set_xlabel(r"COMSOL calibration fields $N_{\mathrm{cal}}$", fontsize=12.6)
    ax2.set_ylabel(r"Held-out 8-case $L_{2,\mathrm{rel}}$ (%)", fontsize=12.6)
    ax2.grid(True, color="#E2E8F0", lw=0.75)
    ax2.legend(loc="upper right", fontsize=10.0, frameon=True)
    fig2.tight_layout()
    _save_pdf(fig2, "fig10b_sample_eff.pdf")


# ---------------------------------------------------------------------------
# Figure 11 (Split into 4 standalone subfigures)
# ---------------------------------------------------------------------------
def make_figure11_split() -> None:
    exp2 = json.loads((EXP2_DIR / "exp2_complete_results.json").read_text(encoding="utf-8"))
    npz = np.load(EXP2_DIR / "design_chart_1600.npz")
    k_grid, bD_grid = npz["k_grid"] * 1e5, npz["bD_grid"]
    C_uplift, i_max = npz["C_uplift"], npz["i_max"]
    KK, BB = np.meshgrid(k_grid, bD_grid)

    # (a) Uplift C_U
    fig1, ax1 = plt.subplots(figsize=(5.2, 4.15))
    cf1 = ax1.contourf(KK, BB, C_uplift, levels=14, cmap="YlGnBu")
    cs1 = ax1.contour(KK, BB, C_uplift, levels=6, colors="#0F172A", linewidths=0.8)
    ax1.clabel(cs1, inline=True, fontsize=9.8, fmt="%.2f")
    cb1 = fig1.colorbar(cf1, ax=ax1, pad=0.025)
    cb1.set_label(r"Cyclic uplift $C_U$", fontsize=12.0)
    cb1.ax.tick_params(labelsize=10.5)
    ax1.set_xlabel(r"Hydraulic conductivity $k$ ($10^{-5}\ \mathrm{m/s}$)", fontsize=12.4)
    ax1.set_ylabel(r"Burial ratio $b/D$", fontsize=12.4)
    fig1.tight_layout()
    _save_pdf(fig1, "fig11a_uplift_chart.pdf")

    # (b) Crown gradient i_max
    fig2, ax2 = plt.subplots(figsize=(5.2, 4.15))
    cf2 = ax2.contourf(KK, BB, i_max, levels=14, cmap="YlOrRd")
    cs2 = ax2.contour(KK, BB, i_max, levels=6, colors="#450A0A", linewidths=0.8)
    ax2.clabel(cs2, inline=True, fontsize=9.8, fmt="%.2f")
    cs_allow = ax2.contour(KK, BB, i_max, levels=[0.66], colors="#1E3A8A", linestyles="--", linewidths=2.2)
    ax2.annotate(
        r"$i_{\mathrm{allow}}=0.66$",
        xy=(1.95, 1.26),
        xytext=(3.25, 1.36),
        arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28", color="#1E3A8A", lw=1.4),
        fontsize=10.2,
        fontweight="bold",
        color="#1E3A8A",
        bbox=dict(facecolor="#FFFFFF", edgecolor="#93C5FD", boxstyle="round,pad=0.22", lw=0.9, alpha=0.95),
        zorder=7,
    )
    cb2 = fig2.colorbar(cf2, ax=ax2, pad=0.025)
    cb2.set_label(r"Peak crown gradient $i_{\max}$", fontsize=12.0)
    cb2.ax.tick_params(labelsize=10.5)
    ax2.set_xlabel(r"Hydraulic conductivity $k$ ($10^{-5}\ \mathrm{m/s}$)", fontsize=12.4)
    ax2.set_ylabel(r"Burial ratio $b/D$", fontsize=12.4)
    fig2.tight_layout()
    _save_pdf(fig2, "fig11b_gradient_chart.pdf")

    # (c) Turcotte external wave-flume inversion
    lab = exp2["inverse_turcotte_lab"]
    k_scan = (10.0 ** np.array(lab["grid_log10_k"])) * 1e3
    rmse_scan = np.sqrt(np.array(lab["grid_mse"]))
    k_true_mm = lab["k_true_lab_m_per_s"] * 1e3
    k_inv_mm = lab["k_inverted_m_per_s"] * 1e3

    fig3, ax3 = plt.subplots(figsize=(5.2, 4.15))
    ax3.plot(k_scan, rmse_scan, color="#1D4ED8", lw=2.3, label="Physics-only SeaKAN RMSE")
    ax3.axvline(k_true_mm, color="#0F172A", ls="--", lw=1.7, label=f"Measured $k={k_true_mm:.2f}\\times 10^{{-3}}$")
    ax3.axvline(k_inv_mm, color="#DC2626", ls="-.", lw=1.8, label=f"Inverted $\\widehat{{k}}={k_inv_mm:.2f}\\times 10^{{-3}}$ ({lab['E_k_pct']:.1f}%)")
    ax3.set_xscale("log")
    ax3.set_xticks([0.3, 0.5, 1.0, 2.0, 3.0])
    ax3.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax3.set_ylim(0.010, 0.225)
    ax3.set_xlabel(r"Candidate $k$ ($10^{-3}\ \mathrm{m/s}$)", fontsize=12.4)
    ax3.set_ylabel("8-transducer amplitude RMSE", fontsize=12.4)
    ax3.grid(True, which="both", color="#E2E8F0", lw=0.75)
    ax3.legend(loc="upper right", fontsize=9.8, frameon=True)
    fig3.tight_layout()
    _save_pdf(fig3, "fig11c_turcotte_inv.pdf")

    # (d) Matched-accuracy break-even cost
    cost = exp2["cost_and_design_summary"]
    t_fine = cost["comsol_fine_per_case_wall_s_mean"]
    t_med = cost["comsol_medium_matched_per_case_wall_s_mean"]
    t_off = cost["total_offline_s_including_32_comsol"]
    t_trunk = cost["clean_trunk_train_s"]
    t_gpu = cost["mf_seakan_matched_mesh_gpu_ms_mean"] / 1000.0
    n_q = np.logspace(0, 3.3, 200)

    fig4, ax4 = plt.subplots(figsize=(5.2, 4.15))
    ax4.plot(n_q, n_q * t_fine, color="#0F172A", lw=2.2, label=f"COMSOL Fine ({t_fine:.2f} s/case)")
    ax4.plot(n_q, n_q * t_med, color="#475569", ls=":", lw=2.0, label=f"COMSOL Medium ({t_med:.2f} s/case)")
    ax4.plot(n_q, t_off + n_q * t_gpu, color="#1D4ED8", lw=2.2, label=r"MF-SeaKAN ($N_{\mathrm{cal}}=32$)")
    ax4.plot(n_q, t_trunk + n_q * t_gpu, color="#0D9488", ls="--", lw=2.0, label=r"Physics trunk ($N_{\mathrm{cal}}=0$)")

    n_be_fine = cost["forward_breakeven_vs_fine_comsol_gpu"]
    n_be_med = cost["forward_breakeven_vs_matched_medium_comsol_gpu"]
    ax4.scatter([n_be_fine, n_be_med], [n_be_fine * t_fine, n_be_med * t_med], color="#DC2626", s=54, zorder=6)
    ax4.annotate(
        f"$N^*\\!\\approx\\!{int(round(n_be_fine))}$–${int(round(n_be_med))}$",
        xy=(n_be_med, n_be_med * t_med),
        xytext=(430.0, 255.0),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.48,head_width=0.26", color="#DC2626", lw=1.35),
        fontsize=10.6, fontweight="bold", color="#991B1B",
        bbox=dict(facecolor="#FFFFFF", edgecolor="#FCA5A5", boxstyle="round,pad=0.22", lw=0.85, alpha=0.95),
        zorder=7,
    )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel(r"Forward field evaluations $N_{\mathrm{query}}$", fontsize=12.4)
    ax4.set_ylabel("Cumulative wall-clock time (s)", fontsize=12.4)
    ax4.grid(True, which="both", color="#E2E8F0", lw=0.75)
    ax4.legend(loc="upper left", fontsize=9.6, frameon=True)
    fig4.tight_layout()
    _save_pdf(fig4, "fig11d_cost_breakeven.pdf")


FIGURES = {
    1: make_figure1_domain_schematic,
    2: make_architecture_figure,
    5: make_figure5_split,
    6: make_figure6_split,
    7: make_figure7_split,
    8: make_figure8_split,
    9: make_figure9_split,
    10: make_figure10_split,
    11: make_figure11_split,
}


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(
        description="Render the manuscript's revised PDF figure panels from numerical results."
    )
    parser.add_argument("--figures", type=int, nargs="+", choices=tuple(FIGURES),
                        default=list(FIGURES), metavar="N",
                        help="Figure numbers (default: 1 2 5 6 7 8 9 10 11).")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help="Directory for generated PDFs; manuscript files are never edited.")
    args = parser.parse_args()
    OUT_DIR = args.output_dir.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number in args.figures:
        FIGURES[number]()


if __name__ == "__main__":
    main()
