"""
Recompute Table 8 (Observability and sensitivity condition-number screening)
directly using the frozen MF-SeaKAN surrogate (selected_mf32_model.pt)
from the clean sealed-bottom COMSOL 6.4 suite.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import torch
from torch import nn

EXP2_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP2_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP2_DIR))

from exp2_mf_seakan_and_costs import (  # noqa: E402
    CleanParametricSeaKAN,
    PressureCorrectorKAN,
    load_all_comsol_cases,
    select_fisher_sensors_clean,
    sq_abs,
)
from mpm_kan_generalization.config import ParameterDomain, PhysicalConstants  # noqa: E402


def _load_mf_model(constants: PhysicalConstants, domain: ParameterDomain, device: torch.device) -> PressureCorrectorKAN:
    e2 = json.loads((EXP2_DIR / "exp2_complete_results.json").read_text(encoding="utf-8"))
    best_sd = min(e2["physics_trunk_3seed"], key=lambda r: r["val_complex_l2_pct"])["seed"]
    trunk = CleanParametricSeaKAN(constants, domain, dtype=torch.float32).to(device)
    trunk.load_state_dict(torch.load(EXP2_DIR / f"clean_trunk_seed_{best_sd}.pt", map_location=device, weights_only=True))
    trunk.eval()

    mf = PressureCorrectorKAN(constants, domain, trunk, width=24, hidden_layers=2, n_basis=7, dtype=torch.float32).to(device)
    mf.load_state_dict(torch.load(EXP2_DIR / "selected_mf32_model.pt", map_location=device, weights_only=True))
    mf.eval()
    for p in mf.parameters():
        p.requires_grad_(False)
    return mf


def _eval_wall_amp_batched(
    mf: PressureCorrectorKAN,
    theta_t: torch.Tensor,
    T: float,
    kappa: float,
    k_vals: torch.Tensor,   # [B]
    bD_vals: torch.Tensor,  # [B]
    constants: PhysicalConstants,
) -> torch.Tensor:
    B = k_vals.shape[0]
    N = theta_t.shape[0]
    cy = 1.0 - bD_vals * constants.pipe_diameter_m / constants.seabed_thickness_m  # [B]
    cos_t = torch.cos(theta_t).unsqueeze(0).expand(B, N)
    sin_t = torch.sin(theta_t).unsqueeze(0).expand(B, N)
    xy = torch.stack(
        (constants.radius * cos_t, cy.unsqueeze(1) + constants.radius * sin_t),
        dim=2,
    ).reshape(B * N, 2)
    tkb = torch.stack(
        (
            torch.full((B, N), T, device=theta_t.device, dtype=theta_t.dtype),
            k_vals.unsqueeze(1).expand(B, N),
            bD_vals.unsqueeze(1).expand(B, N),
        ),
        dim=2,
    ).reshape(B * N, 3)
    kap = torch.full((B * N,), kappa, device=theta_t.device, dtype=theta_t.dtype)
    cy_v = cy.unsqueeze(1).expand(B, N).reshape(B * N)
    p = mf.forward_pressure(xy, tkb, kap, cy_v).view(B, N)
    return torch.sqrt(sq_abs(p) + 1.0e-12)


def _eval_interior_complex_batched(
    mf: PressureCorrectorKAN,
    sensor_xy: torch.Tensor,  # [N, 2]
    T: float,
    kappa: float,
    k_vals: torch.Tensor,     # [B]
    bD_vals: torch.Tensor,    # [B]
    constants: PhysicalConstants,
) -> torch.Tensor:
    B = k_vals.shape[0]
    N = sensor_xy.shape[0]
    cy = 1.0 - bD_vals * constants.pipe_diameter_m / constants.seabed_thickness_m  # [B]
    xy = sensor_xy.unsqueeze(0).expand(B, N, 2).reshape(B * N, 2)
    tkb = torch.stack(
        (
            torch.full((B, N), T, device=sensor_xy.device, dtype=sensor_xy.dtype),
            k_vals.unsqueeze(1).expand(B, N),
            bD_vals.unsqueeze(1).expand(B, N),
        ),
        dim=2,
    ).reshape(B * N, 3)
    kap = torch.full((B * N,), kappa, device=sensor_xy.device, dtype=sensor_xy.dtype)
    cy_v = cy.unsqueeze(1).expand(B, N).reshape(B * N)
    return mf.forward_pressure(xy, tkb, kap, cy_v).view(B, N)


def compute_condition_numbers(
    mf: PressureCorrectorKAN,
    case: dict[str, object],
    sensor_xy: torch.Tensor,
    theta_t: torch.Tensor,
    constants: PhysicalConstants,
    domain: ParameterDomain,
    device: torch.device,
) -> dict[str, float]:
    T = float(case["T"])
    kappa = float(case["kappa"])
    k_true = float(case["k"])
    bD_true = float(case["bD"])
    lk_low, lk_high = math.log10(domain.permeability_m_per_s[0]), math.log10(domain.permeability_m_per_s[1])
    b_low, b_high = domain.burial_center_depth_over_diameter

    u_k = (math.log10(k_true) - lk_low) / (lk_high - lk_low)
    u_b = (bD_true - b_low) / (b_high - b_low)
    u_vec = torch.tensor([u_k, u_b], device=device, dtype=torch.float32)

    def wall_resp(u: torch.Tensor) -> torch.Tensor:
        kv = torch.pow(10.0, lk_low + (lk_high - lk_low) * u[0:1])
        bv = b_low + (b_high - b_low) * u[1:2]
        return _eval_wall_amp_batched(mf, theta_t, T, kappa, kv, bv, constants)[0]

    def int_resp(u: torch.Tensor) -> torch.Tensor:
        kv = torch.pow(10.0, lk_low + (lk_high - lk_low) * u[0:1])
        bv = b_low + (b_high - b_low) * u[1:2]
        p = _eval_interior_complex_batched(mf, sensor_xy, T, kappa, kv, bv, constants)[0]
        return torch.cat((p.real, p.imag), dim=0)

    J_wall = torch.autograd.functional.jacobian(wall_resp, u_vec)
    sv_wall = torch.linalg.svdvals(J_wall).detach().cpu().numpy()
    kappa_wall = float(sv_wall.max() / max(sv_wall.min(), 1e-12))

    J_int = torch.autograd.functional.jacobian(int_resp, u_vec)
    sv_int = torch.linalg.svdvals(J_int).detach().cpu().numpy()
    kappa_int = float(sv_int.max() / max(sv_int.min(), 1e-12))

    return {
        "kappa_wall_amp_joint": kappa_wall,
        "sv_wall": sv_wall.tolist(),
        "kappa_interior_complex_joint": kappa_int,
        "sv_int": sv_int.tolist(),
        "kappa_interior_complex_k_only": 1.0,
    }


def run_inversion_config(
    mf: PressureCorrectorKAN,
    case: dict[str, object],
    sensor_xy: torch.Tensor,
    theta_t: torch.Tensor,
    constants: PhysicalConstants,
    domain: ParameterDomain,
    device: torch.device,
    *,
    modality: str,     # "wall_amp" or "interior_complex"
    invert_mode: str,  # "joint" or "k_only"
    obs_source: str,   # "self_consistent" or "comsol"
    noise_rel: float,
    noise_seed: int,
) -> dict[str, float]:
    T = float(case["T"])
    kappa = float(case["kappa"])
    k_true = float(case["k"])
    bD_true = float(case["bD"])
    lk_low, lk_high = math.log10(domain.permeability_m_per_s[0]), math.log10(domain.permeability_m_per_s[1])
    b_low, b_high = domain.burial_center_depth_over_diameter

    # Generate observations
    with torch.no_grad():
        if obs_source == "self_consistent":
            kv_t = torch.tensor([k_true], device=device, dtype=torch.float32)
            bv_t = torch.tensor([bD_true], device=device, dtype=torch.float32)
            if modality == "wall_amp":
                obs_clean = _eval_wall_amp_batched(mf, theta_t, T, kappa, kv_t, bv_t, constants)[0]
            else:
                obs_clean = _eval_interior_complex_batched(mf, sensor_xy, T, kappa, kv_t, bv_t, constants)[0]
        else:
            mesh_xy = case["xy"].numpy()
            mesh_tgt = case["target"].numpy()
            if modality == "wall_amp":
                cy_true = 1.0 - bD_true * constants.pipe_diameter_m / constants.seabed_thickness_m
                th_np = theta_t.cpu().numpy()
                q_xy = np.column_stack(
                    (constants.radius * 1.002 * np.cos(th_np), cy_true + constants.radius * 1.002 * np.sin(th_np))
                )
                ire = LinearNDInterpolator(mesh_xy, mesh_tgt.real)(q_xy)
                iim = LinearNDInterpolator(mesh_xy, mesh_tgt.imag)(q_xy)
                if np.any(np.isnan(ire)):
                    for j in np.where(np.isnan(ire))[0]:
                        d2 = np.sum((mesh_xy - q_xy[j]) ** 2, axis=1)
                        ni = int(np.argmin(d2))
                        ire[j], iim[j] = mesh_tgt[ni].real, mesh_tgt[ni].imag
                obs_clean = torch.from_numpy(np.abs(ire + 1j * iim).astype(np.float32)).to(device)
            else:
                q_xy = sensor_xy.cpu().numpy()
                ire = LinearNDInterpolator(mesh_xy, mesh_tgt.real)(q_xy)
                iim = LinearNDInterpolator(mesh_xy, mesh_tgt.imag)(q_xy)
                obs_clean = torch.from_numpy((ire + 1j * iim).astype(np.complex64)).to(device)

    obs = obs_clean.clone()
    if noise_rel > 0.0:
        nrng = np.random.default_rng(noise_seed)
        if modality == "wall_amp":
            rms = float(torch.sqrt(obs.square().mean()).item())
            obs = obs + torch.from_numpy(nrng.normal(0.0, noise_rel * rms, size=obs.shape).astype(np.float32)).to(device)
        else:
            rms = float(torch.sqrt(sq_abs(obs).mean()).item())
            sig = noise_rel * rms / math.sqrt(2.0)
            n_re = nrng.normal(0.0, sig, size=obs.shape).astype(np.float32)
            n_im = nrng.normal(0.0, sig, size=obs.shape).astype(np.float32)
            obs = obs + torch.from_numpy(n_re + 1j * n_im).to(device)

    # Vectorized multi-start Adam across all starts in parallel!
    if invert_mode == "k_only":
        fk_init = np.linspace(0.08, 0.92, 8)
        z_init = np.log(fk_init / (1.0 - fk_init))[:, None]  # [8, 1]
    else:
        grid = [(fk, fb) for fk in (0.15, 0.50, 0.85) for fb in (0.15, 0.50, 0.85)]
        z_init = np.array([[math.log(fk / (1.0 - fk)), math.log(fb / (1.0 - fb))] for fk, fb in grid])  # [9, 2]

    B = z_init.shape[0]
    z_batch = nn.Parameter(torch.tensor(z_init, device=device, dtype=torch.float32))
    bv_fixed_b = torch.full((B,), bD_true, device=device, dtype=torch.float32)

    def per_start_losses(zb: torch.Tensor) -> torch.Tensor:
        kv = torch.pow(10.0, lk_low + (lk_high - lk_low) * torch.sigmoid(zb[:, 0]))
        bv = bv_fixed_b[: zb.shape[0]] if invert_mode == "k_only" else (b_low + (b_high - b_low) * torch.sigmoid(zb[:, 1]))
        if modality == "wall_amp":
            pred = _eval_wall_amp_batched(mf, theta_t, T, kappa, kv, bv, constants)
            return (pred - obs.unsqueeze(0)).square().mean(dim=1)
        pred = _eval_interior_complex_batched(mf, sensor_xy, T, kappa, kv, bv, constants)
        return sq_abs(pred - obs.unsqueeze(0)).mean(dim=1)

    opt = torch.optim.Adam([z_batch], lr=0.08)
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        losses = per_start_losses(z_batch)
        losses.sum().backward()
        opt.step()

    with torch.no_grad():
        adam_losses = per_start_losses(z_batch).cpu().numpy()
    top_idx = np.argsort(adam_losses)[:3]

    # Polish top-3 starts with exact L-BFGS (strong_wolfe)
    best_loss = float("inf")
    best_k = k_true
    best_bD = bD_true
    for idx_s in top_idx:
        z_one = nn.Parameter(z_batch[idx_s : idx_s + 1].detach().clone())
        lbfgs = torch.optim.LBFGS([z_one], lr=0.5, max_iter=30, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            lv = per_start_losses(z_one)[0]
            lv.backward()
            return lv

        lbfgs.step(closure)
        with torch.no_grad():
            lf = float(per_start_losses(z_one)[0].item())
            if lf < best_loss:
                best_loss = lf
                best_k = float(torch.pow(10.0, lk_low + (lk_high - lk_low) * torch.sigmoid(z_one[0, 0])).item())
                best_bD = (
                    bD_true
                    if invert_mode == "k_only"
                    else float((b_low + (b_high - b_low) * torch.sigmoid(z_one[0, 1])).item())
                )

    return {
        "k_true": k_true,
        "k_inv": best_k,
        "bD_true": bD_true,
        "bD_inv": best_bD,
        "E_k_pct": 100.0 * abs(best_k - k_true) / k_true,
        "E_bD_pct": 100.0 * abs(best_bD - bD_true) / bD_true,
        "loss": best_loss,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constants = PhysicalConstants()
    domain = ParameterDomain()
    mf = _load_mf_model(constants, domain, device)
    splits = load_all_comsol_cases(constants)
    test8 = splits["blind_sealed"] + splits["prospective"]

    theta_np = np.linspace(0.0, 2.0 * math.pi, 65)[:-1]
    theta_t = torch.from_numpy(theta_np.astype(np.float32)).to(device)

    print("=== 1. Jacobian Condition Numbers Across Held-Out Test Cases (B1-B8) ===", flush=True)
    kappas_wall, kappas_int = [], []
    per_case_sensors: dict[str, torch.Tensor] = {}
    for idx, c in enumerate(test8, 1):
        s_xy = select_fisher_sensors_clean(
            mf, float(c["T"]), float(c["bD"]), float(c["kappa"]), float(c["pipe_center_y"]), constants, domain, device, count=64
        )
        per_case_sensors[c["case_id"]] = torch.from_numpy(s_xy).to(device=device, dtype=torch.float32)
        cn = compute_condition_numbers(mf, c, per_case_sensors[c["case_id"]], theta_t, constants, domain, device)
        kappas_wall.append(cn["kappa_wall_amp_joint"])
        kappas_int.append(cn["kappa_interior_complex_joint"])
        print(
            f"  B{idx} ({c['case_id']:10s}): T={c['T']:.2f}, k={c['k']:.2e}, b/D={c['bD']:.2f} | "
            f"kappa_wall_amp={cn['kappa_wall_amp_joint']:7.2f} | kappa_int_cplx={cn['kappa_interior_complex_joint']:6.2f}",
            flush=True,
        )
    print(f"  Mean across B1-B8: kappa_wall_amp={np.mean(kappas_wall):.2f}, kappa_int_cplx={np.mean(kappas_int):.2f}", flush=True)

    configs = [
        ("wall_amp_joint_0pct", "wall_amp", "joint", 0.0),
        ("wall_amp_joint_1pct", "wall_amp", "joint", 0.01),
        ("interior_complex_joint_0pct", "interior_complex", "joint", 0.0),
        ("interior_complex_joint_1pct", "interior_complex", "joint", 0.01),
        ("interior_complex_k_only_1pct", "interior_complex", "k_only", 0.01),
    ]

    results: dict[str, object] = {
        "per_case_condition_numbers": [
            {
                "paper_id": f"B{i}",
                "case_id": c["case_id"],
                "kappa_wall_amp_joint": float(kw),
                "kappa_interior_complex_joint": float(ki),
                "kappa_interior_complex_k_only": 1.0,
            }
            for i, (c, kw, ki) in enumerate(zip(test8, kappas_wall, kappas_int), 1)
        ],
        "mean_condition_numbers": {
            "wall_amp_joint": float(np.mean(kappas_wall)),
            "interior_complex_joint": float(np.mean(kappas_int)),
            "interior_complex_k_only": 1.0,
        },
        "comsol_8case_summary": {},
        "self_consistent_8case_summary": {},
    }

    for obs_src in ("self_consistent", "comsol"):
        print(f"\n=== 2. 8-Case Held-Out Observability Screening (obs_source = {obs_src}) ===", flush=True)
        summary_dict = {}
        for name, mod, inv_m, nlvl in configs:
            ek_list, eb_list = [], []
            per_case_rows = []
            for idx, c in enumerate(test8):
                n_reps = 5 if nlvl > 0.0 else 1
                rep_ek, rep_eb = [], []
                for rep in range(n_reps):
                    r = run_inversion_config(
                        mf,
                        c,
                        per_case_sensors[c["case_id"]],
                        theta_t,
                        constants,
                        domain,
                        device,
                        modality=mod,
                        invert_mode=inv_m,
                        obs_source=obs_src,
                        noise_rel=nlvl,
                        noise_seed=9100 + 100 * idx + rep,
                    )
                    rep_ek.append(r["E_k_pct"])
                    rep_eb.append(r["E_bD_pct"])
                ek_list.append(float(np.mean(rep_ek)))
                eb_list.append(float(np.mean(rep_eb)))
                per_case_rows.append(
                    {
                        "paper_id": f"B{idx+1}",
                        "case_id": c["case_id"],
                        "E_k_pct": float(np.mean(rep_ek)),
                        "E_bD_pct": float(np.mean(rep_eb)),
                    }
                )
            kap_mean = (
                float(np.mean(kappas_wall))
                if mod == "wall_amp"
                else (float(np.mean(kappas_int)) if inv_m == "joint" else 1.0)
            )
            summary_dict[name] = {
                "modality": mod,
                "invert_mode": inv_m,
                "noise_pct": 100.0 * nlvl,
                "E_k_pct_mean": float(np.mean(ek_list)),
                "E_k_pct_sd": float(np.std(ek_list, ddof=1)),
                "E_bD_pct_mean": float(np.mean(eb_list)),
                "E_bD_pct_sd": float(np.std(eb_list, ddof=1)),
                "kappa_inv_mean": kap_mean,
                "per_case": per_case_rows,
            }
            print(
                f"  {name:30s} | noise={100*nlvl:.0f}% | E_k={np.mean(ek_list):6.2f}% (sd={np.std(ek_list, ddof=1):5.2f}%) | "
                f"E_bD={np.mean(eb_list):6.2f}% (sd={np.std(eb_list, ddof=1):5.2f}%) | kappa={kap_mean:.2f}",
                flush=True,
            )
        results[f"{obs_src}_8case_summary"] = summary_dict

    out_path = EXP2_DIR / "table8_observability_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    bundle_path = EXP2_DIR / "exp2_complete_results.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["table8_observability_screening"] = results
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved Table 8 results to {out_path} and updated {bundle_path} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
