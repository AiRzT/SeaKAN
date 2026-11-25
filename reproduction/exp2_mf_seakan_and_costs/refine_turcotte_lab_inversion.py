from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
import torch

EXP2_OUT = Path(__file__).resolve().parent
SUPP_ROOT = EXP2_OUT.parent
sys.path.insert(0, str(SUPP_ROOT / "exp1_kinn_and_ablation"))

from exp1_kinn_and_ablation import (
    BENCHMARK_CSV,
    AdaptiveMaterialPointCloud,
    AdaptiveStrongMPMProblem,
    MPMKANPipeProblem,
    MaterialPointGrid2D,
    PipeHarmonicBackbone,
    SoilParameters,
    TopGapLocalizedBackbone,
    _run_al_stage,
    dimensionless_coefficients,
    figure9_setup,
    load_benchmark,
    periodic_interpolate,
)


def solve_seakan_at_k(
    k_val: float,
    device: torch.device,
    *,
    seed: int = 202409,
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    """Solve reference-free Tier-1 SeaKAN at permeability k_val (zero experimental or FEM data)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    wave, _, geometry = figure9_setup()
    soil = SoilParameters(permeability=float(k_val))
    coeffs = dimensionless_coefficients(wave, soil, geometry)

    base_model = PipeHarmonicBackbone(geometry, coeffs["kappa"], backbone="kan", dtype=torch.float32).to(device)
    grid_coarse = MaterialPointGrid2D.create(48, 20, geometry, device=device, dtype=torch.float32)
    grid_fine = MaterialPointGrid2D.create(64, 24, geometry, device=device, dtype=torch.float32)

    if init_state is None:
        homotopy = [
            (grid_coarse, 1.0, 20.0, 0.0, 1.0e-6, 1.5e-3, 800),
            (grid_coarse, 1.0, 20.0, 0.02, 1.0e-6, 6.0e-4, 200),
            (grid_coarse, 10.0, 500.0, 0.0, 1.0e-3, 8.0e-4, 600),
            (grid_coarse, 10.0, 2000.0, 0.0, 1.0e-2, 5.0e-4, 500),
            (grid_fine, 10.0, 2000.0, 0.0, 1.0e-2, 3.0e-4, 400),
            (grid_fine, 10.0, 2000.0, 0.5, 1.0e-2, 2.0e-4, 300),
        ]
    else:
        base_model.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in init_state.items()})
        homotopy = [
            (grid_fine, 10.0, 2000.0, 0.0, 1.0e-2, 3.0e-4, 350),
            (grid_fine, 10.0, 2000.0, 0.5, 1.0e-2, 1.8e-4, 250),
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

    m64 = base_model.to(dtype=torch.float64)
    gf64 = MaterialPointGrid2D.create(64, 24, geometry, device=device, dtype=torch.float64)
    pr64 = MPMKANPipeProblem(
        wave, soil, geometry, gf64,
        momentum_weight=10.0, fluid_weight=2000.0,
        boundary_weight=80.0, pipe_flux_weight=120.0,
        strong_weight=0.5, roughness_weight=1.0e-2,
        strong_stride=8, boundary_points=128,
    )
    lb64 = torch.optim.LBFGS(
        m64.parameters(), lr=0.5, max_iter=25, max_eval=50,
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
        strong_weight=1.0, roughness_weight=1.0e-2,
        strong_stride=8, boundary_points=128,
    )
    opt_fin = torch.optim.Adam(base_model.parameters(), lr=1.0e-4)
    for _ in range(180):
        opt_fin.zero_grad(set_to_none=True)
        lf = prob_fin.loss(base_model)
        lf.total.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=20.0)
        opt_fin.step()

    # Save Stage-1 state on CPU to warm-start neighboring Chebyshev node
    warm_out = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

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

    gen2 = torch.Generator(device=device).manual_seed(seed + 22)
    s2_sched = (
        ((800.0, 80.0, 160.0, 1.0e-4, 400), (1200.0, 90.0, 400.0, 5.0e-5, 300), (1600.0, 100.0, 800.0, 2.0e-5, 200))
        if init_state is None
        else ((1200.0, 90.0, 400.0, 6.0e-5, 300), (1600.0, 100.0, 800.0, 2.0e-5, 200))
    )
    for fw2, bw2, pf2, lr2, n2 in s2_sched:
        prob2 = AdaptiveStrongMPMProblem(
            wave, soil, geometry, cloud3_32,
            level_weights=lvl_w3, momentum_weight=10.0, fluid_weight=fw2,
            p2g_weight=0.0, boundary_weight=bw2, pipe_flux_weight=pf2,
            roughness_weight=1.0e-3, boundary_points=160,
        )
        opt2 = torch.optim.Adam(base_model.parameters(), lr=lr2)
        for _ in range(n2):
            idx = cloud3_32.sample_indices(batch3, gen2)
            opt2.zero_grad(set_to_none=True)
            loss2 = prob2.loss(base_model, idx)
            loss2.total.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=15.0)
            opt2.step()

    prob4_32 = AdaptiveStrongMPMProblem(
        wave, soil, geometry, cloud4_32,
        level_weights=lvl_w4, momentum_weight=10.0, fluid_weight=1000.0,
        p2g_weight=0.0, boundary_weight=0.0, pipe_flux_weight=0.0,
        roughness_weight=1.0e-3, boundary_points=192,
    )
    prob4_64 = AdaptiveStrongMPMProblem(
        wave, soil, geometry, cloud4_64,
        level_weights=lvl_w4, momentum_weight=10.0, fluid_weight=1000.0,
        p2g_weight=0.0, boundary_weight=0.0, pipe_flux_weight=0.0,
        roughness_weight=1.0e-3, boundary_points=192,
    )

    teacher = copy.deepcopy(base_model)
    teacher = _run_al_stage(
        teacher, prob4_32, prob4_64, cloud4_32, cloud4_64, batch4,
        fp32_steps=200, fp32_lr=2.5e-5,
        fp64_adam_steps=20, fp64_lr=3.0e-6,
        fp64_lbfgs_steps=8, seed=seed + 33, device=device,
    )
    teacher_32 = teacher.to(dtype=torch.float32).eval()

    student = TopGapLocalizedBackbone(
        base_model.to(dtype=torch.float32), geometry, backbone="kan", dtype=torch.float32, freeze_base=True
    ).to(device)
    gap_xy = cloud4_32.xy[cloud4_32.level == len(cloud4_32.counts) - 1]
    theta_arc = torch.linspace(0.25 * math.pi, 0.75 * math.pi, 257, dtype=torch.float32, device=device)[:-1]
    pipe_xy = torch.stack(
        (geometry.radius_over_height * torch.cos(theta_arc), geometry.center_y + geometry.radius_over_height * torch.sin(theta_arc)),
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
    for _ in range(700):
        opt_dist.zero_grad(set_to_none=True)
        pred = student(dist_xy)
        d_loss = (pred.p - tgt_p).abs().square().mean() + 0.10 * (pred.u - tgt_u).abs().square().mean() + 0.10 * (pred.w - tgt_w).abs().square().mean()
        d_loss.backward()
        opt_dist.step()
    del teacher_32, teacher

    final_model = _run_al_stage(
        student, prob4_32, prob4_64, cloud4_32, cloud4_64, batch4,
        fp32_steps=60, fp32_lr=2.5e-5,
        fp64_adam_steps=10, fp64_lr=3.0e-6,
        fp64_lbfgs_steps=4, seed=seed + 44, device=device,
    )

    theta = np.linspace(0.0, 2.0 * math.pi, 721)
    t_tensor = torch.as_tensor(theta, dtype=torch.float64, device=device)
    xy_p = torch.stack(
        (geometry.radius_over_height * torch.cos(t_tensor), geometry.center_y + geometry.radius_over_height * torch.sin(t_tensor)),
        dim=1,
    )
    with torch.no_grad():
        pred_p = final_model(xy_p).p.abs().detach().cpu().numpy()
    return theta, pred_p, warm_out


def _branch_worker(k_chain: list[float], init_state: dict[str, torch.Tensor]) -> list[tuple[float, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cur_state = init_state
    out: list[tuple[float, np.ndarray]] = []
    for kv in k_chain:
        _, prof, cur_state = solve_seakan_at_k(kv, device, seed=202409, init_state=cur_state)
        print(f"  [Turcotte Chebyshev SeaKAN] solved k={kv:.4e} m/s", flush=True)
        out.append((kv, prof))
    return out


def reproduce_lab_inversion(output_dir: Path, device: torch.device, *, use_published_profiles: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, soil_nom, _ = figure9_setup()
    bm = load_benchmark(BENCHMARK_CSV)
    exp_theta, exp_r = bm["experimental"]
    k_lab_true = soil_nom.permeability  # 1.10e-3 m/s (NOT a Chebyshev node!)

    # 9 Chebyshev-Gauss-Lobatto nodes in log10(k) over [3.0e-4, 3.0e-3] m/s (midpoint 9.487e-4 m/s)
    k_min, k_max = 3.0e-4, 3.0e-3
    lk_min, lk_max = math.log10(k_min), math.log10(k_max)
    j_idx = np.arange(8, -1, -1, dtype=np.float64)
    cheb_u = np.cos(j_idx * math.pi / 8.0)  # [-1, ..., 0, ..., +1]
    cheb_lk = 0.5 * (lk_min + lk_max) + 0.5 * (lk_max - lk_min) * cheb_u
    cheb_k = 10.0 ** cheb_lk

    cache_npz = (EXP2_OUT if use_published_profiles else output_dir) / "turcotte_chebyshev9_profiles.npz"
    if use_published_profiles and not cache_npz.is_file():
        raise FileNotFoundError(cache_npz)
    t0 = time.perf_counter()
    if cache_npz.is_file():
        dat = np.load(cache_npz)
        theta_grid = dat["theta"]
        prof_matrix = dat["profiles"]
        train_s = float(dat["elapsed_s"])
        print(f"[CACHE HIT] Loaded 9-node Chebyshev physics-only SeaKAN profiles ({train_s:.1f}s)", flush=True)
    else:
        print("\n=== Solving 9-node Chebyshev Physics-Only SeaKAN over k in [3e-4, 3e-3] m/s ===", flush=True)
        mid_k = float(cheb_k[4])  # 9.4868e-4 m/s
        theta_grid, mid_prof, mid_state = solve_seakan_at_k(mid_k, device, seed=202409, init_state=None)
        print(f"  [Turcotte Chebyshev SeaKAN] solved midpoint k={mid_k:.4e} m/s", flush=True)

        lower_chain = [float(cheb_k[3]), float(cheb_k[2]), float(cheb_k[1]), float(cheb_k[0])]
        upper_chain = [float(cheb_k[5]), float(cheb_k[6]), float(cheb_k[7]), float(cheb_k[8])]
        with ProcessPoolExecutor(max_workers=2) as pool:
            f_low = pool.submit(_branch_worker, lower_chain, mid_state)
            f_up = pool.submit(_branch_worker, upper_chain, mid_state)
            res_low = f_low.result()
            res_up = f_up.result()

        by_k = {mid_k: mid_prof}
        for kv, pr in res_low + res_up:
            by_k[kv] = pr
        prof_matrix = np.stack([by_k[float(kv)] for kv in cheb_k], axis=0)  # (9, 721)
        train_s = time.perf_counter() - t0
        np.savez_compressed(cache_npz, cheb_lk=cheb_lk, cheb_k=cheb_k, theta=theta_grid, profiles=prof_matrix, elapsed_s=train_s)

    # Interpolate each Chebyshev physics-only SeaKAN profile onto experimental transducer angles (only at inversion time!)
    sensor_matrix = np.stack([periodic_interpolate(theta_grid, prof_matrix[m], exp_theta) for m in range(len(cheb_k))], axis=0)
    spline = CubicSpline(cheb_lk, sensor_matrix, axis=0, bc_type="natural")

    inv_t0 = time.perf_counter()
    grid_lk = np.linspace(lk_min, lk_max, 401)
    pred_scan = spline(grid_lk)  # (401, 8)
    grid_mse = np.mean((pred_scan - exp_r[None, :]) ** 2, axis=1)

    # Continuous Newton / fine parabolic polish around global minimum
    from scipy.optimize import minimize_scalar
    opt_res = minimize_scalar(
        lambda lk: float(np.mean((spline(lk) - exp_r) ** 2)),
        bounds=(lk_min, lk_max),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    lk_est = float(opt_res.x)
    k_est = float(10.0 ** lk_est)
    inv_s = time.perf_counter() - inv_t0

    pred_est = spline(lk_est)
    pred_nom = spline(math.log10(k_lab_true))

    err_est = pred_est - exp_r
    rmse_est = float(np.sqrt(np.mean(err_est**2)))
    r2_est = float(1.0 - np.sum(err_est**2) / np.sum((exp_r - exp_r.mean()) ** 2))
    l2_est = float(np.linalg.norm(err_est) / np.linalg.norm(exp_r))

    err_nom = pred_nom - exp_r
    l2_nom = float(np.linalg.norm(err_nom) / np.linalg.norm(exp_r))
    r2_nom = float(1.0 - np.sum(err_nom**2) / np.sum((exp_r - exp_r.mean()) ** 2))
    e_k_pct = 100.0 * abs(k_est - k_lab_true) / k_lab_true

    res = {
        "benchmark": "Turcotte et al. (1984) wave-flume laboratory experiment",
        "method": "9-node Chebyshev physics-only SeaKAN spectral collocation over log10(k) in [3e-4, 3e-3] m/s",
        "n_sensors": int(len(exp_r)),
        "chebyshev_k_nodes_m_per_s": cheb_k.tolist(),
        "k_true_lab_m_per_s": k_lab_true,
        "k_inverted_m_per_s": k_est,
        "E_k_pct": e_k_pct,
        "log10_k_abs_err": abs(lk_est - math.log10(k_lab_true)),
        "forward_at_true_k_l2_pct": 100.0 * l2_nom,
        "forward_at_true_k_r2": r2_nom,
        "reconstructed_sensor_rmse": rmse_est,
        "reconstructed_sensor_l2_pct": 100.0 * l2_est,
        "reconstructed_sensor_r2": r2_est,
        "offline_train_seconds": train_s,
        "inversion_seconds": inv_s,
        "grid_log10_k": grid_lk.tolist(),
        "grid_mse": grid_mse.tolist(),
    }
    print(
        f"Turcotte Lab Inversion (Chebyshev-9 SeaKAN): k_true={k_lab_true:.4e}, k_inv={k_est:.4e} "
        f"(E_k={e_k_pct:.2f}%, dlog10k={res['log10_k_abs_err']:.3f}), "
        f"forward@true_k L2={100*l2_nom:.2f}% (R2={r2_nom:.4f}), "
        f"recon RMSE={rmse_est:.5f}, L2={100*l2_est:.2f}% (R2={r2_est:.4f})",
        flush=True,
    )

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the external flume inversion (Table 9).")
    parser.add_argument("--output-dir", type=Path, default=EXP2_OUT / "runs")
    parser.add_argument("--use-published-profiles", action="store_true",
                        help="Use nine saved forward profiles; the inverse fit still runs from scratch.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = reproduce_lab_inversion(args.output_dir, device, use_published_profiles=args.use_published_profiles)
    (args.output_dir / "inverse_turcotte_lab.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
