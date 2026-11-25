"""Re-evaluate the five available Table 6 weights on CPU and check their profiles."""
import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import torch

from verify_tier1 import DATA, TABLE6, LEGACY, check_printed, load_benchmark, metrics, read_profile, seakan_model, seakan_profile
from figure9_comparison.common import DeepXDEFNN, TorchMLP, physics_constants, pipe_profile
from figure9_comparison.pointwise import PiecewiseXPINN
from figure9_comparison.pino import PINO2d, PINOOptions, _case_arrays
from mpm_kan_figure9.config import figure9_setup
from mpm_kan_figure9.reference import _sample_wall


def restore(model, name):
    with np.load(DATA / (name + ".npz"), allow_pickle=False) as weights:
        model.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights})
    model.eval()
    return sum(p.numel() for p in model.parameters())


def pino_profile(wave, soil, geom):
    options = PINOOptions(width=32, modes_x=12, modes_y=8, nx=65, ny=25, layers=4)
    model = PINO2d(options)
    count = restore(model, "pino")
    inputs, _, _ = _case_arrays(wave, soil, geom, options.nx, options.ny)
    with torch.no_grad():
        field = model(torch.from_numpy(inputs[None]))[0].numpy()
    active = inputs[7].astype(bool)
    pressure = np.where(active, field[4] + 1j * field[5], np.nan + 0j)
    x = np.linspace(-geom.half_width_over_height, geom.half_width_over_height, options.nx)
    y = np.linspace(0., 1., options.ny)
    t = np.linspace(0., 2 * np.pi, 721)
    p = _sample_wall(t, pressure, active, x, y, geom, max(x[1]-x[0], y[1]-y[0]))
    return (t, p), count


def main():
    # DeepXDE may change PyTorch's default to CUDA when imported on a GPU host.
    import deepxde
    torch.set_default_device("cpu")
    wave, soil, geom = figure9_setup()
    const = physics_constants(wave, soil, geom)
    selected = seakan_model()
    profiles = {"seakan": (seakan_profile(selected), sum(p.numel() for p in selected.parameters()))}
    for method, model in (("deepxde_pinn", DeepXDEFNN(64, 4)), ("gpinn_rar", TorchMLP(64, 4)),
                          ("xpinn", PiecewiseXPINN(40, 4, 0.35 / const.half_width))):
        model = model.to(dtype=torch.float64)
        count = restore(model, method)
        profiles[method] = (pipe_profile(model, const), count)
    profiles["pino"] = pino_profile(wave, soil, geom)
    ref = load_benchmark(LEGACY / "figure9_mpm_kan/data/figure9.csv")
    for method, (profile, count) in profiles.items():
        values = metrics(profile, ref)
        print(f"{method}: {count} parameter tensor elements; relative L2 {(values[0], values[2])} percent")
        if method != "seakan":
            theta, expected = read_profile(DATA / "profiles" / f"{method}.csv")
            np.testing.assert_allclose(profile[0], theta, atol=1e-7, rtol=0.)
            diff = np.max(np.abs(profile[1] - expected))
            print(f"  max absolute difference from archived profile: {diff:.8g}")
            tolerance = 5e-4 if method == "pino" else 1e-10
            np.testing.assert_allclose(profile[1], expected, atol=tolerance, rtol=0.)
        if method == "pino":
            # FP32/complex64 spectral convolution is sensitive to the FFT and GEMM backend.
            original = metrics(read_profile(DATA / "profiles/pino.csv"), ref)
            print(f"  archived L2 (%) = {(original[0], original[2])}; CPU replay R2 = {(values[1], values[3])}")
            if not np.isfinite(values).all():
                raise ValueError("Non-finite PINO output")
        else:
            check_printed(method, values, TABLE6[method], (2, 4, 2, 4))


if __name__ == "__main__":
    main()
