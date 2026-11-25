from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "deepxde_pinn": "DeepXDE-PINN",
    "gpinn_rar": "gPINN-RAR",
    "xpinn": "XPINN",
    "pino": "PINO",
    "seakan": "SeaKAN",
}


def periodic_interpolate(
    source_theta: np.ndarray, source_value: np.ndarray, target_theta: np.ndarray
) -> np.ndarray:
    theta = np.mod(np.asarray(source_theta, dtype=float), 2.0 * np.pi)
    value = np.asarray(source_value, dtype=float)
    order = np.argsort(theta)
    theta, value = theta[order], value[order]
    theta_ext = np.concatenate((theta[-1:] - 2.0 * np.pi, theta, theta[:1] + 2.0 * np.pi))
    value_ext = np.concatenate((value[-1:], value, value[:1]))
    return np.interp(np.mod(target_theta, 2.0 * np.pi), theta_ext, value_ext)


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    error = prediction - target
    sse = float(np.sum(error**2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    mse = float(np.mean(error**2))
    return {
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(target)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - sse / denominator) if denominator > 0.0 else float("nan"),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def load_benchmark(data_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = pd.read_csv(data_path)
    mapping = {
        "experimental": ("Experimental_data_theta", "Experimental_data_r"),
        "zhu_numerical": (
            "Numerical_result_of_this_study_theta",
            "Numerical_result_of_this_study_r",
        ),
    }
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (theta_column, value_column) in mapping.items():
        valid = data[[theta_column, value_column]].dropna()
        result[name] = (
            valid[theta_column].to_numpy(float),
            valid[value_column].to_numpy(float),
        )
    return result


def evaluate_all(
    output_dir: Path,
    benchmark_path: Path,
    seakan_profile_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = load_benchmark(benchmark_path)
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in ("deepxde_pinn", "gpinn_rar", "xpinn", "pino"):
        profile_path = output_dir / method / "profile.csv"
        profile = pd.read_csv(profile_path)
        profiles[method] = (
            profile["theta_rad"].to_numpy(float),
            profile["normalized_pressure"].to_numpy(float),
        )
    seakan_data = pd.read_csv(seakan_profile_path)
    seakan_data = seakan_data[seakan_data["source"] == "mpm_kan_pinn"]
    profiles["seakan"] = (
        seakan_data["theta_rad"].to_numpy(float),
        seakan_data["normalized_pressure"].to_numpy(float),
    )

    metric_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    common_theta = np.linspace(0.0, 2.0 * np.pi, 721)
    for method, (theta, value) in profiles.items():
        common_value = periodic_interpolate(theta, value, common_theta)
        profile_rows.extend(
            {
                "method": method,
                "theta_rad": angle,
                "normalized_pressure": pressure,
            }
            for angle, pressure in zip(common_theta, common_value)
        )
        for reference_name, (reference_theta, reference_value) in benchmark.items():
            prediction = periodic_interpolate(theta, value, reference_theta)
            metric_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "reference": reference_name,
                    "n": len(reference_theta),
                    **error_metrics(prediction, reference_value),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    profiles_frame = pd.DataFrame(profile_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    profiles_frame.to_csv(output_dir / "profiles.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metric_rows, indent=2), encoding="utf-8"
    )
    run_rows: list[dict[str, object]] = []
    for method in ("deepxde_pinn", "gpinn_rar", "xpinn", "pino"):
        metadata_path = output_dir / method / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("profile", None)
        metadata.pop("lbfgs_final", None)
        run_rows.append(metadata)
    pd.DataFrame(run_rows).to_csv(output_dir / "run_summary.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_rows, indent=2), encoding="utf-8"
    )
    return metrics, profiles_frame


def plot_comparison(
    output_dir: Path,
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    benchmark_path: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    colors = {
        "deepxde_pinn": "#4C78A8",
        "gpinn_rar": "#E45756",
        "xpinn": "#B279A2",
        "pino": "#F2CF5B",
        "seakan": "#222222",
    }
    styles = {
        "deepxde_pinn": (0, (4, 2)),
        "gpinn_rar": (0, (5, 1, 1, 1)),
        "xpinn": (0, (2, 1)),
        "pino": (0, (7, 2)),
        "seakan": "solid",
    }
    benchmark = load_benchmark(benchmark_path)

    fig = plt.figure(figsize=(4.1, 4.1))
    ax = fig.add_subplot(111, projection="polar")
    for method in METHOD_LABELS:
        subset = profiles[profiles["method"] == method]
        ax.plot(
            subset["theta_rad"],
            subset["normalized_pressure"],
            color=colors[method],
            linestyle=styles[method],
            linewidth=1.5 if method == "seakan" else 1.15,
            label=METHOD_LABELS[method],
            zorder=3 if method == "seakan" else 2,
        )
    zhu_theta, zhu_value = benchmark["zhu_numerical"]
    ax.plot(
        zhu_theta,
        zhu_value,
        color="#666666",
        linewidth=1.0,
        linestyle=(0, (1, 1)),
        label="Zhu et al. numerical",
    )
    exp_theta, exp_value = benchmark["experimental"]
    ax.scatter(
        exp_theta,
        exp_value,
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=5,
        label="Experiment",
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        np.arange(0, 360, 45),
        ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$"],
    )
    ax.set_ylim(0.0, 1.25)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.grid(color="#c8c8c8", linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handlelength=2.8,
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.25)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"figure9_method_comparison_polar.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)

    zhu = metrics[metrics["reference"] == "zhu_numerical"].copy()
    order = list(METHOD_LABELS)
    zhu["order"] = zhu["method"].map({name: i for i, name in enumerate(order)})
    zhu = zhu.sort_values("order")
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    y = np.arange(len(zhu))
    values = 100.0 * zhu["relative_l2"].to_numpy()
    bars = ax.barh(
        y,
        values,
        color=[colors[name] for name in zhu["method"]],
        edgecolor="black",
        linewidth=0.45,
        height=0.62,
    )
    ax.set_yticks(y, zhu["method_label"])
    ax.invert_yaxis()
    ax.set_xlabel(r"Relative $L_2$ error against Zhu et al. numerical solution (\%)")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.5)
    ax.set_axisbelow(True)
    maximum = max(values) * 1.18
    ax.set_xlim(0.0, maximum)
    for bar, value in zip(bars, values):
        ax.text(
            value + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"figure9_method_comparison_error.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


def write_isolation_manifest(
    output_dir: Path,
    benchmark_path: Path,
    seakan_profile_path: Path,
) -> None:
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "evaluation_benchmark": str(benchmark_path.resolve()),
        "evaluation_benchmark_sha256": sha256(benchmark_path),
        "benchmark_access": (
            "Loaded only by evaluation.py after all trainable models had written "
            "their checkpoints and profiles."
        ),
        "pointwise_training": (
            "DeepXDE-PINN, gPINN-RAR and XPINN use only Biot strong residuals "
            "and boundary/interface conditions."
        ),
        "pino_training": (
            "Uses 26 finite-difference parameter cases; the exact nominal "
            "Figure 9 parameter tuple is programmatically excluded."
        ),
        "seakan_profile": str(seakan_profile_path.resolve()),
        "seakan_profile_sha256": sha256(seakan_profile_path),
    }
    (output_dir / "data_isolation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
