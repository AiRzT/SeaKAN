from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Geometry


SERIES_COLUMNS = {
    "experimental": ("Experimental_data_theta", "Experimental_data_r"),
    "poroelasticity": (
        "Poroelasticity_solution_theta",
        "Poroelasticity_solution_r",
    ),
    "darcy": ("Darcy's_solution_theta", "Darcy's_solution_r"),
    "jeng_numerical": ("Numerical_result_theta", "Numerical_result_r"),
    "paper_this_study": (
        "Numerical_result_of_this_study_theta",
        "Numerical_result_of_this_study_r",
    ),
}


def load_figure9_data(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Figure 9 data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Figure 9 data has no header: {path}")
        normalized_fields = [name.strip() for name in reader.fieldnames]
        rows = [
            {
                (key.strip() if key is not None else key): value
                for key, value in row.items()
            }
            for row in reader
        ]
    required = {
        name for pair in SERIES_COLUMNS.values() for name in pair
    }
    missing = sorted(required - set(normalized_fields))
    if missing:
        raise ValueError(f"Figure 9 data is missing columns: {missing}")

    data: dict[str, dict[str, np.ndarray]] = {}
    source_quality: dict[str, Any] = {}
    for source, (theta_column, radius_column) in SERIES_COLUMNS.items():
        theta: list[float] = []
        radius: list[float] = []
        for row in rows:
            try:
                angle = float(row[theta_column])
                value = float(row[radius_column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(angle) and np.isfinite(value):
                theta.append(angle)
                radius.append(value)
        theta_array = np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi)
        radius_array = np.asarray(radius, dtype=float)
        order = np.argsort(theta_array)
        theta_array = theta_array[order]
        radius_array = radius_array[order]
        if theta_array.size < 2:
            raise ValueError(f"Figure 9 {source} has fewer than two points")
        data[source] = {"theta": theta_array, "radius": radius_array}
        source_quality[source] = {
            "n": int(theta_array.size),
            "theta_range": [float(theta_array.min()), float(theta_array.max())],
            "radius_range": [float(radius_array.min()), float(radius_array.max())],
            "duplicate_angles": int(
                theta_array.size - np.unique(theta_array).size
            ),
        }
    return data, {
        "file": str(path),
        "rows": len(rows),
        "columns": normalized_fields,
        "sources": source_quality,
        "warnings": [],
    }


def circular_interpolate(
    source_theta: np.ndarray,
    source_value: np.ndarray,
    target_theta: np.ndarray,
) -> np.ndarray:
    order = np.argsort(np.mod(source_theta, 2.0 * np.pi))
    theta = np.mod(source_theta[order], 2.0 * np.pi)
    value = source_value[order]
    theta_extended = np.concatenate(
        (theta[-1:] - 2.0 * np.pi, theta, theta[:1] + 2.0 * np.pi)
    )
    value_extended = np.concatenate((value[-1:], value, value[:1]))
    return np.interp(np.mod(target_theta, 2.0 * np.pi), theta_extended, value_extended)


def error_metrics(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    residual = prediction - observed
    denominator = float(np.linalg.norm(observed))
    return {
        "n": int(observed.size),
        "relative_l2": (
            float(np.linalg.norm(residual) / denominator)
            if denominator > 0.0
            else float(np.linalg.norm(residual))
        ),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def calculate_metrics(
    prediction: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
    quality: dict[str, Any],
    reference: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "metric_definition": {
            "relative_l2": "||model-data||_2 / ||data||_2",
            "interpolation": "periodic linear interpolation at digitized angles",
            "training_use": (
                "digitized Figure 9 series are loaded only after physics training"
            ),
        },
        "mpm_kan": {},
        "data_quality": quality,
    }
    for source, observed in extracted.items():
        values = circular_interpolate(
            prediction["theta"],
            prediction["normalized_pressure"],
            observed["theta"],
        )
        metrics["mpm_kan"][source] = error_metrics(values, observed["radius"])
    if reference is not None:
        reference_on_network = circular_interpolate(
            reference["theta"],
            reference["normalized_pressure"],
            prediction["theta"],
        )
        metrics["mpm_kan"]["finite_difference_reference"] = error_metrics(
            prediction["normalized_pressure"], reference_on_network
        )
    return metrics


def save_profiles(
    path: Path,
    prediction: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray] | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "theta_rad", "normalized_pressure"])
        sources: list[tuple[str, dict[str, np.ndarray]]] = [
            ("mpm_kan_pinn", prediction)
        ]
        if reference is not None:
            sources.append(("finite_difference_reference", reference))
        sources.extend(extracted.items())
        for name, profile in sources:
            value_key = (
                "normalized_pressure" if "normalized_pressure" in profile else "radius"
            )
            for theta, value in zip(profile["theta"], profile[value_key]):
                writer.writerow([name, f"{theta:.12g}", f"{value:.12g}"])


def plot_polar(
    path: Path,
    prediction: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray] | None = None,
    baseline: dict[str, np.ndarray] | None = None,
    prediction_label: str = "2-D MPM-KAN PINN (physics-only)",
    baseline_label: str = "Previous uniform weak-form MPM-KAN",
) -> None:
    fig, axis = plt.subplots(
        figsize=(7.4, 7.0), subplot_kw={"projection": "polar"}
    )
    styles = {
        "poroelasticity": dict(
            color="#2ca02c", linestyle="-.", linewidth=1.5,
            label="Digitized poroelasticity solution"
        ),
        "darcy": dict(
            color="#1f77b4", linestyle=":", linewidth=1.7,
            label="Digitized Darcy solution"
        ),
        "jeng_numerical": dict(
            color="#222222", linestyle="--", linewidth=1.5,
            label="Digitized Jeng et al. numerical result"
        ),
        "paper_this_study": dict(
            color="#e24a33", linestyle="-", linewidth=1.5, alpha=0.72,
            label="Digitized Zhu et al. numerical result"
        ),
    }
    for key in ("poroelasticity", "darcy", "jeng_numerical", "paper_this_study"):
        profile = extracted[key]
        closed_theta = np.concatenate((profile["theta"], profile["theta"][:1] + 2 * np.pi))
        closed_radius = np.concatenate((profile["radius"], profile["radius"][:1]))
        axis.plot(closed_theta, closed_radius, **styles[key])

    if baseline is not None:
        axis.plot(
            baseline["theta"],
            baseline["normalized_pressure"],
            color="#a05a2c",
            linewidth=1.6,
            linestyle=(0, (5, 3)),
            label=baseline_label,
            zorder=6,
        )
    axis.plot(
        prediction["theta"],
        prediction["normalized_pressure"],
        color="#7b2cbf",
        linewidth=2.8,
        label=prediction_label,
        zorder=7,
    )
    if reference is not None:
        axis.plot(
            reference["theta"],
            reference["normalized_pressure"],
            color="#666666",
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            label="Independent finite-difference reference",
            zorder=5,
        )
    experimental = extracted["experimental"]
    axis.scatter(
        experimental["theta"],
        experimental["radius"],
        s=40,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.25,
        label="Digitized experiment",
        zorder=9,
    )
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    axis.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
    axis.set_ylim(0.0, 0.85)
    axis.set_yticks([0.2, 0.4, 0.6, 0.8])
    axis.set_rlabel_position(22.5)
    axis.grid(color="#cfd3da", linewidth=0.8, alpha=0.8)
    axis.set_title(
        "Figure 9: Maximum normalized pore pressure around the pipeline",
        fontsize=13,
        pad=24,
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8.2,
        frameon=True,
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.25)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_field(
    path: Path,
    field: dict[str, np.ndarray],
    geometry: Geometry,
) -> None:
    x_m = field["x"] * geometry.trench_height
    z_m = (field["y"] - 1.0) * geometry.trench_height
    fig, axis = plt.subplots(figsize=(10.0, 3.7))
    image = axis.pcolormesh(
        x_m,
        z_m,
        field["pressure"],
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.85, float(np.nanpercentile(field["pressure"], 99))),
    )
    pipe = plt.Circle(
        (0.0, -geometry.pipe_center_depth),
        geometry.pipe_radius,
        facecolor="white",
        edgecolor="#171717",
        linewidth=1.2,
    )
    axis.add_patch(pipe)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z below mudline (m)")
    axis.set_title("MPM-KAN predicted harmonic pore-pressure amplitude")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"$|p|/P_0$")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
