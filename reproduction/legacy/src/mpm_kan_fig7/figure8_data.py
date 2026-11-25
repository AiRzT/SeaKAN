from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_COLUMNS = {
    "experimental": ("Experimental_data_x", "Experimental_data_y"),
    "analytical": ("Analytical_Solution_x", "Analytical_Solution_y"),
    "paper_numerical": ("Numerical_Solution_x", "Numerical_Solution_y"),
}


def load_figure8_data(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Figure 8 data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Figure 8 data has no header: {path}")
        missing = sorted(
            {
                column
                for columns in SOURCE_COLUMNS.values()
                for column in columns
                if column not in reader.fieldnames
            }
        )
        if missing:
            raise ValueError(f"Figure 8 data is missing columns: {missing}")
        rows = list(reader)

    data: dict[str, dict[str, np.ndarray]] = {}
    quality_sources: dict[str, Any] = {}
    for source, (value_column, depth_column) in SOURCE_COLUMNS.items():
        values: list[float] = []
        depths: list[float] = []
        for row in rows:
            try:
                value = float(row[value_column])
                depth = float(row[depth_column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and np.isfinite(depth):
                values.append(value)
                depths.append(depth)
        value_array = np.asarray(values)
        depth_array = np.asarray(depths)
        if value_array.size < 2:
            raise ValueError(f"Figure 8 {source} series has fewer than two points")
        data[source] = {
            "pore_pressure": value_array,
            "z_over_h": depth_array,
        }
        quality_sources[source] = {
            "n": int(value_array.size),
            "pressure_range": [
                float(value_array.min()),
                float(value_array.max()),
            ],
            "depth_range": [
                float(depth_array.min()),
                float(depth_array.max()),
            ],
            "duplicate_depths": int(
                depth_array.size - np.unique(depth_array).size
            ),
        }

    quality = {
        "file": str(path),
        "rows": len(rows),
        "columns": list(reader.fieldnames),
        "sources": quality_sources,
        "warnings": [],
    }
    return data, quality


def _error_metrics(
    prediction: np.ndarray, observed: np.ndarray
) -> dict[str, float | int]:
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
    }


def calculate_figure8_metrics(
    prediction: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "metric_definition": {
            "relative_l2": "||model - data||_2 / ||data||_2",
            "interpolation": "model linearly interpolated at each digitized depth",
            "training_use": "digitized curves are validation only and are not in the loss",
        },
        "mpm_kan": {},
        "finite_difference_reference": {},
        "data_quality": quality,
    }
    for source, observed in extracted.items():
        depths = np.clip(observed["z_over_h"], -1.0, 0.0)
        network_values = np.interp(
            depths, prediction["z_over_h"], prediction["pore_pressure"]
        )
        reference_values = np.interp(
            depths, reference["z_over_h"], reference["pore_pressure"]
        )
        metrics["mpm_kan"][source] = _error_metrics(
            network_values, observed["pore_pressure"]
        )
        metrics["finite_difference_reference"][source] = _error_metrics(
            reference_values, observed["pore_pressure"]
        )
    reference_on_network = np.interp(
        prediction["z_over_h"],
        reference["z_over_h"],
        reference["pore_pressure"],
    )
    metrics["mpm_kan"]["finite_difference_reference"] = _error_metrics(
        prediction["pore_pressure"], reference_on_network
    )
    return metrics


def save_figure8_profiles(
    prediction: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "z_over_h", "normalized_pore_pressure"])
        for source, profile in (
            ("mpm_kan_pinn", prediction),
            ("finite_difference_reference", reference),
            *extracted.items(),
        ):
            for depth, value in zip(
                profile["z_over_h"], profile["pore_pressure"]
            ):
                writer.writerow([source, f"{depth:.12g}", f"{value:.12g}"])


def plot_figure8(
    prediction: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    analytical = extracted["analytical"]
    paper_numerical = extracted["paper_numerical"]
    experimental = extracted["experimental"]
    analytical_order = np.argsort(analytical["z_over_h"])
    numerical_order = np.argsort(paper_numerical["z_over_h"])

    fig, axis = plt.subplots(figsize=(6.7, 7.0))
    axis.plot(
        prediction["pore_pressure"],
        prediction["z_over_h"],
        color="#1f77b4",
        linewidth=2.6,
        label="MPM-KAN PINN",
        zorder=5,
    )
    axis.plot(
        analytical["pore_pressure"][analytical_order],
        analytical["z_over_h"][analytical_order],
        color="#262626",
        linewidth=1.6,
        linestyle=(0, (5, 3)),
        label="Digitized analytical solution",
        zorder=3,
    )
    axis.plot(
        paper_numerical["pore_pressure"][numerical_order],
        paper_numerical["z_over_h"][numerical_order],
        color="#d62728",
        linewidth=1.8,
        label="Digitized paper numerical solution",
        zorder=4,
    )
    axis.plot(
        reference["pore_pressure"],
        reference["z_over_h"],
        color="#777777",
        linewidth=1.2,
        linestyle=":",
        label="Independent FD reference",
        zorder=2,
    )
    axis.plot(
        experimental["pore_pressure"],
        experimental["z_over_h"],
        linestyle="None",
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        markeredgecolor="#111111",
        label="Digitized experimental data",
        zorder=6,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.9, 0.0)
    axis.set_xlabel(r"$|p|/P_0$", fontsize=12)
    axis.set_ylabel(r"$z/h$", fontsize=12)
    fig.suptitle(
        "Figure 8: Maximum normalized pore-water pressure",
        fontsize=14,
        y=0.975,
    )
    fig.text(
        0.5,
        0.945,
        "Zhu et al. (2024), Table 4 parameters",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    axis.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    axis.legend(loc="lower right", frameon=True, fontsize=8.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
