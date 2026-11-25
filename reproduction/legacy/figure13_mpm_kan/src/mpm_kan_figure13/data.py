from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SERIES_COLUMNS = {
    "numerical": ("Numerical_result_x", "Numerical_result_y"),
    "analytical": ("Analytical_solution_x", "Analytical_solution_y"),
}

QUANTITY_LABELS = {
    "pressure": r"$|p|/P_0$",
    "vertical_effective_stress": r"$|\sigma_z^\prime|/P_0$",
}


def _load_panel(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Figure 13 data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Figure 13 data has no header: {path}")
        fields = [field.strip() for field in reader.fieldnames]
        rows = [
            {
                (key.strip() if key is not None else key): value
                for key, value in row.items()
            }
            for row in reader
        ]
    required = {column for pair in SERIES_COLUMNS.values() for column in pair}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Figure 13 data is missing columns: {missing}")

    panel: dict[str, dict[str, np.ndarray]] = {}
    source_quality: dict[str, Any] = {}
    for source, (value_column, z_column) in SERIES_COLUMNS.items():
        values: list[float] = []
        depths: list[float] = []
        for row in rows:
            try:
                value = float(row[value_column])
                z = float(row[z_column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and np.isfinite(z):
                values.append(value)
                depths.append(z)
        value_array = np.asarray(values, dtype=float)
        z_array = np.asarray(depths, dtype=float)
        order = np.argsort(z_array)
        value_array = value_array[order]
        z_array = z_array[order]
        if z_array.size < 2:
            raise ValueError(f"Figure 13 {source} has fewer than two points")
        panel[source] = {"z": z_array, "value": value_array}
        source_quality[source] = {
            "n": int(z_array.size),
            "z_range": [float(z_array.min()), float(z_array.max())],
            "value_range": [
                float(value_array.min()),
                float(value_array.max()),
            ],
            "duplicate_depths": int(z_array.size - np.unique(z_array).size),
        }
    return panel, {
        "file": str(path),
        "rows": len(rows),
        "columns": fields,
        "sources": source_quality,
    }


def load_figure13_data(
    pressure_path: Path,
    stress_path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, Any],
]:
    pressure, pressure_quality = _load_panel(pressure_path)
    stress, stress_quality = _load_panel(stress_path)
    warnings: list[str] = []
    for quantity, quality in (
        ("pressure", pressure_quality),
        ("vertical_effective_stress", stress_quality),
    ):
        for source, details in quality["sources"].items():
            z_min, z_max = details["z_range"]
            if z_min < -35.05 or z_max > -9.95:
                warnings.append(
                    f"{quantity}/{source} extends slightly outside "
                    "the nominal [-35, -10] m seabed interval"
                )
    return {
        "pressure": pressure,
        "vertical_effective_stress": stress,
    }, {
        "pressure": pressure_quality,
        "vertical_effective_stress": stress_quality,
        "warnings": warnings,
    }


def interpolate_profile(
    source_z: np.ndarray,
    source_value: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    order = np.argsort(source_z)
    return np.interp(
        target_z,
        source_z[order],
        source_value[order],
    )


def error_metrics(
    prediction: np.ndarray,
    observed: np.ndarray,
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
        "bias": float(np.mean(residual)),
    }


def calculate_metrics(
    prediction: dict[str, np.ndarray],
    extracted: dict[str, dict[str, dict[str, np.ndarray]]],
    quality: dict[str, Any],
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "metric_definition": {
            "relative_l2": "||model-data||_2 / ||data||_2",
            "interpolation": (
                "linear depth interpolation at each digitized source depth"
            ),
            "training_use": (
                "Figure 13 CSV files and the independent BVP solution are "
                "loaded only after physics-only MPM-KAN training"
            ),
        },
        "mpm_kan": {},
        "digitized_numerical_vs_analytical": {},
        "data_quality": quality,
    }
    for quantity in ("pressure", "vertical_effective_stress"):
        metrics["mpm_kan"][quantity] = {}
        for source, observed in extracted[quantity].items():
            values = interpolate_profile(
                prediction["z"],
                prediction[quantity],
                observed["z"],
            )
            metrics["mpm_kan"][quantity][source] = error_metrics(
                values, observed["value"]
            )

        reference_on_network = interpolate_profile(
            reference["z"],
            reference[quantity],
            prediction["z"],
        )
        metrics["mpm_kan"][quantity][
            "independent_bvp_reference"
        ] = error_metrics(
            prediction[quantity],
            reference_on_network,
        )

        numerical = extracted[quantity]["numerical"]
        analytical = extracted[quantity]["analytical"]
        analytical_at_numerical = interpolate_profile(
            analytical["z"],
            analytical["value"],
            numerical["z"],
        )
        metrics["digitized_numerical_vs_analytical"][
            quantity
        ] = error_metrics(
            numerical["value"], analytical_at_numerical
        )
    return metrics


def save_profiles(
    path: Path,
    prediction: dict[str, np.ndarray],
    extracted: dict[str, dict[str, dict[str, np.ndarray]]],
    reference: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["quantity", "source", "z_m", "normalized_amplitude"]
        )
        for quantity in ("pressure", "vertical_effective_stress"):
            profiles = [
                (
                    "mpm_kan_pinn",
                    prediction["z"],
                    prediction[quantity],
                ),
                (
                    "independent_bvp_reference",
                    reference["z"],
                    reference[quantity],
                ),
            ]
            profiles.extend(
                (
                    f"digitized_{source}",
                    values["z"],
                    values["value"],
                )
                for source, values in extracted[quantity].items()
            )
            for source, z, values in profiles:
                for depth, value in zip(z, values):
                    writer.writerow(
                        [
                            quantity,
                            source,
                            f"{depth:.12g}",
                            f"{value:.12g}",
                        ]
                    )


def plot_figure13(
    path: Path,
    prediction: dict[str, np.ndarray],
    extracted: dict[str, dict[str, dict[str, np.ndarray]]],
    reference: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 6.0),
        sharey=True,
    )
    panels = (
        ("pressure", "(a) Pore pressure", (0.0, 1.04)),
        (
            "vertical_effective_stress",
            "(b) Vertical effective normal stress",
            (0.0, 0.72),
        ),
    )
    for axis, (quantity, title, x_limits) in zip(axes, panels):
        numerical = extracted[quantity]["numerical"]
        analytical = extracted[quantity]["analytical"]
        axis.plot(
            numerical["value"],
            numerical["z"],
            color="#d62728",
            linewidth=1.5,
            label="Digitized numerical result",
            zorder=4,
        )
        axis.plot(
            analytical["value"],
            analytical["z"],
            color="#171717",
            linewidth=1.45,
            linestyle="--",
            label="Digitized analytical solution",
            zorder=3,
        )
        axis.plot(
            reference[quantity],
            reference["z"],
            color="#777777",
            linewidth=1.25,
            linestyle=(0, (1.5, 2.2)),
            label="Independent BVP reference",
            zorder=2,
        )
        axis.plot(
            prediction[quantity],
            prediction["z"],
            color="#6f2dbd",
            linewidth=2.55,
            label="MPM-KAN PINN (physics-only)",
            zorder=5,
        )
        axis.set_title(title, fontsize=11.5, pad=10)
        axis.set_xlabel(QUANTITY_LABELS[quantity], fontsize=11.5)
        axis.set_xlim(*x_limits)
        axis.set_ylim(-35.0, -10.0)
        axis.grid(color="#d8dce2", linewidth=0.75, alpha=0.8)
    axes[0].set_ylabel(r"$z$ (m)", fontsize=11.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        fontsize=9.0,
        frameon=True,
    )
    fig.suptitle(
        "Figure 13: Maximum normalized seabed responses at x = 50 m",
        fontsize=13.2,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.88,
        bottom=0.19,
        wspace=0.17,
    )
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
