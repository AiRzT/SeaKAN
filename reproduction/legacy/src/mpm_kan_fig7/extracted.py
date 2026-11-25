from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Fig7Case
from .plotting import FIELDS


PANEL_FILES = {
    "pore_pressure": "figure7a.csv",
    "sigma_x_eff": "figure7b.csv",
    "sigma_z_eff": "figure7c.csv",
    "tau_xz": "figure7d.csv",
}
CSV_CASE_ORDER = (
    "saturated_coarse",
    "saturated_fine",
    "unsaturated_coarse",
    "unsaturated_fine",
)
CSV_CASE_PREFIX = {
    "saturated_coarse": "Saturated_Coarse",
    "saturated_fine": "Saturated_Fine",
    "unsaturated_coarse": "Unsaturated_Coarse",
    "unsaturated_fine": "Unsaturated_Fine",
}
SOURCES = ("analytical", "numerical")


def _finite_pairs(
    rows: list[list[str]], x_column: int, y_column: int
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        if max(x_column, y_column) >= len(row):
            continue
        try:
            x_value = float(row[x_column])
            y_value = float(row[y_column])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x_value) and np.isfinite(y_value):
            x_values.append(x_value)
            y_values.append(y_value)
    return np.asarray(x_values), np.asarray(y_values)


def load_extracted_figure7(
    data_dir: Path,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, np.ndarray]]]], dict[str, Any]
]:
    """Load WebPlotDigitizer exports by column position.

    The last analytical-y headers in Figure 7b and 7c are mislabeled. Their
    values and positions are valid, so positional parsing is intentional.
    """

    data_dir = data_dir.resolve()
    extracted: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
        key: {} for key in CSV_CASE_ORDER
    }
    quality: dict[str, Any] = {
        "data_dir": str(data_dir),
        "parsing": "positional (four x/y column pairs per case)",
        "depth_policy": "raw values retained; clipped to [-1, 0] only for interpolation",
        "files": {},
        "warnings": [],
    }

    for field, file_name in PANEL_FILES.items():
        path = data_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing extracted Figure 7 data: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(f"Empty extracted-data file: {path}") from error
            rows = list(reader)
        if len(header) != 16:
            raise ValueError(
                f"{path.name} has {len(header)} columns; expected 16 columns"
            )

        header_warnings: list[str] = []
        source_counts: dict[str, dict[str, int]] = {}
        duplicate_depths: dict[str, dict[str, int]] = {}
        raw_depth_min = np.inf
        raw_depth_max = -np.inf
        for case_index, key in enumerate(CSV_CASE_ORDER):
            prefix = CSV_CASE_PREFIX[key]
            base = 4 * case_index
            expected = (
                f"{prefix}_Analytical_x",
                f"{prefix}_Analytical_y",
                f"{prefix}_Numerical_x",
                f"{prefix}_Numerical_y",
            )
            for offset, (actual, expected_value) in enumerate(
                zip(header[base : base + 4], expected)
            ):
                if actual != expected_value:
                    header_warnings.append(
                        f"column {base + offset + 1}: "
                        f"{actual!r} parsed as {expected_value!r}"
                    )

            extracted[key][field] = {}
            source_counts[key] = {}
            duplicate_depths[key] = {}
            for source_index, source in enumerate(SOURCES):
                x_column = base + 2 * source_index
                y_column = x_column + 1
                x_values, y_values = _finite_pairs(rows, x_column, y_column)
                if x_values.size < 2:
                    raise ValueError(
                        f"{path.name} {key} {source} has fewer than two points"
                    )
                extracted[key][field][source] = {
                    "value": x_values,
                    "z_over_h": y_values,
                }
                source_counts[key][source] = int(x_values.size)
                duplicate_depths[key][source] = int(
                    y_values.size - np.unique(y_values).size
                )
                raw_depth_min = min(raw_depth_min, float(y_values.min()))
                raw_depth_max = max(raw_depth_max, float(y_values.max()))

        file_quality = {
            "rows": len(rows),
            "columns": len(header),
            "point_counts": source_counts,
            "duplicate_depth_counts": duplicate_depths,
            "raw_depth_range": [float(raw_depth_min), float(raw_depth_max)],
            "header_warnings": header_warnings,
        }
        quality["files"][file_name] = file_quality
        quality["warnings"].extend(f"{file_name}: {item}" for item in header_warnings)

    return extracted, quality


def _error_metrics(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    residual = prediction - observed
    denominator = float(np.linalg.norm(observed))
    relative_l2 = (
        float(np.linalg.norm(residual) / denominator)
        if denominator > 0.0
        else float(np.linalg.norm(residual))
    )
    return {
        "n": int(observed.size),
        "relative_l2": relative_l2,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def calculate_extracted_metrics(
    predictions: dict[str, dict[str, np.ndarray]],
    extracted: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    case_metrics: dict[str, Any] = {}
    all_analytical_errors: list[float] = []
    for key, prediction in predictions.items():
        field_metrics: dict[str, Any] = {}
        for field in FIELDS:
            source_metrics: dict[str, Any] = {}
            for source in SOURCES:
                observed = extracted[key][field][source]
                clipped_depth = np.clip(observed["z_over_h"], -1.0, 0.0)
                predicted = np.interp(
                    clipped_depth, prediction["z_over_h"], prediction[field]
                )
                source_metrics[source] = _error_metrics(
                    predicted, observed["value"]
                )
            all_analytical_errors.append(
                float(source_metrics["analytical"]["relative_l2"])
            )
            field_metrics[field] = source_metrics
        case_metrics[key] = field_metrics

    return {
        "metric_definition": {
            "relative_l2": "||prediction - digitized curve||_2 / ||digitized curve||_2",
            "interpolation": "MPM-KAN prediction linearly interpolated at digitized depths",
            "selection_role": "validation/parameter selection only; not included in training loss",
        },
        "summary": {
            "analytical_mean_relative_l2": float(np.mean(all_analytical_errors)),
            "analytical_max_relative_l2": float(np.max(all_analytical_errors)),
        },
        "cases": case_metrics,
        "data_quality": quality,
    }


def plot_extracted_comparison(
    predictions: dict[str, dict[str, np.ndarray]],
    extracted: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    cases: dict[str, Fig7Case],
    output_path: Path,
) -> None:
    panels = [
        ("a", "pore_pressure", r"$|p|/P_0$", (0.2, 1.0)),
        ("b", "sigma_x_eff", r"$|\sigma'_x|/P_0$", (0.0, 0.4)),
        ("c", "sigma_z_eff", r"$|\sigma'_z|/P_0$", (0.0, 0.8)),
        ("d", "tau_xz", r"$|\tau_{xz}|/P_0$", (0.0, 0.4)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.2), sharey=True)
    for axis, (letter, field, x_label, x_limits) in zip(axes.flat, panels):
        for key, case in cases.items():
            prediction = predictions[key]
            analytical = extracted[key][field]["analytical"]
            numerical = extracted[key][field]["numerical"]
            numerical_order = np.argsort(numerical["z_over_h"])
            axis.plot(
                prediction[field],
                prediction["z_over_h"],
                color=case.color,
                linewidth=2.0,
            )
            axis.plot(
                numerical["value"][numerical_order],
                numerical["z_over_h"][numerical_order],
                color=case.color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )
            axis.plot(
                analytical["value"],
                analytical["z_over_h"],
                linestyle="None",
                marker="D",
                markersize=3.5,
                markerfacecolor=case.color,
                markeredgecolor=case.color,
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(-1.0, 0.0)
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$z/h$")
        axis.grid(True, alpha=0.28)
        axis.text(
            -0.12,
            1.02,
            letter,
            transform=axis.transAxes,
            fontsize=16,
            fontweight="bold",
        )

    style_handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=2.0, label="MPM-KAN PINN"),
        plt.Line2D(
            [0], [0], color="#333333", linewidth=1.0, linestyle="--",
            label="Digitized numerical",
        ),
        plt.Line2D(
            [0], [0], color="#333333", linestyle="None", marker="D",
            markersize=4, label="Digitized analytical",
        ),
    ]
    case_handles = [
        plt.Line2D([0], [0], color=case.color, linewidth=2.0, label=case.label)
        for case in cases.values()
    ]
    fig.legend(
        handles=style_handles + case_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
    )
    fig.suptitle("Figure 7: MPM-KAN PINN vs digitized paper curves")
    fig.tight_layout(rect=(0.02, 0.09, 1.0, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
