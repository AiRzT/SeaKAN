from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite_pairs(
    rows: list[list[str]],
    theta_index: int,
    radius_index: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    theta: list[float] = []
    radius: list[float] = []
    orphan_theta = 0
    orphan_radius = 0
    for row in rows:
        try:
            angle = float(row[theta_index])
            angle_ok = np.isfinite(angle)
        except (ValueError, IndexError):
            angle_ok = False
            angle = 0.0
        try:
            value = float(row[radius_index])
            value_ok = np.isfinite(value)
        except (ValueError, IndexError):
            value_ok = False
            value = 0.0
        if angle_ok and value_ok:
            theta.append(angle % (2.0 * math.pi))
            radius.append(value)
        elif angle_ok:
            orphan_theta += 1
        elif value_ok:
            orphan_radius += 1
    theta_array = np.asarray(theta, dtype=float)
    radius_array = np.asarray(radius, dtype=float)
    order = np.argsort(theta_array)
    return (
        theta_array[order],
        radius_array[order],
        orphan_theta,
        orphan_radius,
    )


def _load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Figure 15 data file not found: {path}"
        )
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2:
        raise ValueError(f"Figure 15 data is empty: {path}")
    return rows[0], rows[1:]


def _series_quality(
    theta: np.ndarray,
    radius: np.ndarray,
    orphan_theta: int,
    orphan_radius: int,
) -> dict[str, Any]:
    return {
        "n": int(theta.size),
        "theta_range": [
            float(theta.min()),
            float(theta.max()),
        ],
        "radius_range": [
            float(radius.min()),
            float(radius.max()),
        ],
        "orphan_theta": orphan_theta,
        "orphan_radius": orphan_radius,
        "duplicate_theta": int(
            theta.size - np.unique(theta).size
        ),
    }


def load_figure15_data(
    test10_path: Path,
    test49_path: Path,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    header10, rows10 = _load_csv(test10_path)
    header49, rows49 = _load_csv(test49_path)
    if len(header10) != 4 or len(header49) != 6:
        raise ValueError(
            "Figure 15 CSV column count does not match the two panels"
        )
    exp10 = _finite_pairs(rows10, 0, 1)
    pore10 = _finite_pairs(rows10, 2, 3)
    exp49 = _finite_pairs(rows49, 0, 1)
    pore49_raw = _finite_pairs(rows49, 2, 3)
    water49_raw = _finite_pairs(rows49, 4, 5)

    pore_theta, pore_radius = pore49_raw[:2]
    pore_keep = pore_theta >= math.pi - 0.05
    pore49 = (
        pore_theta[pore_keep],
        pore_radius[pore_keep],
        pore49_raw[2],
        pore49_raw[3],
    )
    water_theta, water_radius = water49_raw[:2]
    wrap = water_theta >= 2.0 * math.pi - 0.05
    water_theta = water_theta.copy()
    water_theta[wrap] = 0.0
    water_keep = water_theta <= math.pi + 0.05
    water_theta = water_theta[water_keep]
    water_radius = water_radius[water_keep]
    order = np.argsort(water_theta)
    water49 = (
        water_theta[order],
        water_radius[order],
        water49_raw[2],
        water49_raw[3],
    )

    def profile(
        values: tuple[np.ndarray, np.ndarray, int, int]
    ) -> dict[str, np.ndarray]:
        return {"theta": values[0], "radius": values[1]}

    extracted = {
        "test10": {
            "experimental": profile(exp10),
            "pore_numerical": profile(pore10),
        },
        "test49": {
            "experimental": profile(exp49),
            "pore_numerical": profile(pore49),
            "hydrodynamic_numerical": profile(water49),
        },
    }
    warnings: list[str] = []
    if pore10[2] > 0:
        warnings.append(
            f"figure15a contains {pore10[2]} numerical theta "
            "entries without matching radius values; they were ignored"
        )
    quality = {
        "test10": {
            "file": str(test10_path.resolve()),
            "rows": len(rows10),
            "headers": header10,
            "experimental": _series_quality(*exp10),
            "pore_numerical": _series_quality(*pore10),
        },
        "test49": {
            "file": str(test49_path.resolve()),
            "rows": len(rows49),
            "headers": header49,
            "experimental": _series_quality(*exp49),
            "pore_numerical": _series_quality(*pore49),
            "hydrodynamic_numerical": _series_quality(*water49),
        },
        "warnings": warnings,
    }
    return extracted, quality


def circular_interpolate(
    source_theta: np.ndarray,
    source_value: np.ndarray,
    target_theta: np.ndarray,
) -> np.ndarray:
    theta = np.mod(source_theta, 2.0 * math.pi)
    order = np.argsort(theta)
    theta = theta[order]
    value = source_value[order]
    extended_theta = np.concatenate(
        (
            theta[-1:] - 2.0 * math.pi,
            theta,
            theta[:1] + 2.0 * math.pi,
        )
    )
    extended_value = np.concatenate(
        (value[-1:], value, value[:1])
    )
    return np.interp(
        np.mod(target_theta, 2.0 * math.pi),
        extended_theta,
        extended_value,
    )


def error_metrics(
    prediction: np.ndarray, observed: np.ndarray
) -> dict[str, float | int | None]:
    residual = prediction - observed
    denominator = float(np.linalg.norm(observed))
    mse = float(np.mean(residual**2))
    observed_variance = float(np.var(observed))
    residual_variance = float(np.var(residual))
    observed_mean = float(np.mean(observed))
    sst = float(np.sum((observed - observed_mean) ** 2))
    r_squared = (
        float(1.0 - np.sum(residual**2) / sst)
        if sst > np.finfo(float).eps
        else None
    )
    explained_variance = (
        float(1.0 - residual_variance / observed_variance)
        if observed_variance > np.finfo(float).eps
        else None
    )
    return {
        "n": int(observed.size),
        "relative_l2": (
            float(np.linalg.norm(residual) / denominator)
            if denominator > 0.0
            else float(np.linalg.norm(residual))
        ),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r_squared": r_squared,
        "explained_variance": explained_variance,
        "reference_mean": observed_mean,
        "reference_standard_deviation": float(
            np.sqrt(observed_variance)
        ),
        "reference_coefficient_of_variation": (
            float(np.sqrt(observed_variance) / abs(observed_mean))
            if abs(observed_mean) > np.finfo(float).eps
            else None
        ),
    }


def calculate_metrics(
    predictions: dict[str, dict[str, np.ndarray]],
    extracted: dict[str, dict[str, dict[str, np.ndarray]]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "metric_definition": {
            "relative_l2": "||model-data||_2 / ||data||_2",
            "r_squared": "1 - SSE/SST using the untransformed amplitudes",
            "explained_variance": (
                "1 - Var(model-data)/Var(data); reported as a "
                "bias-insensitive diagnostic, not as R^2"
            ),
            "interpolation": (
                "periodic interpolation for full-circle profiles and "
                "linear interpolation on each exposed/embedded semicircle"
            ),
            "training_use": (
                "Figure 15 CSV files are opened only after coupled "
                "water-seabed physics training"
            ),
        },
        "mpm_kan": {"test10": {}, "test49": {}},
        "data_quality": quality,
    }
    prediction10 = predictions["test10"]
    for source in ("experimental", "pore_numerical"):
        observed = extracted["test10"][source]
        values = circular_interpolate(
            prediction10["theta"],
            prediction10["combined_pressure"],
            observed["theta"],
        )
        metrics["mpm_kan"]["test10"][source] = error_metrics(
            values, observed["radius"]
        )

    prediction49 = predictions["test49"]
    experimental = extracted["test49"]["experimental"]
    combined = circular_interpolate(
        prediction49["theta"],
        prediction49["combined_pressure"],
        experimental["theta"],
    )
    metrics["mpm_kan"]["test49"]["experimental"] = error_metrics(
        combined, experimental["radius"]
    )
    upper = experimental["theta"] <= math.pi + 0.05
    metrics["mpm_kan"]["test49"][
        "experimental_upper_hydrodynamic"
    ] = error_metrics(
        combined[upper], experimental["radius"][upper]
    )
    metrics["mpm_kan"]["test49"][
        "experimental_lower_pore"
    ] = error_metrics(
        combined[~upper], experimental["radius"][~upper]
    )
    numerical_predictions: list[np.ndarray] = []
    numerical_observations: list[np.ndarray] = []
    for source, prediction_key in (
        ("pore_numerical", "pore_pressure"),
        (
            "hydrodynamic_numerical",
            "hydrodynamic_pressure",
        ),
    ):
        observed = extracted["test49"][source]
        finite = np.isfinite(prediction49[prediction_key])
        values = np.interp(
            observed["theta"],
            prediction49["theta"][finite],
            prediction49[prediction_key][finite],
        )
        metrics["mpm_kan"]["test49"][source] = error_metrics(
            values, observed["radius"]
        )
        numerical_predictions.append(values)
        numerical_observations.append(observed["radius"])
    metrics["mpm_kan"]["test49"]["numerical_combined"] = (
        error_metrics(
            np.concatenate(numerical_predictions),
            np.concatenate(numerical_observations),
        )
    )
    return metrics


def save_profiles(
    path: Path,
    predictions: dict[str, dict[str, np.ndarray]],
    extracted: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "source",
                "theta_rad",
                "normalized_pressure",
            ]
        )
        for case_name, prediction in predictions.items():
            for key in (
                "combined_pressure",
                "pore_pressure",
                "hydrodynamic_pressure",
            ):
                for theta, value in zip(
                    prediction["theta"], prediction[key]
                ):
                    if np.isfinite(value):
                        writer.writerow(
                            [
                                case_name,
                                f"mpm_kan_{key}",
                                f"{theta:.12g}",
                                f"{value:.12g}",
                            ]
                        )
            for source, profile in extracted[case_name].items():
                for theta, value in zip(
                    profile["theta"], profile["radius"]
                ):
                    writer.writerow(
                        [
                            case_name,
                            f"digitized_{source}",
                            f"{theta:.12g}",
                            f"{value:.12g}",
                        ]
                    )


def plot_figure15(
    path: Path,
    predictions: dict[str, dict[str, np.ndarray]],
    extracted: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 6.3),
        subplot_kw={"projection": "polar"},
    )
    for axis, case_name, panel in zip(
        axes,
        ("test10", "test49"),
        ("(a) Test 10 - fully buried", "(b) Test 49 - partially buried"),
    ):
        prediction = predictions[case_name]
        observed = extracted[case_name]
        if case_name == "test10":
            paper = observed["pore_numerical"]
            axis.plot(
                paper["theta"],
                paper["radius"],
                color="#d62728",
                linewidth=1.55,
                label="Digitized pore-pressure result",
            )
            axis.plot(
                prediction["theta"],
                prediction["combined_pressure"],
                color="#6f2dbd",
                linewidth=2.5,
                label="MPM-KAN pore pressure",
                zorder=5,
            )
        else:
            paper_p = observed["pore_numerical"]
            paper_water = observed["hydrodynamic_numerical"]
            axis.plot(
                paper_p["theta"],
                paper_p["radius"],
                color="#d62728",
                linewidth=1.55,
                label="Digitized pore-pressure result",
            )
            axis.plot(
                paper_water["theta"],
                paper_water["radius"],
                color="#d62728",
                linewidth=1.55,
                linestyle=(0, (4, 3)),
                label="Digitized hydrodynamic result",
            )
            pore_finite = np.isfinite(
                prediction["pore_pressure"]
            )
            water_finite = np.isfinite(
                prediction["hydrodynamic_pressure"]
            )
            axis.plot(
                prediction["theta"][pore_finite],
                prediction["pore_pressure"][pore_finite],
                color="#6f2dbd",
                linewidth=2.5,
                label="MPM-KAN pore pressure",
                zorder=5,
            )
            axis.plot(
                prediction["theta"][water_finite],
                prediction["hydrodynamic_pressure"][water_finite],
                color="#6f2dbd",
                linewidth=2.5,
                linestyle=(0, (4, 3)),
                label="MPM-KAN hydrodynamic pressure",
                zorder=5,
            )
        experimental = observed["experimental"]
        axis.scatter(
            experimental["theta"],
            experimental["radius"],
            s=43,
            facecolor="#b8b8b8",
            edgecolor="#252525",
            linewidth=1.25,
            label="Digitized experiment",
            zorder=8,
        )
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_xticks(
            [
                0.0,
                0.25 * math.pi,
                0.5 * math.pi,
                0.75 * math.pi,
                math.pi,
                1.25 * math.pi,
                1.5 * math.pi,
                1.75 * math.pi,
            ]
        )
        axis.set_xticklabels(
            [
                r"$0$",
                r"$\pi/4$",
                r"$\pi/2$",
                r"$3\pi/4$",
                r"$\pi$",
                r"$5\pi/4$",
                r"$3\pi/2$",
                r"$7\pi/4$",
            ],
            fontsize=9.0,
        )
        axis.set_ylim(0.0, 1.04)
        axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_rlabel_position(83)
        axis.grid(color="#d4d8de", linewidth=0.75)
        axis.set_title(panel, fontsize=11.7, pad=22)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8.5,
        frameon=True,
    )
    fig.suptitle(
        "Figure 15: Normalized pressure around the trenched pipeline",
        fontsize=13.3,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.87,
        bottom=0.19,
        wspace=0.25,
    )
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
