from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Fig7Case


FIELDS = ("pore_pressure", "sigma_x_eff", "sigma_z_eff", "tau_xz")


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float(np.linalg.norm(prediction - reference))
    return float(np.linalg.norm(prediction - reference) / denominator)


def interpolate_reference(
    reference: dict[str, np.ndarray], z_over_h: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        field: np.interp(z_over_h, reference["z_over_h"], reference[field])
        for field in FIELDS
    }


def calculate_metrics(
    predictions: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key, prediction in predictions.items():
        reference = interpolate_reference(references[key], prediction["z_over_h"])
        metrics[key] = {
            f"{field}_relative_l2": relative_l2(prediction[field], reference[field])
            for field in FIELDS
        }
        metrics[key]["mean_relative_l2"] = float(
            np.mean(list(metrics[key].values()))
        )
    return metrics


def save_profiles(
    predictions: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
    cases: dict[str, Fig7Case],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "label", "source", "z_over_h", *FIELDS])
        for key in cases:
            for source, profile in (
                ("mpm_kan_pinn", predictions[key]),
                ("finite_difference_reference", references[key]),
            ):
                for index, z_value in enumerate(profile["z_over_h"]):
                    writer.writerow(
                        [
                            key,
                            cases[key].label,
                            source,
                            f"{z_value:.10g}",
                            *[f"{profile[field][index]:.10g}" for field in FIELDS],
                        ]
                    )


def plot_figure7(
    predictions: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
    cases: dict[str, Fig7Case],
    output_path: Path,
) -> None:
    panels = [
        ("a", "pore_pressure", r"$|p|/P_0$", (0.2, 1.0)),
        ("b", "sigma_x_eff", r"$|\sigma'_x|/P_0$", (0.0, 0.4)),
        ("c", "sigma_z_eff", r"$|\sigma'_z|/P_0$", (0.0, 0.8)),
        ("d", "tau_xz", r"$|\tau_{xz}|/P_0$", (0.0, 0.4)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 10.0), sharey=True)
    marker_depth = np.linspace(-1.0, 0.0, 11)

    for axis, (letter, field, x_label, x_limits) in zip(axes.flat, panels):
        for key, case in cases.items():
            prediction = predictions[key]
            reference_value = np.interp(
                marker_depth, references[key]["z_over_h"], references[key][field]
            )
            axis.plot(
                prediction[field],
                prediction["z_over_h"],
                color=case.color,
                linewidth=2.0,
                label=f"{case.label} MPM-KAN PINN",
            )
            axis.plot(
                reference_value,
                marker_depth,
                linestyle="None",
                marker="D",
                markersize=4.5,
                markerfacecolor=case.color,
                markeredgecolor=case.color,
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(-1.0, 0.0)
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$z/h$")
        axis.grid(True, alpha=0.28)
        axis.text(-0.12, 1.02, letter, transform=axis.transAxes, fontsize=16, fontweight="bold")

    case_handles = [
        plt.Line2D([0], [0], color=case.color, linewidth=2.0, label=case.label)
        for case in cases.values()
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=2.0, label="MPM-KAN PINN"),
        plt.Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="None",
            marker="D",
            markersize=5,
            label="Independent FD reference",
        ),
    ]
    fig.legend(
        handles=style_handles + case_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
    )
    fig.suptitle("Zhu et al. (2024) Fig. 7 - MPM-KAN physics-informed reproduction")
    fig.tight_layout(rect=(0.02, 0.08, 1.0, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_metrics(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

