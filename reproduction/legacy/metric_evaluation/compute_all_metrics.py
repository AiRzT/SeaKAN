from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def regression_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int | None]:
    truth = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("paired arrays must have the same non-zero shape")
    if not np.all(np.isfinite(truth)) or not np.all(
        np.isfinite(estimate)
    ):
        raise ValueError("paired arrays must be finite")
    residual = estimate - truth
    squared_error = residual**2
    sse = float(squared_error.sum())
    mse = float(squared_error.mean())
    reference_energy = float(np.square(truth).sum())
    total_variation = float(
        np.square(truth - truth.mean()).sum()
    )
    epsilon = np.finfo(np.float64).eps
    return {
        "n": int(truth.size),
        "r2": (
            float(1.0 - sse / total_variation)
            if total_variation > epsilon
            else None
        ),
        "relative_l2": (
            float(np.sqrt(sse / reference_energy))
            if reference_energy > epsilon
            else None
        ),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def _unique_average(
    coordinate: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    values = values[order]
    unique, inverse = np.unique(
        coordinate, return_inverse=True
    )
    sums = np.zeros_like(unique, dtype=np.float64)
    counts = np.zeros_like(unique, dtype=np.float64)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return unique, sums / counts


def linear_interpolate(
    source_coordinate: np.ndarray,
    source_value: np.ndarray,
    target_coordinate: np.ndarray,
) -> np.ndarray:
    coordinate, values = _unique_average(
        np.asarray(source_coordinate, dtype=np.float64),
        np.asarray(source_value, dtype=np.float64),
    )
    return np.interp(target_coordinate, coordinate, values)


def periodic_interpolate(
    source_theta: np.ndarray,
    source_value: np.ndarray,
    target_theta: np.ndarray,
) -> np.ndarray:
    period = 2.0 * np.pi
    theta, values = _unique_average(
        np.mod(source_theta, period), source_value
    )
    if theta.size < 2:
        raise ValueError("periodic interpolation needs two angles")
    extended_theta = np.concatenate(
        (theta[-1:] - period, theta, theta[:1] + period)
    )
    extended_value = np.concatenate(
        (values[-1:], values, values[:1])
    )
    return np.interp(
        np.mod(target_theta, period),
        extended_theta,
        extended_value,
    )


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def _finite_pairs(
    rows: Iterable[dict[str, str]],
    coordinate_column: str,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate: list[float] = []
    values: list[float] = []
    for row in rows:
        try:
            x = float(row[coordinate_column])
            y = float(row[value_column])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            coordinate.append(x)
            values.append(y)
    return np.asarray(coordinate), np.asarray(values)


def _metric_row(
    *,
    figure: str,
    case: str,
    quantity: str,
    reference_source: str,
    coordinate_method: str,
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_path: Path,
    prediction_path: Path,
) -> dict[str, object]:
    metrics = regression_metrics(reference, prediction)
    return {
        "figure": figure,
        "case": case,
        "quantity": quantity,
        "reference_source": reference_source,
        "coordinate_method": coordinate_method,
        **metrics,
        "l2_percent": (
            None
            if metrics["relative_l2"] is None
            else 100.0 * float(metrics["relative_l2"])
        ),
        "reference_path": reference_path.relative_to(ROOT).as_posix(),
        "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
    }


def score_figure7() -> list[dict[str, object]]:
    prediction_path = (
        ROOT / "src" / "outputs" / "figure7"
        / "figure7_profiles.csv"
    )
    rows = _read_dict_rows(prediction_path)
    cases = (
        "saturated_coarse",
        "saturated_fine",
        "unsaturated_coarse",
        "unsaturated_fine",
    )
    fields = (
        "pore_pressure",
        "sigma_x_eff",
        "sigma_z_eff",
        "tau_xz",
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, dict[str, np.ndarray]] = {}
    for case in cases:
        model = [
            row for row in rows
            if row["case"] == case
            and row["source"] == "mpm_kan_pinn"
        ]
        fd = [
            row for row in rows
            if row["case"] == case
            and row["source"] == "finite_difference_reference"
        ]
        predictions[case] = {
            "z": np.asarray(
                [float(row["z_over_h"]) for row in model]
            ),
            **{
                field: np.asarray(
                    [float(row[field]) for row in model]
                )
                for field in fields
            },
        }
        references[case] = {
            "z": np.asarray(
                [float(row["z_over_h"]) for row in fd]
            ),
            **{
                field: np.asarray(
                    [float(row[field]) for row in fd]
                )
                for field in fields
            },
        }
    output: list[dict[str, object]] = []
    for case in cases:
        prediction = predictions[case]
        fd = references[case]
        for field in fields:
            estimate = linear_interpolate(
                prediction["z"],
                prediction[field],
                fd["z"],
            )
            output.append(
                _metric_row(
                    figure="figure7",
                    case=case,
                    quantity=field,
                    reference_source="finite_difference_reference",
                    coordinate_method="linear_depth",
                    reference=fd[field],
                    prediction=estimate,
                    reference_path=prediction_path,
                    prediction_path=prediction_path,
                )
            )

    panel_files = {
        "pore_pressure": "figure7a.csv",
        "sigma_x_eff": "figure7b.csv",
        "sigma_z_eff": "figure7c.csv",
        "tau_xz": "figure7d.csv",
    }
    csv_case_order = (
        "saturated_coarse",
        "saturated_fine",
        "unsaturated_coarse",
        "unsaturated_fine",
    )
    for field, filename in panel_files.items():
        source_path = ROOT / "figure7_data" / filename
        with source_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            raw = list(csv.reader(stream))
        data_rows = raw[1:]
        for case_index, case in enumerate(csv_case_order):
            for source_index, source in enumerate(
                ("analytical", "numerical")
            ):
                x_column = 4 * case_index + 2 * source_index
                y_column = x_column + 1
                values: list[float] = []
                depths: list[float] = []
                for row in data_rows:
                    try:
                        value = float(row[x_column])
                        depth = float(row[y_column])
                    except (IndexError, TypeError, ValueError):
                        continue
                    if np.isfinite(value) and np.isfinite(depth):
                        values.append(value)
                        depths.append(depth)
                truth = np.asarray(values)
                target = np.clip(
                    np.asarray(depths), -1.0, 0.0
                )
                prediction = predictions[case]
                estimate = linear_interpolate(
                    prediction["z"],
                    prediction[field],
                    target,
                )
                output.append(
                    _metric_row(
                        figure="figure7",
                        case=case,
                        quantity=field,
                        reference_source=f"digitized_{source}",
                        coordinate_method="linear_depth",
                        reference=truth,
                        prediction=estimate,
                        reference_path=source_path,
                        prediction_path=prediction_path,
                    )
                )
    return output


def score_profile_file(
    *,
    figure: str,
    path: Path,
    coordinate_column: str,
    value_column: str,
    prediction_source: str,
    reference_sources: Iterable[str],
    periodic: bool,
    quantity: str,
) -> list[dict[str, object]]:
    rows = _read_dict_rows(path)
    model = [
        row for row in rows
        if row["source"] == prediction_source
    ]
    model_x, model_y = _finite_pairs(
        model, coordinate_column, value_column
    )
    interpolate = (
        periodic_interpolate if periodic else linear_interpolate
    )
    output = []
    for source in reference_sources:
        observed = [
            row for row in rows if row["source"] == source
        ]
        target_x, truth = _finite_pairs(
            observed, coordinate_column, value_column
        )
        estimate = interpolate(model_x, model_y, target_x)
        output.append(
            _metric_row(
                figure=figure,
                case="main",
                quantity=quantity,
                reference_source=source,
                coordinate_method=(
                    "periodic_angle" if periodic
                    else "linear_depth"
                ),
                reference=truth,
                prediction=estimate,
                reference_path=path,
                prediction_path=path,
            )
        )
    return output


def score_figure8() -> list[dict[str, object]]:
    path = (
        ROOT / "figure8_mpm_kan" / "artifacts"
        / "figure8_profiles.csv"
    )
    output = score_profile_file(
        figure="figure8",
        path=path,
        coordinate_column="z_over_h",
        value_column="normalized_pore_pressure",
        prediction_source="mpm_kan_pinn",
        reference_sources=(
            "experimental",
            "analytical",
            "paper_numerical",
        ),
        periodic=False,
        quantity="pore_pressure",
    )
    rows = _read_dict_rows(path)
    model = [
        row for row in rows
        if row["source"] == "mpm_kan_pinn"
    ]
    reference = [
        row for row in rows
        if row["source"] == "finite_difference_reference"
    ]
    model_z, estimate = _finite_pairs(
        model, "z_over_h", "normalized_pore_pressure"
    )
    reference_z, reference_value = _finite_pairs(
        reference, "z_over_h", "normalized_pore_pressure"
    )
    truth = linear_interpolate(
        reference_z, reference_value, model_z
    )
    output.insert(
        0,
        _metric_row(
            figure="figure8",
            case="main",
            quantity="pore_pressure",
            reference_source="finite_difference_reference",
            coordinate_method="linear_depth_on_prediction_grid",
            reference=truth,
            prediction=estimate,
            reference_path=path,
            prediction_path=path,
        ),
    )
    return output


def score_figure9() -> list[dict[str, object]]:
    path = (
        ROOT / "figure9_mpm_kan" / "artifacts"
        / "figure9_profiles.csv"
    )
    return score_profile_file(
        figure="figure9",
        path=path,
        coordinate_column="theta_rad",
        value_column="normalized_pressure",
        prediction_source="mpm_kan_pinn",
        reference_sources=(
            "finite_difference_reference",
            "experimental",
            "poroelasticity",
            "darcy",
            "jeng_numerical",
            "paper_this_study",
        ),
        periodic=True,
        quantity="pipe_pressure",
    )


def score_figure13() -> list[dict[str, object]]:
    path = (
        ROOT / "figure13_mpm_kan" / "artifacts"
        / "figure13_profiles.csv"
    )
    rows = _read_dict_rows(path)
    output = []
    for quantity in (
        "pressure",
        "vertical_effective_stress",
    ):
        quantity_rows = [
            row for row in rows if row["quantity"] == quantity
        ]
        model = [
            row for row in quantity_rows
            if row["source"] == "mpm_kan_pinn"
        ]
        model_z, model_y = _finite_pairs(
            model, "z_m", "normalized_amplitude"
        )
        for source in (
            "independent_bvp_reference",
            "digitized_numerical",
            "digitized_analytical",
        ):
            observed = [
                row for row in quantity_rows
                if row["source"] == source
            ]
            target_z, truth = _finite_pairs(
                observed, "z_m", "normalized_amplitude"
            )
            if source == "independent_bvp_reference":
                truth = linear_interpolate(
                    target_z, truth, model_z
                )
                target_z = model_z
                estimate = model_y
                method = "linear_depth_on_prediction_grid"
            else:
                estimate = linear_interpolate(
                    model_z, model_y, target_z
                )
                method = "linear_depth"
            output.append(
                _metric_row(
                    figure="figure13",
                    case="main",
                    quantity=quantity,
                    reference_source=source,
                    coordinate_method=method,
                    reference=truth,
                    prediction=estimate,
                    reference_path=path,
                    prediction_path=path,
                )
            )
    return output


def score_figure15() -> list[dict[str, object]]:
    path = (
        ROOT / "figure15_mpm_kan" / "artifacts"
        / "figure15_profiles.csv"
    )
    rows = _read_dict_rows(path)
    output = []

    def series(
        case: str, source: str
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = [
            row for row in rows
            if row["case"] == case
            and row["source"] == source
        ]
        return _finite_pairs(
            selected, "theta_rad", "normalized_pressure"
        )

    def add(
        case: str,
        prediction_source: str,
        reference_source: str,
        *,
        method: str,
        suffix: str = "",
        mask=None,
    ) -> None:
        model_x, model_y = series(case, prediction_source)
        target_x, truth = series(case, reference_source)
        if mask is not None:
            keep = mask(target_x)
            target_x, truth = target_x[keep], truth[keep]
        estimate = (
            periodic_interpolate(
                model_x, model_y, target_x
            )
            if method == "periodic_angle"
            else linear_interpolate(
                model_x, model_y, target_x
            )
        )
        output.append(
            _metric_row(
                figure="figure15",
                case=case,
                quantity=prediction_source.removeprefix(
                    "mpm_kan_"
                ),
                reference_source=reference_source + suffix,
                coordinate_method=method,
                reference=truth,
                prediction=estimate,
                reference_path=path,
                prediction_path=path,
            )
        )

    add(
        "test10",
        "mpm_kan_combined_pressure",
        "digitized_experimental",
        method="periodic_angle",
    )
    add(
        "test10",
        "mpm_kan_combined_pressure",
        "digitized_pore_numerical",
        method="periodic_angle",
    )
    add(
        "test49",
        "mpm_kan_combined_pressure",
        "digitized_experimental",
        method="periodic_angle",
    )
    add(
        "test49",
        "mpm_kan_combined_pressure",
        "digitized_experimental",
        method="periodic_angle",
        suffix="_upper",
        mask=lambda theta: theta <= math.pi + 0.05,
    )
    add(
        "test49",
        "mpm_kan_combined_pressure",
        "digitized_experimental",
        method="periodic_angle",
        suffix="_lower",
        mask=lambda theta: theta > math.pi + 0.05,
    )
    add(
        "test49",
        "mpm_kan_pore_pressure",
        "digitized_pore_numerical",
        method="linear_semicircle",
    )
    add(
        "test49",
        "mpm_kan_hydrodynamic_pressure",
        "digitized_hydrodynamic_numerical",
        method="linear_semicircle",
    )
    return output


def _summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    by_figure: dict[str, dict[str, object]] = {}
    for figure in sorted({str(row["figure"]) for row in rows}):
        selected = [
            row for row in rows if row["figure"] == figure
        ]
        by_figure[figure] = {
            "comparisons": len(selected),
            "samples": int(
                sum(int(row["n"]) for row in selected)
            ),
            "macro_mean": {
                metric: float(
                    np.mean(
                        [
                            float(row[metric])
                            for row in selected
                            if row[metric] is not None
                        ]
                    )
                )
                for metric in (
                    "r2",
                    "relative_l2",
                    "mse",
                    "rmse",
                )
            },
            "median": {
                metric: float(
                    np.median(
                        [
                            float(row[metric])
                            for row in selected
                            if row[metric] is not None
                        ]
                    )
                )
                for metric in (
                    "r2",
                    "relative_l2",
                    "mse",
                    "rmse",
                )
            },
        }
    finite_r2 = [
        row for row in rows if row["r2"] is not None
    ]
    return {
        "figures": by_figure,
        "comparison_count": len(rows),
        "sample_count_with_repeated_reference_series": int(
            sum(int(row["n"]) for row in rows)
        ),
        "best_r2": max(
            finite_r2, key=lambda row: float(row["r2"])
        ),
        "worst_r2": min(
            finite_r2, key=lambda row: float(row["r2"])
        ),
        "lowest_relative_l2": min(
            (
                row
                for row in rows
                if row["relative_l2"] is not None
            ),
            key=lambda row: float(row["relative_l2"]),
        ),
        "highest_relative_l2": max(
            (
                row
                for row in rows
                if row["relative_l2"] is not None
            ),
            key=lambda row: float(row["relative_l2"]),
        ),
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile_existing_artifacts(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Cross-check L2/RMSE against the projects' saved metrics."""

    lookup = {
        (
            row["figure"],
            row["case"],
            row["quantity"],
            row["reference_source"],
        ): row
        for row in rows
    }
    checks: list[dict[str, object]] = []

    def compare(
        key: tuple[str, str, str, str],
        expected: dict[str, object],
        source_path: Path,
    ) -> None:
        actual = lookup[key]
        for metric in ("relative_l2", "rmse"):
            if metric not in expected:
                continue
            difference = abs(
                float(actual[metric]) - float(expected[metric])
            )
            checks.append(
                {
                    "key": "|".join(key),
                    "metric": metric,
                    "absolute_difference": difference,
                    "source": source_path.relative_to(ROOT).as_posix(),
                }
            )

    figure8_path = (
        ROOT / "figure8_mpm_kan" / "artifacts"
        / "figure8_metrics.json"
    )
    figure8 = _load_json(figure8_path)["mpm_kan"]
    for source in (
        "experimental",
        "analytical",
        "paper_numerical",
        "finite_difference_reference",
    ):
        compare(
            ("figure8", "main", "pore_pressure", source),
            figure8[source],
            figure8_path,
        )

    figure9_path = (
        ROOT / "figure9_mpm_kan" / "artifacts"
        / "figure9_metrics.json"
    )
    figure9 = _load_json(figure9_path)["mpm_kan"]
    for source in (
        "experimental",
        "poroelasticity",
        "darcy",
        "jeng_numerical",
        "paper_this_study",
        "finite_difference_reference",
    ):
        compare(
            ("figure9", "main", "pipe_pressure", source),
            figure9[source],
            figure9_path,
        )

    figure13_path = (
        ROOT / "figure13_mpm_kan" / "artifacts"
        / "figure13_metrics.json"
    )
    figure13 = _load_json(figure13_path)["mpm_kan"]
    figure13_quantities = {
        "pressure": "pressure",
        "vertical_effective_stress": "vertical_effective_stress",
    }
    figure13_sources = {
        "numerical": "digitized_numerical",
        "analytical": "digitized_analytical",
        "independent_bvp_reference": "independent_bvp_reference",
    }
    for artifact_quantity, quantity in figure13_quantities.items():
        for artifact_source, source in figure13_sources.items():
            compare(
                ("figure13", "main", quantity, source),
                figure13[artifact_quantity][artifact_source],
                figure13_path,
            )

    figure15_path = (
        ROOT / "figure15_mpm_kan" / "artifacts"
        / "figure15_metrics.json"
    )
    figure15 = _load_json(figure15_path)["mpm_kan"]
    figure15_map = {
        (
            "figure15",
            "test10",
            "combined_pressure",
            "digitized_experimental",
        ): figure15["test10"]["experimental"],
        (
            "figure15",
            "test10",
            "combined_pressure",
            "digitized_pore_numerical",
        ): figure15["test10"]["pore_numerical"],
        (
            "figure15",
            "test49",
            "combined_pressure",
            "digitized_experimental",
        ): figure15["test49"]["experimental"],
        (
            "figure15",
            "test49",
            "combined_pressure",
            "digitized_experimental_upper",
        ): figure15["test49"]["experimental_upper_hydrodynamic"],
        (
            "figure15",
            "test49",
            "combined_pressure",
            "digitized_experimental_lower",
        ): figure15["test49"]["experimental_lower_pore"],
        (
            "figure15",
            "test49",
            "pore_pressure",
            "digitized_pore_numerical",
        ): figure15["test49"]["pore_numerical"],
        (
            "figure15",
            "test49",
            "hydrodynamic_pressure",
            "digitized_hydrodynamic_numerical",
        ): figure15["test49"]["hydrodynamic_numerical"],
    }
    for key, expected in figure15_map.items():
        compare(key, expected, figure15_path)

    figure7_path = (
        ROOT / "src" / "outputs" / "figure7"
        / "extracted_data_metrics.json"
    )
    figure7 = _load_json(figure7_path)["cases"]
    for case, quantities in figure7.items():
        for quantity, sources in quantities.items():
            for source, expected in sources.items():
                compare(
                    (
                        "figure7",
                        case,
                        quantity,
                        f"digitized_{source}",
                    ),
                    expected,
                    figure7_path,
                )

    differences = [
        float(check["absolute_difference"])
        for check in checks
    ]
    tolerance = 1.0e-8
    return {
        "status": (
            "passed"
            if max(differences, default=0.0) <= tolerance
            else "failed"
        ),
        "checks": len(checks),
        "tolerance": tolerance,
        "max_absolute_difference": max(
            differences, default=0.0
        ),
    }


def compute_all() -> dict[str, object]:
    rows = [
        *score_figure7(),
        *score_figure8(),
        *score_figure9(),
        *score_figure13(),
        *score_figure15(),
    ]
    return {
        "metric_definitions": {
            "r2": "1 - sum((prediction-reference)^2) / sum((reference-mean(reference))^2)",
            "relative_l2": "sqrt(sum((prediction-reference)^2) / sum(reference^2))",
            "mse": "mean((prediction-reference)^2)",
            "rmse": "sqrt(mse)",
        },
        "scope": (
            "MPM-KAN outputs versus every reliably paired reference "
            "series in Figure 7, 8, 9, 13 and 15 data folders"
        ),
        "generalization_shadow_set": (
            "not scored: no matching independent solution values exist"
        ),
        "data_quality_notes": [
            (
                "Figure 15a has 28 orphan numerical theta entries "
                "without paired radii; they are excluded."
            ),
            (
                "Negative R2 is valid when the reference varies little "
                "and the model bias exceeds that variation."
            ),
            (
                "Per-figure macro means give every reference curve equal "
                "weight; medians are also reported because sample counts "
                "and physical quantities differ."
            ),
        ],
        "summary": _summary(rows),
        "reconciliation": reconcile_existing_artifacts(rows),
        "comparisons": rows,
    }


def save_results(payload: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "all_metrics.json"
    csv_path = OUTPUT_DIR / "all_metrics.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = payload["comparisons"]
    with csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = (
            "figure",
            "comparisons",
            "samples",
            "r2_mean",
            "r2_median",
            "relative_l2_mean",
            "relative_l2_median",
            "mse_mean",
            "mse_median",
            "rmse_mean",
            "rmse_median",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for figure, item in payload["summary"]["figures"].items():
            writer.writerow(
                {
                    "figure": figure,
                    "comparisons": item["comparisons"],
                    "samples": item["samples"],
                    **{
                        f"{metric}_{statistic}": item[
                            "macro_mean"
                            if statistic == "mean"
                            else "median"
                        ][metric]
                        for metric in (
                            "r2",
                            "relative_l2",
                            "mse",
                            "rmse",
                        )
                        for statistic in ("mean", "median")
                    },
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-save", action="store_true"
    )
    args = parser.parse_args()
    payload = compute_all()
    if not args.no_save:
        save_results(payload)
    print(
        json.dumps(payload["summary"], indent=2)
    )


if __name__ == "__main__":
    main()
