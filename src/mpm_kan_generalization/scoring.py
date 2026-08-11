from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import regression_metrics


def _case_index(row: dict[str, str]) -> int:
    if row.get("case_index", "").strip():
        return int(float(row["case_index"]))
    case_id = row.get("case_id", "").strip()
    if case_id.startswith("blind_"):
        return int(case_id.removeprefix("blind_"))
    raise ValueError(
        "each reference row needs case_index or blind_NNN case_id"
    )


def _read_predictions(
    path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") != "blind":
                continue
            case = int(float(row["case_index"]))
            grouped.setdefault(case, []).append(
                (
                    float(row["theta_rad"]),
                    float(row["normalized_pressure"]),
                )
            )
    if not grouped:
        raise ValueError("no blind predictions were found")
    result = {}
    for case, values in grouped.items():
        values.sort()
        result[case] = (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
        )
    return result


def _read_reference(
    path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            value = row.get(
                "reference_normalized_pressure", ""
            ).strip()
            if not value:
                continue
            try:
                case = _case_index(row)
                grouped.setdefault(case, []).append(
                    (float(row["theta_rad"]), float(value))
                )
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"invalid reference row {line}: {error}"
                ) from error
    if not grouped:
        raise ValueError(
            "no filled reference_normalized_pressure values found"
        )
    result = {}
    for case, values in grouped.items():
        values.sort()
        result[case] = (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
        )
    return result


def _periodic_interpolate(
    query: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    period = 2.0 * np.pi
    wrapped = np.mod(theta, period)
    order = np.argsort(wrapped)
    x = wrapped[order]
    y = values[order]
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    if unique.size < 2:
        raise ValueError(
            "each case requires at least two distinct theta values"
        )
    x_periodic = np.concatenate(
        (unique[-1:] - period, unique, unique[:1] + period)
    )
    y_periodic = np.concatenate((y[-1:], y, y[:1]))
    return np.interp(
        np.mod(query, period), x_periodic, y_periodic
    )


def score_pipe_profiles(
    prediction_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    predictions = _read_predictions(prediction_path)
    references = _read_reference(reference_path)
    paired_reference: list[np.ndarray] = []
    paired_prediction: list[np.ndarray] = []
    cases: list[dict[str, object]] = []
    missing = sorted(set(references) - set(predictions))
    if missing:
        raise ValueError(
            f"predictions missing reference cases: {missing}"
        )
    for case in sorted(references):
        query, truth = references[case]
        theta, prediction = predictions[case]
        estimate = _periodic_interpolate(
            query, theta, prediction
        )
        metrics = regression_metrics(truth, estimate)
        cases.append(
            {
                "case_id": f"blind_{case:03d}",
                **metrics.as_dict(),
            }
        )
        paired_reference.append(truth)
        paired_prediction.append(estimate)
    overall = regression_metrics(
        np.concatenate(paired_reference),
        np.concatenate(paired_prediction),
    )
    macro = {
        name: float(
            np.mean(
                [
                    row[name]
                    for row in cases
                    if row[name] is not None
                ]
            )
        )
        for name in ("r2", "relative_l2", "mse", "rmse")
    }
    return {
        "status": "scored",
        "scope": (
            "one-dimensional pipe-profile magnitude; "
            "not a full-field complex-solution score"
        ),
        "reference_used_in_training": False,
        "predictions": str(prediction_path.resolve()),
        "reference": str(reference_path.resolve()),
        "overall_micro": overall.as_dict(),
        "macro_case_mean": macro,
        "cases": cases,
    }


def export_template(
    prediction_path: Path,
    output_path: Path,
    cases: list[int] | None,
) -> int:
    predictions = _read_predictions(prediction_path)
    selected = sorted(predictions) if cases is None else cases
    missing = sorted(set(selected) - set(predictions))
    if missing:
        raise ValueError(f"unknown prediction cases: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "theta_rad",
                "reference_normalized_pressure",
            ),
        )
        writer.writeheader()
        for case in selected:
            theta = predictions[case][0]
            for value in theta:
                writer.writerow(
                    {
                        "case_id": f"blind_{case:03d}",
                        "theta_rad": f"{value:.16g}",
                        "reference_normalized_pressure": "",
                    }
                )
    return sum(predictions[case][0].size for case in selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score frozen blind pipe-pressure predictions against "
            "independent reference data."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/blind_pipe_profiles.csv"),
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/accuracy_metrics.json"),
    )
    parser.add_argument(
        "--export-template",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="*",
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.export_template is not None:
        rows = export_template(
            args.predictions, args.export_template, args.cases
        )
        print(
            f"Exported {rows} empty reference rows to "
            f"{args.export_template.resolve()}"
        )
        return
    if args.reference is None:
        raise SystemExit(
            "--reference is required unless --export-template is used"
        )
    result = score_pipe_profiles(
        args.predictions, args.reference
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
