from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .config import (
    PARAMETER_NAMES,
    ParameterDomain,
    PhysicalConstants,
    derived_case_columns,
)


@dataclass(frozen=True)
class ExperimentDesign:
    train: np.ndarray
    validation: np.ndarray
    blind: np.ndarray

    def split(self, name: str) -> np.ndarray:
        if name not in {"train", "validation", "blind"}:
            raise KeyError(name)
        return getattr(self, name)


def _latin_hypercube(
    count: int,
    dimensions: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values = (
            np.arange(count, dtype=np.float64)
            + generator.random(count)
        ) / count
        generator.shuffle(values)
        result[:, dimension] = values
    return result


def _scale(
    unit: np.ndarray,
    domain: ParameterDomain,
    *,
    interior_margin: float,
) -> np.ndarray:
    ranges = [
        domain.wave_height_m,
        domain.wave_period_s,
        domain.permeability_m_per_s,
        domain.burial_center_depth_over_diameter,
        *(
            [domain.random_coefficient]
            * domain.random_modes
        ),
    ]
    clipped = (
        interior_margin
        + (1.0 - 2.0 * interior_margin) * unit
    )
    columns: list[np.ndarray] = []
    for index, bounds in enumerate(ranges):
        if index == 2:
            low, high = np.log10(bounds)
            columns.append(10.0 ** (
                low + clipped[:, index] * (high - low)
            ))
        elif index >= 4:
            normal = NormalDist()
            probability_low = normal.cdf(bounds[0])
            probability_high = normal.cdf(bounds[1])
            probability = (
                probability_low
                + clipped[:, index]
                * (probability_high - probability_low)
            )
            columns.append(
                np.asarray(
                    [
                        normal.inv_cdf(float(value))
                        for value in probability
                    ],
                    dtype=np.float64,
                )
            )
        else:
            low, high = bounds
            columns.append(
                low + clipped[:, index] * (high - low)
            )
    return np.stack(columns, axis=1)


def create_design(
    domain: ParameterDomain,
    *,
    train_cases: int = 16,
    validation_cases: int = 6,
    blind_cases: int = 8,
    seed: int = 202416,
    train_margin: float = 0.08,
    training_boundary_anchors: bool = False,
) -> ExperimentDesign:
    dimensions = 4 + domain.random_modes
    train = _scale(
        _latin_hypercube(train_cases, dimensions, seed),
        domain,
        interior_margin=train_margin,
    )
    validation = _scale(
        _latin_hypercube(
            validation_cases, dimensions, seed + 1
        ),
        domain,
        interior_margin=0.04,
    )
    blind = _scale(
        _latin_hypercube(blind_cases, dimensions, seed + 2),
        domain,
        interior_margin=0.0,
    )
    if blind_cases >= 4:
        blind[0, :4] = [
            domain.wave_height_m[0],
            domain.wave_period_s[0],
            domain.permeability_m_per_s[0],
            domain.burial_center_depth_over_diameter[0],
        ]
        blind[1, :4] = [
            domain.wave_height_m[1],
            domain.wave_period_s[1],
            domain.permeability_m_per_s[1],
            domain.burial_center_depth_over_diameter[1],
        ]
        blind[2, :4] = [
            domain.wave_height_m[0],
            domain.wave_period_s[1],
            domain.permeability_m_per_s[1],
            domain.burial_center_depth_over_diameter[0],
        ]
        blind[3, :4] = [
            domain.wave_height_m[1],
            domain.wave_period_s[0],
            domain.permeability_m_per_s[0],
            domain.burial_center_depth_over_diameter[1],
        ]
    if training_boundary_anchors and train_cases >= 8:
        low = np.asarray(
            [
                domain.wave_height_m[0],
                domain.wave_period_s[0],
                domain.permeability_m_per_s[0],
                domain.burial_center_depth_over_diameter[0],
            ]
        )
        high = np.asarray(
            [
                domain.wave_height_m[1],
                domain.wave_period_s[1],
                domain.permeability_m_per_s[1],
                domain.burial_center_depth_over_diameter[1],
            ]
        )
        binary = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        train[:8, :4] = low + binary * (high - low)
    return ExperimentDesign(
        train=train,
        validation=validation,
        blind=blind,
    )


def save_design(
    design: ExperimentDesign,
    path: Path,
    domain: ParameterDomain,
    constants: PhysicalConstants,
) -> None:
    physical_low = np.asarray(
        [
            domain.wave_height_m[0],
            domain.wave_period_s[0],
            domain.permeability_m_per_s[0],
            domain.burial_center_depth_over_diameter[0],
        ]
    )
    physical_high = np.asarray(
        [
            domain.wave_height_m[1],
            domain.wave_period_s[1],
            domain.permeability_m_per_s[1],
            domain.burial_center_depth_over_diameter[1],
        ]
    )
    train_on_boundary = np.logical_or(
        np.isclose(design.train[:, :4], physical_low),
        np.isclose(design.train[:, :4], physical_high),
    ).all(axis=1)
    payload: dict[str, object] = {
        "training_labels": "none",
        "split_policy": {
            "train": (
                "Latin hypercube with declared parameter-boundary "
                "anchors when present; participates in physics loss"
            ),
            "validation": "held-out interior Latin hypercube; physics metrics only",
            "blind": "held-out full-domain Latin hypercube with four parameter-boundary cases",
        },
        "random_coefficient_sampling": (
            "Latin hypercube in the CDF of a standard normal "
            "truncated to the declared coefficient bounds"
        ),
        "training_boundary_anchor_count": int(
            train_on_boundary.sum()
        ),
        "parameter_names": list(PARAMETER_NAMES),
        "domain": domain.as_dict(),
        "constants": constants.as_dict(),
        "splits": {},
    }
    for name in ("train", "validation", "blind"):
        values = design.split(name)
        derived = derived_case_columns(values, constants)
        rows = []
        for index, row in enumerate(values):
            record = {
                "case_id": f"{name}_{index:03d}",
                **dict(zip(PARAMETER_NAMES, row.tolist())),
            }
            record.update(
                {
                    key: float(value[index])
                    for key, value in derived.items()
                }
            )
            rows.append(record)
        payload["splits"][name] = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    blind_rows = payload["splits"]["blind"]
    reference_path = path.with_name(
        "blind_reference_cases.csv"
    )
    with reference_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(blind_rows[0])
        )
        writer.writeheader()
        writer.writerows(blind_rows)
