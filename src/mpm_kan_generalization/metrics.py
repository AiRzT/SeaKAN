from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    samples: int
    r2: float | None
    relative_l2: float | None
    mse: float
    rmse: float

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def regression_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> RegressionMetrics:
    """Return R², relative L2, MSE and RMSE on paired values.

    Complex arrays are supported.  R² uses squared complex magnitudes and
    the complex reference mean, so phase error is retained.
    """

    truth = np.asarray(reference)
    estimate = np.asarray(prediction)
    if truth.shape != estimate.shape:
        raise ValueError(
            "reference and prediction must have the same shape"
        )
    if truth.size == 0:
        raise ValueError("at least one paired sample is required")
    finite = np.isfinite(truth) & np.isfinite(estimate)
    if not np.all(finite):
        raise ValueError("paired values must all be finite")
    if weights is None:
        sample_weights = np.ones(
            truth.shape, dtype=np.float64
        )
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != truth.shape:
            raise ValueError(
                "weights must have the same shape as the values"
            )
        if (
            not np.all(np.isfinite(sample_weights))
            or np.any(sample_weights < 0.0)
            or sample_weights.sum() <= 0.0
        ):
            raise ValueError(
                "weights must be finite, non-negative and non-zero"
            )
    normalised = sample_weights / sample_weights.sum()
    squared_error = np.abs(estimate - truth) ** 2
    mse = float(np.sum(normalised * squared_error))
    rmse = float(np.sqrt(mse))
    reference_energy = float(
        np.sum(sample_weights * np.abs(truth) ** 2)
    )
    error_energy = float(
        np.sum(sample_weights * squared_error)
    )
    relative_l2 = (
        float(np.sqrt(error_energy / reference_energy))
        if reference_energy > np.finfo(np.float64).eps
        else None
    )
    reference_mean = np.sum(normalised * truth)
    total_variation = float(
        np.sum(
            sample_weights
            * np.abs(truth - reference_mean) ** 2
        )
    )
    r2 = (
        float(1.0 - error_energy / total_variation)
        if total_variation > np.finfo(np.float64).eps
        else None
    )
    return RegressionMetrics(
        samples=int(truth.size),
        r2=r2,
        relative_l2=relative_l2,
        mse=mse,
        rmse=rmse,
    )
