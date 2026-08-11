from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .config import PARAMETER_NAMES, derived_case_columns
from .model import ComsolCalibratedSoilKAN
from .training import load_checkpoint


@dataclass(frozen=True)
class CalibrationOptions:
    steps: int = 600
    learning_rate: float = 2.0e-3
    minimum_lr_ratio: float = 0.05
    batch_per_case: int = 160
    width: int = 24
    hidden_layers: int = 2
    n_basis: int = 7
    condition_wave_height: bool = False
    condition_random_field: bool = False
    roughness_weight: float = 2.0e-6
    weight_decay: float = 1.0e-6
    validation_every: int = 20
    seed: int = 202438


@dataclass
class ComplexCase:
    case_id: str
    split: str
    xy: torch.Tensor
    target: torch.Tensor
    parameters: torch.Tensor
    kappa: float
    pipe_center_y: float
    energy: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_case(
    row: dict[str, str],
    data_dir: Path,
    constants,
) -> ComplexCase:
    csv_path = data_dir / f"{row['case_id']}_complex_domain.csv"
    values = np.loadtxt(csv_path, delimiter=",", comments="%")
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"Unexpected COMSOL table: {csv_path}")
    parameters = np.asarray(
        [float(row[name]) for name in PARAMETER_NAMES],
        dtype=np.float64,
    )
    derived = derived_case_columns(parameters[None, :], constants)
    xy = values[:, 2:4] / constants.seabed_thickness_m
    target = values[:, 4] + 1j * values[:, 5]
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(target)
    xy = xy[finite]
    target = target[finite]
    return ComplexCase(
        case_id=row["case_id"],
        split=row["split"],
        xy=torch.from_numpy(xy),
        target=torch.from_numpy(target),
        parameters=torch.from_numpy(parameters),
        kappa=float(derived["kappa"][0]),
        pipe_center_y=float(derived["pipe_center_y"][0]),
        energy=float(np.mean(np.abs(target) ** 2)),
    )


def _predict(
    model: ComsolCalibratedSoilKAN,
    case: ComplexCase,
    indices: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    xy = case.xy[indices].to(device=device, dtype=dtype)
    target = case.target[indices].to(
        device=device,
        dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
    )
    count = xy.shape[0]
    parameters = case.parameters.to(device=device, dtype=dtype).expand(
        count, -1
    )
    kappa = torch.full(
        (count,), case.kappa, device=device, dtype=dtype
    )
    center = torch.full(
        (count,), case.pipe_center_y, device=device, dtype=dtype
    )
    return model(xy, parameters, kappa, center).p, target


def _metrics(
    model: ComsolCalibratedSoilKAN,
    cases: list[ComplexCase],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    rows: list[dict[str, float | str]] = []
    model.eval()
    with torch.no_grad():
        for case in cases:
            predicted_parts = []
            target_parts = []
            for start in range(0, case.xy.shape[0], 4096):
                indices = np.arange(
                    start, min(start + 4096, case.xy.shape[0])
                )
                predicted, target = _predict(
                    model, case, indices, device, dtype
                )
                predicted_parts.append(predicted.cpu())
                target_parts.append(target.cpu())
            predicted = torch.cat(predicted_parts)
            target = torch.cat(target_parts)
            residual = predicted - target
            amplitude_residual = predicted.abs() - target.abs()
            complex_l2 = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(
                target
            ).clamp_min(1.0e-12)
            amplitude_l2 = torch.linalg.vector_norm(
                amplitude_residual
            ) / torch.linalg.vector_norm(target.abs()).clamp_min(1.0e-12)
            complex_mse = residual.abs().square().mean()
            amplitude_mse = amplitude_residual.square().mean()
            amplitude_sst = (
                target.abs() - target.abs().mean()
            ).square().sum()
            amplitude_r2 = 1.0 - amplitude_residual.square().sum() / (
                amplitude_sst.clamp_min(1.0e-12)
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "complex_l2": float(complex_l2),
                    "amplitude_l2": float(amplitude_l2),
                    "complex_mse": float(complex_mse),
                    "complex_rmse": float(torch.sqrt(complex_mse)),
                    "amplitude_mse": float(amplitude_mse),
                    "amplitude_rmse": float(torch.sqrt(amplitude_mse)),
                    "amplitude_r2": float(amplitude_r2),
                    "points": int(target.numel()),
                }
            )
    return {
        "mean_complex_l2": float(
            np.mean([float(row["complex_l2"]) for row in rows])
        ),
        "mean_amplitude_l2": float(
            np.mean([float(row["amplitude_l2"]) for row in rows])
        ),
        "cases": rows,
    }


def calibrate(
    *,
    base_checkpoint: Path,
    manifest_path: Path,
    data_dir: Path,
    output_dir: Path,
    options: CalibrationOptions,
    device: torch.device,
) -> Path:
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    base, design, domain, constants, base_options, augmented, _ = (
        load_checkpoint(base_checkpoint, device)
    )
    base = base.to(dtype=torch.float32)
    model = ComsolCalibratedSoilKAN(
        base,
        width=options.width,
        hidden_layers=options.hidden_layers,
        n_basis=options.n_basis,
        condition_wave_height=options.condition_wave_height,
        condition_random_field=options.condition_random_field,
        dtype=torch.float32,
    ).to(device)
    dtype = torch.float32

    manifest = _read_manifest(manifest_path)
    train_rows = [row for row in manifest if row["split"] == "train"]
    validation_rows = [
        row for row in manifest if row["split"] == "validation"
    ]
    blind_ids = [
        row["case_id"] for row in manifest if row["split"] == "blind"
    ]
    if not train_rows or not validation_rows or not blind_ids:
        raise ValueError("Manifest must contain train, validation and blind cases")
    # Deliberately load only train/validation files. Blind CSV files remain
    # unopened until the selected checkpoint has been frozen.
    train_cases = [
        _load_case(row, data_dir, constants) for row in train_rows
    ]
    validation_cases = [
        _load_case(row, data_dir, constants) for row in validation_rows
    ]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(options.steps, 1),
        eta_min=options.learning_rate * options.minimum_lr_ratio,
    )
    rng = np.random.default_rng(options.seed)
    history: list[dict[str, float | int]] = []
    initial_validation = _metrics(
        model, validation_cases, device, dtype
    )
    best_score = float(initial_validation["mean_complex_l2"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()

    for step in range(1, options.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        xy_parts = []
        target_parts = []
        parameter_parts = []
        kappa_parts = []
        center_parts = []
        counts = []
        for case in train_cases:
            count = min(options.batch_per_case, case.xy.shape[0])
            indices = rng.choice(case.xy.shape[0], size=count, replace=False)
            xy_parts.append(
                case.xy[indices].to(device=device, dtype=dtype)
            )
            target_parts.append(
                case.target[indices].to(
                    device=device, dtype=torch.complex64
                )
            )
            parameter_parts.append(
                case.parameters.to(device=device, dtype=dtype).expand(
                    count, -1
                )
            )
            kappa_parts.append(
                torch.full(
                    (count,), case.kappa, device=device, dtype=dtype
                )
            )
            center_parts.append(
                torch.full(
                    (count,),
                    case.pipe_center_y,
                    device=device,
                    dtype=dtype,
                )
            )
            counts.append(count)
        predicted = model(
            torch.cat(xy_parts),
            torch.cat(parameter_parts),
            torch.cat(kappa_parts),
            torch.cat(center_parts),
        ).p
        target = torch.cat(target_parts)
        losses = []
        offset = 0
        for case, count in zip(train_cases, counts):
            case_residual = predicted[offset : offset + count] - target[
                offset : offset + count
            ]
            losses.append(
                case_residual.abs().square().mean()
                / max(case.energy, 1.0e-12)
            )
            offset += count
        data_loss = torch.stack(losses).mean()
        roughness = model.roughness()
        loss = data_loss + options.roughness_weight * roughness
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        scheduler.step()

        if step % options.validation_every == 0 or step == options.steps:
            validation = _metrics(
                model, validation_cases, device, dtype
            )
            score = float(validation["mean_complex_l2"])
            row = {
                "step": step,
                "train_relative_mse": float(data_loss.detach()),
                "validation_complex_l2": score,
                "validation_amplitude_l2": float(
                    validation["mean_amplitude_l2"]
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    selected_validation = _metrics(
        model, validation_cases, device, dtype
    )
    selected_train = _metrics(model, train_cases, device, dtype)
    elapsed = time.perf_counter() - started
    model = model.to(dtype=torch.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "comsol_calibrated.pt"
    base_state = torch.load(
        base_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint = {
        "checkpoint_type": "comsol_calibrated",
        "model": model.state_dict(),
        "domain": domain.as_dict(),
        "constants": constants.as_dict(),
        "options": asdict(base_options),
        "design": asdict(design),
        "augmented": asdict(augmented),
        "history": history,
        "calibration_architecture": {
            "width": options.width,
            "hidden_layers": options.hidden_layers,
            "n_basis": options.n_basis,
            "condition_wave_height": options.condition_wave_height,
            "condition_random_field": options.condition_random_field,
        },
        "calibration_options": asdict(options),
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "calibration_manifest": str(manifest_path.resolve()),
        "calibration_train_cases": [c.case_id for c in train_cases],
        "calibration_validation_cases": [c.case_id for c in validation_cases],
        "sealed_blind_cases": blind_ids,
        "reference_data_used_in_training": True,
        "training_labels": "COMSOL complex pore-pressure fields",
        "selected_step": best_step,
        "initial_validation": initial_validation,
        "selected_validation": selected_validation,
        "selected_train": selected_train,
        "elapsed_seconds": elapsed,
        "base_reference_data_used_in_training": base_state.get(
            "reference_data_used_in_training", False
        ),
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        key: checkpoint[key]
        for key in (
            "checkpoint_type",
            "base_checkpoint",
            "base_checkpoint_sha256",
            "calibration_train_cases",
            "calibration_validation_cases",
            "sealed_blind_cases",
            "reference_data_used_in_training",
            "selected_step",
            "initial_validation",
            "selected_validation",
            "selected_train",
            "elapsed_seconds",
        )
    }
    summary["checkpoint_sha256"] = _sha256(checkpoint_path)
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return checkpoint_path
