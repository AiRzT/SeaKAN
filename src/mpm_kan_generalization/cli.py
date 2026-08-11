from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ParameterDomain, PhysicalConstants
from .design import create_design, save_design
from .training import (
    TrainingOptions,
    evaluate_and_save,
    evaluate_selection_and_save,
    load_checkpoint,
    sha256,
    train,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one physics-only conditional MPM-KAN across wave, "
            "permeability, burial-depth, and random-field cases."
        )
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/generalization")
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="physics-only checkpoint used only to initialise new training",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="evaluate train/validation only; never open blind cases",
    )
    parser.add_argument(
        "--shadow-blind-seed",
        type=int,
        default=None,
        help="replace the embedded blind split during evaluate-only",
    )
    parser.add_argument(
        "--shadow-blind-cases", type=int, default=16
    )
    parser.add_argument("--train-cases", type=int, default=16)
    parser.add_argument("--validation-cases", type=int, default=6)
    parser.add_argument("--blind-cases", type=int, default=8)
    parser.add_argument("--train-margin", type=float, default=0.08)
    parser.add_argument(
        "--training-boundary-anchors", action="store_true"
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--float64-steps", type=int, default=20)
    parser.add_argument(
        "--learning-rate", type=float, default=4.0e-4
    )
    parser.add_argument(
        "--float64-learning-rate",
        type=float,
        default=2.0e-5,
    )
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--n-basis", type=int, default=7)
    parser.add_argument("--local-width", type=int, default=16)
    parser.add_argument("--cases-per-step", type=int, default=4)
    parser.add_argument(
        "--robust-case-alpha", type=float, default=0.0
    )
    parser.add_argument(
        "--robust-case-temperature", type=float, default=0.75
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument(
        "--minimum-lr-ratio", type=float, default=0.08
    )
    parser.add_argument(
        "--backbone", choices=("kan", "mlp"), default="kan"
    )
    parser.add_argument(
        "--no-random-field-conditioning",
        action="store_true",
        help=(
            "mask xi_1..xi_4 at the network input while retaining the "
            "true random field in the governing equation"
        ),
    )
    parser.add_argument(
        "--particle-weight", type=float, default=1.0
    )
    parser.add_argument("--p2g-weight", type=float, default=0.15)
    parser.add_argument(
        "--side-boundary",
        choices=("bloch", "incident_fixed"),
        default="bloch",
        help="lateral pressure boundary-value problem",
    )
    parser.add_argument(
        "--al-update-every", type=int, default=100
    )
    parser.add_argument("--seed", type=int, default=202416)
    parser.add_argument(
        "--design-seed",
        type=int,
        default=None,
        help=(
            "fixed train/validation/blind split seed; defaults to "
            "--seed for backward compatibility"
        ),
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--version",
        action="version",
        version="mpm-kan-generalization 0.2.1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else output_dir / "checkpoints" / "generalization.pt"
    )
    if args.evaluate_only:
        (
            model,
            design,
            domain,
            constants,
            options,
            augmented,
            history,
        ) = load_checkpoint(checkpoint, device)
        if args.shadow_blind_seed is not None:
            shadow = create_design(
                domain,
                train_cases=design.train.shape[0],
                validation_cases=design.validation.shape[0],
                blind_cases=args.shadow_blind_cases,
                seed=args.shadow_blind_seed,
            )
            design = type(design)(
                train=design.train,
                validation=design.validation,
                blind=shadow.blind,
            )
    else:
        domain = ParameterDomain()
        constants = PhysicalConstants()
        design_seed = (
            args.seed
            if args.design_seed is None
            else args.design_seed
        )
        design = create_design(
            domain,
            train_cases=args.train_cases,
            validation_cases=args.validation_cases,
            blind_cases=args.blind_cases,
            seed=design_seed,
            train_margin=args.train_margin,
            training_boundary_anchors=(
                args.training_boundary_anchors
            ),
        )
        options = TrainingOptions(
            steps=args.steps,
            float64_steps=args.float64_steps,
            learning_rate=args.learning_rate,
            float64_learning_rate=(
                args.float64_learning_rate
            ),
            width=args.width,
            hidden_layers=args.hidden_layers,
            n_basis=args.n_basis,
            local_width=args.local_width,
            cases_per_step=args.cases_per_step,
            robust_case_alpha=args.robust_case_alpha,
            robust_case_temperature=(
                args.robust_case_temperature
            ),
            weight_decay=args.weight_decay,
            minimum_lr_ratio=args.minimum_lr_ratio,
            backbone=args.backbone,
            condition_random_field=(
                not args.no_random_field_conditioning
            ),
            particle_weight=args.particle_weight,
            p2g_weight=args.p2g_weight,
            side_boundary_mode=args.side_boundary,
            al_update_every=args.al_update_every,
            seed=args.seed,
            log_every=args.log_every,
        )
        model, augmented, history = train(
            design,
            domain,
            constants,
            options,
            device,
            output_dir,
            initial_checkpoint=(
                None
                if args.initial_checkpoint is None
                else args.initial_checkpoint.resolve()
            ),
        )
    save_design(
        design,
        output_dir / "experiment_design.json",
        domain,
        constants,
    )
    if args.selection_only:
        metrics = evaluate_selection_and_save(
            model,
            design,
            domain,
            constants,
            options,
            augmented,
            history,
            device,
            output_dir,
        )
    else:
        metrics = evaluate_and_save(
            model,
            design,
            domain,
            constants,
            options,
            augmented,
            history,
            device,
            output_dir,
        )
        if args.shadow_blind_seed is not None:
            scores = sorted(
                float(row["physics"][
                    "selection_score_unaugmented"
                ])
                for row in metrics["blind_cases"]
            )
            position = 0.9 * (len(scores) - 1)
            lower = int(position)
            upper = min(lower + 1, len(scores) - 1)
            fraction = position - lower
            p90 = (
                scores[lower] * (1.0 - fraction)
                + scores[upper] * fraction
            )
            train_score = metrics["splits"]["train"][
                "selection_score_unaugmented"
            ]
            ratios = {
                "mean": metrics["splits"]["blind"][
                    "selection_score_unaugmented"
                ]
                / train_score,
                "p90": p90 / train_score,
                "maximum": scores[-1] / train_score,
            }
            thresholds = {
                "mean": 1.05,
                "p90": 1.20,
                "maximum": 1.35,
            }
            metrics["shadow_evaluation"] = {
                "seed": args.shadow_blind_seed,
                "cases": len(scores),
                "checkpoint_frozen_before_evaluation": True,
                "ratios_over_train": ratios,
                "preregistered_thresholds": thresholds,
                "passed": all(
                    ratios[name] <= thresholds[name]
                    for name in thresholds
                ),
                "claim_scope": (
                    "physics-residual generalization only"
                ),
            }
            (
                output_dir / "generalization_metrics.json"
            ).write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
    run = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "single_checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_labels": "none",
        "reference_data_used_in_training": False,
        "initial_checkpoint": (
            None
            if args.evaluate_only or args.initial_checkpoint is None
            else str(args.initial_checkpoint.resolve())
        ),
        "per_case_retraining": False,
        "per_case_hyperparameter_tuning": False,
        "blind_evaluated": not args.selection_only,
        "shadow_blind_seed": args.shadow_blind_seed,
        "design_seed": (
            None
            if args.evaluate_only
            else (
                args.seed
                if args.design_seed is None
                else args.design_seed
            )
        ),
        "training_seed": options.seed,
        "design": {
            "train_cases": int(design.train.shape[0]),
            "validation_cases": int(
                design.validation.shape[0]
            ),
            "blind_cases": int(design.blind.shape[0]),
            "requested_train_margin": (
                None if args.evaluate_only else args.train_margin
            ),
            "requested_training_boundary_anchors": (
                None
                if args.evaluate_only
                else args.training_boundary_anchors
            ),
        },
        "options": asdict(options),
        "ablation_controls": {
            "backbone": options.backbone,
            "condition_random_field": (
                options.condition_random_field
            ),
            "particle_weight": options.particle_weight,
            "p2g_weight": options.p2g_weight,
            "al_update_every": options.al_update_every,
        },
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "domain": domain.as_dict(),
        "constants": constants.as_dict(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    if args.selection_only:
        print(
            "Selection metrics: "
            f"{output_dir / 'selection_metrics.json'}"
        )
    else:
        print(
            "Summary: "
            f"{output_dir / 'generalization_summary.png'}"
        )


if __name__ == "__main__":
    main()
