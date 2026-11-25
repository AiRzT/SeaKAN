from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .adaptive_mpm import AdaptiveMaterialPointCloud
from .adaptive_training import (
    AdaptiveTrainingOptions,
    load_adaptive_checkpoint,
    make_adaptive_problem,
    train_adaptive_strong,
)
from .cli import packaged_resource, project_root, select_device
from .config import figure9_setup
from .data import (
    calculate_metrics,
    load_figure9_data,
    plot_field,
    plot_polar,
    save_json,
    save_profiles,
)
from .reference import solve_reference
from .training import load_checkpoint, make_problem


def main() -> None:
    args = build_parser().parse_args()
    root = project_root()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "outputs" / "figure9_adaptive_strong"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline_checkpoint:
        baseline_checkpoint = Path(args.baseline_checkpoint).resolve()
    else:
        baseline_checkpoint = root / "artifacts" / "checkpoints" / "figure9.pt"
        if not baseline_checkpoint.is_file():
            baseline_checkpoint = packaged_resource(
                "artifacts/checkpoints/figure9.pt"
            )
    adaptive_checkpoint = output_dir / "checkpoints" / "figure9_adaptive_strong.pt"
    if args.evaluate_only and not adaptive_checkpoint.is_file():
        released = (
            root
            / "artifacts"
            / "adaptive_strong"
            / "checkpoints"
            / "figure9_adaptive_strong.pt"
        )
        adaptive_checkpoint = (
            released
            if released.is_file()
            else packaged_resource(
                "artifacts/adaptive_strong/checkpoints/"
                "figure9_adaptive_strong.pt"
            )
        )
    if args.data_file:
        data_path = Path(args.data_file).resolve()
    else:
        data_path = root / "data" / "figure9.csv"
        if not data_path.is_file():
            data_path = packaged_resource("figure9.csv")
    device = select_device(args.device)
    wave, soil, geometry = figure9_setup()
    _, baseline_options, dtype, _ = load_checkpoint(
        wave, soil, geometry, baseline_checkpoint, device
    )
    cloud = AdaptiveMaterialPointCloud.create(
        geometry,
        device,
        dtype,
        base_x=args.base_x,
        base_y=args.base_y,
        middle_radial=args.middle_radial,
        middle_theta=args.middle_theta,
        fine_radial=args.fine_radial,
        fine_theta=args.fine_theta,
        p2g_x=args.p2g_x,
        p2g_y=args.p2g_y,
    )
    options = AdaptiveTrainingOptions(
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_background=args.batch_background,
        batch_middle=args.batch_middle,
        batch_fine=args.batch_fine,
        momentum_weight=args.momentum_weight,
        fluid_weight=args.fluid_weight,
        p2g_weight=args.p2g_weight,
        boundary_weight=args.boundary_weight,
        pipe_flux_weight=args.pipe_flux_weight,
        roughness_weight=args.roughness_weight,
        seed=args.seed,
        log_every=args.log_every,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Adaptive cloud: {cloud.summary()}")
    print(
        "Solver: full strong-form 2-D Biot residual at adaptive MPM particles "
        "plus volume-weighted P2G residual."
    )
    print("Digitized Figure 9 data remain withheld until training finishes.")

    if args.evaluate_only:
        model, baseline_options, options, dtype, history = (
            load_adaptive_checkpoint(
                wave,
                soil,
                geometry,
                baseline_checkpoint,
                adaptive_checkpoint,
                device,
            )
        )
    else:
        model, baseline_options, dtype, history = train_adaptive_strong(
            wave,
            soil,
            geometry,
            cloud,
            baseline_checkpoint,
            output_dir,
            options,
            device,
            resume=args.resume,
        )

    baseline_model, _, _, _ = load_checkpoint(
        wave, soil, geometry, baseline_checkpoint, device
    )
    _, evaluation_problem = make_problem(
        wave, soil, geometry, baseline_options, device, dtype
    )
    adaptive_tensor = evaluation_problem.evaluate_pipe(
        model, n_theta=args.theta_points
    )
    baseline_tensor = evaluation_problem.evaluate_pipe(
        baseline_model, n_theta=args.theta_points
    )
    prediction = {
        key: value.cpu().numpy() for key, value in adaptive_tensor.items()
    }
    baseline = {
        key: value.cpu().numpy() for key, value in baseline_tensor.items()
    }
    field_tensor = evaluation_problem.evaluate_field(
        model, nx=args.field_x, ny=args.field_y
    )
    field = {key: value.cpu().numpy() for key, value in field_tensor.items()}
    reference_result = solve_reference(
        wave,
        soil,
        geometry,
        nx=args.reference_x,
        ny=args.reference_y,
        n_theta=args.theta_points,
    )
    reference = reference_result.profile()
    extracted, quality = load_figure9_data(data_path)
    metrics = calculate_metrics(prediction, extracted, quality, reference)
    baseline_metrics = calculate_metrics(
        baseline, extracted, quality, reference
    )
    full_indices = torch.arange(cloud.n_particles, device=device)
    strong_problem = make_adaptive_problem(
        wave, soil, geometry, cloud, options
    )
    metrics["adaptive_strong_physics_loss"] = (
        strong_problem.loss(model, full_indices).detached()
    )
    metrics["adaptive_cloud"] = cloud.summary()
    metrics["baseline_mpm_kan"] = baseline_metrics["mpm_kan"]
    metrics["relative_l2_change"] = {
        key: (
            metrics["mpm_kan"][key]["relative_l2"]
            - baseline_metrics["mpm_kan"][key]["relative_l2"]
        )
        for key in metrics["mpm_kan"]
    }
    metrics["finite_difference_reference"] = {
        "degrees_of_freedom": reference_result.degrees_of_freedom,
        "relative_linear_residual": reference_result.relative_linear_residual,
    }
    metrics["prediction_range"] = {
        "minimum": float(np.min(prediction["normalized_pressure"])),
        "maximum": float(np.max(prediction["normalized_pressure"])),
    }

    plot_polar(
        output_dir / "figure9_adaptive_strong_mpm_kan.png",
        prediction,
        extracted,
        reference,
        baseline=baseline,
        prediction_label="Adaptive strong-form MPM-KAN",
    )
    plot_field(
        output_dir / "figure9_adaptive_strong_field.png", field, geometry
    )
    save_profiles(
        output_dir / "figure9_adaptive_strong_profiles.csv",
        prediction,
        extracted,
        reference,
    )
    save_json(metrics, output_dir / "figure9_adaptive_strong_metrics.json")
    (output_dir / "adaptive_training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "adaptive_run_config.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "baseline_checkpoint": str(baseline_checkpoint),
                "adaptive_checkpoint": str(adaptive_checkpoint),
                "adaptive_options": asdict(options),
                "cloud": cloud.summary(),
                "data_loaded_after_training": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Adaptive strong-form run completed.")
    for source in (
        "paper_this_study",
        "experimental",
        "finite_difference_reference",
    ):
        new_value = metrics["mpm_kan"][source]["relative_l2"]
        old_value = baseline_metrics["mpm_kan"][source]["relative_l2"]
        print(
            f"{source}: {100.0 * old_value:.3f}% -> "
            f"{100.0 * new_value:.3f}% "
            f"(delta {100.0 * (new_value - old_value):+.3f} pp)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive-refined strong-form MPM-KAN solve for Figure 9."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--batch-background", type=int, default=384)
    parser.add_argument("--batch-middle", type=int, default=640)
    parser.add_argument("--batch-fine", type=int, default=896)
    parser.add_argument("--momentum-weight", type=float, default=10.0)
    parser.add_argument("--fluid-weight", type=float, default=800.0)
    parser.add_argument("--p2g-weight", type=float, default=0.20)
    parser.add_argument("--boundary-weight", type=float, default=80.0)
    parser.add_argument("--pipe-flux-weight", type=float, default=160.0)
    parser.add_argument("--roughness-weight", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=202409)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--base-x", type=int, default=48)
    parser.add_argument("--base-y", type=int, default=18)
    parser.add_argument("--middle-radial", type=int, default=8)
    parser.add_argument("--middle-theta", type=int, default=128)
    parser.add_argument("--fine-radial", type=int, default=7)
    parser.add_argument("--fine-theta", type=int, default=192)
    parser.add_argument("--p2g-x", type=int, default=96)
    parser.add_argument("--p2g-y", type=int, default=36)
    parser.add_argument("--theta-points", type=int, default=721)
    parser.add_argument("--field-x", type=int, default=241)
    parser.add_argument("--field-y", type=int, default=81)
    parser.add_argument("--reference-x", type=int, default=257)
    parser.add_argument("--reference-y", type=int, default=65)
    return parser


if __name__ == "__main__":
    main()
