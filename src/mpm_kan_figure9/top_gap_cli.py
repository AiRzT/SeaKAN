from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .adaptive_mpm import AdaptiveMaterialPointCloud
from .adaptive_training import load_adaptive_checkpoint
from .cli import packaged_resource, project_root, select_device
from .config import figure9_setup
from .data import (
    calculate_metrics,
    circular_interpolate,
    load_figure9_data,
    plot_field,
    plot_polar,
    save_json,
    save_profiles,
)
from .reference import solve_reference
from .top_gap_training import (
    TopGapTrainingOptions,
    load_top_gap_checkpoint,
    make_top_gap_problem,
    train_top_gap_augmented,
)
from .training import make_problem


def _top_point_metrics(
    prediction: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    extracted: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    experimental = extracted["experimental"]
    nearest = int(np.argmin(np.abs(experimental["theta"] - 0.5 * np.pi)))
    theta = float(experimental["theta"][nearest])
    observed = float(experimental["radius"][nearest])
    current = float(
        circular_interpolate(
            prediction["theta"],
            prediction["normalized_pressure"],
            np.asarray([theta]),
        )[0]
    )
    old = float(
        circular_interpolate(
            previous["theta"],
            previous["normalized_pressure"],
            np.asarray([theta]),
        )[0]
    )
    paper = float(
        circular_interpolate(
            extracted["paper_this_study"]["theta"],
            extracted["paper_this_study"]["radius"],
            np.asarray([theta]),
        )[0]
    )
    pi_over_two = np.asarray([0.5 * np.pi])
    return {
        "nearest_experimental_top_point": {
            "theta_rad": theta,
            "experimental": observed,
            "prediction": current,
            "previous_adaptive_strong": old,
            "zhu_numerical_at_same_angle": paper,
            "absolute_error": abs(current - observed),
            "previous_absolute_error": abs(old - observed),
        },
        "at_pi_over_two": {
            "prediction": float(
                circular_interpolate(
                    prediction["theta"],
                    prediction["normalized_pressure"],
                    pi_over_two,
                )[0]
            ),
            "previous_adaptive_strong": float(
                circular_interpolate(
                    previous["theta"],
                    previous["normalized_pressure"],
                    pi_over_two,
                )[0]
            ),
            "experimental_periodic_interpolation": float(
                circular_interpolate(
                    experimental["theta"],
                    experimental["radius"],
                    pi_over_two,
                )[0]
            ),
            "zhu_numerical": float(
                circular_interpolate(
                    extracted["paper_this_study"]["theta"],
                    extracted["paper_this_study"]["radius"],
                    pi_over_two,
                )[0]
            ),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    root = project_root()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "outputs" / "figure9_top_gap_augmented"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = (
        Path(args.baseline_checkpoint).resolve()
        if args.baseline_checkpoint
        else root / "artifacts" / "checkpoints" / "figure9.pt"
    )
    if not baseline_checkpoint.is_file():
        baseline_checkpoint = packaged_resource(
            "artifacts/checkpoints/figure9.pt"
        )
    adaptive_checkpoint = (
        Path(args.adaptive_checkpoint).resolve()
        if args.adaptive_checkpoint
        else (
            root
            / "artifacts"
            / "adaptive_strong"
            / "checkpoints"
            / "figure9_adaptive_strong.pt"
        )
    )
    if not adaptive_checkpoint.is_file():
        adaptive_checkpoint = packaged_resource(
            "artifacts/adaptive_strong/checkpoints/"
            "figure9_adaptive_strong.pt"
        )
    if args.teacher_checkpoint:
        teacher_checkpoint: Path | None = Path(
            args.teacher_checkpoint
        ).resolve()
    else:
        released_teacher = (
            root
            / "artifacts"
            / "top_gap_teacher"
            / "checkpoints"
            / "figure9_top_gap_teacher.pt"
        )
        teacher_checkpoint = (
            released_teacher if released_teacher.is_file() else None
        )
        if teacher_checkpoint is None:
            packaged_teacher = packaged_resource(
                "artifacts/top_gap_teacher/checkpoints/"
                "figure9_top_gap_teacher.pt"
            )
            if packaged_teacher.is_file():
                teacher_checkpoint = packaged_teacher
    top_gap_checkpoint = (
        output_dir / "checkpoints" / "figure9_top_gap_augmented.pt"
    )
    if args.evaluate_only and not top_gap_checkpoint.is_file():
        released = (
            root
            / "artifacts"
            / "top_gap_augmented"
            / "checkpoints"
            / "figure9_top_gap_augmented.pt"
        )
        top_gap_checkpoint = (
            released
            if released.is_file()
            else packaged_resource(
                "artifacts/top_gap_augmented/checkpoints/"
                "figure9_top_gap_augmented.pt"
            )
        )
    data_path = (
        Path(args.data_file).resolve()
        if args.data_file
        else root / "data" / "figure9.csv"
    )
    if not data_path.is_file():
        data_path = packaged_resource("figure9.csv")

    device = select_device(args.device)
    wave, soil, geometry = figure9_setup()
    cloud_kwargs = {
        "base_x": args.base_x,
        "base_y": args.base_y,
        "middle_radial": args.middle_radial,
        "middle_theta": args.middle_theta,
        "fine_radial": args.fine_radial,
        "fine_theta": args.fine_theta,
        "gap_normal": args.gap_normal,
        "gap_theta": args.gap_theta,
        "p2g_x": args.p2g_x,
        "p2g_y": args.p2g_y,
    }
    cloud_float64 = AdaptiveMaterialPointCloud.create(
        geometry, device, torch.float64, **cloud_kwargs
    )
    options = TopGapTrainingOptions(
        float32_steps=args.float32_steps,
        float32_learning_rate=args.float32_learning_rate,
        float64_adam_steps=args.float64_adam_steps,
        float64_learning_rate=args.float64_learning_rate,
        float64_lbfgs_steps=args.float64_lbfgs_steps,
        batch_background=args.batch_background,
        batch_middle=args.batch_middle,
        batch_fine=args.batch_fine,
        batch_gap=args.batch_gap,
        momentum_weight=args.momentum_weight,
        fluid_weight=args.fluid_weight,
        p2g_weight=args.p2g_weight,
        roughness_weight=args.roughness_weight,
        boundary_points=args.boundary_points,
        al_update_every=args.al_update_every,
        rho_top=args.rho_top,
        rho_outer=args.rho_outer,
        rho_pipe=args.rho_pipe,
        x_support_radii=args.x_support_radii,
        y_support_center_gap=args.y_support_center_gap,
        distillation_steps=args.distillation_steps,
        distillation_learning_rate=args.distillation_learning_rate,
        seed=args.seed,
        log_every=args.log_every,
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Four-level cloud: {cloud_float64.summary()}")
    print(
        "Solver: top-gap level-4 refinement + augmented-Lagrangian "
        "boundaries + deterministic float64 full-batch finish."
    )
    print("Digitized Figure 9 data remain withheld until training finishes.")

    if args.evaluate_only:
        (
            model,
            baseline_options,
            _,
            options,
            dtype,
            history,
            augmented,
        ) = load_top_gap_checkpoint(
            wave,
            soil,
            geometry,
            baseline_checkpoint,
            adaptive_checkpoint,
            top_gap_checkpoint,
            device,
        )
    else:
        cloud_float32 = AdaptiveMaterialPointCloud.create(
            geometry, device, torch.float32, **cloud_kwargs
        )
        (
            model,
            baseline_options,
            _,
            dtype,
            history,
            augmented,
        ) = train_top_gap_augmented(
            wave,
            soil,
            geometry,
            cloud_float32,
            cloud_float64,
            baseline_checkpoint,
            adaptive_checkpoint,
            teacher_checkpoint,
            output_dir,
            options,
            device,
        )

    previous_model, _, _, _, _ = load_adaptive_checkpoint(
        wave,
        soil,
        geometry,
        baseline_checkpoint,
        adaptive_checkpoint,
        device,
    )
    previous_model = previous_model.to(dtype=torch.float64)
    _, evaluation_problem = make_problem(
        wave, soil, geometry, baseline_options, device, dtype
    )
    prediction_tensor = evaluation_problem.evaluate_pipe(
        model, n_theta=args.theta_points
    )
    previous_tensor = evaluation_problem.evaluate_pipe(
        previous_model, n_theta=args.theta_points
    )
    prediction = {
        key: value.cpu().numpy() for key, value in prediction_tensor.items()
    }
    previous = {
        key: value.cpu().numpy() for key, value in previous_tensor.items()
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
    previous_metrics = calculate_metrics(
        previous, extracted, quality, reference
    )
    full_indices = torch.arange(
        cloud_float64.n_particles, device=device, dtype=torch.long
    )
    final_problem = make_top_gap_problem(
        wave, soil, geometry, cloud_float64, options
    )
    metrics["top_gap_augmented_physics_loss"] = final_problem.loss(
        model, full_indices, augmented
    ).detached()
    metrics["top_point"] = _top_point_metrics(
        prediction, previous, extracted
    )
    metrics["four_level_cloud"] = cloud_float64.summary()
    metrics["augmented_boundary"] = augmented.as_dict()
    metrics["previous_adaptive_strong"] = previous_metrics["mpm_kan"]
    metrics["relative_l2_change"] = {
        key: (
            metrics["mpm_kan"][key]["relative_l2"]
            - previous_metrics["mpm_kan"][key]["relative_l2"]
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
        output_dir / "figure9_top_gap_augmented_mpm_kan.png",
        prediction,
        extracted,
        reference,
        baseline=previous,
        prediction_label="Top-gap AL + float64 MPM-KAN",
        baseline_label="Previous adaptive strong-form MPM-KAN",
    )
    plot_field(
        output_dir / "figure9_top_gap_augmented_field.png", field, geometry
    )
    save_profiles(
        output_dir / "figure9_top_gap_augmented_profiles.csv",
        prediction,
        extracted,
        reference,
    )
    save_json(
        metrics, output_dir / "figure9_top_gap_augmented_metrics.json"
    )
    (output_dir / "top_gap_training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "top_gap_run_config.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "dtype_finish": str(dtype),
                "baseline_checkpoint": str(baseline_checkpoint),
                "adaptive_checkpoint": str(adaptive_checkpoint),
                "teacher_checkpoint": (
                    str(teacher_checkpoint)
                    if teacher_checkpoint is not None
                    else None
                ),
                "top_gap_checkpoint": str(top_gap_checkpoint),
                "top_gap_options": asdict(options),
                "cloud_parameters": cloud_kwargs,
                "cloud": cloud_float64.summary(),
                "data_loaded_after_training": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    top = metrics["top_point"]["nearest_experimental_top_point"]
    print("Top-gap augmented run completed.")
    print(
        "Nearest top experimental point: "
        f"{top['previous_adaptive_strong']:.6f} -> "
        f"{top['prediction']:.6f}; observed={top['experimental']:.6f}; "
        f"|error|={top['absolute_error']:.6f}"
    )
    for source in (
        "paper_this_study",
        "experimental",
        "finite_difference_reference",
    ):
        new_value = metrics["mpm_kan"][source]["relative_l2"]
        old_value = previous_metrics["mpm_kan"][source]["relative_l2"]
        print(
            f"{source}: {100.0 * old_value:.3f}% -> "
            f"{100.0 * new_value:.3f}% "
            f"(delta {100.0 * (new_value - old_value):+.3f} pp)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 9 top-gap level-4 MPM refinement, augmented boundary "
            "constraints, and float64 full-batch finish."
        )
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--adaptive-checkpoint", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--float32-steps", type=int, default=60)
    parser.add_argument("--float32-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--float64-adam-steps", type=int, default=10)
    parser.add_argument("--float64-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--float64-lbfgs-steps", type=int, default=4)
    parser.add_argument("--batch-background", type=int, default=320)
    parser.add_argument("--batch-middle", type=int, default=480)
    parser.add_argument("--batch-fine", type=int, default=704)
    parser.add_argument("--batch-gap", type=int, default=1024)
    parser.add_argument("--momentum-weight", type=float, default=10.0)
    parser.add_argument("--fluid-weight", type=float, default=1000.0)
    parser.add_argument("--p2g-weight", type=float, default=0.25)
    parser.add_argument("--roughness-weight", type=float, default=1.0e-3)
    parser.add_argument("--boundary-points", type=int, default=192)
    parser.add_argument("--al-update-every", type=int, default=40)
    parser.add_argument("--rho-top", type=float, default=160.0)
    parser.add_argument("--rho-outer", type=float, default=160.0)
    parser.add_argument("--rho-pipe", type=float, default=320.0)
    parser.add_argument("--x-support-radii", type=float, default=0.60)
    parser.add_argument("--y-support-center-gap", type=float, default=1.40)
    parser.add_argument("--distillation-steps", type=int, default=800)
    parser.add_argument(
        "--distillation-learning-rate", type=float, default=1.0e-3
    )
    parser.add_argument("--seed", type=int, default=202409)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--base-x", type=int, default=48)
    parser.add_argument("--base-y", type=int, default=18)
    parser.add_argument("--middle-radial", type=int, default=8)
    parser.add_argument("--middle-theta", type=int, default=128)
    parser.add_argument("--fine-radial", type=int, default=7)
    parser.add_argument("--fine-theta", type=int, default=192)
    parser.add_argument("--gap-normal", type=int, default=10)
    parser.add_argument("--gap-theta", type=int, default=128)
    parser.add_argument("--p2g-x", type=int, default=128)
    parser.add_argument("--p2g-y", type=int, default=48)
    parser.add_argument("--theta-points", type=int, default=721)
    parser.add_argument("--field-x", type=int, default=241)
    parser.add_argument("--field-y", type=int, default=81)
    parser.add_argument("--reference-x", type=int, default=257)
    parser.add_argument("--reference-y", type=int, default=65)
    return parser


if __name__ == "__main__":
    main()
