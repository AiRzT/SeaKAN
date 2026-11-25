from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mpm_kan_figure15.config import figure15_setups
from mpm_kan_figure15.training import (
    _soil_row,
    load_checkpoint,
    make_soil_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the near-pipe Soil-KAN branch under an already "
            "physics-trained contact-enriched interface."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--float64-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument(
        "--float64-learning-rate", type=float, default=3.0e-6
    )
    parser.add_argument(
        "--train-global",
        action="store_true",
        help="Unfreeze both the global and near-pipe Soil-KAN branches.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20241551)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enriched soil training requires CUDA.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    models, _, options, histories, augmented = load_checkpoint(
        args.checkpoint, device
    )
    model = models["test49"].to(dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.local_kan.parameters():
        parameter.requires_grad_(True)
    if args.train_global:
        for parameter in model.kan.parameters():
            parameter.requires_grad_(True)
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    wave, soil, geometry = figure15_setups()["test49"]
    cloud, problem = make_soil_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float32,
    )
    state = augmented["test49"]
    optimizer = torch.optim.Adam(
        trainable, lr=args.learning_rate
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    history: list[dict[str, float | str]] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = cloud.sample_indices(
            (
                options.batch_background,
                options.batch_middle,
                options.batch_fine,
            ),
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss(model, indices, state)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 8.0)
        optimizer.step()
        if (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        ):
            row = _soil_row(
                "test49",
                (
                    "contact_global_local_soil_float32_adam"
                    if args.train_global
                    else "contact_local_soil_float32_adam"
                ),
                step,
                loss,
                state,
            )
            row["elapsed_seconds"] = (
                time.perf_counter() - started
            )
            history.append(row)
            print(json.dumps(row))

    model = model.to(dtype=torch.float64)
    cloud64, problem64 = make_soil_problem(
        wave,
        soil,
        geometry,
        options,
        device,
        torch.float64,
    )
    indices64 = torch.arange(
        cloud64.n_particles, device=device
    )
    trainable64 = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer64 = torch.optim.Adam(
        trainable64, lr=args.float64_learning_rate
    )
    for step in range(1, args.float64_steps + 1):
        optimizer64.zero_grad(set_to_none=True)
        loss = problem64.loss(model, indices64, state)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable64, 6.0)
        optimizer64.step()
        if (
            step == 1
            or step % 10 == 0
            or step == args.float64_steps
        ):
            row = _soil_row(
                "test49",
                (
                    "contact_global_local_soil_float64_full_adam"
                    if args.train_global
                    else "contact_local_soil_float64_full_adam"
                ),
                step,
                loss,
                state,
            )
            row["elapsed_seconds"] = (
                time.perf_counter() - started
            )
            history.append(row)
            print(json.dumps(row))

    final = problem64.loss(model, indices64, state)
    final_row = _soil_row(
        "test49",
        "contact_local_soil_final",
        0,
        final,
        state,
    )
    final_row["elapsed_seconds"] = (
        time.perf_counter() - started
    )
    history.append(final_row)
    payload = dict(source)
    payload["soil_test49"] = model.state_dict()
    released_histories = dict(histories)
    released_histories["contact_local_soil_training"] = history
    payload["histories"] = released_histories
    payload["training_data"] = (
        "no_figure15_csv; water/contact physics plus local strong "
        "Biot MPM-KAN refinement"
    )
    payload["contact_local_soil_training"] = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "figure15_data_used": False,
        "arguments": {
            name: (
                str(value.resolve())
                if isinstance(value, Path)
                else value
            )
            for name, value in vars(args).items()
        },
        "final": final_row,
    }
    checkpoint = args.output / "checkpoints" / "figure15.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "device": torch.cuda.get_device_name(device),
        "training_data": "physics_only_no_figure15_csv",
        "final": final_row,
        "cloud": cloud64.summary(),
    }
    (args.output / "enriched_soil_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "enriched_soil_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
