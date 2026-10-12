"""Run the mandatory small-data overfit test for C1 Enemy Detection."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from c1_detection import (
    C1CocoDetectionDataset,
    build_c1_detector,
    detection_collate,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "c1_annotations" / "day1_seed_v1" / "normalized"
DEFAULT_ANNOTATIONS = DATASET_ROOT / "annotations" / "instances_train.json"
DEFAULT_IMAGES = DATASET_ROOT / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "c1_detection" / "overfit_final_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny C1 detection subset")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--positive-samples", type=int, default=1)
    parser.add_argument("--negative-samples", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("even", "largest-box"),
        default="largest-box",
        help="Use largest-box for the minimal pipeline-memory diagnostic",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261007)
    return parser.parse_args()


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if count > len(values):
        raise ValueError(f"Cannot select {count} items from {len(values)}")
    if count == 0:
        return []
    indices = [
        min(len(values) - 1, int((index + 0.5) * len(values) / count))
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError("Tiny-subset selection produced duplicate indices")
    return [values[index] for index in indices]


def choose_subset(
    dataset: C1CocoDetectionDataset,
    positives: int,
    negatives: int,
    selection: str,
) -> list[int]:
    positive_indices = [index for index in range(len(dataset)) if dataset.box_count(index)]
    negative_indices = [index for index in range(len(dataset)) if not dataset.box_count(index)]
    if selection == "largest-box":
        positive_indices.sort(key=dataset.largest_box_area, reverse=True)
        selected_positives = positive_indices[:positives]
    else:
        selected_positives = evenly_spaced(positive_indices, positives)
    selected = selected_positives + evenly_spaced(negative_indices, negatives)
    return sorted(selected)


def move_targets(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict]:
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The C1 overfit test requires the configured CUDA GPU")

    dataset = C1CocoDetectionDataset(args.images, args.annotations)
    selected_indices = choose_subset(
        dataset, args.positive_samples, args.negative_samples, args.selection
    )
    subset = Subset(dataset, selected_indices)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        generator=generator,
        pin_memory=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    selected = [
        {
            "dataset_index": index,
            "file_name": dataset.image_info(index)["file_name"],
            "encounter_id": dataset.image_info(index).get("encounter_id"),
            "enemy_boxes": dataset.box_count(index),
        }
        for index in selected_indices
    ]
    (args.output / "selected_samples.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Device: {device} — {torch.cuda.get_device_name(0)}")
    print(f"Dataset images: {len(dataset)}")
    print(
        f"Overfit subset: {len(subset)} "
        f"({args.positive_samples} positive, {args.negative_samples} negative)"
    )
    print("Building pretrained detector...", flush=True)
    model = build_c1_detector(pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 32], gamma=0.1
    )

    history = []
    best_loss = math.inf
    started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        component_sums: dict[str, float] = {}
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = move_targets(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite detection loss: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            for name, value in loss_dict.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    value.detach().cpu()
                )

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        record = {
            "epoch": epoch,
            "mean_loss": mean_loss,
            **{
                name: value / len(epoch_losses)
                for name, value in sorted(component_sums.items())
            },
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} — loss {mean_loss:.6f}",
            flush=True,
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "mean_loss": mean_loss,
                    "model_name": "fasterrcnn_mobilenet_v3_large_fpn_small_anchors",
                    "num_classes": 2,
                },
                args.output / "best.pt",
            )
        scheduler.step()

    elapsed_seconds = time.perf_counter() - started_at
    window = min(5, len(history))
    initial_loss = history[0]["mean_loss"]
    final_loss = sum(item["mean_loss"] for item in history[-window:]) / window
    reduction = 1.0 - best_loss / initial_loss
    passed = reduction >= 0.60

    summary = {
        "test": "C1 Enemy Detection small-data overfit",
        "stage": "Stage 1 — Local Combat Policy",
        "model": "fasterrcnn_mobilenet_v3_large_fpn_small_anchors",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dataset_images": len(dataset),
        "overfit_images": len(subset),
        "positive_images": args.positive_samples,
        "negative_images": args.negative_samples,
        "epochs": args.epochs,
        "initial_mean_loss": initial_loss,
        "final_mean_loss": final_loss,
        "best_epoch_loss": best_loss,
        "best_loss_reduction_fraction": reduction,
        "pass_threshold_fraction": 0.60,
        "passed": passed,
        "elapsed_seconds": elapsed_seconds,
    }
    (args.output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("Small-data overfit loss did not meet the 60% reduction threshold")


if __name__ == "__main__":
    main()
