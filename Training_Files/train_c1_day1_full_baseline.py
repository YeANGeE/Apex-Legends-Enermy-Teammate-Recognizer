"""Train the first full Day1 baseline for C1 Enemy Detection.

Run this file from PyCharm with the CUDA-enabled Python 3.14 interpreter.
The 58-image test split is intentionally not referenced anywhere in this file.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from c1_detection import C1CocoDetectionDataset, build_c1_detector, detection_collate
from train_c1_seed_baseline import evaluate


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations" / "day1_full_v1" / "normalized"
OUTPUT = ROOT / "runs" / "c1_detection" / "baseline_day1_full_v1"

EPOCHS = 24
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
RANDOM_SEED = 20261007
MAX_NON_FINITE_BATCHES_PER_EPOCH = 3
RESUME_LR_FACTOR_AFTER_NAN = 0.5


def make_loader(dataset: C1CocoDetectionDataset, shuffle: bool) -> DataLoader:
    """Create a Windows-safe detection DataLoader."""
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In PyCharm, select the Python 3.14 interpreter "
            "whose PyTorch build reports +cu132."
        )
    device = torch.device("cuda:0")
    print(f"Device: {device} — {torch.cuda.get_device_name(0)}")

    train_dataset = C1CocoDetectionDataset(
        DATA / "images", DATA / "annotations" / "instances_train.json"
    )
    val_dataset = C1CocoDetectionDataset(
        DATA / "images", DATA / "annotations" / "instances_val.json"
    )
    print(
        f"Dataset: train={len(train_dataset)} images, "
        f"val={len(val_dataset)} images, test=unused"
    )
    train_loader = make_loader(train_dataset, shuffle=True)

    # build_c1_detector(pretrained=True) starts from COCO person features and
    # copies the person classifier/regressor into the single enemy foreground
    # class. This is a clean baseline; it does not reuse the seed checkpoint.
    model = build_c1_detector(pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[14, 20], gamma=0.1
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUT / "history.json"
    last_path = OUTPUT / "last.pt"
    history: list[dict] = []
    best_map = -1.0
    best_epoch = 0
    start_epoch = 1
    resume_adjustment = None
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best_record = max(history, key=lambda item: item["mAP_50_95"])
            best_map = float(best_record["mAP_50_95"])
            best_epoch = int(best_record["epoch"])

        # Checkpoints made by the first script version were written just
        # before scheduler.step(). Bring that state forward to the completed
        # epoch, then halve the already-small LR after the observed NaN.
        while scheduler.last_epoch < completed_epoch:
            scheduler.step()
        previous_adjustment = checkpoint.get("resume_adjustment")
        if previous_adjustment:
            resume_adjustment = previous_adjustment
            print(
                f"Resuming after epoch {completed_epoch} at learning rate "
                f"{optimizer.param_groups[0]['lr']:g}."
            )
        else:
            old_lr = float(optimizer.param_groups[0]["lr"])
            for group in optimizer.param_groups:
                group["lr"] *= RESUME_LR_FACTOR_AFTER_NAN
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            resume_adjustment = {
                "resumed_from_epoch": completed_epoch,
                "old_learning_rate": old_lr,
                "new_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "reason": "non-finite ROI loss observed after epoch 15",
            }
            print(
                f"Resuming after epoch {completed_epoch}; learning rate reduced "
                f"from {old_lr:g} to {optimizer.param_groups[0]['lr']:g}."
            )

    if start_epoch > EPOCHS:
        print(f"All {EPOCHS} epochs are already complete: {OUTPUT}")
        return

    image_names_by_id = {
        int(item["id"]): item["file_name"] for item in train_dataset.images
    }
    non_finite_log_path = OUTPUT / "non_finite_batches.json"
    if non_finite_log_path.is_file():
        non_finite_log = json.loads(
            non_finite_log_path.read_text(encoding="utf-8")
        )
    else:
        non_finite_log = []
    started = time.perf_counter()

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        batch_losses = []
        component_totals: dict[str, float] = {}
        skipped_batches = 0

        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                skipped_batches += 1
                affected_ids = [int(target["image_id"].detach().cpu()) for target in targets]
                event = {
                    "epoch": epoch,
                    "image_ids": affected_ids,
                    "file_names": [image_names_by_id[value] for value in affected_ids],
                    "losses": {
                        name: float(value.detach().cpu())
                        for name, value in loss_dict.items()
                    },
                }
                non_finite_log.append(event)
                save_json(non_finite_log_path, non_finite_log)
                optimizer.zero_grad(set_to_none=True)
                print(f"Skipped non-finite batch: {event}", flush=True)
                if skipped_batches > MAX_NON_FINITE_BATCHES_PER_EPOCH:
                    raise FloatingPointError(
                        f"More than {MAX_NON_FINITE_BATCHES_PER_EPOCH} non-finite "
                        f"batches in epoch {epoch}; stopping for diagnosis."
                    )
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0, error_if_nonfinite=False
            )
            if not torch.isfinite(gradient_norm):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped_batches > MAX_NON_FINITE_BATCHES_PER_EPOCH:
                    raise FloatingPointError(
                        f"Repeated non-finite gradients in epoch {epoch}."
                    )
                continue
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu()))
            for name, value in loss_dict.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(
                    value.detach().cpu()
                )

        metrics = evaluate(model, val_dataset, device)
        mean_loss = sum(batch_losses) / len(batch_losses)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": mean_loss,
            "skipped_non_finite_batches": skipped_batches,
            "loss_components": {
                name: total / len(batch_losses)
                for name, total in sorted(component_totals.items())
            },
            **metrics,
        }
        history.append(record)
        save_json(history_path, history)

        # Advance the schedule before checkpointing so a resumed run starts
        # with exactly the learning rate intended for the next epoch.
        scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_name": "fasterrcnn_mobilenet_v3_large_fpn_small_anchors",
            "num_classes": 2,
            "dataset": "day1_full_v1",
            "resume_adjustment": resume_adjustment,
        }
        torch.save(checkpoint, last_path)
        if metrics["mAP_50_95"] > best_map:
            best_map = metrics["mAP_50_95"]
            best_epoch = epoch
            torch.save(checkpoint, OUTPUT / "best.pt")

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={mean_loss:.4f} "
            f"mAP50={metrics['mAP_50']:.4f} "
            f"mAP50:95={metrics['mAP_50_95']:.4f} "
            f"P={metrics['precision']:.3f} "
            f"R={metrics['recall']:.3f}",
            flush=True,
        )

    best = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    summary = {
        "stage": "Stage 1 — Local Combat Policy",
        "task": "C1 Enemy Detection",
        "dataset": "day1_full_v1",
        "initialization": "COCO person weights mapped to enemy class",
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "test_images_used": 0,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        **best["metrics"],
        "elapsed_seconds": time.perf_counter() - started,
        "resume_adjustment": resume_adjustment,
        "non_finite_batches_logged": len(non_finite_log),
    }
    save_json(OUTPUT / "summary.json", summary)
    print("Training complete. Best validation result:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
