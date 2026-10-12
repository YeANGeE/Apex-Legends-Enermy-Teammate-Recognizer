"""Train and evaluate the first C1 seed detector (no test-set access)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from c1_detection import C1CocoDetectionDataset, build_c1_detector, detection_collate


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations" / "day1_seed_v1" / "normalized"
OUT = ROOT / "runs" / "c1_detection" / "baseline_seed_v1"
EPOCHS = 24
SCORE_THRESHOLD = 0.25
IOU_THRESHOLDS = [0.50 + 0.05 * index for index in range(10)]


def loader(dataset: C1CocoDetectionDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate,
        generator=torch.Generator().manual_seed(20261007),
    )


def average_precision(tp: list[int], fp: list[int], total_gt: int) -> float:
    if total_gt == 0:
        return 0.0
    cumulative_tp, cumulative_fp = 0, 0
    precisions, recalls = [], []
    for is_tp, is_fp in zip(tp, fp, strict=True):
        cumulative_tp += is_tp
        cumulative_fp += is_fp
        precisions.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / total_gt)
    return sum(
        max((p for p, r in zip(precisions, recalls, strict=True) if r >= level), default=0.0)
        for level in (index / 100 for index in range(101))
    ) / 101


def evaluate(model, dataset, device) -> dict:
    model.eval()
    old_threshold = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001
    predictions = []
    ground_truth = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model([image.to(device)])[0]
            image_id = int(target["image_id"])
            ground_truth[image_id] = target["boxes"].cpu()
            for box, score in zip(output["boxes"].cpu(), output["scores"].cpu(), strict=True):
                predictions.append((float(score), image_id, box))
    model.roi_heads.score_thresh = old_threshold
    predictions.sort(key=lambda item: item[0], reverse=True)
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    aps = []
    fixed = None
    for threshold in IOU_THRESHOLDS:
        matched = {image_id: set() for image_id in ground_truth}
        tp, fp, scores = [], [], []
        for score, image_id, box in predictions:
            targets = ground_truth[image_id]
            hit = False
            if len(targets):
                ious = box_iou(box.unsqueeze(0), targets)[0]
                best_iou, best_index = ious.max(dim=0)
                best = int(best_index)
                if float(best_iou) >= threshold and best not in matched[image_id]:
                    matched[image_id].add(best)
                    hit = True
            tp.append(int(hit)); fp.append(int(not hit)); scores.append(score)
        aps.append(average_precision(tp, fp, total_gt))
        if threshold == 0.50:
            keep = [i for i, score in enumerate(scores) if score >= SCORE_THRESHOLD]
            fixed_tp = sum(tp[i] for i in keep)
            fixed_fp = sum(fp[i] for i in keep)
            fixed_fn = total_gt - fixed_tp
            fixed = {
                "tp": fixed_tp, "fp": fixed_fp, "fn": fixed_fn,
                "precision": fixed_tp / max(1, fixed_tp + fixed_fp),
                "recall": fixed_tp / max(1, total_gt),
            }
    return {
        **fixed,
        "mAP_50": aps[0],
        "mAP_50_95": sum(aps) / len(aps),
        "score_threshold": SCORE_THRESHOLD,
        "ground_truth_boxes": total_gt,
    }


def main() -> None:
    random.seed(20261007); torch.manual_seed(20261007); torch.cuda.manual_seed_all(20261007)
    device = torch.device("cuda")
    train = C1CocoDetectionDataset(DATA / "images", DATA / "annotations" / "instances_train.json")
    val = C1CocoDetectionDataset(DATA / "images", DATA / "annotations" / "instances_val.json")
    train_loader = loader(train, True)
    model = build_c1_detector(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14, 20], gamma=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    history, best_map, started = [], -1.0, time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets); loss = sum(loss_dict.values())
            if not torch.isfinite(loss): raise FloatingPointError(str(loss))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = evaluate(model, val, device)
        record = {"epoch": epoch, "train_loss": sum(losses)/len(losses), **metrics}
        history.append(record)
        print(f"Epoch {epoch:02d}/{EPOCHS} loss={record['train_loss']:.4f} mAP50={metrics['mAP_50']:.4f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}", flush=True)
        if metrics["mAP_50_95"] > best_map:
            best_map = metrics["mAP_50_95"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                        "model_name": "fasterrcnn_mobilenet_v3_large_fpn_small_anchors",
                        "num_classes": 2}, OUT / "best.pt")
    best = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
    summary = {
        "stage": "Stage 1 — Local Combat Policy", "task": "C1 Enemy Detection",
        "train_images": len(train), "val_images": len(val), "test_images_used": 0,
        "epochs": EPOCHS, "best_epoch": best["epoch"], **best["metrics"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
