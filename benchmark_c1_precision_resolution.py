"""Benchmark C1 inference precision and input resolution on Day1 validation data.

This is an inference-only experiment. It never updates model weights and never
opens the held-out test split. Run it with the CUDA-enabled Python interpreter.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torchvision.ops import box_iou

from c1_detection import C1CocoDetectionDataset, build_c1_detector
from train_c1_seed_baseline import average_precision


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations" / "day1_full_v1" / "normalized"
CHECKPOINT = ROOT / "runs" / "c1_detection" / "baseline_day1_full_v1" / "best.pt"
OUTPUT = ROOT / "runs" / "c1_detection" / "precision_resolution_benchmark"

SCORE_THRESHOLD = 0.25
IOU_THRESHOLDS = [0.50 + 0.05 * index for index in range(10)]
WARMUP_RUNS = 8
CONFIGURATIONS = (
    {"name": "default_fp32", "min_size": 800, "max_size": 1333, "dtype": "fp32"},
    {"name": "default_fp16", "min_size": 800, "max_size": 1333, "dtype": "fp16"},
    {"name": "default_bf16", "min_size": 800, "max_size": 1333, "dtype": "bf16"},
    {"name": "640_fp32", "min_size": 640, "max_size": 1138, "dtype": "fp32"},
    {"name": "640_fp16", "min_size": 640, "max_size": 1138, "dtype": "fp16"},
    {"name": "640_bf16", "min_size": 640, "max_size": 1138, "dtype": "bf16"},
)


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def autocast_context(dtype_name: str):
    if dtype_name == "fp32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def calculate_metrics(
    predictions: list[tuple[float, int, torch.Tensor]],
    ground_truth: dict[int, torch.Tensor],
) -> dict[str, float | int]:
    predictions.sort(key=lambda item: item[0], reverse=True)
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    aps: list[float] = []
    fixed: dict[str, float | int] | None = None

    for iou_threshold in IOU_THRESHOLDS:
        matched = {image_id: set() for image_id in ground_truth}
        true_positives: list[int] = []
        false_positives: list[int] = []
        scores: list[float] = []
        for score, image_id, box in predictions:
            targets = ground_truth[image_id]
            hit = False
            if len(targets):
                overlaps = box_iou(box.unsqueeze(0), targets)[0]
                best_iou, best_index = overlaps.max(dim=0)
                target_index = int(best_index)
                if (
                    float(best_iou) >= iou_threshold
                    and target_index not in matched[image_id]
                ):
                    matched[image_id].add(target_index)
                    hit = True
            true_positives.append(int(hit))
            false_positives.append(int(not hit))
            scores.append(score)

        aps.append(average_precision(true_positives, false_positives, total_gt))
        if iou_threshold == 0.50:
            kept = [i for i, score in enumerate(scores) if score >= SCORE_THRESHOLD]
            tp = sum(true_positives[i] for i in kept)
            fp = sum(false_positives[i] for i in kept)
            fn = total_gt - tp
            fixed = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, total_gt),
            }

    assert fixed is not None
    return {
        **fixed,
        "mAP_50": aps[0],
        "mAP_50_95": sum(aps) / len(aps),
        "score_threshold": SCORE_THRESHOLD,
        "ground_truth_boxes": total_gt,
    }


def benchmark_configuration(
    model: torch.nn.Module,
    dataset: C1CocoDetectionDataset,
    device: torch.device,
    config: dict,
) -> dict:
    model.transform.min_size = (config["min_size"],)
    model.transform.max_size = config["max_size"]
    model.roi_heads.score_thresh = 0.001
    if "rpn_pre_nms" in config:
        model.rpn._pre_nms_top_n["testing"] = config["rpn_pre_nms"]
    if "rpn_post_nms" in config:
        model.rpn._post_nms_top_n["testing"] = config["rpn_post_nms"]
    if "detections_per_img" in config:
        model.roi_heads.detections_per_img = config["detections_per_img"]

    warmup_image, _ = dataset[0]
    warmup_image = warmup_image.to(device)
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            with autocast_context(config["dtype"]):
                model([warmup_image])
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    predictions: list[tuple[float, int, torch.Tensor]] = []
    ground_truth: dict[int, torch.Tensor] = {}
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            image = image.to(device, non_blocking=False)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with autocast_context(config["dtype"]):
                output = model([image])[0]
            torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - started) * 1000)

            image_id = int(target["image_id"])
            ground_truth[image_id] = target["boxes"].cpu()
            boxes = output["boxes"].float().cpu()
            scores = output["scores"].float().cpu()
            predictions.extend(
                (float(score), image_id, box)
                for box, score in zip(boxes, scores, strict=True)
            )

    metrics = calculate_metrics(predictions, ground_truth)
    return {
        **config,
        "status": "ok",
        "validation_images": len(dataset),
        "warmup_runs": WARMUP_RUNS,
        "latency_mean_ms": statistics.fmean(timings_ms),
        "latency_median_ms": statistics.median(timings_ms),
        "latency_p90_ms": percentile(timings_ms, 0.90),
        "latency_p95_ms": percentile(timings_ms, 0.95),
        "latency_min_ms": min(timings_ms),
        "latency_max_ms": max(timings_ms),
        "model_fps_from_median": 1000 / statistics.median(timings_ms),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        **metrics,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。请使用已经验证过的 Python 3.14 + cu132 解释器。"
        )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{CHECKPOINT}")

    device = torch.device("cuda:0")
    dataset = C1CocoDetectionDataset(
        DATA / "images", DATA / "annotations" / "instances_val.json"
    )
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_c1_detector(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation images: {len(dataset)} | test split: unused")
    print("计时仅包含模型推理；图片解码和 CPU→GPU 传输不计入本轮。\n")

    rows: list[dict] = []
    for config in CONFIGURATIONS:
        print(
            f"[{config['name']}] min={config['min_size']} "
            f"max={config['max_size']} dtype={config['dtype']}"
        )
        try:
            row = benchmark_configuration(model, dataset, device, config)
            rows.append(row)
            print(
                f"  median={row['latency_median_ms']:.2f} ms "
                f"p95={row['latency_p95_ms']:.2f} ms "
                f"mAP50={row['mAP_50']:.4f} "
                f"mAP50:95={row['mAP_50_95']:.4f}\n"
            )
        except (RuntimeError, NotImplementedError) as error:
            torch.cuda.empty_cache()
            row = {**config, "status": "failed", "error": str(error)}
            rows.append(row)
            print(f"  FAILED: {error}\n")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": "Stage 1 — Local Combat Policy",
        "task": "C1 Enemy Detection inference benchmark",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "validation_images": len(dataset),
        "test_images_used": 0,
        "timing_scope": "model forward only; excludes decode and host-to-device copy",
        "results": rows,
    }
    json_path = OUTPUT / "results.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = OUTPUT / "results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
