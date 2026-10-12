"""Review the selected first YOLO baseline and benchmark batch-1 inference.

Stage 1 — Local Combat Policy / C1 Enemy Detection. Validation only.
Run from PyCharm with .venv-yolo; no model training is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from c1_detection import C1CocoDetectionDataset
from evaluate_c1_day1_full_baseline import greedy_match


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "c1_yolo" / "day1_yolo26n_1280_v1"
CHECKPOINT = RUN / "weights" / "best.pt"
DATA = ROOT / "c1_annotations" / "day1_full_v1" / "normalized"
DISPLAY_THRESHOLD = 0.25
IMAGE_SIZE = 1280


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size=size)


def render(source: Path, target: torch.Tensor, match: dict, destination: Path,
           threshold: float = DISPLAY_THRESHOLD) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = font(22)

    def box(coords: torch.Tensor, color: str, label: str) -> None:
        x1, y1, x2, y2 = map(float, coords)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1, max(40, y1 - 28)), label, fill=color, font=label_font,
                  stroke_width=1, stroke_fill="black")

    for i, coords in enumerate(target):
        missed = i in match["missed_ground_truth"]
        box(coords, "yellow" if missed else "lime", "漏检敌人" if missed else "人工标注")
    for i, (coords, score) in enumerate(zip(match["boxes"], match["scores"], strict=True)):
        correct = i in match["true_predictions"]
        box(coords, "cyan" if correct else "red",
            f"{'正确检出敌人' if correct else '误检'} {float(score):.2f}")
    draw.rectangle((0, 0, image.width, 38), fill="black")
    draw.text((8, 4), f"绿色：人工框  青色：正确检出  红色：误检  黄色：漏检  阈值：{threshold:.2f}，IoU：0.50",
              fill="white", font=label_font)
    image.save(destination, quality=92)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(model: YOLO) -> dict:
    # predict() has already prepared/fused the same FP16 inference backend.
    backend = model.predictor.model
    dtype = torch.float16 if backend.fp16 else torch.float32
    inputs = torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=backend.device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(30):
            backend(inputs)
        torch.cuda.synchronize()
        timings = []
        for _ in range(200):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            backend(inputs)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    return {
        "measurement": "CUDA events, warmed model forward only",
        "batch": 1, "input_shape": list(inputs.shape), "dtype": str(dtype),
        "warmup_iterations": 30, "measured_iterations": 200,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": min(timings), "max_ms": max(timings),
        "p95_target_ms": 15.0,
        "p95_target_met": float(np.percentile(timings, 95)) <= 15.0,
        "excludes": "capture, decoding, preprocessing, transfer, external NMS, drawing, controller output",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=DISPLAY_THRESHOLD,
                        help="Confidence threshold used for rendered images and review counts")
    parser.add_argument("--run-dir", type=Path, default=RUN,
                        help="Training output directory containing weights/best.pt")
    args = parser.parse_args()
    if not 0.05 <= args.threshold <= 0.95:
        parser.error("--threshold must be between 0.05 and 0.95")
    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / "weights" / "best.pt"
    if not checkpoint.is_file() or not torch.cuda.is_available():
        raise RuntimeError("请确认所选训练的 best.pt 存在，并使用 .venv-yolo 解释器。")
    dataset = C1CocoDetectionDataset(DATA / "images", DATA / "annotations" / "instances_val.json")
    assert len(dataset) == 62 and sum(dataset.box_count(i) for i in range(len(dataset))) == 29
    if args.check_only:
        print(f"检查通过：{checkpoint}；62 张验证图片；29 个敌人框；测试集使用 0 张。")
        return

    output = run_dir / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    rendered = output / "rendered"
    rendered.mkdir(parents=True)
    model = YOLO(str(checkpoint))
    predictions, targets, review_rows = [], [], []
    print(f"正在检查 62 张验证图片，权重：{checkpoint}")
    # Warm up the prediction/postprocessing path before evaluating the first image.
    model.predict(source=np.zeros((1080, 1920, 3), dtype=np.uint8),
                  imgsz=IMAGE_SIZE, rect=False, device=0, half=True,
                  conf=0.05, iou=0.5, max_det=100, verbose=False)
    for i in range(len(dataset)):
        info = dataset.image_info(i)
        _, target = dataset[i]
        result = model.predict(
            source=str(DATA / "images" / info["file_name"]), imgsz=IMAGE_SIZE,
            rect=False, device=0, half=True, conf=0.05, iou=0.5,
            max_det=100, verbose=False,
        )[0]
        order = torch.argsort(result.boxes.conf, descending=True)
        pred = {"boxes": result.boxes.xyxy[order].cpu(), "scores": result.boxes.conf[order].cpu()}
        truth = target["boxes"].cpu()
        predictions.append(pred)
        targets.append(truth)
        match = greedy_match(pred, truth, args.threshold)
        row = {"image_id": i + 1, "file_name": info["file_name"], "tp": len(match["true_predictions"]),
               "fp": len(match["false_predictions"]), "fn": len(match["missed_ground_truth"])}
        review_rows.append(row)
        render(DATA / "images" / info["file_name"], truth, match,
               rendered / info["file_name"], args.threshold)
        print(f"验证 {i + 1}/62：正确 {row['tp']}，误检 {row['fp']}，漏检 {row['fn']}")

    sweep = []
    for threshold in (round(i * 0.05, 2) for i in range(1, 20)):
        tp = fp = fn = 0
        for pred, truth in zip(predictions, targets, strict=True):
            matched = greedy_match(pred, truth, threshold)
            tp += len(matched["true_predictions"])
            fp += len(matched["false_predictions"])
            fn += len(matched["missed_ground_truth"])
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        sweep.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                      "precision": precision, "recall": recall,
                      "f1": 2 * precision * recall / max(1e-12, precision + recall)})
    write_csv(output / "threshold_sweep.csv", sweep, list(sweep[0]))
    write_csv(output / "image_review.csv", review_rows, list(review_rows[0]))

    failures = sorted((r for r in review_rows if r["fp"] or r["fn"]),
                      key=lambda r: r["fp"] + r["fn"], reverse=True)
    for offset in range(0, len(failures), 12):
        sheet = Image.new("RGB", (1920, 1280), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(failures[offset:offset + 12]):
            with Image.open(rendered / row["file_name"]) as source:
                tile = source.copy()
                tile.thumbnail((640, 280))
            x, y = (j % 3) * 640, (j // 3) * 320
            sheet.paste(tile, (x, y))
            draw.text((x + 5, y + 280), f"编号 {row['image_id']:02d} | 正确 {row['tp']} 误检 {row['fp']} 漏检 {row['fn']} | 阈值 {args.threshold:.2f}",
                      fill="black", font=font(20))
        sheet.save(output / f"contact_sheet_{offset // 12 + 1:02d}.jpg", quality=92)

    print("正在预热并测量单张模型前向延迟，请保持 GPU 空闲。")
    latency = benchmark(model)
    summary = {
        "stage": "Stage 1 — Local Combat Policy", "task": "C1 Enemy Detection",
        "checkpoint": str(checkpoint), "validation_images": 62,
        "ground_truth_boxes": 29, "test_images_used": 0,
        "prediction_settings": {"imgsz": IMAGE_SIZE, "rect": False, "half": True, "nms_iou": 0.5},
        "metrics_at_0_25": next(r for r in sweep if r["threshold"] == 0.25),
        "display_threshold": args.threshold,
        "metrics_at_display_threshold": {
            "threshold": args.threshold,
            "tp": sum(r["tp"] for r in review_rows),
            "fp": sum(r["fp"] for r in review_rows),
            "fn": sum(r["fn"] for r in review_rows),
        },
        "best_validation_f1": max(sweep, key=lambda r: r["f1"]),
        "failure_images": len(failures), "gpu": torch.cuda.get_device_name(0),
        "latency": latency,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()
