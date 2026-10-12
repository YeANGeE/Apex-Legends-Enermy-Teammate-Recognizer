"""Stage 1 — Local Combat Policy: user-run common holdout and predictor timing.

Evaluate the Day3 best Faster R-CNN and both YOLO baselines. No training.
--check-only audits inputs and checkpoint loading without validation/benchmark.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from c1_detection.two_class_dataset import audit_training_data, digest
from c1_detection.two_class_metrics import coco_metrics, fixed_metrics
from evaluate_c1_yolo_teammates import draw_review, match_predictions
from train_c1_fasterrcnn_day3_teammates import DATA as TRAIN_DATA, ROOT, SETTINGS, build_model, save_json

HOLDOUT = ROOT / "c1_yolo_dataset/day2_validation_v1"
DEFAULT_FRCNN = ROOT / "runs/c1_detection/day3_fasterrcnn_enemy_teammate_20261009_160808_243868/best.pt"
YOLO_WEIGHTS = {
    "yolo_day1_v3": ROOT / "runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt",
    "yolo_day2_v4": ROOT / "runs/c1_yolo/day2_yolo26n_1280_v4_real_match-2/weights/best.pt",
}
DISPLAY = {"fasterrcnn_day3": "Faster R-CNN Day3", "yolo_day1_v3": "YOLO Day1 V3", "yolo_day2_v4": "YOLO Day2 V4"}


def normalize_prediction(boxes, scores, labels, yolo=False):
    """All evaluators use CPU absolute xyxy, 1=enemy, 2=teammate."""
    boxes = boxes.detach().float().cpu().reshape(-1, 4)
    scores = scores.detach().float().cpu().reshape(-1)
    raw_labels = labels.detach().cpu().reshape(-1)
    if not torch.isfinite(raw_labels).all() or not torch.equal(raw_labels, raw_labels.long().to(raw_labels.dtype)):
        raise ValueError("Noninteger prediction labels")
    labels = raw_labels.long() + int(yolo)
    if len(boxes) != len(scores) or len(scores) != len(labels):
        raise ValueError("Prediction lengths differ")
    if (not set(labels.tolist()).issubset({1, 2}) or not torch.isfinite(boxes).all() or
            not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or
            (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("Invalid detector prediction")
    return {"boxes": boxes, "scores": scores, "labels": labels}


def load_holdout():
    authority = json.loads((HOLDOUT / "dataset_summary.json").read_text(encoding="utf-8"))
    records = json.loads((HOLDOUT / "source_manifest.json").read_text(encoding="utf-8"))
    records = sorted((r for r in records if r["eligible"] and r["scene_type"] == "real_match"), key=lambda r: r["preview_id"])
    if len(records) != 38 or len({r["preview_id"] for r in records}) != len(records):
        raise ValueError("Unexpected holdout roster")
    prior = json.loads((TRAIN_DATA / "source_manifest.json").read_text(encoding="utf-8"))
    old_hashes = {r["image_sha256"] for r in prior}
    old_pixels = {r.get("pixel_sha256") for r in prior if r.get("pixel_sha256")}
    old_groups = {r["encounter_id"] for r in prior}
    targets, sizes = [], []
    for i, record in enumerate(records, 1):
        image, label = Path(record["normalized_image"]), Path(record["normalized_label"])
        if record["split"] != "comparison_holdout" or record["quality"] != "usable":
            raise ValueError("Ineligible holdout record")
        if digest(image) != record["image_sha256"] or digest(label) != record["label_sha256"]:
            raise ValueError("Holdout content changed")
        if (record["image_sha256"] in old_hashes or record.get("pixel_sha256") in old_pixels or
                record["tentative_group"] in old_groups):
            raise ValueError("Holdout overlaps training/validation")
        with Image.open(image) as im:
            width, height = im.size
        sizes.append((width, height))
        boxes, labels = [], []
        for b in record["boxes"]:
            if b["class_id"] not in (0, 1):
                raise ValueError("Unexpected target identity")
            x, y, w, h = b["xywh"]
            coords = np.array([x-w/2, y-h/2, x+w/2, y+h/2])
            if not np.isfinite(coords).all() or min(w, h) <= 0 or coords.min() < -1e-8 or coords.max() > 1+1e-8:
                raise ValueError("Invalid holdout box")
            boxes.append((coords*np.array([width, height, width, height])).tolist())
            labels.append(b["class_id"]+1)
        targets.append({"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                        "labels": torch.tensor(labels, dtype=torch.int64), "image_id": torch.tensor(i)})
        # Serialized YOLO labels must describe the same authority used for metrics.
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(record["boxes"]):
            raise ValueError("Manifest/label box count differs")
        for row, b in zip(rows, record["boxes"], strict=True):
            if len(row) != 5 or int(row[0]) != b["class_id"] or not np.allclose(list(map(float, row[1:])), b["xywh"], atol=1e-6, rtol=0):
                raise ValueError("Manifest/label coordinates differ")
    counts = {"images": len(records), "enemy_boxes": sum(int((t["labels"] == 1).sum()) for t in targets),
              "teammate_boxes": sum(int((t["labels"] == 2).sum()) for t in targets)}
    if any(v != authority["statistics"]["real_match"][k] for k, v in counts.items()):
        raise ValueError("Holdout counts differ from authority")
    return records, targets, sizes


class Predictor:
    def __init__(self, key, weights, audit=None):
        self.key = key
        if key == "fasterrcnn_day3":
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            if (checkpoint.get("experiment") != "day3_fasterrcnn_enemy_teammate" or
                    checkpoint.get("num_classes") != 3 or checkpoint.get("classes") != {0: "background", 1: "enemy", 2: "teammate"} or
                    checkpoint.get("settings") != SETTINGS):
                raise ValueError("Wrong Faster R-CNN experiment/classes/settings")
            if audit and any(checkpoint["data_audit"][k] != audit[k] for k in ("manifest_sha256", "summary_sha256", "statistics")):
                raise ValueError("Checkpoint training data differs")
            self.model = build_model(pretrained=False)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.model.cuda().eval()
            self.epoch = checkpoint["epoch"]
            self.dtype = "float32"
        else:
            self.model = YOLO(str(weights))
            if self.model.names != {0: "enemy", 1: "teammate"}:
                raise ValueError("Wrong YOLO classes")
            self.epoch = None
            self.dtype = "float16"

    def __call__(self, rgb, conf=.001):
        with torch.inference_mode():
            if self.key == "fasterrcnn_day3":
                self.model.roi_heads.score_thresh = conf
                tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).cuda().float().div_(255)
                result = self.model([tensor])[0]
                return normalize_prediction(result["boxes"], result["scores"], result["labels"])
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            result = self.model.predict(bgr, imgsz=1280, rect=False, device=0, quantize=16,
                                        conf=conf, iou=.5, max_det=100, verbose=False)[0]
            return normalize_prediction(result.boxes.xyxy, result.boxes.conf, result.boxes.cls, yolo=True)


def read_rgb(record):
    with Image.open(record["normalized_image"]) as im:
        return np.array(im.convert("RGB"))


def benchmark_predictor(predictor, rgbs, warmup=30, iterations=200):
    """Identical timing boundaries: host RGB input -> CPU original-pixel boxes."""
    for index in range(warmup):
        predictor(rgbs[index % len(rgbs)], .05)
    torch.cuda.synchronize()
    timings = []
    for index in range(iterations):
        rgb = rgbs[index % len(rgbs)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        predictor(rgb, .05)
        torch.cuda.synchronize()
        timings.append((time.perf_counter()-start)*1000)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "min_ms": min(timings), "max_ms": max(timings), "warmup": warmup, "iterations": iterations,
            "batch": 1, "confidence_threshold": .05, "dtype": predictor.dtype,
            "measurement": "Synchronized wall time: host RGB ndarray to CPU boxes/classes/scores",
            "includes": "format conversion, normalization, resize/padding, H2D, model, proposals/NMS, D2H",
            "excludes": "image decoding/capture, rendering, tracking, controller output",
            "note": "Predictor latency, not model-forward-only or capture-to-action; do not compare directly to 15ms forward budget."}


def review_image(record, target, pred, folder, threshold):
    targets = [{"box": b, "class": int(c)-1} for b, c in zip(target["boxes"].tolist(), target["labels"].tolist(), strict=True)]
    keep = pred["scores"] >= threshold
    predictions = [{"box": b, "class": int(c)-1, "score": s} for b, c, s in
                   zip(pred["boxes"][keep].tolist(), pred["labels"][keep].tolist(), pred["scores"][keep].tolist(), strict=True)]
    correct, wrong, missed = match_predictions(predictions, targets)
    row = {"validation_id": record["preview_id"], "gallery_id": record["preview_id"],
           "image_name": record["image_name"], "correct": len(correct), "wrong_identity": len(wrong),
           "unmatched_predictions": len(predictions)-len(correct)-len(wrong), "unmatched_targets": len(missed)}
    draw_review(Path(record["normalized_image"]), folder / record["image_name"], targets, predictions, correct, wrong, row, threshold)
    return row


def make_report(output, summaries, records, threshold):
    keys = list(summaries)
    lines = ["# Day3 三模型 holdout 对照", "", "同一38张图，37个敌人框、10个队友框；所有AP统一使用COCOeval。", "",
             "| 指标 | " + " | ".join(DISPLAY[k] for k in keys) + " |", "|---|---:|---:|---:|"]
    def row(label, values, percent=False):
        cells = ["未定义" if v is None else f"{v:.1%}" if percent else f"{v:.4f}" for v in values]
        lines.append(f"| {label} | " + " | ".join(cells)+" |")
    for metric in ("map50", "map50_95"):
        row(metric, [summaries[k][metric] for k in keys])
    for conf in ("0.05", "0.25"):
        for name, title in (("enemy", "敌人"), ("teammate", "队友")):
            for metric, label in (("precision", "精确率"), ("recall", "召回率")):
                row(f"{title} {label} conf={conf}", [summaries[k]["operating_points"][conf]["classes"][name][metric] for k in keys], True)
        for metric, label in (("person_recall", "人物召回"), ("identity_accuracy_on_located", "已定位人物身份准确率")):
            row(f"{label} conf={conf}", [summaries[k]["operating_points"][conf][metric] for k in keys], True)
        row(f"未匹配预测/100帧 conf={conf}", [summaries[k]["operating_points"][conf]["unmatched_predictions_per_100_frames"] for k in keys])
    for name, label in (("median_ms", "完整预测中位延迟ms"), ("p95_ms", "完整预测p95延迟ms")):
        row(label, [summaries[k]["predictor_latency"][name] for k in keys])
    lines += ["", "计时从已解码CPU RGB画面到CPU原图坐标预测结果，包含预处理/传输/后处理。Faster R-CNN为FP32，YOLO为FP16。",
              "各模型都以长边1280处理画面，padding和内部处理不同；这是当前配置的实用比较。不能用此p95直接判断15ms纯前向预算。",
              "", "单次录制、相关帧、队友仅10个实例，且该holdout已用于Day2和跟踪迭代；不能视为全新封存测试集。",
              "未匹配预测包含重复框、定位不合格和背景误检。身份准确率只覆盖已定位人物。", "",
              f"下方逐图可视化阈值为{threshold}。使用原有类别优先匹配着色；汇总的人物/身份诊断则采用类别无关匹配。", ""]
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)
    for offset in range(0, len(records), 4):
        sheet = Image.new("RGB", (5760, 4720), "#111820")
        draw = ImageDraw.Draw(sheet)
        for j, record in enumerate(records[offset:offset+4]):
            for column, key in enumerate(keys):
                x, y = column*1920, j*1180
                with Image.open(output/key/"rendered"/record["image_name"]) as im:
                    sheet.paste(im.convert("RGB"), (x, y))
                draw.text((x+12, y+1100), f"编号{record['preview_id']:02d} {DISPLAY[key]}", font=font, fill="white")
        path = output / f"三模型对照_{offset//4+1:02d}.jpg"
        sheet.save(path, quality=95, subsampling=0)
        lines.append(f"- [对照 {offset//4+1:02d}]({path.as_posix()})")
    (output / "对比报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--fasterrcnn-weights", type=Path, default=DEFAULT_FRCNN)
    parser.add_argument("--display-threshold", type=float, choices=(.05, .25), default=.05)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 .venv-yolo CUDA环境")
    audit = audit_training_data(TRAIN_DATA)
    records, targets, sizes = load_holdout()
    weights = {"fasterrcnn_day3": args.fasterrcnn_weights, **YOLO_WEIGHTS}
    hashes = {k: digest(path) for k, path in weights.items()}
    if args.check_only:
        for key, path in weights.items():
            predictor = Predictor(key, path, audit)
            print(f"检查通过：{DISPLAY[key]} 权重加载成功；轮次={predictor.epoch}", flush=True)
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
        print("数据与三个模型检查通过：38张holdout，37个敌人框、10个队友框。")
        return
    output = ROOT / "runs/c1_detection" / ("day3_holdout_comparison_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    save_json(output / "run_config.json", {"stage": "Stage 1 — Local Combat Policy", "weights": {k: str(p) for k, p in weights.items()},
        "checkpoint_sha256": hashes, "holdout_manifest_sha256": digest(HOLDOUT/"source_manifest.json"),
        "training_manifest_sha256": audit["manifest_sha256"], "preview_ids": [r["preview_id"] for r in records],
        "display_threshold": args.display_threshold, "training_started": False,
        "small_or_blurry_targets_removed": False, "test_split_used": False,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
    # Decode before timing; exactly the same RGB frames for each predictor.
    rgbs = [read_rgb(r) for r in records]
    summaries = {}
    for key, path in weights.items():
        print(f"开始评估：{DISPLAY[key]}", flush=True)
        predictor = Predictor(key, path, audit)
        folder = output / key
        rendered = folder / "rendered"
        rendered.mkdir(parents=True)
        predictions, rows = [], []
        for i, (record, target, rgb) in enumerate(zip(records, targets, rgbs, strict=True), 1):
            pred = predictor(rgb, .001)
            predictions.append(pred)
            rows.append(review_image(record, target, pred, rendered, args.display_threshold))
            print(f"  图片{i}/{len(records)} 编号{record['preview_id']:02d}", flush=True)
        metrics = {**coco_metrics(predictions, targets, sizes),
                   "operating_points": {str(t): fixed_metrics(predictions, targets, t) for t in (.05, .25)}}
        print("  预热30次，完整预测测速200次……", flush=True)
        latency = benchmark_predictor(predictor, rgbs)
        summary = {**metrics, "checkpoint": str(path), "checkpoint_sha256": hashes[key],
                   "epoch": predictor.epoch, "images": len(records), "predictor_latency": latency}
        save_json(folder / "summary.json", summary)
        details = [{"preview_id": r["preview_id"], "prediction": {k: v.tolist() for k, v in p.items()},
                    "target": {k: v.tolist() for k, v in t.items()}}
                   for r, p, t in zip(records, predictions, targets, strict=True)]
        save_json(folder / "prediction_details.json", details)
        with (folder / "image_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries[key] = summary
        print(f"{DISPLAY[key]} mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f} 完整预测p95={latency['p95_ms']:.2f}ms", flush=True)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    save_json(output / "comparison.json", {"models": summaries, "images": len(records),
        "metric_method": "Shared pycocotools COCOeval and fixed_metrics for all models",
        "limitation": "Previously inspected Day2 holdout; single recording session, correlated frames, only 10 teammate instances"})
    make_report(output, summaries, records, args.display_threshold)
    print(f"验收完成，结果目录：{output}\n对比报告：{output / '对比报告.md'}", flush=True)


if __name__ == "__main__":
    main()
