"""Stage 1 — Local Combat Policy: fixed-threshold two-class visual acceptance.

Validation inference only; no training or WebUI replacement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluate_c1_yolo_day1 import benchmark, font
from train_c1_yolo_day1_teammates import DATA, ROOT, check_configuration

WEIGHTS = ROOT / "runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt"

NAMES = {0: "敌人", 1: "队友"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(1e-12, area-intersection)


def match_predictions(predictions, targets):
    """Score-ordered same-class matches first; leftover cross-class overlaps are diagnostic."""
    remaining = set(range(len(targets)))
    correct, wrong = {}, {}
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True):
        p = predictions[index]
        candidates = [t for t in remaining if targets[t]["class"] == p["class"] and iou(p["box"], targets[t]["box"]) >= .5]
        if candidates:
            target = max(candidates, key=lambda t: iou(p["box"], targets[t]["box"]))
            correct[index] = target
            remaining.remove(target)
    for index in sorted(set(range(len(predictions)))-set(correct), key=lambda i: predictions[i]["score"], reverse=True):
        p = predictions[index]
        candidates = [t for t in remaining if targets[t]["class"] != p["class"] and iou(p["box"], targets[t]["box"]) >= .5]
        if candidates:
            target = max(candidates, key=lambda t: iou(p["box"], targets[t]["box"]))
            wrong[index] = target
            remaining.remove(target)
    return correct, wrong, sorted(remaining)


def draw_review(source, destination, targets, predictions, correct, wrong, row, threshold):
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    text_font = font(21)
    matched_targets = set(correct.values())
    def box(coords, color, label, width):
        x1, y1, x2, y2 = coords
        draw.rectangle(coords, outline=color, width=width)
        draw.text((x1, max(76, y1-25)), label, fill=color, font=text_font, stroke_width=1, stroke_fill="black")
    for index, target in enumerate(targets):
        color = "lime" if index in matched_targets else "yellow"
        box(target["box"], color, f"人工：{NAMES[target['class']]}", 3)
    for index, pred in enumerate(predictions):
        color = "cyan" if index in correct else "magenta" if index in wrong else "red"
        label = "正确" if index in correct else "敌我认反" if index in wrong else "误检/框偏差"
        # Place predicted labels below the box so manual labels remain readable.
        draw.rectangle(pred["box"], outline=color, width=4)
        draw.text((pred["box"][0], min(image.height-28, pred["box"][3]+2)),
                  f"{label}：{NAMES[pred['class']]} {pred['score']:.2f}", fill=color,
                  font=text_font, stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, image.width, 72), fill="black")
    draw.text((8, 3), f"验证编号 {row['validation_id']:02d} / 筛选编号 {row['gallery_id']:03d}  阈值{threshold:.2f}  匹配IoU≥0.50", font=text_font, fill="white")
    draw.text((8, 36), "青：正确  紫：敌我认反  红：误检/框偏差  绿：已匹配人工框  黄：未正确检出的人工框", font=text_font, fill="white")
    image.save(destination, quality=93)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=.05)
    parser.add_argument("--validation-labels", type=Path,
                        help="Versioned human-repaired validation labels directory")
    args = parser.parse_args()
    if not .001 <= args.threshold <= .95:
        parser.error("Threshold must be between .001 and .95")
    check_configuration()
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    data_root = DATA.parent
    manifest = json.loads((data_root / "source_manifest.json").read_text(encoding="utf-8"))
    records = sorted((r for r in manifest if r["split"] == "val"), key=lambda r: r["gallery_id"])
    labels_dir = data_root / "labels/val"
    if args.validation_labels:
        labels_dir = args.validation_labels.resolve()
        correction = json.loads((labels_dir.parent / "summary.json").read_text(encoding="utf-8"))
        if correction["training_changed"] or correction["test_images_used"] or correction["validation_images"] != len(records):
            raise ValueError("Invalid validation correction metadata")
        if correction["base_manifest_sha256"] != hashlib.sha256((data_root / "source_manifest.json").read_bytes()).hexdigest():
            raise ValueError("Correction belongs to a different dataset")
        for name, digest in correction["label_sha256"].items():
            if Path(name).name != name or hashlib.sha256((labels_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Corrected label checksum mismatch")
        if {p.stem for p in labels_dir.glob("*.txt")} != {Path(r["image_name"]).stem for r in records}:
            raise ValueError("Corrected label roster mismatch")
    print(f"评估权重：{WEIGHTS}；验证标签：{labels_dir}；阈值：{args.threshold}")
    if args.check_only:
        return
    output = WEIGHTS.parent.parent / ("review_two_class_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    rendered = output / "rendered"
    rendered.mkdir(parents=True)
    model = YOLO(str(WEIGHTS))
    if model.names != {0: "enemy", 1: "teammate"}:
        raise ValueError(f"Checkpoint classes differ: {model.names}")
    model.predict(np.zeros((1080, 1920, 3), dtype=np.uint8), imgsz=1280, rect=False,
                  device=0, half=True, conf=args.threshold, iou=.5, max_det=100, verbose=False)
    rows, details = [], []
    totals = {cls: Counter(tp=0, fp=0, fn=0) for cls in NAMES}
    diagnostic = Counter(teammate_as_enemy=0, enemy_as_teammate=0, unmatched_predictions=0,
                         unmatched_targets=0)
    for ordinal, record in enumerate(records, 1):
        source = data_root / "images/val" / record["image_name"]
        with Image.open(source) as image:
            width, height = image.size
        targets = []
        for line in (labels_dir / Path(source.name).with_suffix(".txt")).read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            targets.append({"class": int(cls), "box": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
        result = model.predict(str(source), imgsz=1280, rect=False, device=0, half=True,
                               conf=args.threshold, iou=.5, max_det=100, verbose=False)[0]
        predictions = [{"box": box, "score": score, "class": int(cls)} for box, score, cls in
                       zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
        correct, wrong, missed = match_predictions(predictions, targets)
        row = {"validation_id": ordinal, "gallery_id": record["gallery_id"], "image_name": source.name,
               "correct": len(correct), "wrong_identity": len(wrong),
               "unmatched_predictions": len(predictions)-len(correct)-len(wrong), "unmatched_targets": len(missed)}
        for cls in NAMES:
            tp = sum(predictions[i]["class"] == cls for i in correct)
            fp = sum(p["class"] == cls for p in predictions)-tp
            fn = sum(t["class"] == cls for t in targets)-tp
            totals[cls].update(tp=tp, fp=fp, fn=fn)
        for pred_index, target_index in wrong.items():
            key = "teammate_as_enemy" if targets[target_index]["class"] == 1 else "enemy_as_teammate"
            diagnostic[key] += 1
        diagnostic["unmatched_predictions"] += row["unmatched_predictions"]
        diagnostic["unmatched_targets"] += row["unmatched_targets"]
        draw_review(source, rendered / source.name, targets, predictions, correct, wrong, row, args.threshold)
        rows.append(row)
        details.append({**row, "targets": targets, "predictions": predictions,
                        "correct_matches": correct, "wrong_identity_matches": wrong,
                        "unmatched_target_indices": missed})
    for offset in range(0, len(rows), 16):
        sheet = Image.new("RGB", (2560, 1600), "#10161e")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows[offset:offset+16]):
            with Image.open(rendered / row["image_name"]) as source:
                tile = source.copy()
                tile.thumbnail((640, 360))
            x, y = index % 4 * 640, index // 4 * 400
            sheet.paste(tile, (x, y))
            draw.text((x+5, y+361), f"{row['validation_id']:02d}（筛选{row['gallery_id']}）正确{row['correct']} 认反{row['wrong_identity']} 多框{row['unmatched_predictions']} 漏框{row['unmatched_targets']}",
                      fill="white", font=font(18))
        sheet.save(output / f"contact_sheet_{offset//16+1:02d}.jpg", quality=93)
    class_metrics = {}
    for cls, count in totals.items():
        p = count["tp"] / max(1, count["tp"]+count["fp"])
        r = count["tp"] / max(1, count["tp"]+count["fn"])
        class_metrics[NAMES[cls]] = {**count, "precision": p, "recall": r}
    summary = {"checkpoint": str(WEIGHTS), "stage": "Stage 1 — Local Combat Policy",
               "validation_images": len(records), "confidence_threshold": args.threshold, "matching_iou": .5,
               "validation_labels": str(labels_dir),
               "class_metrics": class_metrics, "identity_diagnostics": dict(diagnostic),
               "diagnostic_note": "Wrong-class matches count as FP for predicted class and FN for true class. Unmatched predictions may be background, duplicates, or localization errors; not all are background errors.",
               "validation_only": True, "test_images_used": 0, "latency": benchmark(model)}
    for name, value in (("summary.json", summary), ("prediction_details.json", details)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "image_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()
