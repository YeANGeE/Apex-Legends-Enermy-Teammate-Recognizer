"""Stage 1 — Local Combat Policy: bounded unlabelled video-frame diagnostics.

No AP/recall/accuracy claims without human ground truth. No training or services.
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from c1_detection.two_class_dataset import digest
from compare_c1_day3_fasterrcnn_yolo import DEFAULT_FRCNN, DISPLAY, Predictor, YOLO_WEIGHTS

NAMES = {1: "敌人", 2: "队友"}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def detection_flags(predictions, size):
    boxes = torch.tensor(predictions["boxes"]).reshape(-1, 4)
    labels = predictions["labels"]
    overlaps = box_iou(boxes, boxes)
    # Signals for manual review, not certified duplicate errors.
    pairs = [(i, j) for i in range(len(boxes)) for j in range(i+1, len(boxes)) if overlaps[i, j] >= .5]
    return {"overlapping_pairs": len(pairs),
            "cross_class_overlap_pairs": sum(labels[i] != labels[j] for i, j in pairs),
            "large_box_predictions": sum((b[3]-b[1])/size[1] >= .35 for b in predictions["boxes"]),
            "small_box_predictions": sum((b[3]-b[1])/size[1] < .1 for b in predictions["boxes"])}


def disagreement(a, b):
    aa = torch.tensor(a["boxes"]).reshape(-1, 4)
    bb = torch.tensor(b["boxes"]).reshape(-1, 4)
    overlap = box_iou(aa, bb)
    remaining = set(range(len(bb)))
    matched, identity_conflicts = 0, 0
    for i in sorted(range(len(aa)), key=lambda i: a["scores"][i], reverse=True):
        candidates = [j for j in remaining if overlap[i, j] >= .3]
        if candidates:
            j = max(candidates, key=lambda j: float(overlap[i, j]))
            remaining.remove(j)
            matched += 1
            identity_conflicts += a["labels"][i] != b["labels"][j]
    return {"a_unmatched": len(aa)-matched, "b_unmatched": len(bb)-matched,
            "matched_identity_conflicts": identity_conflicts, "matching_iou": .3,
            "note": "Model disagreement only; not FP/FN or ground truth."}


def draw_predictions(source, pred, title):
    image = source.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    for box, score, cls in zip(pred["boxes"], pred["scores"], pred["labels"], strict=True):
        color = "#ff6464" if cls == 1 else "#4bbfff"
        draw.rectangle(box, outline=color, width=4)
        x, y = box[0], max(44, box[1]-26)
        draw.text((x, y), f"{NAMES[cls]} {score:.2f}", font=font, fill=color, stroke_width=2, stroke_fill="black")
    draw.rectangle((0, 0, image.width, 42), fill="black")
    draw.text((10, 4), title, font=font, fill="white")
    return image


def render(frames, all_predictions, output):
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 21)
    groups = {}
    for rec in frames:
        groups.setdefault(rec["clip_id"], []).append(rec)
    keys = list(all_predictions)
    index_by_id = {rec["id"]: i for i, rec in enumerate(frames)}
    for clip_id, records in groups.items():
        folder = output / clip_id
        folder.mkdir()
        sheet = None
        for sequence, record in enumerate(records):
            i = index_by_id[record["id"]]
            with Image.open(record["image_path"]) as im:
                source = im.convert("RGB")
            strip = Image.new("RGB", (source.width*3, source.height+48), "#10151c")
            row_thumb = Image.new("RGB", (1920, 410), "#10151c")
            for column, key in enumerate(keys):
                pred = all_predictions[key][i]["prediction"]
                image = draw_predictions(source, pred, f"{record['id']} {record['elapsed_s']:.3f}s · {DISPLAY[key]}")
                strip.paste(image, (column*source.width, 0))
                image.thumbnail((640, 363))
                row_thumb.paste(image, (column*640, 0))
                text = f"敌人{pred['labels'].count(1)} 队友{pred['labels'].count(2)}  conf≥0.05"
                ImageDraw.Draw(strip).text((column*source.width+10, source.height+7), text, font=font, fill="white")
                ImageDraw.Draw(row_thumb).text((column*640+10, 371), text, font=font, fill="white")
            strip.save(folder / f"{record['id']}_compare.jpg", quality=96, subsampling=0)
            if sequence % 4 == 0:
                sheet = Image.new("RGB", (1920, 1640), "#10151c")
            sheet.paste(row_thumb, (0, (sequence % 4)*410))
            if sequence % 4 == 3 or sequence == len(records)-1:
                sheet.save(folder / f"overview_{sequence//4+1:02d}.jpg", quality=94, subsampling=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    if not 1 <= len(frames) <= 300 or manifest["action_labels_available"]:
        raise ValueError("Expected bounded, unlabelled third-party diagnostic frames")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = args.output or args.manifest.parent / "three_model_diagnosis"
    output.mkdir(parents=True, exist_ok=False)
    for record in frames:
        if digest(Path(record["image_path"])) != record["image_sha256"]:
            raise ValueError("Extracted frame changed")
    weights = {"yolo_day1_v3": YOLO_WEIGHTS["yolo_day1_v3"],
               "yolo_day2_v4": YOLO_WEIGHTS["yolo_day2_v4"], "fasterrcnn_day3": DEFAULT_FRCNN}
    save(output / "run_config.json", {"stage": "Stage 1 — Local Combat Policy",
         "manifest": str(args.manifest.resolve()), "manifest_sha256": digest(args.manifest),
         "weights": {k: {"path": str(v), "sha256": digest(v)} for k, v in weights.items()},
         "confidence": .05, "source_timestamps": "media PTS, not native QPC/action timestamps",
         "ground_truth_available": False, "training_started": False, "latency_benchmark": False})
    all_predictions, summary = {}, {}
    for key, weight in weights.items():
        predictor = Predictor(key, weight)
        predictions, per_clip = [], {}
        print(f"诊断 {DISPLAY[key]}：{len(frames)}张", flush=True)
        for i, record in enumerate(frames, 1):
            with Image.open(record["image_path"]) as im:
                rgb = np.array(im.convert("RGB"))
            pred = predictor(rgb, .05)
            pred = {k: v.tolist() for k, v in pred.items()}
            flags = detection_flags(pred, (record["width"], record["height"]))
            predictions.append({"id": record["id"], "clip_id": record["clip_id"],
                                "elapsed_s": record["elapsed_s"], "media_pts_s": record["media_pts_s"],
                                "prediction": pred, "diagnostic_signals": flags})
            counts = per_clip.setdefault(record["clip_id"], Counter())
            counts["sampled_frames"] += 1
            counts["frames_with_predictions"] += bool(pred["boxes"])
            counts["enemy_predictions"] += pred["labels"].count(1)
            counts["teammate_predictions"] += pred["labels"].count(2)
            for flag, value in flags.items():
                counts[flag] += value
            for cls, name in ((1, "enemy"), (2, "teammate")):
                counts[f"{name}_predictions_ge025"] += sum(c == cls and s >= .25 for c, s in zip(pred["labels"], pred["scores"], strict=True))
            if i % 20 == 0 or i == len(frames):
                print(f"  已完成{i}/{len(frames)}", flush=True)
        all_predictions[key] = predictions
        summary[key] = {c: dict(v) for c, v in per_clip.items()}
        save(output / f"{key}_predictions.json", predictions)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    signals = []
    for i, record in enumerate(frames):
        preds = {k: all_predictions[k][i]["prediction"] for k in weights}
        signals.append({"id": record["id"], "clip_id": record["clip_id"], "elapsed_s": record["elapsed_s"],
            "v3_vs_v4": disagreement(preds["yolo_day1_v3"], preds["yolo_day2_v4"]),
            "v3_vs_fasterrcnn": disagreement(preds["yolo_day1_v3"], preds["fasterrcnn_day3"])})
    save(output / "disagreement_signals.json", signals)
    save(output / "summary.json", {"models": summary, "sampled_frames": len(frames),
         "ground_truth_available": False, "metric_note": "Prediction counts/signals are NOT precision, recall, AP or ID continuity."})
    print("生成高清三模型对照和总览……", flush=True)
    render(frames, all_predictions, output)
    print(f"诊断产物：{output}", flush=True)


if __name__ == "__main__":
    main()
