"""Evaluate the best C1 Day1 detector on validation data only.

Produces threshold metrics and visual failure cases without touching test data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from c1_detection import C1CocoDetectionDataset, build_c1_detector


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations" / "day1_full_v1" / "normalized"
RUN = ROOT / "runs" / "c1_detection" / "baseline_day1_full_v1"
CHECKPOINT = RUN / "best.pt"
OUTPUT = RUN / "validation_failures"
IOU_THRESHOLD = 0.50
DISPLAY_THRESHOLD = 0.25
SWEEP_THRESHOLDS = [round(index * 0.05, 2) for index in range(1, 20)]


def greedy_match(
    prediction: dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
    score_threshold: float,
) -> dict:
    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    matched_ground_truth: set[int] = set()
    true_predictions: set[int] = set()
    for prediction_index, box in enumerate(boxes):
        if len(ground_truth) == 0:
            continue
        overlaps = box_iou(box.unsqueeze(0), ground_truth)[0]
        order = torch.argsort(overlaps, descending=True)
        for ground_truth_index_tensor in order:
            ground_truth_index = int(ground_truth_index_tensor)
            if float(overlaps[ground_truth_index]) < IOU_THRESHOLD:
                break
            if ground_truth_index not in matched_ground_truth:
                matched_ground_truth.add(ground_truth_index)
                true_predictions.add(prediction_index)
                break
    return {
        "boxes": boxes,
        "scores": scores,
        "true_predictions": true_predictions,
        "false_predictions": set(range(len(boxes))) - true_predictions,
        "matched_ground_truth": matched_ground_truth,
        "missed_ground_truth": set(range(len(ground_truth))) - matched_ground_truth,
    }


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: torch.Tensor,
    color: tuple[int, int, int],
    label: str,
    width: int = 4,
) -> None:
    x1, y1, x2, y2 = (float(value) for value in box)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    text_box = draw.textbbox((x1, y1), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_y = max(0, y1 - text_height - 5)
    draw.rectangle(
        (x1, text_y, x1 + text_width + 8, text_y + text_height + 5), fill=color
    )
    draw.text((x1 + 4, text_y + 2), label, fill=(0, 0, 0))


def render_failure(
    source: Path,
    destination: Path,
    ground_truth: torch.Tensor,
    match: dict,
    summary: str,
) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(ground_truth):
        if index in match["missed_ground_truth"]:
            draw_box(draw, box, (255, 215, 0), "MISSED GT", width=6)
        else:
            draw_box(draw, box, (0, 220, 80), "GT", width=4)
    for index, (box, score) in enumerate(
        zip(match["boxes"], match["scores"], strict=True)
    ):
        if index in match["true_predictions"]:
            draw_box(draw, box, (0, 220, 255), f"TP {float(score):.2f}", width=4)
        else:
            draw_box(draw, box, (255, 60, 60), f"FP {float(score):.2f}", width=6)
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
    draw.text((10, 9), summary, fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)


def make_contact_sheets(rows: list[dict], rendered_dir: Path) -> int:
    if not rows:
        return 0
    columns, rows_per_page = 3, 4
    tile_width, tile_height = 480, 300
    per_page = columns * rows_per_page
    font = ImageFont.load_default()
    pages = 0
    for page_start in range(0, len(rows), per_page):
        page_rows = rows[page_start : page_start + per_page]
        sheet = Image.new(
            "RGB", (columns * tile_width, rows_per_page * tile_height), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page_rows):
            with Image.open(rendered_dir / row["rendered_file"]) as opened:
                thumbnail = opened.convert("RGB")
                thumbnail.thumbnail((tile_width, tile_height - 30))
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            sheet.paste(thumbnail, (x, y))
            caption = (
                f"{row['file_name']} | TP {row['tp']} FP {row['fp']} FN {row['fn']}"
            )
            draw.text((x + 4, y + tile_height - 25), caption, fill="black", font=font)
        pages += 1
        sheet.save(OUTPUT / f"contact_sheet_{pages:02d}.jpg", quality=90)
    return pages


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script")
    dataset = C1CocoDetectionDataset(
        DATA / "images", DATA / "annotations" / "instances_val.json"
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cuda:0", weights_only=False)
    model = build_c1_detector(pretrained=False).to("cuda:0")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.roi_heads.score_thresh = min(SWEEP_THRESHOLDS)

    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[torch.Tensor] = []
    print(f"Evaluating {len(dataset)} validation images; test split is unused.")
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model([image.to("cuda:0")])[0]
            predictions.append(
                {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                }
            )
            targets.append(target["boxes"].cpu())

    sweep_rows = []
    for threshold in SWEEP_THRESHOLDS:
        tp = fp = fn = 0
        for prediction, ground_truth in zip(predictions, targets, strict=True):
            match = greedy_match(prediction, ground_truth, threshold)
            tp += len(match["true_predictions"])
            fp += len(match["false_predictions"])
            fn += len(match["missed_ground_truth"])
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        sweep_rows.append(
            {
                "score_threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "threshold_sweep.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    rendered_dir = OUTPUT / "rendered"
    failure_rows = []
    display_totals = {"tp": 0, "fp": 0, "fn": 0}
    for index, (prediction, ground_truth) in enumerate(
        zip(predictions, targets, strict=True)
    ):
        match = greedy_match(prediction, ground_truth, DISPLAY_THRESHOLD)
        tp = len(match["true_predictions"])
        fp = len(match["false_predictions"])
        fn = len(match["missed_ground_truth"])
        display_totals["tp"] += tp
        display_totals["fp"] += fp
        display_totals["fn"] += fn
        if fp == 0 and fn == 0:
            continue
        info = dataset.image_info(index)
        top_fp_score = max(
            (float(match["scores"][item]) for item in match["false_predictions"]),
            default=0.0,
        )
        rendered_file = info["file_name"]
        summary = (
            f"TP={tp} FP={fp} FN={fn} threshold={DISPLAY_THRESHOLD:.2f} "
            "| GT green / TP cyan / FP red / missed GT yellow"
        )
        render_failure(
            DATA / "images" / info["file_name"],
            rendered_dir / rendered_file,
            ground_truth,
            match,
            summary,
        )
        failure_rows.append(
            {
                "file_name": info["file_name"],
                "split": info["split"],
                "encounter_id": info["encounter_id"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "top_fp_score": top_fp_score,
                "rendered_file": rendered_file,
                "manual_fp_type": "",
                "manual_fn_type": "",
                "notes": "",
            }
        )
    failure_rows.sort(
        key=lambda row: (row["fp"] + row["fn"], row["top_fp_score"]), reverse=True
    )
    with (OUTPUT / "failure_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(failure_rows[0]))
        writer.writeheader()
        writer.writerows(failure_rows)

    pages = make_contact_sheets(failure_rows, rendered_dir)
    best_f1 = max(sweep_rows, key=lambda row: row["f1"])
    summary = {
        "stage": "Stage 1 — Local Combat Policy",
        "task": "C1 Enemy Detection",
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_images": len(dataset),
        "validation_ground_truth_boxes": sum(len(item) for item in targets),
        "test_images_used": 0,
        "display_threshold": DISPLAY_THRESHOLD,
        "display_metrics": display_totals,
        "failure_images": len(failure_rows),
        "contact_sheets": pages,
        "best_validation_f1_threshold": best_f1,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Failure review: {OUTPUT / 'failure_review.csv'}")
    print(f"Contact sheets: {OUTPUT}")


if __name__ == "__main__":
    main()
