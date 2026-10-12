"""Evaluate a C1 overfit checkpoint on its exact tiny training subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from c1_detection import C1CocoDetectionDataset, build_c1_detector


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "c1_annotations" / "day1_seed_v1" / "normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the tiny C1 overfit set")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DATASET_ROOT / "annotations" / "instances_train.json",
    )
    parser.add_argument("--images", type=Path, default=DATASET_ROOT / "images")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def greedy_match(
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    if prediction_boxes.numel() == 0:
        return 0, 0, len(target_boxes), []
    if target_boxes.numel() == 0:
        return 0, len(prediction_boxes), 0, []

    overlaps = box_iou(prediction_boxes, target_boxes)
    matched_targets: set[int] = set()
    true_positives = 0
    matched_ious = []
    for prediction_index in torch.argsort(prediction_scores, descending=True).tolist():
        best_iou, target_index = overlaps[prediction_index].max(dim=0)
        target_index_value = int(target_index)
        if float(best_iou) >= iou_threshold and target_index_value not in matched_targets:
            matched_targets.add(target_index_value)
            true_positives += 1
            matched_ious.append(float(best_iou))
    false_positives = len(prediction_boxes) - true_positives
    false_negatives = len(target_boxes) - true_positives
    return true_positives, false_positives, false_negatives, matched_ious


def draw_result(
    source_path: Path,
    target_boxes: torch.Tensor,
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    output_path: Path,
) -> None:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = load_font(26)
    for box in target_boxes.tolist():
        draw.rectangle(box, outline=(30, 220, 90), width=5)
        draw.text((box[0], max(0, box[1] - 30)), "GT", fill=(30, 220, 90), font=label_font)
    for box, score in zip(prediction_boxes.tolist(), prediction_scores.tolist(), strict=True):
        draw.rectangle(box, outline=(235, 57, 53), width=4)
        draw.text(
            (box[0], min(image.height - 30, box[3] + 2)),
            f"P {score:.2f}",
            fill=(235, 57, 53),
            font=label_font,
        )
    image.save(output_path, quality=94, subsampling=0)


def main() -> None:
    args = parse_args()
    run_dir = args.run.resolve()
    selected = json.loads((run_dir / "selected_samples.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    dataset = C1CocoDetectionDataset(args.images, args.annotations)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_c1_detector(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    # Retain raw low-confidence candidates here; the explicit CLI threshold
    # below is the single source of truth for evaluation filtering.
    model.roi_heads.score_thresh = 0.0
    model.to(device).eval()

    visual_dir = run_dir / "evaluation"
    visual_dir.mkdir(parents=True, exist_ok=True)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    all_matched_ious = []
    per_image = []
    with torch.inference_mode():
        for item in selected:
            index = item["dataset_index"]
            image, target = dataset[index]
            output = model([image.to(device)])[0]
            keep = output["scores"] >= args.score_threshold
            boxes = output["boxes"][keep].cpu()
            scores = output["scores"][keep].cpu()
            target_boxes = target["boxes"].cpu()
            tp, fp, fn, matched_ious = greedy_match(
                boxes, scores, target_boxes, args.iou_threshold
            )
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            all_matched_ious.extend(matched_ious)
            per_image.append(
                {
                    "file_name": item["file_name"],
                    "ground_truth_boxes": len(target_boxes),
                    "predicted_boxes": len(boxes),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "matched_ious": matched_ious,
                    "prediction_scores": scores.tolist(),
                }
            )
            draw_result(
                args.images / item["file_name"],
                target_boxes,
                boxes,
                scores,
                visual_dir / item["file_name"],
            )

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    mean_iou = sum(all_matched_ious) / max(1, len(all_matched_ious))
    metrics = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_loss": checkpoint["mean_loss"],
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        **totals,
        "precision": precision,
        "recall": recall,
        "mean_matched_iou": mean_iou,
        "passed": precision >= 0.80 and recall >= 0.80,
        "per_image": per_image,
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_image"}, indent=2))
    print(f"Visualizations: {visual_dir}")


if __name__ == "__main__":
    main()
