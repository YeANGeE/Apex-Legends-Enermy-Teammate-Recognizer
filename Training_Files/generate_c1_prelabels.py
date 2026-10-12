"""Generate Label Studio preannotations for remaining Day1 train/val images."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from c1_detection import build_c1_detector


ROOT = Path(__file__).resolve().parent
FORMAL_MANIFEST = ROOT / "c1_day1_formal_manifest.csv"
SEED_MANIFEST = ROOT / "c1_day1_seed_manifest.csv"
CHECKPOINT = ROOT / "runs" / "c1_detection" / "baseline_seed_v1" / "best.pt"
OUTPUT_ROOT = ROOT / "c1_prelabels" / "day1_remaining_v1"
TASKS_JSON = OUTPUT_ROOT / "label_studio_tasks.json"
MANIFEST_CSV = OUTPUT_ROOT / "prelabel_manifest.csv"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
MODEL_VERSION = "c1_baseline_seed_v1_epoch12"
SCORE_THRESHOLD = 0.15


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def prediction_id(image_name: str, index: int) -> str:
    return hashlib.sha1(f"{image_name}:{index}".encode()).hexdigest()[:10]


def main() -> None:
    formal = load_csv(FORMAL_MANIFEST)
    seed_names = {row["image_name"] for row in load_csv(SEED_MANIFEST)}
    remaining = [
        row for row in formal
        if row["image_name"] not in seed_names and row["split"] in {"train", "val"}
    ]
    remaining.sort(key=lambda row: (row["split"], row["encounter_id"], int(row["frame_timestamp_ns"])))
    if len(remaining) != 422:
        raise ValueError(f"Expected 422 remaining train/val images, found {len(remaining)}")
    if any(row["split"] == "test" for row in remaining):
        raise RuntimeError("Test image entered prelabel inference")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = build_c1_detector(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.roi_heads.score_thresh = SCORE_THRESHOLD
    model.to(device).eval()

    tasks, output_rows = [], []
    with torch.inference_mode():
        for number, row in enumerate(remaining, start=1):
            image_path = Path(row["image_path"])
            with Image.open(image_path) as source:
                rgb = source.convert("RGB")
                width, height = rgb.size
                tensor = pil_to_tensor(rgb).float().div_(255).to(device)
            output = model([tensor])[0]
            keep = output["scores"] >= SCORE_THRESHOLD
            boxes = output["boxes"][keep].cpu().tolist()
            scores = output["scores"][keep].cpu().tolist()
            results = []
            for index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                x1, y1, x2, y2 = box
                results.append({
                    "id": prediction_id(row["image_name"], index),
                    "from_name": "label", "to_name": "image",
                    "type": "rectanglelabels", "score": score,
                    "original_width": width, "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": 100 * x1 / width, "y": 100 * y1 / height,
                        "width": 100 * (x2 - x1) / width,
                        "height": 100 * (y2 - y1) / height,
                        "rotation": 0, "rectanglelabels": ["enemy"],
                    },
                })
            relative = image_path.resolve().relative_to(ROOT.resolve()).as_posix()
            tasks.append({
                "data": {"image": f"/data/local-files/?d={quote(relative)}"},
                "meta": {"split": row["split"], "encounter_id": row["encounter_id"],
                         "image_name": row["image_name"]},
                "predictions": [{
                    "model_version": MODEL_VERSION,
                    "score": max(scores, default=0.0), "result": results,
                }],
            })
            output_rows.append({
                "image_name": row["image_name"], "split": row["split"],
                "encounter_id": row["encounter_id"],
                "predicted_boxes": len(results),
                "max_score": max(scores, default=0.0),
            })
            if number % 50 == 0 or number == len(remaining):
                print(f"Prelabelled: {number}/{len(remaining)}", flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TASKS_JSON.write_text(json.dumps(tasks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0]); writer.writeheader(); writer.writerows(output_rows)
    summary = {
        "model_version": MODEL_VERSION, "checkpoint": str(CHECKPOINT),
        "score_threshold": SCORE_THRESHOLD, "tasks": len(tasks),
        "train_tasks": sum(row["split"] == "train" for row in output_rows),
        "val_tasks": sum(row["split"] == "val" for row in output_rows),
        "test_tasks": 0,
        "tasks_with_predictions": sum(row["predicted_boxes"] > 0 for row in output_rows),
        "predicted_boxes": sum(row["predicted_boxes"] for row in output_rows),
        "tasks_without_predictions": sum(row["predicted_boxes"] == 0 for row in output_rows),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Import into Label Studio: {TASKS_JSON}")


if __name__ == "__main__":
    main()
