"""Convert the frozen Label Studio JSON export into clean single-class COCO."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_EXPORT_DIR = (
    PROJECT_ROOT / "c1_annotations" / "day1_calibration_v1" / "raw_exports"
)
RAW_JSON = RAW_EXPORT_DIR / "day1_calibration_v1_label_studio.json"
SOURCE_IMAGES = PROJECT_ROOT / "c1_calibration_images"
OUTPUT_ROOT = PROJECT_ROOT / "c1_annotations" / "day1_calibration_v1" / "normalized"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_ANNOTATIONS = OUTPUT_ROOT / "annotations"
QUALITY_OVERRIDES = (
    PROJECT_ROOT
    / "c1_annotations"
    / "day1_calibration_v1"
    / "quality_overrides.csv"
)

UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_image_name(task: dict) -> str:
    uploaded_name = Path(task["data"]["image"].replace("\\", "/")).name
    return UPLOAD_PREFIX.sub("", uploaded_name, count=1)


def active_annotation(task: dict) -> dict:
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]
    if len(annotations) != 1:
        raise ValueError(
            f"Task {task.get('id')} has {len(annotations)} active annotations; expected 1"
        )
    return annotations[0]


def load_quality_overrides() -> dict[str, str]:
    if not QUALITY_OVERRIDES.is_file():
        return {}

    overrides = {}
    with QUALITY_OVERRIDES.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            image_name = row["file_name"].strip()
            quality = row["frame_quality"].strip()
            if quality not in {"usable", "uncertain"}:
                raise ValueError(
                    f"Invalid quality override for {image_name}: {quality}"
                )
            if image_name in overrides:
                raise ValueError(f"Duplicate quality override: {image_name}")
            overrides[image_name] = quality
    return overrides


def build_dataset(
    tasks: list[dict], quality_overrides: dict[str, str]
) -> tuple[dict, list[dict]]:
    images = []
    annotations = []
    quality_rows = []
    annotation_id = 1

    sorted_tasks = sorted(tasks, key=original_image_name)
    for image_id, task in enumerate(sorted_tasks, start=1):
        image_name = original_image_name(task)
        source_image = SOURCE_IMAGES / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing source calibration image: {source_image}")

        with Image.open(source_image) as image:
            width, height = image.size
        if (width, height) != (1920, 1080):
            raise ValueError(f"Unexpected dimensions for {image_name}: {width}x{height}")

        annotation = active_annotation(task)
        quality_values = []
        rectangle_results = []
        for result in annotation.get("result", []):
            if result.get("type") == "choices" and result.get("from_name") == "frame_quality":
                quality_values.extend(result.get("value", {}).get("choices", []))
            elif result.get("type") == "rectanglelabels":
                rectangle_results.append(result)

        if len(quality_values) != 1 or quality_values[0] not in {"usable", "uncertain"}:
            raise ValueError(
                f"Task {task.get('id')} has invalid frame quality: {quality_values}"
            )
        quality = quality_overrides.get(image_name, quality_values[0])

        destination_image = OUTPUT_IMAGES / image_name
        shutil.copy2(source_image, destination_image)
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "frame_quality": quality,
                "label_studio_task_id": task.get("id"),
            }
        )

        box_count = 0
        for result in rectangle_results:
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if labels != ["enemy"]:
                raise ValueError(
                    f"Task {task.get('id')} contains unexpected rectangle labels: {labels}"
                )
            if float(value.get("rotation", 0)) != 0:
                raise ValueError(f"Rotated box found in task {task.get('id')}")

            x = float(value["x"]) * width / 100.0
            y = float(value["y"]) * height / 100.0
            box_width = float(value["width"]) * width / 100.0
            box_height = float(value["height"]) * height / 100.0

            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid box in task {task.get('id')}: {value}")
            if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
                raise ValueError(f"Out-of-bounds box in task {task.get('id')}: {value}")

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            box_count += 1

        quality_rows.append(
            {
                "image_id": image_id,
                "file_name": image_name,
                "frame_quality": quality,
                "enemy_box_count": box_count,
                "label_studio_task_id": task.get("id"),
            }
        )

    dataset = {
        "info": {
            "description": "FPS C1 Enemy Detection Day1 calibration v1",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "enemy", "supercategory": "enemy"}],
    }
    known_images = {row["file_name"] for row in quality_rows}
    unknown_overrides = sorted(set(quality_overrides) - known_images)
    if unknown_overrides:
        raise ValueError(f"Quality overrides reference unknown images: {unknown_overrides}")

    return dataset, quality_rows


def subset_usable(dataset: dict) -> dict:
    usable_images = [
        image for image in dataset["images"] if image["frame_quality"] == "usable"
    ]
    usable_ids = {image["id"] for image in usable_images}
    usable_annotations = [
        annotation
        for annotation in dataset["annotations"]
        if annotation["image_id"] in usable_ids
    ]
    return {
        **dataset,
        "info": {**dataset["info"], "description": dataset["info"]["description"] + " — usable only"},
        "images": usable_images,
        "annotations": usable_annotations,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    tasks = json.loads(RAW_JSON.read_text(encoding="utf-8-sig"))
    if len(tasks) != 30:
        raise ValueError(f"Expected 30 calibration tasks, found {len(tasks)}")

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANNOTATIONS.mkdir(parents=True, exist_ok=True)

    quality_overrides = load_quality_overrides()
    dataset, quality_rows = build_dataset(tasks, quality_overrides)
    usable_dataset = subset_usable(dataset)

    all_json = OUTPUT_ANNOTATIONS / "instances_all.json"
    usable_json = OUTPUT_ANNOTATIONS / "instances_usable.json"
    quality_csv = OUTPUT_ROOT / "frame_quality.csv"
    summary_json = OUTPUT_ROOT / "dataset_summary.json"

    write_json(all_json, dataset)
    write_json(usable_json, usable_dataset)

    with quality_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=quality_rows[0])
        writer.writeheader()
        writer.writerows(quality_rows)

    summary = {
        "raw_label_studio_json": str(RAW_JSON),
        "raw_label_studio_json_sha256": sha256(RAW_JSON),
        "all_images": len(dataset["images"]),
        "all_enemy_boxes": len(dataset["annotations"]),
        "usable_images": len(usable_dataset["images"]),
        "usable_enemy_boxes": len(usable_dataset["annotations"]),
        "uncertain_images": len(dataset["images"]) - len(usable_dataset["images"]),
        "quality_overrides": len(quality_overrides),
        "categories": dataset["categories"],
    }
    write_json(summary_json, summary)

    print("Output:", OUTPUT_ROOT)
    for key, value in summary.items():
        if not key.endswith("sha256") and key != "raw_label_studio_json":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
