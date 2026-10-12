"""Build the complete reviewed Day1 C1 train/val dataset.

This combines the 120-image manual seed with the 422-image model-assisted
review export.  The 58-image test split remains untouched and is never added
to a training annotation file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

from PIL import Image


ROOT = Path(__file__).resolve().parent
FORMAL_MANIFEST = ROOT / "c1_day1_formal_manifest.csv"
SEED_COCO = (
    ROOT
    / "c1_annotations"
    / "day1_seed_v1"
    / "normalized"
    / "annotations"
    / "instances_all.json"
)
REVIEW_EXPORT_DIR = ROOT / "C1_422IMG_EXPORT"
OUTPUT_ROOT = ROOT / "c1_annotations" / "day1_full_v1" / "normalized"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_ANNOTATIONS = OUTPUT_ROOT / "annotations"
RAW_OUTPUT = ROOT / "c1_annotations" / "day1_full_v1" / "raw_exports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_export(suffix: str) -> Path:
    matches = sorted(REVIEW_EXPORT_DIR.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {suffix} export in {REVIEW_EXPORT_DIR}, "
            f"found {len(matches)}"
        )
    return matches[0]


def task_image_name(task: dict) -> str:
    image_url = task.get("data", {}).get("image")
    if not image_url:
        raise ValueError(f"Task {task.get('id')} has no data.image")
    parsed = urlparse(image_url)
    query_path = parse_qs(parsed.query).get("d", [None])[0]
    path = query_path if query_path is not None else parsed.path
    return PurePosixPath(path.replace("\\", "/")).name


def active_annotation(task: dict) -> dict:
    active = [
        item
        for item in task.get("annotations", [])
        if not item.get("was_cancelled", False)
    ]
    if len(active) != 1:
        raise ValueError(
            f"Task {task.get('id')} has {len(active)} active annotations; expected 1"
        )
    return active[0]


def read_manifest() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    with FORMAL_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 600:
        raise ValueError(f"Expected 600 formal images, found {len(rows)}")
    by_name = {row["image_name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("Duplicate image names in the formal manifest")
    return by_name, rows


def read_seed() -> dict[str, dict]:
    dataset = json.loads(SEED_COCO.read_text(encoding="utf-8-sig"))
    annotations = defaultdict(list)
    for item in dataset["annotations"]:
        annotations[item["image_id"]].append(item)
    result = {}
    for image in dataset["images"]:
        name = image["file_name"]
        if name in result:
            raise ValueError(f"Duplicate seed image: {name}")
        result[name] = {
            "quality": image["frame_quality"],
            "boxes": [item["bbox"] for item in annotations[image["id"]]],
            "origins": ["manual_seed"] * len(annotations[image["id"]]),
            "task_id": image.get("label_studio_task_id"),
            "review_source": "manual_seed",
        }
    if len(result) != 120:
        raise ValueError(f"Expected 120 seed annotations, found {len(result)}")
    return result


def rectangle_to_bbox(result: dict, width: int, height: int, task_id: int) -> list[float]:
    value = result.get("value", {})
    if value.get("rectanglelabels") != ["enemy"]:
        raise ValueError(
            f"Task {task_id} has unexpected labels: {value.get('rectanglelabels')}"
        )
    x_percent = float(value["x"])
    y_percent = float(value["y"])
    width_percent = float(value["width"])
    height_percent = float(value["height"])
    rotation = float(value.get("rotation", 0))
    if rotation:
        # Label Studio rotates a rectangle clockwise around its top-left
        # corner. Detection targets must be axis aligned, so calculate the
        # smallest horizontal rectangle containing all four rotated corners.
        angle = math.radians(rotation)
        cosine, sine = math.cos(angle), math.sin(angle)
        corners = (
            (0.0, 0.0),
            (width_percent * cosine, width_percent * sine),
            (-height_percent * sine, height_percent * cosine),
            (
                width_percent * cosine - height_percent * sine,
                width_percent * sine + height_percent * cosine,
            ),
        )
        x_values = [x_percent + corner_x for corner_x, _ in corners]
        y_values = [y_percent + corner_y for _, corner_y in corners]
        x_percent = min(x_values)
        y_percent = min(y_values)
        width_percent = max(x_values) - x_percent
        height_percent = max(y_values) - y_percent
    x = x_percent * width / 100.0
    y = y_percent * height / 100.0
    box_width = width_percent * width / 100.0
    box_height = height_percent * height / 100.0
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid box in task {task_id}: {value}")
    if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
        raise ValueError(f"Out-of-bounds box in task {task_id}: {value}")
    return [x, y, box_width, box_height]


def read_reviewed(path: Path) -> dict[str, dict]:
    tasks = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(tasks) != 422:
        raise ValueError(f"Expected 422 reviewed tasks, found {len(tasks)}")
    result = {}
    for task in tasks:
        name = task_image_name(task)
        if name in result:
            raise ValueError(f"Duplicate reviewed image: {name}")
        annotation = active_annotation(task)
        qualities = [
            choice
            for item in annotation.get("result", [])
            if item.get("type") == "choices"
            and item.get("from_name") == "frame_quality"
            for choice in item.get("value", {}).get("choices", [])
        ]
        if len(qualities) != 1 or qualities[0] not in {"usable", "uncertain"}:
            raise ValueError(f"Task {task.get('id')} has invalid quality: {qualities}")
        rectangles = [
            item
            for item in annotation.get("result", [])
            if item.get("type") == "rectanglelabels"
        ]
        result[name] = {
            "quality": qualities[0],
            "rectangles": rectangles,
            "task_id": task.get("id"),
            "review_source": "model_assisted_manual_review",
        }
    return result


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing normalized image differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def subset(dataset: dict, split: str | None = None, usable_only: bool = False) -> dict:
    images = [
        image
        for image in dataset["images"]
        if (split is None or image["split"] == split)
        and (not usable_only or image["frame_quality"] == "usable")
    ]
    ids = {image["id"] for image in images}
    annotations = [
        item for item in dataset["annotations"] if item["image_id"] in ids
    ]
    return {
        **dataset,
        "images": images,
        "annotations": annotations,
        "info": {
            **dataset["info"],
            "description": dataset["info"]["description"]
            + (f" — {split}" if split else "")
            + (" — usable only" if usable_only else ""),
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    raw_json = one_export(".json")
    raw_zip = one_export(".zip")
    manifest, manifest_rows = read_manifest()
    seed = read_seed()
    reviewed = read_reviewed(raw_json)

    overlap = sorted(set(seed) & set(reviewed))
    if overlap:
        raise ValueError(f"Seed/review overlap: {overlap[:10]}")
    annotated_names = set(seed) | set(reviewed)
    expected_names = {
        row["image_name"] for row in manifest_rows if row["split"] in {"train", "val"}
    }
    if annotated_names != expected_names:
        missing = sorted(expected_names - annotated_names)
        extra = sorted(annotated_names - expected_names)
        raise ValueError(f"Train/val coverage mismatch; missing={missing}, extra={extra}")

    test_names = {
        row["image_name"] for row in manifest_rows if row["split"] == "test"
    }
    if annotated_names & test_names:
        raise ValueError("Test images appeared in reviewed train/val annotations")

    encounter_splits = defaultdict(set)
    for row in manifest_rows:
        encounter_splits[row["encounter_id"]].add(row["split"])
    leakage = {
        encounter: sorted(splits)
        for encounter, splits in encounter_splits.items()
        if len(splits) != 1
    }
    if leakage:
        raise ValueError(f"Encounter split leakage: {leakage}")

    images = []
    annotations = []
    quality_rows = []
    origin_counts = Counter()
    annotation_id = 1
    for image_id, name in enumerate(sorted(annotated_names), start=1):
        row = manifest[name]
        source = Path(row["image_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing formal image: {source}")
        with Image.open(source) as image:
            width, height = image.size
        if (width, height) != (1920, 1080):
            raise ValueError(f"Unexpected dimensions for {name}: {width}x{height}")
        link_or_copy(source, OUTPUT_IMAGES / name)

        record = seed.get(name) or reviewed[name]
        if name in seed:
            boxes = record["boxes"]
            origins = record["origins"]
        else:
            boxes = [
                rectangle_to_bbox(item, width, height, record["task_id"])
                for item in record["rectangles"]
            ]
            origins = [item.get("origin", "unknown") for item in record["rectangles"]]

        images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": width,
                "height": height,
                "frame_quality": record["quality"],
                "split": row["split"],
                "encounter_id": row["encounter_id"],
                "clip_id": row["clip_id"],
                "frame_id": int(row["frame_id"]),
                "frame_timestamp_ns": int(row["frame_timestamp_ns"]),
                "review_source": record["review_source"],
                "label_studio_task_id": record["task_id"],
            }
        )
        for box, origin in zip(boxes, origins, strict=True):
            x, y, box_width, box_height = (float(value) for value in box)
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid seed box for {name}: {box}")
            if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
                raise ValueError(f"Out-of-bounds seed box for {name}: {box}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "review_origin": origin,
                }
            )
            annotation_id += 1
            origin_counts[origin] += 1
        quality_rows.append(
            {
                "image_id": image_id,
                "file_name": name,
                "split": row["split"],
                "encounter_id": row["encounter_id"],
                "frame_quality": record["quality"],
                "enemy_box_count": len(boxes),
                "review_source": record["review_source"],
                "label_studio_task_id": record["task_id"],
            }
        )

    dataset = {
        "info": {
            "description": "FPS C1 Enemy Detection Day1 fully reviewed train/val v1",
            "version": "1.0",
            "stage": "Stage 1 — Local Combat Policy",
            "task": "C1 Enemy Detection",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "enemy", "supercategory": "enemy"}],
    }
    datasets = {
        "instances_all.json": dataset,
        "instances_usable.json": subset(dataset, usable_only=True),
        "instances_train.json": subset(dataset, split="train", usable_only=True),
        "instances_val.json": subset(dataset, split="val", usable_only=True),
    }
    for name, value in datasets.items():
        write_json(OUTPUT_ANNOTATIONS / name, value)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "frame_quality.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)

    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)
    for source in (raw_json, raw_zip):
        link_or_copy(source, RAW_OUTPUT / source.name)

    usable = datasets["instances_usable.json"]
    train = datasets["instances_train.json"]
    val = datasets["instances_val.json"]
    usable_ids = {image["id"] for image in usable["images"]}
    summary = {
        "dataset": "C1 Enemy Detection Day1 fully reviewed train/val v1",
        "stage": "Stage 1 — Local Combat Policy",
        "task": "C1 Enemy Detection",
        "raw_json": str(raw_json),
        "raw_json_sha256": sha256(raw_json),
        "raw_zip": str(raw_zip),
        "raw_zip_sha256": sha256(raw_zip),
        "seed_images": len(seed),
        "model_assisted_review_images": len(reviewed),
        "all_train_val_images": len(images),
        "usable_images": len(usable["images"]),
        "uncertain_images": len(images) - len(usable["images"]),
        "usable_enemy_boxes": len(usable["annotations"]),
        "usable_zero_box_images": sum(
            row["frame_quality"] == "usable" and int(row["enemy_box_count"]) == 0
            for row in quality_rows
        ),
        "train_images": len(train["images"]),
        "train_enemy_boxes": len(train["annotations"]),
        "val_images": len(val["images"]),
        "val_enemy_boxes": len(val["annotations"]),
        "test_images_untouched": len(test_names),
        "encounter_split_leakage": False,
        "box_review_origins": dict(sorted(origin_counts.items())),
        "usable_annotation_ids_valid": all(
            item["image_id"] in usable_ids for item in usable["annotations"]
        ),
    }
    write_json(OUTPUT_ROOT / "dataset_summary.json", summary)
    print(f"Output: {OUTPUT_ROOT}")
    for key, value in summary.items():
        if not key.endswith("sha256") and key not in {"raw_json", "raw_zip"}:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
