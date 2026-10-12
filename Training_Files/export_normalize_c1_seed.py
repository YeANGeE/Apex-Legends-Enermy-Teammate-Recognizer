"""Export and normalize the completed C1 Day1 Seed v1 Label Studio project."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE = PROJECT_ROOT / ".label-studio" / "label_studio.sqlite3"
SEED_MANIFEST = PROJECT_ROOT / "c1_day1_seed_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "c1_annotations" / "day1_seed_v1"
RAW_EXPORT_DIR = OUTPUT_ROOT / "raw_exports"
RAW_JSON = RAW_EXPORT_DIR / "day1_seed_v1_label_studio.json"
NORMALIZED_ROOT = OUTPUT_ROOT / "normalized"
NORMALIZED_IMAGES = NORMALIZED_ROOT / "images"
NORMALIZED_ANNOTATIONS = NORMALIZED_ROOT / "annotations"
PROJECT_TITLE = "C1 Day1 Seed v1"
UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_image_name(data: dict) -> str:
    if not data:
        raise ValueError("Label Studio task has empty data")
    uploaded = Path(next(iter(data.values())).replace("\\", "/")).name
    return UPLOAD_PREFIX.sub("", uploaded, count=1)


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


def read_seed_manifest() -> dict[str, dict[str, str]]:
    with SEED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 120:
        raise ValueError(f"Expected 120 seed rows, found {len(rows)}")
    by_name = {row["image_name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("Duplicate image names in seed manifest")
    return by_name


def export_from_database() -> list[dict]:
    connection = sqlite3.connect(
        f"file:{DATABASE.resolve().as_posix()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    projects = connection.execute(
        "select id from project where title = ? and deleted_at is null",
        (PROJECT_TITLE,),
    ).fetchall()
    if len(projects) != 1:
        raise ValueError(f"Expected one active project named {PROJECT_TITLE!r}")
    project_id = projects[0]["id"]

    task_rows = connection.execute(
        "select * from task where project_id = ? order by inner_id", (project_id,)
    ).fetchall()
    if len(task_rows) != 120:
        raise ValueError(f"Expected 120 Label Studio tasks, found {len(task_rows)}")

    completions_by_task = defaultdict(list)
    for completion in connection.execute(
        "select * from task_completion where project_id = ? order by id",
        (project_id,),
    ):
        completions_by_task[completion["task_id"]].append(completion)

    tasks = []
    for task in task_rows:
        active = [
            completion
            for completion in completions_by_task[task["id"]]
            if not completion["was_cancelled"]
        ]
        if len(active) != 1:
            raise ValueError(
                f"Task {task['id']} has {len(active)} active annotations; expected 1"
            )
        completion = active[0]
        original_data = json.loads(task["data"])
        image_url = next(iter(original_data.values()))
        tasks.append(
            {
                "id": task["id"],
                "inner_id": task["inner_id"],
                "data": {"image": image_url},
                "annotations": [
                    {
                        "id": completion["id"],
                        "result": json.loads(completion["result"]),
                        "was_cancelled": bool(completion["was_cancelled"]),
                        "lead_time": completion["lead_time"],
                        "result_count": completion["result_count"],
                        "unique_id": completion["unique_id"],
                    }
                ],
                "is_labeled": bool(task["is_labeled"]),
                "total_annotations": task["total_annotations"],
                "cancelled_annotations": task["cancelled_annotations"],
            }
        )
    return tasks


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_coco(tasks: list[dict], seed_by_name: dict[str, dict[str, str]]) -> tuple[dict, list[dict]]:
    images = []
    annotations = []
    quality_rows = []
    annotation_id = 1

    sorted_tasks = sorted(tasks, key=lambda task: task_image_name(task["data"]))
    seen_names = set()
    for image_id, task in enumerate(sorted_tasks, start=1):
        image_name = task_image_name(task["data"])
        if image_name in seen_names:
            raise ValueError(f"Duplicate Label Studio image: {image_name}")
        seen_names.add(image_name)
        seed = seed_by_name.get(image_name)
        if seed is None:
            raise ValueError(f"Task image is absent from seed manifest: {image_name}")
        source_image = Path(seed["image_path"])
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing seed image: {source_image}")
        with Image.open(source_image) as image:
            width, height = image.size
        if (width, height) != (1920, 1080):
            raise ValueError(f"Unexpected dimensions for {image_name}: {width}x{height}")
        link_or_copy(source_image, NORMALIZED_IMAGES / image_name)

        result_items = task["annotations"][0]["result"]
        qualities = [
            value
            for result in result_items
            if result.get("type") == "choices"
            and result.get("from_name") == "frame_quality"
            for value in result.get("value", {}).get("choices", [])
        ]
        if len(qualities) != 1 or qualities[0] not in {"usable", "uncertain"}:
            raise ValueError(f"Task {task['id']} has invalid quality: {qualities}")
        quality = qualities[0]

        image_entry = {
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
            "frame_quality": quality,
            "split": seed["split"],
            "encounter_id": seed["encounter_id"],
            "clip_id": seed["clip_id"],
            "frame_id": int(seed["frame_id"]),
            "frame_timestamp_ns": int(seed["frame_timestamp_ns"]),
            "label_studio_task_id": task["id"],
            "label_studio_inner_id": task["inner_id"],
        }
        images.append(image_entry)

        box_count = 0
        for result in result_items:
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value", {})
            if value.get("rectanglelabels") != ["enemy"]:
                raise ValueError(
                    f"Task {task['id']} has unexpected labels: "
                    f"{value.get('rectanglelabels')}"
                )
            if float(value.get("rotation", 0)) != 0:
                raise ValueError(f"Rotated box in task {task['id']}")
            x = float(value["x"]) * width / 100.0
            y = float(value["y"]) * height / 100.0
            box_width = float(value["width"]) * width / 100.0
            box_height = float(value["height"]) * height / 100.0
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid box in task {task['id']}")
            if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
                raise ValueError(f"Out-of-bounds box in task {task['id']}")
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
                "split": seed["split"],
                "encounter_id": seed["encounter_id"],
                "frame_quality": quality,
                "enemy_box_count": box_count,
                "label_studio_task_id": task["id"],
                "label_studio_inner_id": task["inner_id"],
            }
        )

    missing_tasks = sorted(set(seed_by_name) - seen_names)
    if missing_tasks:
        raise ValueError(f"Seed images absent from Label Studio: {missing_tasks}")

    dataset = {
        "info": {
            "description": "FPS C1 Enemy Detection Day1 manual seed v1",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "enemy", "supercategory": "enemy"}],
    }
    return dataset, quality_rows


def subset(dataset: dict, split: str | None = None, usable_only: bool = False) -> dict:
    images = [
        image
        for image in dataset["images"]
        if (split is None or image["split"] == split)
        and (not usable_only or image["frame_quality"] == "usable")
    ]
    image_ids = {image["id"] for image in images}
    annotations = [
        annotation
        for annotation in dataset["annotations"]
        if annotation["image_id"] in image_ids
    ]
    return {
        **dataset,
        "info": {
            **dataset["info"],
            "description": dataset["info"]["description"]
            + (f" — {split}" if split else "")
            + (" — usable only" if usable_only else ""),
        },
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    seed_by_name = read_seed_manifest()
    tasks = export_from_database()
    write_json(RAW_JSON, tasks)
    dataset, quality_rows = build_coco(tasks, seed_by_name)

    datasets = {
        "instances_all.json": dataset,
        "instances_usable.json": subset(dataset, usable_only=True),
        "instances_train.json": subset(dataset, split="train", usable_only=True),
        "instances_val.json": subset(dataset, split="val", usable_only=True),
    }
    for name, value in datasets.items():
        write_json(NORMALIZED_ANNOTATIONS / name, value)

    quality_csv = NORMALIZED_ROOT / "frame_quality.csv"
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    with quality_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=quality_rows[0])
        writer.writeheader()
        writer.writerows(quality_rows)

    summary = {
        "label_studio_project": PROJECT_TITLE,
        "raw_label_studio_json": str(RAW_JSON),
        "raw_label_studio_json_sha256": sha256(RAW_JSON),
        "all_images": len(dataset["images"]),
        "all_enemy_boxes": len(dataset["annotations"]),
        "usable_images": len(datasets["instances_usable.json"]["images"]),
        "usable_enemy_boxes": len(datasets["instances_usable.json"]["annotations"]),
        "uncertain_images": sum(
            image["frame_quality"] == "uncertain" for image in dataset["images"]
        ),
        "train_images": len(datasets["instances_train.json"]["images"]),
        "train_enemy_boxes": len(datasets["instances_train.json"]["annotations"]),
        "val_images": len(datasets["instances_val.json"]["images"]),
        "val_enemy_boxes": len(datasets["instances_val.json"]["annotations"]),
        "usable_zero_box_images": sum(
            row["frame_quality"] == "usable" and row["enemy_box_count"] == 0
            for row in quality_rows
        ),
        "categories": dataset["categories"],
    }
    write_json(NORMALIZED_ROOT / "dataset_summary.json", summary)

    print(f"Raw export: {RAW_JSON}")
    print(f"Normalized output: {NORMALIZED_ROOT}")
    for key, value in summary.items():
        if key not in {"raw_label_studio_json", "raw_label_studio_json_sha256", "categories"}:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
