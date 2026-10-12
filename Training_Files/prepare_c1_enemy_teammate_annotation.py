"""Stage 1 — Local Combat Policy: create a separate two-class review project.

Historical enemy boxes are suggestions, NOT completed two-class annotations.
Original datasets, projects, and encounter train/val partitions are preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "c1_yolo_dataset/day1_v2_expanded"
OUTPUT = ROOT / "c1_annotations/day1_enemy_teammate_v1"
TITLE = "DAY1 敌我识别 V1"
VERSION = "historical_manual_enemy_boxes"


def prepare_tasks() -> list[dict]:
    metadata = {}
    for path in (ROOT / "c1_day1_formal_manifest.csv", DATASET / "extra_frames_manifest.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row["image_name"]
                if name in metadata:
                    raise ValueError(f"Duplicate source manifest entry: {name}")
                metadata[name] = row
    tasks = []
    box_count = 0
    split_encounters = {}
    for split, expected in (("train", 717), ("val", 62)):
        images = sorted((DATASET / "images" / split).glob("*.jpg"))
        assert len(images) == expected
        split_encounters[split] = set()
        for image in images:
            row = metadata[image.name]
            assert row["split"] == split
            split_encounters[split].add(row["encounter_id"])
            results = []
            for index, line in enumerate((DATASET / "labels" / split / f"{image.stem}.txt").read_text().splitlines()):
                cls, cx, cy, w, h = map(float, line.split())
                assert cls == 0
                results.append({
                    "id": f"oldenemy{index}", "from_name": "label", "to_name": "image",
                    "type": "rectanglelabels", "original_width": 1920,
                    "original_height": 1080, "image_rotation": 0,
                    "value": {"x": (cx-w/2)*100, "y": (cy-h/2)*100,
                              "width": w*100, "height": h*100,
                              "rotation": 0, "rectanglelabels": ["enemy"]},
                })
            box_count += len(results)
            results.append({"id": "quality", "from_name": "frame_quality", "to_name": "image",
                            "type": "choices", "value": {"choices": ["usable"]}})
            tasks.append({
                "data": {"image": "/data/local-files/?d=" + image.relative_to(ROOT).as_posix()},
                "meta": {"split": split, "encounter_id": row["encounter_id"],
                         "image_name": image.name, "frame_timestamp_ns": int(row["frame_timestamp_ns"]),
                         "dataset": "day1_enemy_teammate_v1",
                         "review_status": "teammates_not_yet_annotated"},
                "predictions": [{"model_version": VERSION, "result": results}],
            })
    assert not split_encounters["train"] & split_encounters["val"]
    assert box_count == 471
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "label_studio_tasks.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == tasks
    else:
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(tasks)} tasks: train 717, val 62; historical enemy boxes {box_count}; test 0.")
    return tasks


def create_project(tasks: list[dict], title: str = TITLE, output: Path = OUTPUT, dry_run: bool = False,
                   copy_original_annotations: bool = True) -> None:
    train_count = sum(t["meta"]["split"] == "train" for t in tasks)
    val_count = sum(t["meta"]["split"] == "val" for t in tasks)
    historical_boxes = sum(r["type"] == "rectanglelabels" for t in tasks for r in t["predictions"][0]["result"])
    data_dir = ROOT / ".label-studio"
    os.environ["LABEL_STUDIO_BASE_DATA_DIR"] = str(data_dir)
    os.environ["LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LOCAL_FILES_DOCUMENT_ROOT"] = str(ROOT)
    os.environ["SECRET_KEY"] = (data_dir / "secret_key.txt").read_text().strip()
    os.environ["DJANGO_SETTINGS_MODULE"] = "core.settings.label_studio"
    import label_studio
    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    import django
    django.setup()
    from django.conf import settings
    from django.db import transaction
    from projects.models import Project
    from tasks.models import Task, Prediction
    from data_import.serializers import ImportApiSerializer
    from io_storages.localfiles.models import LocalFilesImportStorage
    from io_storages.localfiles.views import localfiles_data
    from rest_framework.test import APIRequestFactory, force_authenticate

    config = (ROOT / "c1_enemy_teammate_label_config.xml").read_text(encoding="utf-8")
    existing = Project.objects.filter(title=title)
    if existing.exists():
        project = existing.get()
        assert project.label_config == config
        assert Task.objects.filter(project=project).count() == len(tasks)
        assert {t.data["image"] for t in Task.objects.filter(project=project)} == {t["data"]["image"] for t in tasks}
        print("Existing project reused without changing annotations.")
    else:
        reference = Project.objects.get(pk=8, title="YOLO DAY1 VER2")
        backup = data_dir / "backups" / ("before_enemy_teammate_project_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".sqlite3")
        backup.parent.mkdir(exist_ok=True)
        with sqlite3.connect(str(settings.DATABASES["default"]["NAME"])) as source:
            with sqlite3.connect(backup) as destination:
                source.backup(destination)
        with transaction.atomic():
            project = Project.objects.create(
                title=title, description="Stage 1 敌我两类复核：历史敌人框预填，补标队友；保留原交战划分，测试集不使用。",
                label_config=config, organization=reference.organization,
                created_by=reference.created_by, show_collab_predictions=True,
                model_version=VERSION, enable_empty_annotation=True,
            )
            # Preserve any manual work already completed in the original project.
            if title != TITLE and copy_original_annotations:
                original = Project.objects.get(title=TITLE)
                old_tasks = {t.data["image"]: t for t in Task.objects.filter(project=original).prefetch_related("annotations")}
                for task in tasks:
                    source_task = old_tasks[task["data"]["image"]]
                    annotations = list(source_task.annotations.filter(was_cancelled=False))
                    if annotations:
                        latest = max(annotations, key=lambda a: a.updated_at)
                        task["annotations"] = [{"result": latest.result, "completed_by": latest.completed_by_id or reference.created_by_id,
                                                 "was_cancelled": False}]
            serializer = ImportApiSerializer(data=tasks, many=True,
                context={"project": project, "user": reference.created_by})
            serializer.is_valid(raise_exception=True)
            serializer.save()
            for split in ("train", "val"):
                storage = LocalFilesImportStorage.objects.create(
                    project=project, path=str((DATASET / "images" / split).resolve()),
                    title=f"敌我识别图片 {split}", use_blob_urls=True, recursive_scan=False,
                )
                storage.validate_connection()
            assert Task.objects.filter(project=project).count() == len(tasks)
            assert Prediction.objects.filter(project=project).count() == len(tasks)
            if title == TITLE:
                assert not Task.objects.filter(project=project, is_labeled=True).exists()
            if dry_run:
                # Exercise real import/annotation preservation without leaving a QA project.
                transaction.set_rollback(True)
        print("Database backup:", backup)
        if dry_run:
            print("Dry-run import checks passed; new project/tasks rolled back.")
            return
    factory = APIRequestFactory()
    for task in Task.objects.filter(project=project):
        request = factory.get(task.data["image"])
        force_authenticate(request, user=project.created_by)
        response = localfiles_data(request)
        status = response.status_code
        response.close()
        if status != 200:
            raise RuntimeError(f"Image access failed: task {task.pk}, HTTP {status}")
    result = {"project_id": project.pk, "title": title,
              "url": f"http://127.0.0.1:8080/projects/{project.pk}/data",
              "tasks": len(tasks), "train_images": train_count,
              "val_images": val_count,
              "historical_enemy_boxes": historical_boxes,
              "image_access_checked": len(tasks),
              "test_images_used": 0, "classes": {"0": "enemy", "1": "teammate"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "project_info.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-project", action="store_true")
    parser.add_argument("--tasks-json", type=Path)
    parser.add_argument("--project-title", default=TITLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh-review", action="store_true",
                        help="Keep prefilled predictions, but do not copy completed original annotations")
    args = parser.parse_args()
    tasks = json.loads(args.tasks_json.read_text(encoding="utf-8")) if args.tasks_json else prepare_tasks()
    if args.tasks_json:
        allowed = {t["data"]["image"] for t in json.loads((OUTPUT / "label_studio_tasks.json").read_text(encoding="utf-8"))}
        assert tasks and len({t["data"]["image"] for t in tasks}) == len(tasks)
        assert all(t["data"]["image"] in allowed for t in tasks)
    if args.create_project:
        create_project(tasks, args.project_title, args.output.resolve(), args.dry_run,
                       copy_original_annotations=not args.fresh_review)


if __name__ == "__main__":
    main()
