"""Stage 1 — Local Combat Policy: import approved Day2 preview frames.

No split assigned, no training, no copied Day1 annotations or predicted truth.
Discard/uncertain submissions must not become empty-background examples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIEW = ROOT / "c1_expansion/day2_preview_v1"
OUTPUT = ROOT / "c1_annotations/day2_enemy_teammate_v1"
TITLE = "DAY2 敌我识别 V1 — 214张"
TASK_COUNT = 214
DATASET = "day2_enemy_teammate_v1"
SPLIT = "unassigned"
CONFIG = ROOT / "c1_day2_label_config.xml"
DESCRIPTION = "Day2 候选帧；敌人/队友；可用/不确定/不保留。训练场与实战分别记录，尚未划分训练验证测试。"


def main():
    with (PREVIEW/"preview_manifest.csv").open(encoding="utf-8-sig",newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows)==TASK_COUNT and [int(r["preview_id"]) for r in rows]==list(range(1,TASK_COUNT+1))
    tasks = []
    for r in rows:
        image=Path(r["image_path"]).resolve()
        image.relative_to((PREVIEW/"images").resolve())
        if not image.is_file():
            raise FileNotFoundError(image)
        from PIL import Image
        with Image.open(image) as im:
            if hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()!=r["pixel_sha256"]:
                raise ValueError(f"Image content changed: {image}")
        label=f"编号 {int(r['preview_id']):03d} · {r['episode']} · {float(r['duration_s']):.1f}秒"
        if r["audit_status"]=="needs_review":
            label+=" · 原片段有局部画面间隔待复核（不等于本图不能标注）"
        tasks.append({"data":{"image":"/data/local-files/?d="+image.relative_to(ROOT).as_posix(),
                              "preview_label":label},
                      "meta":{"dataset":DATASET,"preview_id":int(r["preview_id"]),
                              "episode":r["episode"],"image_name":image.name,
                              "frame_id":int(r["preview_frame_id"]),
                              "frame_timestamp_ns":int(r["preview_source_present_ns"]),
                              "tentative_group":r["tentative_group"],"split":SPLIT,
                              "audit_status":r["audit_status"],"pixel_sha256":r["pixel_sha256"]}})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    tasks_path=OUTPUT/"label_studio_tasks.json"
    if tasks_path.exists():
        assert json.loads(tasks_path.read_text(encoding="utf-8"))==tasks
    else:
        tasks_path.write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding="utf-8")
    data_dir=ROOT/".label-studio"
    os.environ.update(LABEL_STUDIO_BASE_DATA_DIR=str(data_dir),LOCAL_FILES_SERVING_ENABLED="true",
        LOCAL_FILES_DOCUMENT_ROOT=str(ROOT),SECRET_KEY=(data_dir/"secret_key.txt").read_text().strip(),
        DJANGO_SETTINGS_MODULE="core.settings.label_studio",COLLECT_ANALYTICS="0")
    import label_studio
    sys.path.insert(0,str(Path(label_studio.__file__).parent))
    import django
    django.setup()
    from django.conf import settings
    from django.db import transaction
    from projects.models import Project
    from tasks.models import Task
    from data_import.serializers import ImportApiSerializer
    from io_storages.localfiles.models import LocalFilesImportStorage
    from io_storages.localfiles.views import localfiles_data
    from rest_framework.test import APIRequestFactory,force_authenticate
    config=CONFIG.read_text(encoding="utf-8")
    existing=Project.objects.filter(title=TITLE)
    if existing.exists():
        project=existing.get()
        assert project.label_config==config
        assert {t.data["image"] for t in Task.objects.filter(project=project)}=={t["data"]["image"] for t in tasks}
    else:
        reference=Project.objects.get(pk=8,title="YOLO DAY1 VER2")
        backup=data_dir/"backups"/("before_day2_annotation_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".sqlite3")
        backup.parent.mkdir(exist_ok=True)
        with sqlite3.connect(str(settings.DATABASES["default"]["NAME"])) as source:
            with sqlite3.connect(backup) as destination:
                source.backup(destination)
        print("Database backup:",backup,flush=True)
        with transaction.atomic():
            project=Project.objects.create(title=TITLE,
                description=DESCRIPTION,
                label_config=config,organization=reference.organization,created_by=reference.created_by,
                enable_empty_annotation=True,show_collab_predictions=False)
            serializer=ImportApiSerializer(data=tasks,many=True,
                context={"project":project,"user":reference.created_by})
            serializer.is_valid(raise_exception=True)
            serializer.save()
            storage=LocalFilesImportStorage.objects.create(project=project,
                path=str((PREVIEW/"images").resolve()),title="Day2 原尺寸图片",
                use_blob_urls=True,recursive_scan=False)
            storage.validate_connection()
            assert Task.objects.filter(project=project).count()==TASK_COUNT
            assert not Task.objects.filter(project=project,is_labeled=True).exists()
    factory=APIRequestFactory()
    for task in Task.objects.filter(project=project):
        request=factory.get(task.data["image"])
        force_authenticate(request,user=project.created_by)
        response=localfiles_data(request)
        status=response.status_code
        response.close()
        if status!=200:
            raise RuntimeError(f"Image access failed: {task.pk} HTTP {status}")
    info={"title":TITLE,"project_id":project.pk,"tasks":TASK_COUNT,"images_access_verified":TASK_COUNT,
          "url":f"http://127.0.0.1:8080/projects/{project.pk}/data","split":SPLIT,
          "quality_policy":"Only explicitly usable annotations are eligible; discard and uncertain are excluded, never negatives",
          "source_preserved":True,"old_projects_preserved":True}
    (OUTPUT/"project_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
