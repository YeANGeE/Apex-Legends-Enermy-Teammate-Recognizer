"""Bounded import into the existing Label Studio database; never starts a service."""
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations/external_screenshots_v1"
TITLE = "外部截图三模型诊断 V1 — 27张"


def main():
    tasks = json.loads((DATA / "label_studio_tasks.json").read_text(encoding="utf-8"))
    if len(tasks) != 27:
        raise ValueError("Expected 27 tasks")
    config = (ROOT / "c1_external_screenshots_label_config.xml").read_text(encoding="utf-8")
    local = ROOT / ".label-studio"
    os.environ.update(LABEL_STUDIO_BASE_DATA_DIR=str(local), LOCAL_FILES_SERVING_ENABLED="true",
                      LOCAL_FILES_DOCUMENT_ROOT=str(ROOT), SECRET_KEY=(local/"secret_key.txt").read_text().strip(),
                      DJANGO_SETTINGS_MODULE="core.settings.label_studio", COLLECT_ANALYTICS="0")
    import label_studio
    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    import django
    django.setup()
    from django.conf import settings
    from django.db import transaction
    from projects.models import Project
    from tasks.models import Task
    from data_import.serializers import ImportApiSerializer
    from io_storages.localfiles.models import LocalFilesImportStorage
    from io_storages.localfiles.views import localfiles_data
    from rest_framework.test import APIRequestFactory, force_authenticate
    existing = Project.objects.filter(title=TITLE)
    if existing.exists():
        project = existing.get()
        if project.label_config != config or {t.data["image"] for t in Task.objects.filter(project=project)} != {t["data"]["image"] for t in tasks}:
            raise ValueError("Existing annotation project differs; no changes applied")
    else:
        reference = Project.objects.get(pk=8, title="YOLO DAY1 VER2")
        backup = local / "backups" / ("before_external_screenshots_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".sqlite3")
        backup.parent.mkdir(exist_ok=True)
        with sqlite3.connect(str(settings.DATABASES["default"]["NAME"])) as source:
            with sqlite3.connect(backup) as destination:
                source.backup(destination)
        print(f"Database backup: {backup}", flush=True)
        with transaction.atomic():
            project = Project.objects.create(title=TITLE, description="27张网络截图，人工盲标，仅外部诊断，不参与训练。",
                        label_config=config, organization=reference.organization, created_by=reference.created_by,
                        enable_empty_annotation=True, show_collab_predictions=False)
            serializer = ImportApiSerializer(data=tasks, many=True, context={"project": project, "user": reference.created_by})
            serializer.is_valid(raise_exception=True)
            serializer.save()
            storage = LocalFilesImportStorage.objects.create(project=project, path=str(DATA/"images"),
                        title="外部截图原尺寸副本", use_blob_urls=True, recursive_scan=False)
            storage.validate_connection()
            if Task.objects.filter(project=project).count() != 27:
                raise ValueError("Import count differs")
    factory = APIRequestFactory()
    for task in Task.objects.filter(project=project):
        request = factory.get(task.data["image"])
        force_authenticate(request, user=project.created_by)
        response = localfiles_data(request)
        status = response.status_code
        response.close()
        if status != 200:
            raise ValueError(f"Image access failed: {task.pk}: {status}")
    info = {"project_id": project.pk, "title": TITLE, "tasks": 27, "images_access_verified": 27,
            "url": f"http://127.0.0.1:8080/projects/{project.pk}/data"}
    (DATA/"project_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
