"""Bounded Day3 project import into existing Label Studio; no service startup."""
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations/day3_v4_prelabel_v1"
TITLE = "DAY3 V4预标注复核 V1"
CONFIG_FILE = ROOT / "c1_day3_label_config.xml"
DESCRIPTION = "Day3实战候选；V4预标，人工补漏/改敌我/统一身体框；来源分组尚未确认，不直接训练。"
MODEL_VERSION = "yolo_v4_day3_conf005"


def main():
    tasks = json.loads((DATA/"label_studio_tasks.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA/"prelabel_summary.json").read_text(encoding="utf-8"))
    if len(tasks) != summary["images"] or not tasks:
        raise ValueError("Prepared task count differs")
    config = CONFIG_FILE.read_text(encoding="utf-8")
    local = ROOT/".label-studio"
    os.environ.update(LABEL_STUDIO_BASE_DATA_DIR=str(local), LOCAL_FILES_SERVING_ENABLED="true",
                      LOCAL_FILES_DOCUMENT_ROOT=str(ROOT), SECRET_KEY=(local/"secret_key.txt").read_text().strip(),
                      DJANGO_SETTINGS_MODULE="core.settings.label_studio", COLLECT_ANALYTICS="0")
    import label_studio
    sys.path.insert(0,str(Path(label_studio.__file__).parent))
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
    existing=Project.objects.filter(title=TITLE)
    if existing.exists():
        project=existing.get()
        if project.label_config != config or {t.data["image"] for t in Task.objects.filter(project=project)} != {t["data"]["image"] for t in tasks}:
            raise ValueError("Existing Day3 project differs; no mutation")
    else:
        reference=Project.objects.get(pk=8,title="YOLO DAY1 VER2")
        backup=local/"backups"/("before_day3_v4_prelabels_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".sqlite3")
        backup.parent.mkdir(exist_ok=True)
        with sqlite3.connect(str(settings.DATABASES["default"]["NAME"])) as source:
            with sqlite3.connect(backup) as destination:
                source.backup(destination)
        print(f"Database backup: {backup}",flush=True)
        with transaction.atomic():
            project=Project.objects.create(title=TITLE,description=DESCRIPTION,
                label_config=config,organization=reference.organization,created_by=reference.created_by,
                enable_empty_annotation=True,show_collab_predictions=True,model_version=MODEL_VERSION)
            serializer=ImportApiSerializer(data=tasks,many=True,context={"project":project,"user":reference.created_by})
            serializer.is_valid(raise_exception=True)
            serializer.save()
            storage=LocalFilesImportStorage.objects.create(project=project,path=str(DATA/"images"),title="Day3原尺寸候选帧",
                                                          use_blob_urls=True,recursive_scan=False)
            storage.validate_connection()
            if Task.objects.filter(project=project).count()!=len(tasks):
                raise ValueError("Import count differs")
            if Prediction.objects.filter(project=project).count()!=len(tasks):
                raise ValueError("Prediction count differs")
    factory=APIRequestFactory()
    for task in Task.objects.filter(project=project):
        request=factory.get(task.data["image"])
        force_authenticate(request,user=project.created_by)
        response=localfiles_data(request)
        status=response.status_code
        response.close()
        if status!=200:
            raise ValueError(f"Image access failed: {task.pk}: {status}")
    info={"project_id":project.pk,"title":TITLE,"tasks":len(tasks),"predictions":Prediction.objects.filter(project=project).count(),
          "images_access_verified":len(tasks),"url":f"http://127.0.0.1:8080/projects/{project.pk}/data"}
    (DATA/"project_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
