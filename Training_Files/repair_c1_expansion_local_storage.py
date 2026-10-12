"""Register the expansion image directory with its existing Label Studio project.

Preserves tasks and annotations. Does not sync/create additional tasks.
"""

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
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
    from io_storages.localfiles.models import LocalFilesImportStorage
    from projects.models import Project
    from tasks.models import Task
    from rest_framework.test import APIRequestFactory, force_authenticate
    from io_storages.localfiles.views import localfiles_data

    image_root = (ROOT / "c1_expansion" / "day1_train_extra_v1" / "images").resolve()
    project = Project.objects.get(pk=8, title="YOLO DAY1 VER2")
    tasks = list(Task.objects.filter(project=project))
    assert len(tasks) == 240 and all(
        t.data.get("image", "").startswith("/data/local-files/?d=c1_expansion/day1_train_extra_v1/images/")
        for t in tasks
    )
    assert image_root.is_dir()
    annotation_count = sum(t.total_annotations for t in tasks)
    factory = APIRequestFactory()

    def check_view(url):
        request = factory.get(url)
        force_authenticate(request, user=project.created_by)
        response = localfiles_data(request)
        status = response.status_code
        response.close()
        return status

    print("Image access before repair:", check_view(tasks[0].data["image"]))
    backup = data_dir / "backups" / ("before_expansion_storage_fix_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".sqlite3")
    backup.parent.mkdir(exist_ok=True)
    with sqlite3.connect(str(settings.DATABASES["default"]["NAME"])) as source:
        with sqlite3.connect(backup) as destination:
            source.backup(destination)
    with transaction.atomic():
        storage, created = LocalFilesImportStorage.objects.get_or_create(
            project=project, path=str(image_root),
            defaults={"title": "Day1 训练集扩充图片", "use_blob_urls": True, "recursive_scan": False},
        )
        storage.validate_connection()
        storage.save()
    statuses = [check_view(t.data["image"]) for t in tasks]
    assert all(s == 200 for s in statuses), statuses
    assert Task.objects.filter(project=project).count() == 240
    assert sum(t.total_annotations for t in Task.objects.filter(project=project)) == annotation_count
    print("Storage:", storage.pk, "created:", created)
    print("Image access after repair: 240/240 HTTP 200 (authenticated local view)")
    print("Tasks and annotation counts unchanged.")
    print("Backup:", backup)


if __name__ == "__main__":
    main()
