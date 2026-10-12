"""Change clear zero-box C1 frames from uncertain to usable in Label Studio.

The operation is narrowly scoped to the two known projects and only changes
the frame_quality choice when an active annotation contains zero rectangles.
It creates a consistent SQLite backup before making any database change.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE = PROJECT_ROOT / ".label-studio" / "label_studio.sqlite3"
BACKUP_DIR = PROJECT_ROOT / ".label-studio" / "backups"
AUDIT_CSV = PROJECT_ROOT / "c1_annotations" / "negative_quality_corrections.csv"
EXPECTED_CHANGES = {
    "C1 Day1 Calibration": 4,
    "C1 Day1 Seed v1": 44,
}
UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def image_name(data_json: str) -> str:
    data = json.loads(data_json)
    uploaded = Path(next(iter(data.values())).replace("\\", "/")).name
    return UPLOAD_PREFIX.sub("", uploaded, count=1)


def result_state(result_json: str) -> tuple[list[dict], str | None, int]:
    results = json.loads(result_json)
    qualities = [
        value
        for result in results
        if result.get("type") == "choices"
        and result.get("from_name") == "frame_quality"
        for value in result.get("value", {}).get("choices", [])
    ]
    quality = qualities[0] if len(qualities) == 1 else None
    boxes = sum(result.get("type") == "rectanglelabels" for result in results)
    return results, quality, boxes


def main() -> None:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    candidates = connection.execute(
        """
        select p.title, t.id as task_id, t.inner_id, t.data,
               c.id as completion_id, c.result
        from project p
        join task t on t.project_id = p.id
        join task_completion c on c.task_id = t.id
        where p.title in (?, ?) and p.deleted_at is null and c.was_cancelled = 0
        order by p.title, t.inner_id
        """,
        tuple(EXPECTED_CHANGES),
    ).fetchall()

    changes = []
    remaining_uncertain = {title: 0 for title in EXPECTED_CHANGES}
    already_usable_zero = {title: 0 for title in EXPECTED_CHANGES}
    for row in candidates:
        results, quality, boxes = result_state(row["result"])
        if boxes != 0:
            continue
        if quality == "uncertain":
            remaining_uncertain[row["title"]] += 1
            for result in results:
                if (
                    result.get("type") == "choices"
                    and result.get("from_name") == "frame_quality"
                ):
                    result["value"]["choices"] = ["usable"]
            changes.append(
                {
                    "project": row["title"],
                    "task_id": row["task_id"],
                    "inner_id": row["inner_id"],
                    "completion_id": row["completion_id"],
                    "image_name": image_name(row["data"]),
                    "old_quality": "uncertain",
                    "new_quality": "usable",
                    "enemy_box_count": 0,
                    "new_result": json.dumps(results, ensure_ascii=False),
                }
            )
        elif quality == "usable":
            already_usable_zero[row["title"]] += 1

    if not changes:
        if already_usable_zero != EXPECTED_CHANGES:
            raise ValueError(
                "No changes found, but existing usable zero-box counts do not match "
                f"the expected correction: {already_usable_zero}"
            )
        print("The negative-frame quality correction was already applied.")
        return

    if remaining_uncertain != EXPECTED_CHANGES:
        raise ValueError(
            f"Refusing unexpected correction counts: {remaining_uncertain}"
        )

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / (
        "before_negative_quality_fix_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".sqlite3"
    )
    backup = sqlite3.connect(backup_path)
    connection.backup(backup)
    backup.close()

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        connection.execute("begin immediate")
        for change in changes:
            connection.execute(
                "update task_completion set result = ?, updated_at = ? where id = ?",
                (change["new_result"], updated_at, change["completion_id"]),
            )
            connection.execute(
                "update task set updated_at = ? where id = ?",
                (updated_at, change["task_id"]),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_fields = [
        "project",
        "task_id",
        "inner_id",
        "completion_id",
        "image_name",
        "old_quality",
        "new_quality",
        "enemy_box_count",
    ]
    with AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(
            {key: change[key] for key in audit_fields} for change in changes
        )

    print(f"Database backup: {backup_path}")
    print(f"Corrections: {len(changes)}")
    for title in EXPECTED_CHANGES:
        print(f"{title}: {sum(change['project'] == title for change in changes)}")
    print(f"Audit: {AUDIT_CSV}")


if __name__ == "__main__":
    main()
