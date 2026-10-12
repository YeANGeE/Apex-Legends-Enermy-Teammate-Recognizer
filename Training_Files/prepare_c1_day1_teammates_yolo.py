"""Stage 1 — Local Combat Policy: merge human-reviewed enemy/teammate labels.

Read-only validation precedes creation; old datasets and annotations stay intact.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "c1_yolo_dataset/day1_v2_expanded"
OUTPUT = ROOT / "c1_yolo_dataset/day1_v3_enemy_teammate"
EXPORT = ROOT / "c1_yolo_export/DAY1/DAY1_TEAMMATES"
SCREENING = ROOT / "c1_annotations/day1_enemy_teammate_v1/screening"
ROSTER = ROOT / "c1_annotations/day1_enemy_teammate_v1/label_studio_tasks.json"
CLASSES = {"enemy": 0, "teammate": 1}


def parse_annotation(task: dict, size: tuple[int, int]) -> tuple[str, list[str]]:
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if len(annotations) != 1:
        raise ValueError(f"Expected exactly one completed annotation: {task['id']}")
    results = annotations[0]["result"]
    quality = [r["value"]["choices"] for r in results if r.get("from_name") == "frame_quality"]
    if len(quality) != 1 or quality[0] not in (["usable"], ["uncertain"]):
        raise ValueError(f"Invalid frame quality: {task['id']}")
    lines = []
    for result in results:
        if result["type"] != "rectanglelabels":
            continue
        value = result["value"]
        labels = value["rectanglelabels"]
        if len(labels) != 1 or labels[0] not in CLASSES:
            raise ValueError(f"Unsupported class: {labels}")
        if value.get("rotation", 0) != 0 or result.get("image_rotation", 0) != 0:
            raise ValueError("Rotated boxes/images are unsupported")
        if (result["original_width"], result["original_height"]) != size:
            raise ValueError("Annotation/source dimensions differ")
        x, y, w, h = (float(value[k]) for k in ("x", "y", "width", "height"))
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 100.000001 or y+h > 100.000001:
            raise ValueError(f"Invalid box: {value}")
        lines.append(str(CLASSES[labels[0]]) + " " + " ".join(
            f"{v:.10f}" for v in ((x+w/2)/100, (y+h/2)/100, w/100, h/100)))
    return quality[0][0], lines


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    json_files, zip_files = list(EXPORT.glob("*.json")), list(EXPORT.glob("*.zip"))
    if len(json_files) != 1 or len(zip_files) != 1:
        raise ValueError("Expected exactly one JSON and one ZIP export")
    state = json.loads((SCREENING / "selection.json").read_text(encoding="utf-8"))
    if state.get("completed") is not True:
        raise ValueError("Human screening has not been confirmed")
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    roster_hash = hashlib.sha256("\n".join(t["meta"]["image_name"] for t in roster).encode()).hexdigest()
    if state["roster_hash"] != roster_hash:
        raise ValueError("Screening roster changed")
    snapshot = Path(state["final_result"]["snapshot"])
    decisions = json.loads((snapshot / "screening_decisions.json").read_text(encoding="utf-8"))
    expected_ids = set(state["selected_ids"])
    if len(decisions) != len(roster) or len(expected_ids) != state["final_result"]["selected_count"]:
        raise ValueError("Screening count mismatch")
    by_name = {}
    encounters = {"train": set(), "val": set()}
    for identifier, (task, decision) in enumerate(zip(roster, decisions), 1):
        meta = task["meta"]
        name, split = meta["image_name"], meta["split"]
        expected_status = "needs_teammate_review" if identifier in expected_ids else "user_screened_no_teammate"
        if decision != {"id": identifier, "image_name": name, "split": split,
                        "status": expected_status, "enemy_boxes_preserved": True}:
            raise ValueError(f"Invalid human decision: {identifier}")
        if name in by_name or split not in encounters or Path(name).name != name:
            raise ValueError(f"Invalid roster entry: {name}")
        source = BASE / "images" / split / name
        relative = parse_qs(urlsplit(task["data"]["image"]).query)["d"][0]
        if (ROOT / relative).resolve() != source.resolve():
            raise ValueError("Unexpected source image path")
        by_name[name] = (identifier, task, source)
        encounters[split].add(meta["encounter_id"])
    if encounters["train"] & encounters["val"]:
        raise ValueError("Encounter leakage between training and validation")
    for split in encounters:
        names = {n for n, (_, t, _) in by_name.items() if t["meta"]["split"] == split}
        if names != {p.name for p in (BASE / "images" / split).glob("*.jpg")}:
            raise ValueError("Base image roster mismatch")
        if {Path(n).stem for n in names} != {p.stem for p in (BASE / "labels" / split).glob("*.txt")}:
            raise ValueError("Base image/label pairing mismatch")
    tasks = json.loads(json_files[0].read_text(encoding="utf-8-sig"))
    reviewed, changes = {}, []
    with ZipFile(zip_files[0]) as archive:
        entries = [n for n in archive.namelist() if n.startswith("images/") and not n.endswith("/")]
        for task in tasks:
            name = task["meta"]["image_name"]
            if name not in by_name or name in reviewed:
                raise ValueError(f"Unknown/duplicate export: {name}")
            identifier, original, source = by_name[name]
            if identifier not in expected_ids or task["meta"].get("gallery_id") != identifier:
                raise ValueError("Export is not the confirmed selection")
            for key in ("split", "encounter_id", "frame_timestamp_ns", "image_name"):
                if task["meta"][key] != original["meta"][key]:
                    raise ValueError(f"Source metadata changed: {name}, {key}")
            if task["data"]["image"] != original["data"]["image"]:
                raise ValueError("Export source URL changed")
            with Image.open(source) as image:
                status, lines = parse_annotation(task, image.size)
            matches = [n for n in entries if Path(n).name.endswith(name)]
            if len(matches) != 1 or archive.read(matches[0]) != source.read_bytes():
                raise ValueError(f"ZIP/source image mismatch: {name}")
            old = (BASE / "labels" / task["meta"]["split"] / Path(name).with_suffix(".txt")).read_text().splitlines()
            reviewed[name] = (status, lines)
            changes.append({"gallery_id": identifier, "image_name": name, "split": task["meta"]["split"],
                            "quality": status, "old_enemy_count": len(old),
                            "reviewed_enemy_count": sum(l.startswith("0 ") for l in lines),
                            "teammate_count": sum(l.startswith("1 ") for l in lines)})
    if {by_name[n][0] for n in reviewed} != expected_ids:
        raise ValueError("Export does not cover the complete confirmed selection")
    stats = {s: Counter() for s in encounters}
    audit, excluded = [], []
    for name, (identifier, task, source) in by_name.items():
        split = task["meta"]["split"]
        old_label = BASE / "labels" / split / Path(name).with_suffix(".txt")
        status, lines = reviewed.get(name, ("usable", old_label.read_text().splitlines()))
        if status == "uncertain":
            excluded.append(name)
            continue
        target = OUTPUT / "images" / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        label_target = OUTPUT / "labels" / split / old_label.name
        label_target.parent.mkdir(parents=True, exist_ok=True)
        if name in reviewed:
            # A reviewed task is authoritative: replace, do not append stale boxes.
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            shutil.copy2(old_label, label_target)
            if label_target.read_bytes() != old_label.read_bytes():
                raise ValueError("Unselected enemy labels changed")
        stats[split]["images"] += 1
        stats[split]["negative_images"] += not lines
        for cls, label in ((0, "enemy"), (1, "teammate")):
            count = sum(l.startswith(f"{cls} ") for l in lines)
            stats[split][f"{label}_boxes"] += count
            stats[split][f"{label}_images"] += count > 0
        audit.append({**task["meta"], "gallery_id": identifier,
                      "annotation_source": "selected_human_review" if name in reviewed else "old_enemy_labels_and_human_no_teammate_screening",
                      "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    summary = {"stage": "Stage 1 — Local Combat Policy", "task": "C1 Enemy / Teammate Detection",
               "annotation_authority": str(json_files[0]), "export_sha256": hashlib.sha256(json_files[0].read_bytes()).hexdigest(),
               "screening_snapshot": str(snapshot), "selected_images": len(reviewed),
               "screened_no_teammate_images": len(roster)-len(reviewed), "classes": {0: "enemy", 1: "teammate"},
               "images": {s: stats[s]["images"] for s in stats}, "split_statistics": {s: dict(v) for s, v in stats.items()},
               "original_split_preserved": True, "validation_labels_updated": True,
               "test_images_used": 0, "excluded_uncertain_images": excluded,
               "selected_images_with_no_teammate_box": [r["image_name"] for r in changes if not r["teammate_count"]],
               "selected_enemy_count_changes": [r for r in changes if r["old_enemy_count"] != r["reviewed_enemy_count"]]}
    (OUTPUT / "data.yaml").write_text(f"path: {OUTPUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: enemy\n  1: teammate\n", encoding="utf-8")
    for filename, value in (("dataset_summary.json", summary), ("source_manifest.json", audit), ("selected_annotation_audit.json", changes)):
        (OUTPUT / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
