"""Stage 1 — Local Combat Policy: human Day3 JSON -> audited two-class YOLO labels.

No training or random split. Predictions are never substituted for human truth.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / "c1_yolo_export/DAY3/3.2"
PREPARED = ROOT / "c1_annotations/day3_v4_prelabel_v1"
OUTPUT = ROOT / "c1_yolo_dataset/day3_v1_normalized"
CLASSES = {"enemy": 0, "teammate": 1}
TAGS = {"small_target", "occlusion", "dark", "effects", "ads", "negative"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_annotation(task, record):
    completed = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if len(completed) != 1:
        raise ValueError(f"Exactly one human annotation required: {record['image_name']}")
    annotation = completed[0]
    quality, tags, boxes = None, [], []
    for result in annotation["result"]:
        field, value = result.get("from_name"), result.get("value", {})
        if field == "frame_quality":
            choices = value.get("choices", [])
            if quality is not None or len(choices) != 1 or choices[0] not in {"usable", "uncertain", "discard"}:
                raise ValueError("Invalid/ambiguous frame quality")
            quality = choices[0]
        elif field == "scene_tags":
            choices = value.get("choices", [])
            if not set(choices).issubset(TAGS):
                raise ValueError("Unknown scene tag")
            tags.extend(choices)
        elif field == "label":
            if result.get("type") != "rectanglelabels" or result.get("to_name") != "image":
                raise ValueError("Unexpected region type")
            labels = value.get("rectanglelabels", [])
            if len(labels) != 1 or labels[0] not in CLASSES:
                raise ValueError("Unexpected detection class")
            if (result.get("original_width"), result.get("original_height")) != (record["width"], record["height"]):
                raise ValueError("Annotation/source dimensions differ")
            if result.get("image_rotation", 0) or value.get("rotation", 0):
                raise ValueError("Rotated boxes unsupported")
            x, y, w, h = (float(value[k]) for k in ("x", "y", "width", "height"))
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or max(x+w, y+h) > 100.000001:
                raise ValueError("Invalid box coordinates")
            boxes.append({"class_name": labels[0], "class_id": CLASSES[labels[0]],
                          "xywh": [(x+w/2)/100, (y+h/2)/100, w/100, h/100]})
        else:
            raise ValueError(f"Unexpected annotation field: {field}")
    if quality is None:
        raise ValueError("Missing frame quality")
    return quality, sorted(set(tags)), boxes, annotation


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve existing normalized dataset; use a new version")
    jsons, zips = list(EXPORT.glob("*.json")), list(EXPORT.glob("*.zip"))
    if len(jsons) != 1 or len(zips) != 1:
        raise ValueError("Expected one JSON and one ZIP")
    roster = json.loads((PREPARED/"label_studio_tasks.json").read_text(encoding="utf-8"))
    manifest = json.loads((PREPARED/"manifest.json").read_text(encoding="utf-8"))
    tasks = json.loads(jsons[0].read_text(encoding="utf-8-sig"))
    expected = {t["meta"]["preview_id"]: t for t in roster}
    source_records = {r["preview_id"]: r for r in manifest}
    if len(tasks) != len(expected) or len({t["meta"]["preview_id"] for t in tasks}) != len(tasks) or {t["meta"]["preview_id"] for t in tasks} != set(expected):
        raise ValueError("Incomplete/duplicate task roster")
    records, warnings = [], []
    with ZipFile(zips[0]) as archive:
        # Windows export may use CRCRLF; do not create blank class entries.
        classes = [line.strip() for line in archive.read("classes.txt").decode("utf-8-sig").split("\n") if line.strip()]
        images = [n for n in archive.namelist() if n.startswith("images/") and not n.endswith("/")]
        if len(images) != len(tasks):
            raise ValueError("ZIP image count differs")
        for i, task in enumerate(sorted(tasks, key=lambda t: t["meta"]["preview_id"]), 1):
            identifier = task["meta"]["preview_id"]
            record = source_records[identifier]
            if task["data"] != expected[identifier]["data"] or task["meta"] != expected[identifier]["meta"]:
                raise ValueError("Source metadata/URL changed")
            source = Path(record["image_path"])
            source.resolve().relative_to((PREPARED/"images").resolve())
            if sha(source) != record["image_sha256"]:
                raise ValueError("Prepared image changed")
            with Image.open(source) as im:
                if im.size != (record["width"], record["height"]) or hashlib.sha256(im.convert("RGB").tobytes()).hexdigest() != record["pixel_sha256"]:
                    raise ValueError("Source pixels/dimensions differ")
            matches = [name for name in images if Path(name).name.endswith("__"+source.name) or Path(name).name == source.name]
            if len(matches) != 1 or hashlib.sha256(archive.read(matches[0])).hexdigest() != record["image_sha256"]:
                raise ValueError("ZIP/source bytes differ")
            quality, tags, boxes, annotation = parse_annotation(task, record)
            label_name = "labels/"+Path(matches[0]).stem+".txt"
            exported = []
            if label_name in archive.namelist():
                for line in archive.read(label_name).decode("utf-8-sig").splitlines():
                    fields = line.split()
                    if len(fields) != 5 or not 0 <= int(fields[0]) < len(classes) or classes[int(fields[0])] not in CLASSES:
                        raise ValueError("Invalid ZIP detection label")
                    exported.append((classes[int(fields[0])], list(map(float, fields[1:]))))
            if len(exported) != len(boxes):
                raise ValueError("JSON/ZIP box count differs")
            for box in boxes:
                match = next((j for j, (name, xywh) in enumerate(exported) if name == box["class_name"]
                              and max(abs(a-b) for a, b in zip(xywh, box["xywh"], strict=True)) <= 1e-5), None)
                if match is None:
                    raise ValueError("JSON/ZIP box coordinates differ")
                exported.pop(match)
            if "negative" in tags and boxes and quality == "usable":
                warnings.append({"image_name": source.name, "reason": "negative_tag_with_positive_boxes; boxes remain authoritative"})
            records.append({**record, "quality": quality, "scene_tags": tags,
                            "effective_scene_tags": [t for t in tags if t != "negative" or not boxes],
                            "boxes": boxes, "eligible": quality == "usable",
                            "source_image": str(source), "annotation_id": annotation["id"],
                            "annotation_updated_at": annotation["updated_at"], "split": "unassigned", "match_group": "unconfirmed"})
            if i % 50 == 0 or i == len(tasks):
                print(f"人工JSON/ZIP/原图核对 {i}/{len(tasks)}", flush=True)
    # Write only after every task and archive image passes validation.
    OUTPUT.mkdir(parents=True)
    for directory in ("images", "labels"):
        (OUTPUT/directory).mkdir()
    for record in records:
        if not record["eligible"]:
            continue
        image, label = OUTPUT/"images"/record["image_name"], OUTPUT/"labels"/(Path(record["image_name"]).stem+".txt")
        shutil.copy2(record["source_image"], image)
        label.write_text("".join(str(b["class_id"])+" "+" ".join(f"{v:.10f}" for v in b["xywh"])+"\n" for b in record["boxes"]), encoding="utf-8")
        record.update(normalized_image=str(image), normalized_label=str(label), label_sha256=sha(label))
    eligible = [r for r in records if r["eligible"]]
    summary = {"stage": "Stage 1 — Local Combat Policy", "tasks": len(records), "usable_images": len(eligible),
               "quality_counts": dict(Counter(r["quality"] for r in records)),
               "negative_images": sum(not r["boxes"] for r in eligible),
               "enemy_boxes": sum(b["class_id"] == 0 for r in eligible for b in r["boxes"]),
               "teammate_boxes": sum(b["class_id"] == 1 for r in eligible for b in r["boxes"]),
               "normalized_classes": CLASSES, "export_sha256": sha(jsons[0]), "export_source": str(jsons[0]),
               "zip_source": str(zips[0]), "zip_sha256": sha(zips[0]), "prepared_manifest_sha256": sha(PREPARED/"manifest.json"),
               "split": "unassigned", "match_boundaries_confirmed": False, "training_started": False,
               "prelabel_model": "V4", "warnings": warnings, "policy": "Only explicit usable human annotations retained. Predictions never used as truth; no test split claimed."}
    for name, data in (("source_manifest.json", records), ("dataset_summary.json", summary)):
        (OUTPUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
