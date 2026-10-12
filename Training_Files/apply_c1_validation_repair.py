"""Version corrected validation truth separately; never change training data."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from prepare_c1_day1_teammates_yolo import parse_annotation

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "c1_yolo_dataset/day1_v3_enemy_teammate"
EXPORT = ROOT / "c1_yolo_export/DAY1/10_img"
OUTPUT = ROOT / "c1_annotations/day1_validation_repair_v1/corrected_validation"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    files, archives = list(EXPORT.glob("*.json")), list(EXPORT.glob("*.zip"))
    if len(files) != 1 or len(archives) != 1:
        raise ValueError("Expected one JSON and one ZIP export")
    prepared = json.loads((OUTPUT.parent / "label_studio_tasks.json").read_text(encoding="utf-8"))
    expected = {t["meta"]["image_name"]: t for t in prepared}
    exported = json.loads(files[0].read_text(encoding="utf-8-sig"))
    reviewed = {}
    audit = []
    with ZipFile(archives[0]) as archive:
        entries = [n for n in archive.namelist() if n.startswith("images/") and not n.endswith("/")]
        for task in exported:
            name = task["meta"]["image_name"]
            if name not in expected or name in reviewed:
                raise ValueError(f"Unexpected/duplicate task: {name}")
            original = expected[name]
            if task["data"] != original["data"]:
                raise ValueError("Source URL changed")
            for key in ("split", "image_name", "encounter_id", "frame_timestamp_ns", "validation_id", "gallery_id"):
                if task["meta"][key] != original["meta"][key]:
                    raise ValueError(f"Metadata mismatch: {name}, {key}")
            if task["meta"]["split"] != "val":
                raise ValueError("Only validation tasks are allowed")
            source = BASE / "images/val" / name
            with Image.open(source) as image:
                quality, lines = parse_annotation(task, image.size)
            if quality != "usable":
                raise ValueError(f"Uncertain frame requires an explicit evaluation exclusion decision: {name}")
            matches = [n for n in entries if Path(n).name.endswith(name)]
            if len(matches) != 1 or archive.read(matches[0]) != source.read_bytes():
                raise ValueError(f"Export/source image mismatch: {name}")
            if task["meta"]["validation_id"] == 18 and (len(lines) != 1 or not lines[0].startswith("0 ")):
                raise ValueError("Image 18 must contain exactly one enemy, as confirmed by user")
            reviewed[name] = lines
            audit.append({"image_name": name, "validation_id": task["meta"]["validation_id"],
                          "gallery_id": task["meta"]["gallery_id"], "reviewed_boxes": len(lines)})
    if set(reviewed) != set(expected) or len(reviewed) != 10:
        raise ValueError("All ten tasks must be reviewed")
    target = OUTPUT / "labels"
    target.mkdir(parents=True)
    for source in (BASE / "labels/val").glob("*.txt"):
        shutil.copy2(source, target / source.name)
    for name, lines in reviewed.items():
        (target / Path(name).with_suffix(".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest = json.loads((BASE / "source_manifest.json").read_text(encoding="utf-8"))
    val = [r for r in manifest if r["split"] == "val"]
    if {p.stem for p in target.glob("*.txt")} != {Path(r["image_name"]).stem for r in val}:
        raise ValueError("Validation image/label pairing differs")
    counts = Counter(enemy=0, teammate=0)
    label_hashes = {}
    for record in val:
        name = record["image_name"]
        path = target / Path(name).with_suffix(".txt")
        if name not in reviewed and path.read_bytes() != (BASE / "labels/val" / path.name).read_bytes():
            raise ValueError("Unreviewed labels changed")
        for line in path.read_text().splitlines():
            counts["enemy" if line.split()[0] == "0" else "teammate"] += 1
        label_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"stage": "Stage 1 — Local Combat Policy", "validation_images": len(val),
               "ground_truth_boxes": dict(counts), "annotation_authority": str(files[0]),
               "export_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
               "base_dataset": str(BASE), "base_manifest_sha256": hashlib.sha256((BASE / "source_manifest.json").read_bytes()).hexdigest(),
               "repaired_images": audit, "unchanged_images": len(val)-len(reviewed),
               "label_sha256": label_hashes, "training_changed": False, "test_images_used": 0,
               "note": "Human repair informed by validation predictions; not an independent test set. Original validation labels retained."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "label_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
