"""Merge reviewed extra TRAIN frames; preserve the frozen Day1 validation set."""
from __future__ import annotations

import csv
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
BASE = ROOT / "c1_yolo_dataset/day1_v1_verified"
OUTPUT = ROOT / "c1_yolo_dataset/day1_v2_expanded"
EXPORT = ROOT / "c1_yolo_export/DAY1/DAY1_EXTRA"
EXPANSION = ROOT / "c1_expansion/day1_train_extra_v1"


def main() -> None:
    json_files = list(EXPORT.glob("*.json"))
    zip_files = list(EXPORT.glob("*.zip"))
    if len(json_files) != 1 or len(zip_files) != 1:
        raise ValueError("Expected exactly one JSON and one ZIP export.")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    with (EXPANSION / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = {r["image_name"]: r for r in rows}
    assert len(rows) == len(manifest) == 240
    assert all(r["split"] == "train" for r in rows)
    tasks = json.loads(json_files[0].read_text(encoding="utf-8-sig"))
    reviewed = {}
    counts = Counter()
    with ZipFile(zip_files[0]) as archive:
        for task in tasks:
            image_url = task["data"]["image"]
            name = Path(parse_qs(urlsplit(image_url).query)["d"][0]).name
            if name not in manifest or name in reviewed:
                raise ValueError(f"Unknown or duplicate task: {name}")
            assert task["meta"]["split"] == "train"
            assert int(task["meta"]["frame_timestamp_ns"]) == int(manifest[name]["frame_timestamp_ns"])
            annotations = [a for a in task["annotations"] if not a.get("was_cancelled")]
            if len(annotations) != 1:
                raise ValueError(f"Expected one completed annotation: {name}")
            results = annotations[0]["result"]
            quality = [r["value"]["choices"] for r in results if r["from_name"] == "frame_quality"]
            if len(quality) != 1 or quality[0] not in (["usable"], ["uncertain"]):
                raise ValueError(f"Missing or ambiguous frame quality: {name}")
            status = quality[0][0]
            lines = []
            for r in results:
                if r["type"] != "rectanglelabels":
                    continue
                v = r["value"]
                if v["rectanglelabels"] != ["enemy"] or v.get("rotation", 0) != 0:
                    raise ValueError(f"Unsupported label or rotated box: {name}")
                x, y, w, h = (float(v[k]) for k in ("x", "y", "width", "height"))
                if not all(math.isfinite(n) for n in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 100.000001 or y+h > 100.000001:
                    raise ValueError(f"Invalid bounding box: {name}")
                lines.append("0 " + " ".join(f"{n:.10f}" for n in ((x+w/2)/100, (y+h/2)/100, w/100, h/100)))
            source = EXPANSION / "images" / name
            with Image.open(source) as image:
                image.load()
                pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                if pixels != manifest[name]["pixel_sha256"]:
                    raise ValueError(f"Source image changed: {name}")
                for r in results:
                    if r["type"] == "rectanglelabels" and (r["original_width"], r["original_height"]) != image.size:
                        raise ValueError(f"Annotation image dimensions differ: {name}")
            entries = [n for n in archive.namelist() if n.startswith("images/") and Path(n).name.endswith(name)]
            if len(entries) != 1 or archive.read(entries[0]) != source.read_bytes():
                raise ValueError(f"Export/source image mismatch: {name}")
            reviewed[name] = (status, lines)
            counts[status] += 1
        if set(reviewed) != set(manifest):
            raise ValueError("Export does not cover all 240 extracted frames.")

    # Copy only image/label pairs, not old Ultralytics caches or old metadata.
    for split in ("train", "val"):
        for folder, suffix, expected in (("images", "*.jpg", 478 if split == "train" else 62), ("labels", "*.txt", 478 if split == "train" else 62)):
            files = list((BASE / folder / split).glob(suffix))
            assert len(files) == expected
            target = OUTPUT / folder / split
            target.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy2(file, target / file.name)
    extra_boxes = extra_negatives = 0
    for name, (status, lines) in reviewed.items():
        if status != "usable":
            continue
        target = OUTPUT / "images/train" / name
        if target.exists():
            raise ValueError(f"Extra frame overlaps baseline: {name}")
        shutil.copy2(EXPANSION / "images" / name, target)
        (OUTPUT / "labels/train" / Path(name).with_suffix(".txt").name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        extra_boxes += len(lines)
        extra_negatives += not lines
    # Byte-for-byte verification of frozen validation labels AND images.
    for folder in ("images", "labels"):
        for file in (BASE / folder / "val").iterdir():
            if file.suffix not in {".jpg", ".txt"}:
                continue
            assert file.read_bytes() == (OUTPUT / folder / "val" / file.name).read_bytes()
    summary = {
        "stage": "Stage 1 — Local Combat Policy", "task": "C1 Enemy Detection",
        "baseline": str(BASE), "annotation_authority": str(json_files[0]),
        "extra_quality": dict(counts), "extra_enemy_boxes": extra_boxes,
        "extra_negative_images": extra_negatives,
        "images": {"train": 478 + counts["usable"], "val": 62},
        "validation_unchanged": True, "test_images_used": 0,
        "excluded_images": [n for n, (s, _) in reviewed.items() if s == "uncertain"],
    }
    (OUTPUT / "data.yaml").write_text(f"path: {OUTPUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: enemy\n", encoding="utf-8")
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(EXPANSION / "manifest.csv", OUTPUT / "extra_frames_manifest.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
