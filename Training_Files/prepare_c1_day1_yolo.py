"""Prepare the reviewed Day1 C1 dataset in Ultralytics YOLO format.

This is dataset plumbing only: it preserves the existing encounter-level
train/validation split from the normalized Day1 metadata, excludes frames
marked uncertain, and never touches the sealed test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT_ROOT = ROOT / "c1_yolo_export" / "DAY1"
DEFAULT_OUTPUT = ROOT / "c1_yolo_dataset" / "day1_v1"
QUALITY_CSV = (
    ROOT / "c1_annotations" / "day1_full_v1" / "normalized" / "frame_quality.csv"
)
CANONICAL_COCO = (
    ROOT
    / "c1_annotations"
    / "day1_full_v1"
    / "normalized"
    / "annotations"
    / "instances_usable.json"
)
SOURCE_PROJECTS = ("DAY1_120", "DAY1_422")


def read_reviewed_frames() -> dict[str, dict[str, str]]:
    with QUALITY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = {
        row["file_name"]: row
        for row in rows
        if row["frame_quality"] == "usable" and row["split"] in {"train", "val"}
    }
    if len(reviewed) != 540:
        raise ValueError(f"Expected 540 usable Day1 frames, found {len(reviewed)}")
    return reviewed


def find_single_zip(project_dir: Path) -> Path:
    archives = sorted(project_dir.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Expected one ZIP in {project_dir}, found {len(archives)}")
    return archives[0]


def canonical_name(exported_name: str, reviewed_names: set[str]) -> str | None:
    basename = Path(exported_name).name
    exported_stem = Path(basename).stem
    matches = [
        name
        for name in reviewed_names
        if basename.endswith(name) or exported_stem.endswith(Path(name).stem)
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous exported file name: {exported_name}")
    return matches[0] if matches else None


def prepare(export_root: Path, output: Path) -> dict[str, object]:
    reviewed = read_reviewed_frames()
    reviewed_names = set(reviewed)
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise FileExistsError(
            f"Output directory is not empty: {output}. "
            "Use a new output path to avoid overwriting a reviewed dataset."
        )

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    found_images: dict[str, tuple[Path, str]] = {}
    for project in SOURCE_PROJECTS:
        archive = find_single_zip(export_root / project)
        with ZipFile(archive) as source:
            for entry in source.infolist():
                suffix = Path(entry.filename).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".txt"}:
                    continue
                name = canonical_name(entry.filename, reviewed_names)
                if name is None:
                    continue
                if suffix == ".txt":
                    continue
                else:
                    found_images[name] = (archive, entry.filename)

    missing_images = sorted(reviewed_names - found_images.keys())
    if missing_images:
        raise ValueError(f"Missing exported images: {missing_images}")

    coco = json.loads(CANONICAL_COCO.read_text(encoding="utf-8-sig"))
    image_info = {item["file_name"]: item for item in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    if set(image_info) != reviewed_names:
        raise ValueError("Canonical COCO images do not match reviewed usable images")

    box_counts: Counter[str] = Counter()
    for file_name, row in reviewed.items():
        split = row["split"]
        image_archive, image_entry = found_images[file_name]
        with ZipFile(image_archive) as source:
            image_bytes = source.read(image_entry)

        normalized_lines: list[str] = []
        info = image_info[file_name]
        image_id = int(info["id"])
        image_width = float(info["width"])
        image_height = float(info["height"])
        for annotation in annotations_by_image.get(image_id, []):
            x, y, width, height = (float(value) for value in annotation["bbox"])
            coordinates = (
                (x + width / 2.0) / image_width,
                (y + height / 2.0) / image_height,
                width / image_width,
                height / image_height,
            )
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(f"Out-of-range canonical box in {file_name}")
            if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                raise ValueError(f"Non-positive canonical box in {file_name}")
            normalized_lines.append(
                "0 " + " ".join(f"{value:.10f}" for value in coordinates)
            )
            box_counts[split] += 1

        (output / "images" / split / file_name).write_bytes(image_bytes)
        label_path = output / "labels" / split / Path(file_name).with_suffix(".txt").name
        label_path.write_text(
            "\n".join(normalized_lines) + ("\n" if normalized_lines else ""),
            encoding="utf-8",
        )

    dataset_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: enemy\n"
    )
    (output / "data.yaml").write_text(dataset_yaml, encoding="utf-8")

    image_counts = Counter(row["split"] for row in reviewed.values())
    negative_counts = Counter()
    for split in ("train", "val"):
        for label in (output / "labels" / split).glob("*.txt"):
            if not label.read_text(encoding="utf-8").strip():
                negative_counts[split] += 1

    summary = {
        "stage": "Stage 1 — Local Combat Policy",
        "task": "C1 Enemy Detection",
        "format": "Ultralytics YOLO detection",
        "sources": list(SOURCE_PROJECTS),
        "excluded_source": "DAY1_30 calibration set",
        "excluded_uncertain_images": 2,
        "test_images_used": 0,
        "images": dict(image_counts),
        "enemy_boxes": dict(box_counts),
        "negative_images": dict(negative_counts),
        "class_names": {"0": "enemy"},
        "split_authority": str(QUALITY_CSV),
        "annotation_authority": str(CANONICAL_COCO),
        "output": str(output),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = prepare(args.export_root.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
