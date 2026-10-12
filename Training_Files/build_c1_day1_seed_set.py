"""Build the manual seed subset for the first C1 detector.

The seed contains 100 train images and 20 validation images selected across
their encounter groups. Test images are intentionally excluded at this stage.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FORMAL_MANIFEST = PROJECT_ROOT / "c1_day1_formal_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "c1_seed_images" / "day1_v1"
OUTPUT_MANIFEST = PROJECT_ROOT / "c1_day1_seed_manifest.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "c1_day1_seed_summary.json"
TARGETS = {"train": 100, "val": 20}
MINIMUM_PER_ENCOUNTER = {"train": 1, "val": 2}


def load_formal_rows() -> list[dict[str, str]]:
    with FORMAL_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 600:
        raise ValueError(f"Expected 600 formal rows, found {len(rows)}")
    return rows


def allocate_counts(
    groups: dict[str, list[dict[str, str]]], target: int, minimum: int
) -> dict[str, int]:
    minimum_total = minimum * len(groups)
    if target < minimum_total:
        raise ValueError(
            f"Target {target} cannot provide {minimum} images to {len(groups)} encounters"
        )

    allocation = {encounter_id: minimum for encounter_id in groups}
    remaining = target - minimum_total
    capacity = {
        encounter_id: len(rows) - minimum for encounter_id, rows in groups.items()
    }
    if sum(capacity.values()) < remaining:
        raise ValueError("Not enough formal images to satisfy the seed target")

    total_capacity = sum(capacity.values())
    raw_extra = {
        encounter_id: remaining * capacity[encounter_id] / total_capacity
        for encounter_id in groups
    }
    for encounter_id in groups:
        allocation[encounter_id] += min(
            capacity[encounter_id], int(raw_extra[encounter_id])
        )

    leftover = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda encounter_id: (
            -(raw_extra[encounter_id] - int(raw_extra[encounter_id])),
            encounter_id,
        ),
    )
    while leftover:
        changed = False
        for encounter_id in order:
            if allocation[encounter_id] < len(groups[encounter_id]):
                allocation[encounter_id] += 1
                leftover -= 1
                changed = True
                if leftover == 0:
                    break
        if not changed:
            raise RuntimeError("Unable to distribute all seed images")
    return allocation


def select_evenly(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["frame_timestamp_ns"]))
    indices = [
        min(len(ordered) - 1, int((index + 0.5) * len(ordered) / count))
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError("Seed selection produced duplicate indices")
    return [ordered[index] for index in indices]


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing seed image differs from source: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    formal_rows = load_formal_rows()
    selected_rows = []
    link_methods = defaultdict(int)

    for split, target in TARGETS.items():
        groups = defaultdict(list)
        for row in formal_rows:
            if row["split"] == split:
                groups[row["encounter_id"]].append(row)

        allocation = allocate_counts(
            groups, target, MINIMUM_PER_ENCOUNTER[split]
        )
        for encounter_id in sorted(groups):
            for row in select_evenly(groups[encounter_id], allocation[encounter_id]):
                source = Path(row["image_path"])
                if not source.is_file():
                    raise FileNotFoundError(f"Missing formal image: {source}")
                destination = OUTPUT_ROOT / split / row["image_name"]
                method = link_or_copy(source, destination)
                link_methods[method] += 1
                selected_rows.append(
                    {
                        **row,
                        "formal_image_path": str(source),
                        "image_path": str(destination.resolve()),
                        "annotation_phase": "manual_seed_v1",
                    }
                )

    selected_rows.sort(
        key=lambda row: (
            row["split"],
            row["encounter_id"],
            int(row["frame_timestamp_ns"]),
        )
    )
    if len(selected_rows) != sum(TARGETS.values()):
        raise RuntimeError(f"Unexpected seed size: {len(selected_rows)}")
    if any(row["split"] == "test" for row in selected_rows):
        raise RuntimeError("Test data entered the manual seed set")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=selected_rows[0])
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "dataset": "C1 Enemy Detection Day1 manual seed v1",
        "stage": "Stage 1 — Local Combat Policy",
        "images": len(selected_rows),
        "images_by_split": {
            split: sum(row["split"] == split for row in selected_rows)
            for split in TARGETS
        },
        "encounters_by_split": {
            split: len(
                {row["encounter_id"] for row in selected_rows if row["split"] == split}
            )
            for split in TARGETS
        },
        "minimum_images_per_encounter": MINIMUM_PER_ENCOUNTER,
        "test_images": 0,
        "link_methods": dict(link_methods),
        "source_manifest": str(FORMAL_MANIFEST),
        "output_root": str(OUTPUT_ROOT),
    }
    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("C1 Day1 manual seed set created successfully.")
    print(f"Images: {OUTPUT_ROOT}")
    print(f"Manifest: {OUTPUT_MANIFEST}")
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
