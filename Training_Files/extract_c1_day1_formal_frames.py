"""Build the first formal Day1 frame set for C1 Enemy Detection.

The split is assigned at encounter level before frames are selected, preventing
nearby frames from the same engagement from leaking across train/val/test.
Frame selection uses recorded source timestamps and excludes calibration frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DAY1_ROOT = Path(r"D:\FPS_AI_Dataset\Apex_Day1")
CURATED_MANIFEST = PROJECT_ROOT / "c1_curated_manifest.csv"
CALIBRATION_MANIFEST = PROJECT_ROOT / "c1_calibration_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "c1_formal_images" / "day1_v1"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "c1_day1_formal_manifest.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "c1_day1_formal_summary.json"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a deterministic, encounter-split C1 Day1 dataset."
    )
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", default="c1-day1-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def largest_remainder(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    result = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(
        ratios,
        key=lambda name: (-(raw[name] - result[name]), SPLIT_PRIORITY[name]),
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def load_curated_rows() -> list[dict[str, str]]:
    with CURATED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["c1_include"] == "1"]
    if not rows:
        raise ValueError("The curated manifest contains no included clips")
    return rows


def assign_encounter_splits(
    encounters: dict[str, list[dict[str, str]]], seed: str
) -> tuple[dict[str, str], dict[str, int]]:
    quotas = largest_remainder(len(encounters), SPLIT_RATIOS)
    target_duration = {
        name: sum(float(row["duration_s"]) for rows in encounters.values() for row in rows)
        * ratio
        for name, ratio in SPLIT_RATIOS.items()
    }
    current_duration = {name: 0.0 for name in SPLIT_RATIOS}
    current_count = {name: 0 for name in SPLIT_RATIOS}
    durations = {
        encounter_id: sum(float(row["duration_s"]) for row in rows)
        for encounter_id, rows in encounters.items()
    }
    ordered = sorted(
        encounters,
        key=lambda encounter_id: (
            -durations[encounter_id],
            stable_key(seed, encounter_id),
        ),
    )

    assignments = {}
    for encounter_id in ordered:
        available = [
            name for name in SPLIT_RATIOS if current_count[name] < quotas[name]
        ]
        split = min(
            available,
            key=lambda name: (
                current_duration[name] / target_duration[name],
                current_count[name] / quotas[name],
                SPLIT_PRIORITY[name],
            ),
        )
        assignments[encounter_id] = split
        current_count[split] += 1
        current_duration[split] += durations[encounter_id]

    if current_count != quotas:
        raise RuntimeError(f"Encounter split count mismatch: {current_count} != {quotas}")
    return assignments, quotas


def calibration_timestamps() -> set[int]:
    if not CALIBRATION_MANIFEST.is_file():
        return set()
    with CALIBRATION_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        return {int(row["frame_timestamp_ns"]) for row in csv.DictReader(file)}


def load_episode_frames(clip_id: str) -> list[dict]:
    episode_dir = DAY1_ROOT / clip_id
    frames_csv = episode_dir / "frames.csv"
    video_path = episode_dir / "video.mkv"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")
    with frames_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No frame rows: {frames_csv}")
    timestamp_column = (
        "source_present_ns" if "source_present_ns" in rows[0] else "timestamp_ns"
    )
    return [
        {
            "clip_id": clip_id,
            "frame_id": int(row["frame_id"]),
            "frame_timestamp_ns": int(row[timestamp_column]),
            "video_path": video_path,
        }
        for row in rows
    ]


def build_candidates(
    encounters: dict[str, list[dict[str, str]]], excluded_timestamps: set[int]
) -> dict[str, list[dict]]:
    episode_cache = {}
    candidates = {}
    for encounter_id, clip_rows in encounters.items():
        by_timestamp = {}
        for clip_row in sorted(clip_rows, key=lambda row: row["clip_id"]):
            clip_id = clip_row["clip_id"]
            if clip_id not in episode_cache:
                episode_cache[clip_id] = load_episode_frames(clip_id)
            for frame in episode_cache[clip_id]:
                timestamp = frame["frame_timestamp_ns"]
                if timestamp not in excluded_timestamps:
                    by_timestamp.setdefault(timestamp, frame)
        encounter_candidates = [by_timestamp[key] for key in sorted(by_timestamp)]
        if not encounter_candidates:
            raise ValueError(f"No candidate frames for {encounter_id}")
        candidates[encounter_id] = encounter_candidates
    return candidates


def allocate_frame_counts(
    candidates: dict[str, list[dict]], total: int, minimum_per_encounter: int = 3
) -> dict[str, int]:
    encounter_ids = sorted(candidates)
    minimum_total = minimum_per_encounter * len(encounter_ids)
    if total < minimum_total:
        raise ValueError(
            f"--count must be at least {minimum_total} for {len(encounter_ids)} encounters"
        )
    for encounter_id in encounter_ids:
        if len(candidates[encounter_id]) < minimum_per_encounter:
            raise ValueError(f"Too few frames in {encounter_id}")

    allocation = {name: minimum_per_encounter for name in encounter_ids}
    remaining = total - minimum_total
    weights = {
        name: max(1, len(candidates[name]) - minimum_per_encounter)
        for name in encounter_ids
    }
    total_weight = sum(weights.values())
    raw_extra = {name: remaining * weights[name] / total_weight for name in encounter_ids}
    for name in encounter_ids:
        allocation[name] += int(raw_extra[name])
    leftover = total - sum(allocation.values())
    order = sorted(
        encounter_ids,
        key=lambda name: (-(raw_extra[name] - int(raw_extra[name])), name),
    )
    for name in order[:leftover]:
        allocation[name] += 1
    return allocation


def select_evenly(candidates: list[dict], count: int) -> list[dict]:
    if count > len(candidates):
        raise ValueError(f"Cannot select {count} frames from {len(candidates)}")
    # Midpoint sampling avoids systematically choosing the first/last frame.
    indices = [min(len(candidates) - 1, int((index + 0.5) * len(candidates) / count)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("Frame selection produced duplicate indices")
    return [candidates[index] for index in indices]


def output_name(encounter_id: str, frame: dict) -> str:
    return (
        f"day1_{encounter_id}_{frame['clip_id']}_"
        f"frame_{frame['frame_id']:06d}.jpg"
    )


def extract_one(frame: dict, output_path: Path) -> None:
    if output_path.is_file() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frame["video_path"]),
            "-vf",
            f"select=eq(n\\,{frame['frame_id']})",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create {output_path}")


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH")

    curated_rows = load_curated_rows()
    encounters = defaultdict(list)
    for row in curated_rows:
        encounters[row["encounter_id"]].append(row)

    split_by_encounter, split_quotas = assign_encounter_splits(encounters, args.seed)
    candidates = build_candidates(encounters, calibration_timestamps())
    allocation = allocate_frame_counts(candidates, args.count)

    selected_rows = []
    jobs = []
    output_root = args.output.resolve()
    for encounter_id in sorted(encounters):
        split = split_by_encounter[encounter_id]
        for frame in select_evenly(candidates[encounter_id], allocation[encounter_id]):
            image_name = output_name(encounter_id, frame)
            image_path = output_root / split / image_name
            jobs.append((frame, image_path))
            selected_rows.append(
                {
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "day": "Day1",
                    "split": split,
                    "encounter_id": encounter_id,
                    "clip_id": frame["clip_id"],
                    "frame_id": frame["frame_id"],
                    "frame_timestamp_ns": frame["frame_timestamp_ns"],
                    "selection_method": "encounter_timestamp_midpoint",
                }
            )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_one, frame, image_path): image_path
            for frame, image_path in jobs
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"Extracted/verified: {completed}/{len(futures)}", flush=True)

    for row in selected_rows:
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"Unexpected image size {image.size}: {image_path}")

    selected_rows.sort(key=lambda row: (row["split"], row["encounter_id"], row["frame_timestamp_ns"]))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=selected_rows[0])
        writer.writeheader()
        writer.writerows(selected_rows)

    frame_counts = {
        split: sum(row["split"] == split for row in selected_rows)
        for split in SPLIT_RATIOS
    }
    encounter_counts = {
        split: sum(value == split for value in split_by_encounter.values())
        for split in SPLIT_RATIOS
    }
    leakage_check = all(
        len({row["split"] for row in selected_rows if row["encounter_id"] == encounter_id}) == 1
        for encounter_id in encounters
    )
    summary = {
        "dataset": "C1 Enemy Detection Day1 formal v1",
        "stage": "Stage 1 — Local Combat Policy",
        "seed": args.seed,
        "output_root": str(output_root),
        "images": len(selected_rows),
        "frames_by_split": frame_counts,
        "encounters": len(encounters),
        "encounters_by_split": encounter_counts,
        "expected_encounters_by_split": split_quotas,
        "calibration_timestamps_excluded": len(calibration_timestamps()),
        "encounter_split_leakage": not leakage_check,
        "image_dimensions": [1920, 1080],
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not leakage_check:
        raise RuntimeError("Encounter leakage detected across dataset splits")

    print(f"Output: {output_root}")
    print(f"Manifest: {args.manifest.resolve()}")
    print(f"Summary: {args.summary.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
