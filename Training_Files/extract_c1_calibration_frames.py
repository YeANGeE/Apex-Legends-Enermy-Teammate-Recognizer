"""Extract a deterministic 30-frame Day1 calibration set for C1 annotation."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DAY1_ROOT = Path(r"D:\FPS_AI_Dataset\Apex_Day1")
CURATED_MANIFEST = PROJECT_ROOT / "c1_curated_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "c1_calibration_images"
OUTPUT_MANIFEST = PROJECT_ROOT / "c1_calibration_manifest.csv"
SAMPLE_COUNT = 30


def episode_number(clip_id: str) -> int:
    return int(clip_id.rsplit("_", 1)[1])


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count > length:
        raise ValueError(f"Cannot select {count} items from {length}")
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def load_frame_rows(frames_csv: Path) -> tuple[list[dict[str, str]], str]:
    with frames_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No frame rows in {frames_csv}")
    timestamp_column = (
        "source_present_ns"
        if "source_present_ns" in rows[0]
        else "timestamp_ns"
    )
    return rows, timestamp_column


def select_trigger_frame(episode_dir: Path) -> tuple[int, int, int]:
    metadata = json.loads(
        (episode_dir / "metadata.json").read_text(encoding="utf-8")
    )
    frame_rows, timestamp_column = load_frame_rows(episode_dir / "frames.csv")

    trigger_timestamps = metadata.get("rb_down_timestamps_ns") or []
    if trigger_timestamps:
        target_timestamp_ns = int(trigger_timestamps[0])
    else:
        target_timestamp_ns = (
            int(metadata["episode_start_ns"]) + int(metadata["episode_end_ns"])
        ) // 2

    selected = min(
        frame_rows,
        key=lambda row: abs(int(row[timestamp_column]) - target_timestamp_ns),
    )
    return (
        int(selected["frame_id"]),
        int(selected[timestamp_column]),
        target_timestamp_ns,
    )


def extract_frame(video_path: Path, frame_id: int, output_path: Path) -> None:
    escaped_filter = f"select=eq(n\\,{frame_id})"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            escaped_filter,
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
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH")

    with CURATED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["c1_include"] == "1"]

    encounters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        encounters[row["encounter_id"]].append(row)

    encounter_ids = sorted(encounters)
    selected_encounters = [
        encounter_ids[index]
        for index in evenly_spaced_indices(len(encounter_ids), SAMPLE_COUNT)
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_image in OUTPUT_DIR.glob("*.jpg"):
        old_image.unlink()

    output_rows = []
    for encounter_id in selected_encounters:
        encounter_rows = encounters[encounter_id]
        source_row = max(
            encounter_rows,
            key=lambda row: (float(row["duration_s"]), -episode_number(row["clip_id"])),
        )
        clip_id = source_row["clip_id"]
        episode_dir = DAY1_ROOT / clip_id
        video_path = episode_dir / "video.mkv"
        frame_id, frame_timestamp_ns, target_timestamp_ns = select_trigger_frame(
            episode_dir
        )

        image_name = f"day1_{encounter_id}_{clip_id}_frame_{frame_id:06d}.jpg"
        image_path = OUTPUT_DIR / image_name
        extract_frame(video_path, frame_id, image_path)

        output_rows.append(
            {
                "image_name": image_name,
                "image_path": str(image_path),
                "day": "Day1",
                "encounter_id": encounter_id,
                "clip_id": clip_id,
                "frame_id": frame_id,
                "frame_timestamp_ns": frame_timestamp_ns,
                "selection_target_ns": target_timestamp_ns,
            }
        )

    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0])
        writer.writeheader()
        writer.writerows(output_rows)

    print("Images:", OUTPUT_DIR)
    print("Calibration frames:", len(output_rows))
    print("Source encounters:", len(set(row["encounter_id"] for row in output_rows)))
    print("Manifest:", OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
