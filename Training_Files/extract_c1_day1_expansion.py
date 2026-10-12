"""Extract new timestamp-spaced frames from Day1 TRAIN encounters only.

Stage 1 — Local Combat Policy / C1 Enemy Detection. No predictions or training.
Original formal and calibration timestamps remain excluded.
"""

import bisect
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from extract_c1_day1_formal_frames import load_curated_rows, load_episode_frames, output_name


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "c1_expansion" / "day1_train_extra_v1"
PER_ENCOUNTER = 4
MIN_GAP_NS = 250_000_000


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def distance(timestamp, sorted_times):
    index = bisect.bisect_left(sorted_times, timestamp)
    return min(abs(timestamp - sorted_times[i]) for i in (index - 1, index)
               if 0 <= i < len(sorted_times))


def extract_clip(frames):
    frames = sorted(frames, key=lambda f: f["frame_id"])
    staging = OUTPUT / "extracted" / frames[0]["clip_id"]
    staging.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{f['frame_id']})" for f in frames)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-i", str(frames[0]["video_path"]), "-vf", f"select={expression}",
        "-fps_mode", "vfr", "-frames:v", str(len(frames)), "-q:v", "2",
        "-n", str(staging / "%06d.jpg"),
    ], check=True)
    for index, frame in enumerate(frames, 1):
        source = staging / f"{index:06d}.jpg"
        with Image.open(source) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"Unexpected dimensions: {source}")
            image.verify()
        source.rename(Path(frame["image_path"]))


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Use a new output directory; existing work is preserved: {OUTPUT}")
    formal = csv_rows(ROOT / "c1_day1_formal_manifest.csv")
    calibration = csv_rows(ROOT / "c1_calibration_manifest.csv")
    splits = defaultdict(set)
    old_times = defaultdict(list)
    excluded = set()
    for row in formal:
        splits[row["encounter_id"]].add(row["split"])
        old_times[row["encounter_id"]].append(int(row["frame_timestamp_ns"]))
        excluded.add(int(row["frame_timestamp_ns"]))
    for row in calibration:
        old_times[row["encounter_id"]].append(int(row["frame_timestamp_ns"]))
        excluded.add(int(row["frame_timestamp_ns"]))
    assert all(len(values) == 1 for values in splits.values())
    train_ids = {key for key, values in splits.items() if values == {"train"}}
    assert len(train_ids) == 60
    clips = defaultdict(list)
    for row in load_curated_rows():
        if row["encounter_id"] in train_ids:
            clips[row["encounter_id"]].append(row)

    selected = []
    for encounter in sorted(train_ids):
        original = sorted(set(old_times[encounter]))
        occupied = original.copy()
        candidates = {}
        for row in clips[encounter]:
            for frame in load_episode_frames(row["clip_id"]):
                timestamp = frame["frame_timestamp_ns"]
                if timestamp not in excluded:
                    candidates.setdefault(timestamp, frame)
        for _ in range(PER_ENCOUNTER):
            if not candidates:
                break
            timestamp = max(candidates, key=lambda t: (distance(t, occupied), -t))
            if distance(timestamp, occupied) < MIN_GAP_NS:
                break
            frame = candidates.pop(timestamp)
            image_name = output_name(encounter, frame)
            selected.append({**frame, "encounter_id": encounter, "split": "train",
                             "image_name": image_name,
                             "image_path": str(OUTPUT / "images" / image_name),
                             "nearest_existing_frame_ms": distance(timestamp, original) / 1e6})
            bisect.insort(occupied, timestamp)

    assert selected and all(f["encounter_id"] in train_ids for f in selected)
    assert not ({f["frame_timestamp_ns"] for f in selected} & excluded)
    (OUTPUT / "images").mkdir(parents=True)
    grouped = defaultdict(list)
    for frame in selected:
        grouped[frame["clip_id"]].append(frame)
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(extract_clip, frames): frames for frames in grouped.values()}
        for job in as_completed(jobs):
            job.result()
            completed += len(jobs[job])
            print(f"已抽帧并验证：{completed}/{len(selected)}", flush=True)

    # Audit exact pixel duplicates, not just file names or JPEG byte hashes.
    hashes = set()
    unique, duplicates = [], []
    for frame in sorted(selected, key=lambda f: (f["encounter_id"], f["frame_timestamp_ns"])):
        with Image.open(frame["image_path"]) as image:
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        frame["pixel_sha256"] = digest
        if digest in hashes:
            duplicates.append(frame["image_name"])
        else:
            unique.append(frame)
            hashes.add(digest)
    rows = [{key: str(value) if isinstance(value, Path) else value for key, value in f.items()}
            for f in unique]
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tasks = [{"data": {"image": "/data/local-files/?d=" + Path(f["image_path"]).relative_to(ROOT).as_posix()},
              "meta": {"split": "train", "encounter_id": f["encounter_id"],
                       "image_name": f["image_name"], "frame_timestamp_ns": f["frame_timestamp_ns"],
                       "dataset": "day1_train_extra_v1"}} for f in unique]
    summary = {"stage": "Stage 1 — Local Combat Policy", "task": "C1 Enemy Detection",
               "annotation_images": len(unique), "train_encounters": len({f['encounter_id'] for f in unique}),
               "val_images_used": 0, "test_images_used": 0,
               "existing_timestamp_overlap": 0, "exact_duplicates_excluded": duplicates,
               "minimum_gap_to_existing_ms": min(f["nearest_existing_frame_ms"] for f in unique),
               "selection": "timestamp farthest-point, at most 4 per train encounter, minimum gap 250 ms",
               "output": str(OUTPUT)}
    for name, value in (("label_studio_tasks.json", tasks), ("summary.json", summary)):
        (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
