import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path

dataset_root = Path(r"D:\FPS_AI_Dataset\Apex_Day1")
output_csv = Path(r"C:\Users\15207\Documents\ChatGPT\FPS\c1_clip_manifest.csv")

rows = []

for video_path in sorted(dataset_root.glob("episode_*/video.mkv")):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    probe = json.loads(result.stdout)
    stream = probe["streams"][0]
    episode_dir = video_path.parent

    fps = float(Fraction(stream["r_frame_rate"]))
    duration_s = float(probe["format"]["duration"])

    rows.append(
        {
            "clip_id": episode_dir.name,
            "video_path": str(video_path),
            "duration_s": round(duration_s, 3),
            "fps": round(fps, 3),
            "width": stream["width"],
            "height": stream["height"],
            "reported_frames": stream.get("nb_frames", ""),
            "size_mb": round(video_path.stat().st_size / 1024**2, 3),
            "has_frames_csv": (episode_dir / "frames.csv").exists(),
            "has_gamepad_csv": (episode_dir / "gamepad.csv").exists(),
            "has_metadata_json": (episode_dir / "metadata.json").exists(),
        }
    )

fieldnames = list(rows[0].keys())

with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

total_duration_min = sum(row["duration_s"] for row in rows) / 60
complete_clips = sum(
    row["has_frames_csv"]
    and row["has_gamepad_csv"]
    and row["has_metadata_json"]
    for row in rows
)

print("Manifest:", output_csv)
print("Clips:", len(rows))
print("Total duration (minutes):", round(total_duration_min, 2))
print("Clips with all sidecar files:", complete_clips)
