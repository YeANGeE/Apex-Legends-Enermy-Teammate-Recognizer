import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

manifest_path = Path(
    r"C:\Users\15207\Documents\ChatGPT\FPS\c1_grouped_manifest.csv"
)
output_dir = Path(
    r"C:\Users\15207\Documents\ChatGPT\FPS\c1_encounter_contact_sheets"
)

thumbnail_width = 320
thumbnail_height = 180
caption_height = 24
columns = 4

output_dir.mkdir(parents=True, exist_ok=True)

with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

groups = defaultdict(list)

for row in rows:
    groups[row["encounter_id"]].append(row)

for encounter_id, encounter_rows in groups.items():
    encounter_rows.sort(
        key=lambda row: int(row["clip_id"].split("_")[-1])
    )

    thumbnails = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)

        for index, row in enumerate(encounter_rows):
            video_path = Path(row["video_path"])
            midpoint_seconds = float(row["duration_s"]) / 2
            thumbnail_path = temporary_directory / f"{index:04d}.jpg"

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", f"{midpoint_seconds:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale={thumbnail_width}:{thumbnail_height}",
                "-q:v", "3",
                "-y",
                str(thumbnail_path),
            ]

            subprocess.run(command, check=True)

            with Image.open(thumbnail_path) as image:
                thumbnails.append((row["clip_id"], image.convert("RGB").copy()))

    rows_required = (len(thumbnails) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * thumbnail_width,
            rows_required * (thumbnail_height + caption_height),
        ),
        color=(25, 25, 25),
    )

    draw = ImageDraw.Draw(sheet)

    for index, (clip_id, thumbnail) in enumerate(thumbnails):
        column = index % columns
        row_number = index // columns

        x = column * thumbnail_width
        y = row_number * (thumbnail_height + caption_height)

        sheet.paste(thumbnail, (x, y))

        draw.text(
            (x + 6, y + thumbnail_height + 4),
            clip_id,
            fill=(255, 255, 255),
        )

    output_path = output_dir / f"{encounter_id}.jpg"
    sheet.save(output_path, quality=92)

print("Contact-sheet directory:", output_dir)
print("Encounter groups:", len(groups))
print("Contact sheets created:", len(list(output_dir.glob('*.jpg'))))