"""Create contact sheets for Seed v1 tasks marked uncertain with no boxes."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE = PROJECT_ROOT / ".label-studio" / "label_studio.sqlite3"
SEED_MANIFEST = PROJECT_ROOT / "c1_day1_seed_manifest.csv"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "c1_annotations"
    / "day1_seed_v1"
    / "uncertain_empty_review"
)
PROJECT_TITLE = "C1 Day1 Seed v1"
UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def uploaded_name(data_json: str) -> str:
    data = json.loads(data_json)
    if not data:
        raise ValueError("Label Studio task has empty data")
    uploaded = Path(next(iter(data.values())).replace("\\", "/")).name
    return UPLOAD_PREFIX.sub("", uploaded, count=1)


def main() -> None:
    with SEED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        seed_by_name = {row["image_name"]: row for row in csv.DictReader(file)}

    connection = sqlite3.connect(
        f"file:{DATABASE.resolve().as_posix()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    project = connection.execute(
        "select id from project where title = ? and deleted_at is null",
        (PROJECT_TITLE,),
    ).fetchall()
    if len(project) != 1:
        raise ValueError(f"Expected one active project named {PROJECT_TITLE!r}")
    project_id = project[0]["id"]

    rows = connection.execute(
        """
        select t.id as task_id, t.inner_id, t.data, c.result
        from task t
        join task_completion c on c.task_id = t.id
        where t.project_id = ? and c.was_cancelled = 0
        order by t.inner_id
        """,
        (project_id,),
    ).fetchall()

    review_rows = []
    for row in rows:
        results = json.loads(row["result"])
        qualities = [
            value
            for result in results
            if result.get("type") == "choices"
            for value in result.get("value", {}).get("choices", [])
        ]
        box_count = sum(result.get("type") == "rectanglelabels" for result in results)
        if qualities == ["uncertain"] and box_count == 0:
            image_name = uploaded_name(row["data"])
            seed = seed_by_name.get(image_name)
            if seed is None:
                raise ValueError(f"Task image is absent from seed manifest: {image_name}")
            review_rows.append(
                {
                    "label_studio_task_id": row["task_id"],
                    "label_studio_inner_id": row["inner_id"],
                    "image_name": image_name,
                    "split": seed["split"],
                    "encounter_id": seed["encounter_id"],
                    "image_path": seed["image_path"],
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "uncertain_empty_tasks.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=review_rows[0])
        writer.writeheader()
        writer.writerows(review_rows)

    columns, rows_per_page = 3, 4
    cell_width, image_height, caption_height = 640, 360, 48
    page_size = columns * rows_per_page
    caption_font = font(18)
    sheet_count = 0
    for offset in range(0, len(review_rows), page_size):
        page = review_rows[offset : offset + page_size]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows_per_page * (image_height + caption_height)),
            (25, 27, 31),
        )
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(page):
            column = index % columns
            row_number = index // columns
            x = column * cell_width
            y = row_number * (image_height + caption_height)
            with Image.open(item["image_path"]) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
            sheet.paste(thumbnail, (x, y))
            caption = (
                f"LS #{item['label_studio_inner_id']} | {item['split']} | "
                f"{item['image_name']}"
            )
            draw.text((x + 6, y + image_height + 8), caption, fill="white", font=caption_font)
        sheet_count += 1
        sheet.save(OUTPUT_DIR / f"uncertain_empty_{sheet_count:02d}.jpg", quality=92)

    print(f"Project: {PROJECT_TITLE} (id={project_id})")
    print(f"Uncertain tasks with zero boxes: {len(review_rows)}")
    print(f"Contact sheets: {sheet_count}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
