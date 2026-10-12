"""Create local visual-QA images from the normalized C1 COCO annotations.

This is a Stage 1 — Local Combat Policy data-quality utility. It performs no
model/API calls: all image rendering happens locally with Pillow.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "c1_annotations" / "day1_calibration_v1" / "normalized"
)

QUALITY_COLOURS = {
    "usable": (40, 200, 90),
    "uncertain": (255, 166, 45),
}
BOX_COLOUR = (235, 57, 53)
BACKGROUND = (24, 27, 32)
TEXT_COLOUR = (245, 247, 250)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render C1 COCO boxes and build local QA contact sheets."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "annotations" / "instances_all.json",
        help="COCO annotation JSON (default: normalized instances_all.json)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "images",
        help="Directory containing the source images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "quality_review",
        help="Output directory for rendered QA files",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [Path("C:/Windows/Fonts/arialbd.ttf")] if bold else []
    ) + [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def validate_dataset(dataset: dict, image_dir: Path) -> None:
    category_names = {item["id"]: item["name"] for item in dataset["categories"]}
    if category_names != {1: "enemy"}:
        raise ValueError(f"Expected the single category {{1: 'enemy'}}, got {category_names}")

    image_ids: set[int] = set()
    for image_info in dataset["images"]:
        image_id = image_info["id"]
        if image_id in image_ids:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        image_ids.add(image_id)
        image_path = image_dir / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        quality = image_info.get("frame_quality")
        if quality not in QUALITY_COLOURS:
            raise ValueError(
                f"Image {image_info['file_name']} has invalid frame_quality: {quality}"
            )

    annotation_ids: set[int] = set()
    for annotation in dataset["annotations"]:
        if annotation["id"] in annotation_ids:
            raise ValueError(f"Duplicate COCO annotation id: {annotation['id']}")
        annotation_ids.add(annotation["id"])
        if annotation["image_id"] not in image_ids:
            raise ValueError(
                f"Annotation {annotation['id']} references an unknown image id"
            )
        if annotation["category_id"] != 1:
            raise ValueError(
                f"Annotation {annotation['id']} has category {annotation['category_id']}"
            )
        x, y, width, height = annotation["bbox"]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Annotation {annotation['id']} has invalid bbox")


def render_image(
    image_path: Path, image_info: dict, annotations: list[dict]
) -> Image.Image:
    header_height = 72
    with Image.open(image_path) as source:
        source_rgb = source.convert("RGB")

    expected_size = (image_info["width"], image_info["height"])
    if source_rgb.size != expected_size:
        raise ValueError(
            f"{image_path.name}: JSON size {expected_size} != file size {source_rgb.size}"
        )

    canvas = Image.new(
        "RGB", (source_rgb.width, source_rgb.height + header_height), BACKGROUND
    )
    canvas.paste(source_rgb, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    label_font = load_font(22, bold=True)
    quality = image_info["frame_quality"]
    quality_colour = QUALITY_COLOURS[quality]

    title = (
        f"{image_info['file_name']}   |   quality: {quality.upper()}"
        f"   |   enemy boxes: {len(annotations)}"
    )
    draw.rectangle((0, 0, 14, header_height), fill=quality_colour)
    draw.text((30, 19), title, fill=TEXT_COLOUR, font=title_font)

    for box_number, annotation in enumerate(annotations, start=1):
        x, y, width, height = annotation["bbox"]
        left = round(x)
        top = round(y) + header_height
        right = round(x + width)
        bottom = round(y + height) + header_height
        draw.rectangle((left, top, right, bottom), outline=BOX_COLOUR, width=5)

        label = f"enemy #{box_number}"
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_top = max(header_height, top - label_height - 12)
        draw.rectangle(
            (left, label_top, left + label_width + 16, label_top + label_height + 10),
            fill=BOX_COLOUR,
        )
        draw.text(
            (left + 8, label_top + 3), label, fill=(255, 255, 255), font=label_font
        )

    if not annotations:
        notice = "NEGATIVE FRAME — 0 enemy boxes"
        notice_font = load_font(36, bold=True)
        notice_box = draw.textbbox((0, 0), notice, font=notice_font)
        notice_width = notice_box[2] - notice_box[0]
        x = (canvas.width - notice_width) // 2
        y = header_height + 28
        draw.rectangle(
            (x - 18, y - 10, x + notice_width + 18, y + 48), fill=(20, 20, 20)
        )
        draw.text((x, y), notice, fill=quality_colour, font=notice_font)

    return canvas


def make_contact_sheets(
    rendered_paths: list[Path], output_dir: Path, columns: int, rows: int
) -> list[Path]:
    if columns < 1 or rows < 1:
        raise ValueError("--columns and --rows must both be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = 600
    cell_height = 382
    margin = 18
    page_header = 62
    page_capacity = columns * rows
    contact_paths = []
    page_font = load_font(30, bold=True)

    for page_index, offset in enumerate(
        range(0, len(rendered_paths), page_capacity), start=1
    ):
        page_paths = rendered_paths[offset : offset + page_capacity]
        sheet = Image.new(
            "RGB",
            (
                columns * cell_width + (columns + 1) * margin,
                page_header + rows * cell_height + (rows + 1) * margin,
            ),
            (235, 238, 242),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (margin, 14),
            f"C1 annotation QA — page {page_index}",
            fill=(25, 29, 34),
            font=page_font,
        )

        for cell_index, rendered_path in enumerate(page_paths):
            row, column = divmod(cell_index, columns)
            x = margin + column * (cell_width + margin)
            y = page_header + margin + row * (cell_height + margin)
            with Image.open(rendered_path) as rendered:
                thumbnail = rendered.copy()
            thumbnail.thumbnail(
                (cell_width, cell_height), resample=Image.Resampling.LANCZOS
            )
            paste_x = x + (cell_width - thumbnail.width) // 2
            paste_y = y + (cell_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (paste_x, paste_y))

        contact_path = output_dir / f"contact_sheet_{page_index:03d}.jpg"
        sheet.save(contact_path, quality=92, subsampling=0)
        contact_paths.append(contact_path)

    return contact_paths


def write_html_index(
    output_path: Path, entries: list[dict], contact_paths: list[Path]
) -> None:
    cards = []
    for entry in entries:
        file_name = html.escape(entry["file_name"])
        quality = html.escape(entry["frame_quality"])
        cards.append(
            f'<article class="{quality}">'
            f'<a href="annotated/{file_name}"><img src="annotated/{file_name}" '
            f'alt="{file_name}"></a>'
            f'<p>{file_name}<br><b>{quality}</b> · {entry["enemy_box_count"]} boxes</p>'
            "</article>"
        )
    contacts = "".join(
        f'<li><a href="contact_sheets/{html.escape(path.name)}">'
        f'{html.escape(path.name)}</a></li>'
        for path in contact_paths
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C1 annotation QA</title>
  <style>
    body {{ font: 15px Arial, sans-serif; margin: 24px; background: #171a1f; color: #f5f7fa; }}
    a {{ color: #84b9ff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    article {{ background: #242930; border-left: 7px solid #28c85a; padding: 10px; }}
    article.uncertain {{ border-left-color: #ffa62d; }}
    article img {{ width: 100%; height: auto; display: block; }}
    article p {{ overflow-wrap: anywhere; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>C1 Enemy Detection — annotation QA</h1>
  <p>Green = usable; orange = uncertain. Click an image to inspect it at full resolution.</p>
  <h2>Contact sheets</h2><ul>{contacts}</ul>
  <h2>All annotated frames</h2><div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotations_path = args.annotations.resolve()
    image_dir = args.images.resolve()
    output_root = args.output.resolve()
    annotated_dir = output_root / "annotated"
    contact_dir = output_root / "contact_sheets"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(annotations_path.read_text(encoding="utf-8-sig"))
    validate_dataset(dataset, image_dir)

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    for annotations in annotations_by_image.values():
        annotations.sort(key=lambda item: item["id"])

    entries = []
    rendered_paths = []
    for image_info in sorted(dataset["images"], key=lambda item: item["file_name"]):
        image_annotations = annotations_by_image[image_info["id"]]
        rendered = render_image(
            image_dir / image_info["file_name"], image_info, image_annotations
        )
        rendered_path = annotated_dir / image_info["file_name"]
        rendered.save(rendered_path, quality=95, subsampling=0)
        rendered_paths.append(rendered_path)
        entries.append(
            {
                "file_name": image_info["file_name"],
                "frame_quality": image_info["frame_quality"],
                "enemy_box_count": len(image_annotations),
            }
        )

    contact_paths = make_contact_sheets(
        rendered_paths, contact_dir, args.columns, args.rows
    )
    write_html_index(output_root / "index.html", entries, contact_paths)

    report = {
        "annotations": str(annotations_path),
        "images": len(entries),
        "enemy_boxes": sum(item["enemy_box_count"] for item in entries),
        "usable_images": sum(item["frame_quality"] == "usable" for item in entries),
        "uncertain_images": sum(
            item["frame_quality"] == "uncertain" for item in entries
        ),
        "negative_images": sum(item["enemy_box_count"] == 0 for item in entries),
        "contact_sheets": len(contact_paths),
    }
    (output_root / "review_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("C1 annotation QA created successfully.")
    print(f"Output: {output_root}")
    for key, value in report.items():
        if key != "annotations":
            print(f"{key}: {value}")
    print(f"Open in browser: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
