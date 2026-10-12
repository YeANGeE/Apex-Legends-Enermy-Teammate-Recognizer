"""Recompose existing reviewed frames at native resolution; no inference."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/review_two_class_20261008_183428_922773"


def main():
    with (REVIEW / "image_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = REVIEW / "high_resolution"
    output.mkdir(exist_ok=True)
    for offset in range(0, len(rows), 16):
        group = rows[offset:offset+16]
        sheet = Image.new("RGB", (1920*4, (1080+100)*4), "#10161e")
        draw = ImageDraw.Draw(sheet)
        label_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 44)
        for index, row in enumerate(group):
            with Image.open(REVIEW / "rendered" / row["image_name"]) as source:
                if source.size != (1920, 1080):
                    raise ValueError(f"Unexpected native frame size: {source.size}")
                tile = source.convert("RGB")
            x, y = index % 4 * 1920, index // 4 * 1180
            sheet.paste(tile, (x, y))
            label = (f"验证{int(row['validation_id']):02d} / 筛选{row['gallery_id']}   "
                     f"正确{row['correct']}  认反{row['wrong_identity']}  "
                     f"未匹配预测{row['unmatched_predictions']}  未匹配人工框{row['unmatched_targets']}")
            draw.text((x+12, y+1100), label, fill="white", font=label_font)
        target = output / f"contact_sheet_{offset//16+1:02d}_native.png"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        sheet.save(target, compress_level=3)
        print(f"{target} | 7680x4720 | {target.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
