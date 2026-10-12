"""Freeze the 27 screenshots for blind annotation; no model/service is started."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from c1_detection.external_screenshots import DATA, ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"D:\FPS_AI_Dataset\test"))
    args = parser.parse_args()
    sources = sorted(p for p in args.source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(sources) != 27:
        raise ValueError(f"Expected 27 images, found {len(sources)}")
    image_dir = DATA / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records, tasks = [], []
    for index, source in enumerate(sources, 1):
        name = f"EXT_{index:03d}{source.suffix.lower()}"
        destination = image_dir / name
        source_sha = sha(source)
        if destination.exists():
            if sha(destination) != source_sha:
                raise ValueError(f"Existing frozen image differs: {destination}")
        else:
            shutil.copy2(source, destination)
        with Image.open(destination) as im:
            im.load()
            width, height = im.size
            pixel_sha = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        image_url = "/data/local-files/?d=" + destination.relative_to(ROOT).as_posix()
        records.append({"preview_id": index, "image_name": name, "source": str(source.resolve()),
                        "normalized_image": str(destination), "image_url": image_url,
                        "width": width, "height": height, "image_sha256": source_sha,
                        "pixel_sha256": pixel_sha, "split": "external_diagnostic"})
        tasks.append({"data": {"image": image_url, "preview_label": f"外部截图 {index:03d} / 027"},
                      "meta": {"dataset": "external_screenshots_v1", "preview_id": index,
                               "split": "external_diagnostic"}})
    for name, value in (("manifest.json", records), ("label_studio_tasks.json", tasks)):
        path = DATA / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen roster changed: {path}")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for offset in range(0, len(records), 12):
        sheet = Image.new("RGB", (1920, 1200), "#111820")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(records[offset:offset+12]):
            with Image.open(r["normalized_image"]) as im:
                tile = im.convert("RGB")
                tile.thumbnail((480, 270))
                x, y = j % 4 * 480, j // 4 * 400
                sheet.paste(tile, (x, y))
            draw.text((x+6, y+275), r["image_name"], fill="white")
        sheet.save(DATA / f"contact_{offset//12+1:02d}.jpg", quality=90)
    print(f"Prepared {len(records)} original-resolution images; unique pixels: {len({r['pixel_sha256'] for r in records})}")
    print(f"Import: {DATA / 'label_studio_tasks.json'}")


if __name__ == "__main__":
    main()
