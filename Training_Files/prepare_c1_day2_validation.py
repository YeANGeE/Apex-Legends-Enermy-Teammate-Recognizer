"""Stage 1 — Local Combat Policy: new, human-labelled comparison holdout.

Bounded preparation only. No models used to select frames, no training/service.
Selection is by native presentation timestamps; indices only decode images.
"""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import prepare_c1_day2_preview as preview

ROOT = Path(__file__).resolve().parent
SOURCE = Path("D:/FPS_AI_Dataset/Apex_YOLO_DAY2_VAL")
OUTPUT = ROOT / "c1_expansion/day2_validation_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing output: {OUTPUT}")
    paths = sorted(p for p in SOURCE.glob("episode_*") if (p / "metadata.json").is_file())
    if not paths:
        raise ValueError("No completed validation recordings")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "images").mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        audits = []
        for row in pool.map(preview.audit_episode, paths):
            audits.append(row)
            print(f"检查完整解码 {len(audits)}/{len(paths)}", flush=True)
    (OUTPUT / "recording_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(r["audit_status"] == "failed" for r in audits):
        raise ValueError("Recording failures: inspect audit before continuing")
    # Detect reuse even when source folders/episode numbers have been renamed.
    old_audits = json.loads((ROOT / "c1_expansion/day2_preview_v1/recording_audit.json").read_text(encoding="utf-8"))
    old_hashes = {r.get("source_frames_sha256") for r in old_audits}
    if any(r["source_frames_sha256"] in old_hashes for r in audits):
        raise ValueError("Validation recordings overlap earlier Day2 recordings")
    candidates, seconds = [], []
    for r in audits:
        r = r.copy()
        r.update(split="comparison_holdout", tentative_group="day2_val_session_001")
        candidates.append(r)
        path = Path(r["episode_path"])
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        with (path / "frames.csv").open(newline="") as handle:
            frames = [f for f in csv.DictReader(handle) if not int(f["is_duplicate"])]
        # A second sample from another RB moment, at least 0.75 seconds away.
        targets = [(a + next((b for b in meta["rb_up_timestamps_ns"] if b >= a), meta["episode_end_ns"])) // 2
                   for a in meta["rb_down_timestamps_ns"]]
        if frames and targets:
            target = max(targets, key=lambda t: abs(t-r["preview_source_present_ns"]))
            f = min(frames, key=lambda f: abs(int(f["source_present_ns"])-target))
            separation = abs(int(f["source_present_ns"])-r["preview_source_present_ns"])
            if separation >= 750_000_000:
                extra = r.copy()
                extra.update(preview_frame_id=int(f["frame_id"]), preview_source_present_ns=int(f["source_present_ns"]))
                seconds.append((separation, extra))
    # Cover every clip once, then add maximally separated secondary samples.
    candidates += [r for _, r in sorted(seconds, key=lambda v: (-v[0], v[1]["episode"]))[:max(0,60-len(candidates))]]
    candidates.sort(key=lambda r: (r["episode"], r["preview_source_present_ns"]))
    for i, r in enumerate(candidates, 1):
        r["preview_id"] = i
    preview.OUTPUT = OUTPUT
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(preview.extract, candidates), 1):
            print(f"抽取验证图 {i}/{len(candidates)}", flush=True)
    old_manifest = json.loads((ROOT / "c1_yolo_dataset/day2_v1_normalized/source_manifest.json").read_text(encoding="utf-8"))
    # Serialized manifest search works across its nested schema; exact pixel hash.
    old_serialized = json.dumps(old_manifest)
    pixels = [r["pixel_sha256"] for r in candidates]
    if len(set(pixels)) != len(pixels) or any(h in old_serialized for h in pixels):
        raise ValueError("Repeated validation image or overlap with Day2 labelled images")
    for r in candidates:
        r["image_sha256"] = preview.digest(Path(r["image_path"]))
    with (OUTPUT / "preview_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in candidates for k in r}))
        writer.writeheader()
        writer.writerows(candidates)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 40)
    for offset in range(0, len(candidates), 16):
        sheet = Image.new("RGB", (7680, 4640), "#111820")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(candidates[offset:offset+16]):
            x, y = j % 4 * 1920, j // 4 * 1160
            with Image.open(r["image_path"]) as im:
                sheet.paste(im.convert("RGB"), (x, y))
            draw.text((x+12,y+1090), f"验证 {r['preview_id']:03d} · {r['episode']} · 帧 {r['preview_frame_id']}", font=font, fill="white")
        sheet.save(OUTPUT / f"总览_{offset//16+1:02d}.jpg", quality=95, subsampling=0)
    summary = dict(stage="Stage 1 — Local Combat Policy", source=str(SOURCE), episodes=len(audits),
        frames=sum(r["frame_count"] for r in audits), minutes=sum(r["duration_s"] for r in audits)/60,
        technical_pass=sum(r["audit_status"]=="technical_pass" for r in audits),
        needs_review=sum(r["audit_status"]=="needs_review" for r in audits), images=len(candidates),
        split="comparison_holdout", training_changed=False, model_based_selection=False,
        selection="One longest-RB-hold midpoint per clip; optional second RB midpoint >=0.75s apart, capped at 60",
        limitation="New recording session, not 38 independent matches; within-clip samples correlated. Human labels pending.",
        manifest_sha256=preview.digest(OUTPUT / "preview_manifest.csv"))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
