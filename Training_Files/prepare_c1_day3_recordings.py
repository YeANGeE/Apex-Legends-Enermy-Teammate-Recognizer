"""Stage 1 — Local Combat Policy: bounded Day3 audit, timestamp sampling and V4 prelabels.

Does not start a service or training. Match boundaries are unknown; no random split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from fps_recorder.utils import resolve_executable
from prepare_c1_day2_preview import audit_episode

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\FPS_AI_Dataset\Apex_YOLO_DAY3")
OUTPUT = ROOT / "c1_annotations/day3_v4_prelabel_v1"
WEIGHTS = ROOT / "runs/c1_yolo/day2_yolo26n_1280_v4_real_match-2/weights/best.pt"
MODEL_VERSION = "yolo_v4_day3_conf005"
PRELABEL_NAME = "V4"
PRELABEL_CONFIDENCE = .05
PREVIEW_PREFIX = "Day3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_frames(path, audit):
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    with (path / "frames.csv").open(newline="") as handle:
        frames = list(csv.DictReader(handle))
    available = [f for f in frames if not int(f["is_duplicate"])]
    if not available:
        raise ValueError(f"No unique capture frames: {path}")
    primary = min(available, key=lambda f: abs(int(f["source_present_ns"])-audit["preview_source_present_ns"]))
    choices = [primary]
    starts = meta.get("rb_down_timestamps_ns", [])
    ends = meta.get("rb_up_timestamps_ns", [])
    if starts:
        begin = max(int(available[0]["source_present_ns"]), min(starts)-500_000_000)
        end = min(int(available[-1]["source_present_ns"]), max(ends or starts)+500_000_000)
    else:
        begin, end = int(available[0]["source_present_ns"]), int(available[-1]["source_present_ns"])
    primary_time = int(primary["source_present_ns"])
    pool = [f for f in available if begin <= int(f["source_present_ns"]) <= end
            and abs(int(f["source_present_ns"])-primary_time) >= 750_000_000]
    if pool:
        choices.append(max(pool, key=lambda f: abs(int(f["source_present_ns"])-primary_time)))
    return sorted(choices, key=lambda f: int(f["source_present_ns"]))


def extract_episode(audit):
    path = Path(audit["episode_path"])
    frames = selected_frames(path, audit)
    indices = [int(f["frame_id"]) for f in frames]
    if len(indices) != len(set(indices)):
        raise ValueError("Repeated selected frame")
    pattern = OUTPUT / "images" / f"{path.name}_%02d.png"
    expected = [OUTPUT / "images" / f"{path.name}_{i:02d}.png" for i in range(1, len(frames)+1)]
    if not all(p.exists() for p in expected):
        if any(p.exists() for p in expected):
            raise ValueError("Partial extraction; preserve files and inspect")
        selection = "+".join(f"eq(n\\,{index})" for index in indices)
        subprocess.run([resolve_executable("ffmpeg"), "-v", "error", "-threads", "1", "-i", str(path/"video.mkv"),
                        "-vf", "select="+selection, "-fps_mode", "vfr", "-frames:v", str(len(indices)),
                        "-threads", "1", "-n", str(pattern)], capture_output=True, check=True, timeout=180)
    records = []
    for frame, image in zip(frames, expected, strict=True):
        with Image.open(image) as im:
            im.load()
            width, height = im.size
            if (width, height) != (1920, 1080):
                raise ValueError("Unexpected recorder resolution")
            pixel_sha = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        records.append({"episode": path.name, "episode_path": str(path), "frame_id": int(frame["frame_id"]),
                        "frame_timestamp_ns": int(frame["source_present_ns"]), "image_path": str(image),
                        "image_name": image.name, "width": width, "height": height,
                        "image_sha256": sha(image), "pixel_sha256": pixel_sha, "split": "unassigned",
                        "match_group": "unconfirmed", "qpc_session": audit["qpc_session"],
                        "audit_status": audit["audit_status"], "warnings": audit["warnings"],
                        "source_metadata_sha256": audit["source_metadata_sha256"],
                        "source_frames_sha256": audit["source_frames_sha256"]})
    return records


def contacts(records):
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
    for offset in range(0, len(records), 20):
        sheet = Image.new("RGB", (2400, 1550), "#121820")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(records[offset:offset+20]):
            x, y = i % 5 * 480, i // 5 * 387
            with Image.open(r["image_path"]) as im:
                tile = im.convert("RGB")
                tile.thumbnail((480, 270))
                sheet.paste(tile, (x, y))
            draw.text((x+5, y+276), f"{r['preview_id']:04d} {r['episode']}\n帧{r['frame_id']} {r['audit_status']}", font=font, fill="white")
        sheet.save(OUTPUT/"contacts"/f"contact_{offset//20+1:03d}.jpg", quality=94)


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for directory in ("images", "contacts"):
        (OUTPUT / directory).mkdir(exist_ok=True)
    paths = sorted(p for p in SOURCE.glob("episode_*") if p.is_dir())
    if not paths:
        raise ValueError("No recording episodes")
    audits = []
    audit_file = OUTPUT / "recording_audit.json"
    if audit_file.exists():
        audits = json.loads(audit_file.read_text(encoding="utf-8"))
        if {r["episode"] for r in audits} != {p.name for p in paths}:
            raise ValueError("Recording roster changed; create a new batch")
        for audit in audits:
            path = Path(audit["episode_path"])
            if audit.get("source_metadata_sha256") and sha(path/"metadata.json") != audit["source_metadata_sha256"]:
                raise ValueError("Source metadata changed")
            if audit.get("source_frames_sha256") and sha(path/"frames.csv") != audit["source_frames_sha256"]:
                raise ValueError("Source frame timestamps changed")
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(audit_episode, p) for p in paths]
            for i, job in enumerate(as_completed(jobs), 1):
                audits.append(job.result())
                if i % 10 == 0 or i == len(jobs):
                    print(f"完整解码检查 {i}/{len(jobs)}", flush=True)
        audits.sort(key=lambda r: r["episode"])
        save(audit_file, audits)
    good = [a for a in audits if a["audit_status"] != "failed"]
    all_records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(extract_episode, a) for a in good]
        for i, job in enumerate(as_completed(jobs), 1):
            all_records.extend(job.result())
            if i % 20 == 0 or i == len(jobs):
                print(f"抽帧 {i}/{len(jobs)}段", flush=True)
    all_records.sort(key=lambda r: (r["episode"], r["frame_timestamp_ns"]))
    records, seen, duplicates = [], {}, []
    for r in all_records:
        if r["pixel_sha256"] in seen:
            duplicates.append({"image": r["image_name"], "kept": seen[r["pixel_sha256"]]})
            continue
        seen[r["pixel_sha256"]] = r["image_name"]
        records.append({**r, "preview_id": len(records)+1})
    manifest = OUTPUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != records:
        raise ValueError("Prepared image manifest changed")
    save(manifest, records)
    save(OUTPUT/"exact_duplicates.json", duplicates)
    contacts(records)
    summary = {"source": str(SOURCE), "episodes": len(paths), "audit_status": dict(Counter(a["audit_status"] for a in audits)),
               "images": len(records), "exact_duplicates_removed": len(duplicates),
               "split": "unassigned", "match_boundaries": "unknown; multiple matches confirmed by user",
               "sampling": "Up to two nonduplicate source_present_ns-selected frames per episode, separated by >=750ms; frame_id only retrieves video frame",
               "warning_policy": "Input/timing warnings are visible; not treated as invalid appearance automatically. Failed video/count/metadata audits excluded.",
               "model_selection_used": False, "action_alignment_performed": False, "training_started": False}
    save(OUTPUT/"preparation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def prelabel():
    from ultralytics import YOLO
    records = json.loads((OUTPUT/"manifest.json").read_text(encoding="utf-8"))
    model = YOLO(str(WEIGHTS))
    if model.names != {0: "enemy", 1: "teammate"}:
        raise ValueError("Wrong model identities")
    tasks, details = [], []
    for i, r in enumerate(records, 1):
        if sha(Path(r["image_path"])) != r["image_sha256"]:
            raise ValueError("Prepared image changed")
        result = model.predict(r["image_path"], imgsz=1280, rect=False, device=0, quantize=16,
                               conf=PRELABEL_CONFIDENCE, iou=.5, max_det=100, verbose=False)[0]
        boxes, scores, labels = result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
        regions = []
        for j, (box, score, label) in enumerate(zip(boxes, scores, labels, strict=True)):
            x1, y1, x2, y2 = box
            if int(label) not in (0, 1) or not 0 <= x1 < x2 <= r["width"] or not 0 <= y1 < y2 <= r["height"]:
                raise ValueError("Invalid model region")
            regions.append({"id": f"p{r['preview_id']}_{j}", "from_name": "label", "to_name": "image",
                            "type": "rectanglelabels", "original_width": r["width"], "original_height": r["height"],
                            "image_rotation": 0, "score": score, "value": {"x":100*x1/r["width"], "y":100*y1/r["height"],
                            "width":100*(x2-x1)/r["width"], "height":100*(y2-y1)/r["height"], "rotation":0,
                            "rectanglelabels":["enemy" if int(label)==0 else "teammate"]}})
        url = "/data/local-files/?d="+Path(r["image_path"]).relative_to(ROOT).as_posix()
        preview_label = f"{PREVIEW_PREFIX} {r['preview_id']:04d} · {r['episode']} · 帧{r['frame_id']}"
        if r["warnings"]:
            preview_label += " · 画面时序提示，请检查图像内容"
        tasks.append({"data":{"image":url,"preview_label":preview_label},
                      "meta":{**r, "dataset":OUTPUT.name},
                      "predictions":[{"model_version":MODEL_VERSION, "score":max(scores,default=0), "result":regions}]})
        details.append({"image":r["image_name"], "boxes":boxes,"scores":scores,"labels":labels})
        if i % 20 == 0 or i == len(records):
            print(f"{PRELABEL_NAME}预标注 {i}/{len(records)}", flush=True)
    save(OUTPUT/"label_studio_tasks.json", tasks)
    save(OUTPUT/"prediction_details.json", details)
    summary={"images":len(records), "checkpoint":str(WEIGHTS),"checkpoint_sha256":sha(WEIGHTS),"confidence":PRELABEL_CONFIDENCE,
             "predicted_boxes":sum(len(p["boxes"]) for p in details),"no_prediction_images":sum(not p["boxes"] for p in details),
             "predictions_are_ground_truth":False,"split":"unassigned", "validation_exposure":f"All batch images assisted by {PRELABEL_NAME}; no blind sealed test claimed",
             "manifest_sha256":sha(OUTPUT/"manifest.json")}
    save(OUTPUT/"prelabel_summary.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prelabel-only",action="store_true")
    args=parser.parse_args()
    if not args.prelabel_only:
        prepare()
    prelabel()
