"""Stage 1 — Local Combat Policy: read-only recording audit + C1 preview.

One representative frame per episode, NOT a formal train/val dataset. Original
recordings and old labels are never modified. No inference or training.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from fps_recorder.utils import resolve_executable

ROOT = Path(__file__).resolve().parent
SOURCE = Path("D:/FPS_AI_Dataset/Apex_YOLO_DAY2")
OUTPUT = ROOT / "c1_expansion/day2_preview_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_episode(path):
    row = {"episode": path.name, "episode_path": str(path), "warnings": []}
    try:
        meta_path = path / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with (path / "frames.csv").open(newline="") as handle:
            frames = list(csv.DictReader(handle))
        times = [int(f["source_present_ns"]) for f in frames]
        if not times or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("Empty or nonmonotonic video timestamp records")
        if len(frames) != meta["frame_count"]:
            raise ValueError("frames.csv count differs from metadata")
        row.update(frame_count=len(frames), duration_s=meta["encoded_playback_duration_seconds"],
                   duplicate_percent=100 * sum(int(f["is_duplicate"]) for f in frames)/len(frames),
                   max_frame_gap_ms=max([0]+[(b-a)/1e6 for a,b in zip(times,times[1:])]),
                   pre_roll_s=meta.get("actual_pre_roll_seconds"),
                   post_roll_s=meta.get("actual_post_roll_seconds"),
                   rb_presses=len(meta["rb_down_timestamps_ns"]),
                   source_metadata_sha256=digest(meta_path),
                   source_frames_sha256=digest(path / "frames.csv"))
        if meta["recording_status"] != "complete" or meta["encoder_queue_overflow"]:
            raise ValueError("Incomplete episode or encoder overflow")
        xt = []
        changes = 0
        before_violation = False
        with (path / "xinput.csv").open(newline="") as handle:
            for reading in csv.DictReader(handle):
                t = int(reading["timestamp_ns"])
                before_violation |= int(reading["poll_before_ns"]) > t
                xt.append(t)
                changes += int(reading["state_changed"])
        if not xt or before_violation or any(b <= a for a,b in zip(xt,xt[1:])):
            raise ValueError("Invalid XInput poll bracket or timestamp order")
        if len(xt) != meta["xinput_reading_count"]:
            raise ValueError("XInput count differs from metadata")
        ages = [(t-xt[bisect.bisect_right(xt,t)-1])/1e6
                for t in times if bisect.bisect_right(xt,t)>0]
        row.update(xinput_count=len(xt),
                   xinput_poll_hz=(len(xt)-1)/((xt[-1]-xt[0])/1e9),
                   xinput_change_hz=changes/((xt[-1]-xt[0])/1e9),
                   xinput_max_gap_ms=max([0]+[(b-a)/1e6 for a,b in zip(xt,xt[1:])]),
                   xinput_unmatched_frames=len(times)-len(ages),
                   xinput_max_asof_age_ms=max(ages, default=None))
        with (path / "gamepad.csv").open(newline="") as handle:
            gi = [int(r["timestamp_ns"]) for r in csv.DictReader(handle)]
        row["gameinput_count"] = len(gi)
        if len(gi) != meta["gameinput_reading_count"] or any(b<a for a,b in zip(gi,gi[1:])):
            row["warnings"].append("GameInput count/order needs review")
        if meta.get("input_quality_warning"):
            row["warnings"].append("Recorder input quality warning")
        if row["xinput_unmatched_frames"] or row["xinput_max_gap_ms"] > 5:
            row["warnings"].append("XInput coverage/gap needs review")
        if row["duplicate_percent"] > 20 or row["max_frame_gap_ms"] > 50:
            row["warnings"].append("Video source repetition/gap needs content review")
        for key in ("pre_roll_s", "post_roll_s"):
            if row[key] is not None and row[key] < 1.95:
                row["warnings"].append(f"Short {key}")
        # Independently decode EVERY frame, not just the ffprobe header.
        exe = resolve_executable("ffmpeg")
        result = subprocess.run([exe,"-v","error","-threads","1","-i",str(path/"video.mkv"),
            "-progress","pipe:1","-nostats","-f","null","-"],
            capture_output=True,text=True,timeout=180)
        decoded = [int(line.split("=",1)[1]) for line in result.stdout.splitlines()
                   if line.startswith("frame=")]
        if result.returncode or result.stderr.strip() or not decoded or decoded[-1] != len(frames):
            raise ValueError(f"Full decode failed/count mismatch: {result.stderr[-300:]}")
        row["decoded_frame_count"] = decoded[-1]
        # Midpoint of longest RB hold: source timestamps select the target.
        # Frame index is used ONLY to retrieve that exact encoded image.
        down = meta["rb_down_timestamps_ns"]
        up = meta["rb_up_timestamps_ns"]
        if down:
            starts = [(a, next((b for b in up if b>=a), meta["episode_end_ns"])) for a in down]
            a,b = max(starts, key=lambda pair: pair[1]-pair[0])
            target = (a+b)//2
        else:
            target = (times[0]+times[-1])//2
        eligible = [f for f in frames if not int(f["is_duplicate"])] or frames
        chosen = min(eligible, key=lambda f: abs(int(f["source_present_ns"])-target))
        row.update(preview_frame_id=int(chosen["frame_id"]),
                   preview_source_present_ns=int(chosen["source_present_ns"]),
                   qpc_session=f"{meta.get('process_id')}:{meta.get('gameinput_calibration',{}).get('offset_ns')}",
                   episode_start_ns=meta["episode_start_ns"], episode_end_ns=meta["episode_end_ns"],
                   audit_status="needs_review" if row["warnings"] else "technical_pass")
    except Exception as exc:
        row.update(audit_status="failed", error=str(exc))
    return row


def extract(row):
    image_path = OUTPUT / "images" / f"{row['preview_id']:04d}_{row['episode']}.jpg"
    exe = resolve_executable("ffmpeg")
    expression = f"select=eq(n\\,{row['preview_frame_id']})"
    subprocess.run([exe,"-v","error","-threads","1","-i",
        str(Path(row["episode_path"])/"video.mkv"),"-vf",expression,
        "-frames:v","1","-q:v","2","-n",str(image_path)],check=True,
        capture_output=True,timeout=180)
    with Image.open(image_path) as image:
        if image.size != (1920,1080):
            raise ValueError(f"Unexpected image size: {image_path}")
        row["pixel_sha256"] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    row["image_path"] = str(image_path)
    return row


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing work; choose a new version: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT/"images").mkdir()
    # The initial gameplay acceptance sample (006) is kept separate.
    paths = sorted(p for p in SOURCE.glob("episode_*") if p.name[-6:].isdigit()
                   and int(p.name[-6:]) >= 7 and (p/"metadata.json").is_file())
    if not paths:
        raise ValueError("No new completed episodes")
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(audit_episode,p) for p in paths]
        for job in as_completed(jobs):
            results.append(job.result())
            if len(results)%10==0 or len(results)==len(paths):
                print(f"已完整解码和检查 {len(results)}/{len(paths)} 段",flush=True)
    results.sort(key=lambda r:r["episode"])
    (OUTPUT/"recording_audit.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    candidates=[r.copy() for r in results if r["audit_status"]!="failed"]
    # Tentative conservative time groups; NOT train/val split assignments.
    group=0
    end=0
    session=None
    for i,row in enumerate(candidates,1):
        if session != row["qpc_session"] or row["episode_start_ns"]>end+15_000_000_000:
            group+=1
        session=row["qpc_session"]
        end=max(end,row["episode_end_ns"])
        row.update(preview_id=i,tentative_group=f"day2_group_{group:03d}",split="unassigned")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(extract,row) for row in candidates]
        for i,job in enumerate(as_completed(jobs),1):
            job.result()
            if i%20==0 or i==len(jobs):
                print(f"已抽取代表帧 {i}/{len(jobs)}",flush=True)
    fields=sorted({k for r in candidates for k in r})
    with (OUTPUT/"preview_manifest.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader(); writer.writerows(candidates)
    sheets=[]
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",32)
    for offset in range(0,len(candidates),16):
        subset=candidates[offset:offset+16]
        sheet=Image.new("RGB",(5120,3104),"#111820")
        draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(subset):
            x,y=j%4*1280,j//4*776
            with Image.open(row["image_path"]) as image:
                tile=image.convert("RGB").resize((1280,720),Image.Resampling.LANCZOS)
            sheet.paste(tile,(x,y))
            label=f"{row['preview_id']:03d}  {row['episode']}  {row['duration_s']:.1f}秒"
            if row["warnings"]:
                label+="  待复核"
            draw.text((x+12,y+728),label,font=font,fill="white")
        path=OUTPUT/f"总览_{offset//16+1:02d}.jpg"
        sheet.save(path,quality=95,subsampling=0)
        sheets.append(str(path))
    summary={"stage":"Stage 1 — Local Combat Policy","purpose":"recording audit and clip preview, not training",
        "source":str(SOURCE),"episodes":len(results),
        "technical_pass":sum(r["audit_status"]=="technical_pass" for r in results),
        "needs_review":sum(r["audit_status"]=="needs_review" for r in results),
        "failed":sum(r["audit_status"]=="failed" for r in results),
        "recorded_minutes":sum(r.get("duration_s",0) for r in results)/60,
        "preview_images":len(candidates),"sheets":sheets,"tentative_groups":group,
        "split":"unassigned; requires encounter review before formal split",
        "warning":"A single preview does not prove the whole clip has enemies/teammates; no human-negative labels inferred."}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="sheets"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
