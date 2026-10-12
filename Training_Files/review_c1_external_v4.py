"""Offline V4 error review from saved predictions. No inference, server, or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/c1_detection/external_screenshots_20261009_181134_641513"
NAMES = {1: "敌人", 2: "队友"}
GT_OPTIONS = {"size_only": "同一个人，身份正确，仅框松紧不同", "bad_localization": "同一个人，身份正确，但框明显偏移/截断/包含背景",
              "wrong_identity": "同一个人，敌我认反", "missed": "没有检出这个人", "annotation_issue": "人工框或人工身份需修改", "uncertain": "无法确定/候选不是同一个人"}
PRED_OPTIONS = {"background": "背景/界面误检", "duplicate": "同一人物的重复框",
                "existing_person_correct": "已标过的同一个人，身份正确（可能框大小不同）",
                "existing_person_wrong_identity": "已标过的同一个人，但敌我认反",
                "annotation_issue": "真实人物，但人工标注遗漏", "uncertain": "无法确定"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, b):
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inter/(area_a+area_b-inter), inter/min(area_a, area_b)


def build_cases(records, predictions, threshold):
    roster = {p["image"]: p for p in predictions}
    if len(roster) != len(predictions) or set(roster) != {r["image_name"] for r in records}:
        raise ValueError("Prediction/truth roster differs")
    cases = []
    for record in records:
        pred = roster[record["image_name"]]
        if not (len(pred["boxes"]) == len(pred["labels"]) == len(pred["scores"])):
            raise ValueError("Prediction lengths differ")
        eligible = sorted((i for i, s in enumerate(pred["scores"]) if s >= threshold), key=lambda i: -pred["scores"][i])
        remaining = set(range(len(record["boxes"])))
        matched, used = {}, set()
        # Preserve standard, class-agnostic, confidence-first IoU>=.5 matching.
        for i in eligible:
            candidates = [(overlap(pred["boxes"][i], record["boxes"][j])[0], j) for j in remaining]
            if candidates:
                iou, j = max(candidates)
                if iou >= .5:
                    matched[j] = (i, "strict_match")
                    used.add(i)
                    remaining.remove(j)
        # Pair residual boxes only as review suggestions, never as accepted detections.
        pairs = []
        for j in remaining:
            for i in eligible:
                if i in used:
                    continue
                iou, ios = overlap(pred["boxes"][i], record["boxes"][j])
                if iou >= .1 or ios >= .5:
                    pairs.append((iou, ios, pred["scores"][i], j, i))
        for iou, ios, score, j, i in sorted(pairs, reverse=True):
            if j in remaining and i not in used:
                matched[j] = (i, "geometry_candidate")
                remaining.remove(j)
                used.add(i)
        def case(j, i, kind):
            truth = record["boxes"][j] if j is not None else None
            box = pred["boxes"][i] if i is not None else None
            label = record["labels"][j] if j is not None else None
            pred_label = pred["labels"][i] if i is not None else None
            iou, ios = overlap(truth, box) if truth is not None and box is not None else (None, None)
            return {"case_id": f"{record['image_name'][:-4]}_{'G'+str(j+1) if j is not None else 'P'+str(i+1)}",
                    "image": record["image_name"], "source": record["normalized_image"], "width": record["width"], "height": record["height"],
                    "target_index": j, "prediction_index": i, "truth_box": truth, "prediction_box": box,
                    "truth_label": label, "prediction_label": pred_label, "confidence": pred["scores"][i] if i is not None else None,
                    "iou": iou, "intersection_over_smaller_box": ios, "pairing": kind,
                    "review_required": kind != "strict_match" or label != pred_label,
                    "scene_tags": record["scene_tags"]}
        for j in range(len(record["boxes"])):
            i, kind = matched.get(j, (None, "no_candidate"))
            cases.append(case(j, i, kind))
        for i in eligible:
            if i not in used:
                cases.append(case(None, i, "unmatched_prediction"))
    return cases


def draw_crop(case, destination, record=None):
    boxes = [b for b in (case["truth_box"], case["prediction_box"]) if b is not None]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = max(80, .2*max(right-left, bottom-top))
    bounds = (max(0, math.floor(left-pad)), max(0, math.floor(top-pad)),
              min(case["width"], math.ceil(right+pad)), min(case["height"], math.ceil(bottom+pad)))
    with Image.open(case["source"]) as image:
        source_crop = image.convert("RGB").crop(bounds)
    crop = Image.new("RGB", (max(520, source_crop.width), source_crop.height+96), "#121820")
    crop.paste(source_crop, (0, 96))
    draw = ImageDraw.Draw(crop)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    if record is not None:
        for index, (box, label) in enumerate(zip(record["boxes"], record["labels"], strict=True)):
            if index == case["target_index"]:
                continue
            if box[2] <= bounds[0] or box[0] >= bounds[2] or box[3] <= bounds[1] or box[1] >= bounds[3]:
                continue
            xy = [box[0]-bounds[0], box[1]-bounds[1]+96, box[2]-bounds[0], box[3]-bounds[1]+96]
            draw.rectangle(xy, outline="#00ff80", width=2)
            draw.text((max(0, xy[0]), max(96, xy[1])), f"G{index+1} {NAMES[label]}",
                      font=font, fill="#00ff80", stroke_width=2, stroke_fill="black")
    for key, label_key, color, prefix in (("truth_box", "truth_label", "#00ff80", "人工"), ("prediction_box", "prediction_label", "#ff55ff", "V4")):
        box = case[key]
        if box is None:
            continue
        xy = [box[0]-bounds[0], box[1]-bounds[1]+96, box[2]-bounds[0], box[3]-bounds[1]+96]
        draw.rectangle(xy, outline=color, width=4)
        text = f"{prefix}：{NAMES[case[label_key]]}"
        if prefix == "V4":
            text += f" {case['confidence']:.3f}"
        draw.text((8, 8 if prefix == "人工" else 48), text,
                  font=font, fill=color, stroke_width=2, stroke_fill="black")
    if case["truth_box"] is None:
        draw.text((8, 8), "当前预测未配对；周围人工框仍显示", font=font, fill="#00ff80")
    crop.save(destination, compress_level=3)


def draw_full_truth(record, destination):
    with Image.open(record["normalized_image"]) as im:
        original = im.convert("RGB")
    sheet = Image.new("RGB", (original.width, original.height+96), "#121820")
    sheet.paste(original, (0, 96))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)
    draw.text((12, 12), f"{record['image_name']} · 全部人工框 {len(record['boxes'])} 个 · 原尺寸", font=font, fill="#00ff80")
    for index, (box, label) in enumerate(zip(record["boxes"], record["labels"], strict=True)):
        xy = [box[0], box[1]+96, box[2], box[3]+96]
        draw.rectangle(xy, outline="#00ff80", width=4)
        draw.text((xy[0], max(96, xy[1]-40)), f"G{index+1} {NAMES[label]}", font=font,
                  fill="#00ff80", stroke_width=2, stroke_fill="black")
    sheet.save(destination, compress_level=3)


def make_html(folder, cases, manifest_sha):
    cards = []
    for c in cases:
        options = GT_OPTIONS if c["target_index"] is not None else PRED_OPTIONS
        if c["prediction_index"] is None:
            options = {k: GT_OPTIONS[k] for k in ("missed", "annotation_issue", "uncertain")}
        opts = '<option value="">待复核</option>'+''.join(f'<option value="{k}">{v}</option>' for k, v in options.items())
        detail = f"{c['pairing']} · IoU={c['iou']:.3f} · 预测置信度={c['confidence']:.3f}" if c['iou'] is not None else c['pairing']
        if c["truth_box"] is not None and c["prediction_box"] is not None:
            gt, pred = c["truth_box"], c["prediction_box"]
            detail += f" · 宽比={(pred[2]-pred[0])/(gt[2]-gt[0]):.2f} · 高比={(pred[3]-pred[1])/(gt[3]-gt[1]):.2f}"
        cards.append(f'<article data-id="{c["case_id"]}" data-required="{str(c["review_required"]).lower()}"><h2>{c["case_id"]} {"★需复核" if c["review_required"] else "标准匹配正确，可选复核"}</h2><p>{html.escape(detail)}</p><div class="image"><img src="crops/{c["case_id"]}.png" loading="lazy"></div><p><a href="crops/{c["case_id"]}.png" target="_blank">打开原尺寸局部图</a> · <a href="full_truth/{c["image"]}" target="_blank">查看本图全部人工标注</a> · <a href="../yolo_day2_v4/rendered/{c["image"]}" target="_blank">查看V4整图及其他预测框</a> · <a href="../compare_{c["image"][4:7]}.png" target="_blank">三模型整图对照</a></p><select>{opts}</select><input placeholder="备注：遮挡、人工框原因、其他候选预测等"></article>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V4 人工复核</title>
<style>body{background:#121820;color:#e6edf3;font:17px system-ui;margin:24px}header{position:sticky;top:0;background:#121820;padding:12px;z-index:2}button,select,input{font:inherit;padding:10px;margin:6px}article{border:1px solid #465568;padding:16px;margin:20px 0} .image{overflow:auto;max-height:760px}img{max-width:none}input{width:60%}a{color:#8bc7ff}</style>
<header><h1>V4 人工复核</h1><p>绿框=人工；紫框=V4。局部图保持原像素，不缩放。几何候选只是提示，先确认是不是同一个人，再判断身份与框。</p><p>页面默认显示全部人工目标与额外预测。未匹配预测没有指定人工目标，但局部图显示周围已有人工框；可点“查看本图全部人工标注”。</p><p>候选仅来自conf≥0.05预测。no_candidate表示没有通过几何筛选的候选，不等于人物绝对未检出；请查看整图，若对应其他框，选不确定并备注。</p><p>“只差大小”仅用于身份正确且框松紧差异可接受；明显偏移/截断选定位问题。不要为让模型得分高而改人工标注。</p><label><input type="checkbox" id="only" style="width:auto">仅显示需复核项</label><label><input type="checkbox" id="issues" style="width:auto">只看已选人工标注问题</label><button id="export">保存复核JSON</button><span id="status"></span></header>CARDS
<script>
const manifest=MANIFEST, key='v4-review-'+manifest, cards=[...document.querySelectorAll('article')];
let state=SEED;try{state={...state,...JSON.parse(localStorage.getItem(key)||'{}')}}catch(e){}
function update(){let required=cards.filter(c=>c.dataset.required==='true');document.querySelector('#status').textContent=`需复核 ${required.filter(c=>state[c.dataset.id]?.outcome).length}/${required.length} · 全部项目 ${cards.length}`;cards.forEach(c=>c.hidden=(document.querySelector('#only').checked&&c.dataset.required!=='true')||(document.querySelector('#issues').checked&&state[c.dataset.id]?.outcome!=='annotation_issue'))}
cards.forEach(c=>{const id=c.dataset.id,s=c.querySelector('select'),n=c.querySelector('input');s.value=state[id]?.outcome||'';n.value=state[id]?.note||'';function changed(){state[id]={case_id:id,outcome:s.value,note:n.value};try{localStorage.setItem(key,JSON.stringify(state))}catch(e){}update()}s.onchange=changed;n.oninput=changed});
document.querySelector('#only').onchange=update;document.querySelector('#issues').onchange=update;update();
document.querySelector('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({manifest_sha256:manifest,reviews:Object.values(state).filter(r=>r.outcome)},null,2)],{type:'application/json'}));a.download='v4_manual_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''
    seed = {}
    saved = folder / "v4_manual_review.json"
    if saved.exists():
        payload = json.loads(saved.read_text(encoding="utf-8-sig"))
        if payload["manifest_sha256"] != manifest_sha:
            raise ValueError("Saved review belongs to another case list")
        seed = {r["case_id"]: r for r in payload["reviews"]}
    (folder / "review.html").write_text(page.replace('CARDS', '\n'.join(cards)).replace('MANIFEST', json.dumps(manifest_sha)).replace('SEED', json.dumps(seed, ensure_ascii=False).replace('</', '<\\/')), encoding="utf-8")


def apply_reviews(folder, review_path):
    cases = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    payload = json.loads(review_path.read_text(encoding="utf-8-sig"))
    if payload["manifest_sha256"] != sha(folder / "cases.json"):
        raise ValueError("Review belongs to a different frozen case list")
    indexed = {c["case_id"]: c for c in cases}
    seen = set()
    rows = []
    for r in payload["reviews"]:
        identity = r["case_id"]
        if identity not in indexed or identity in seen:
            raise ValueError("Unknown/duplicate case")
        seen.add(identity)
        c = indexed[identity]
        options = GT_OPTIONS if c["target_index"] is not None else PRED_OPTIONS
        outcome = r["outcome"]
        if outcome not in options or (c["prediction_index"] is None and outcome not in ("missed", "annotation_issue", "uncertain")):
            raise ValueError(f"Invalid outcome: {identity}")
        if outcome in ("size_only", "bad_localization") and c["truth_label"] != c["prediction_label"]:
            raise ValueError(f"Identity differs: {identity}; use wrong_identity or annotation_issue")
        if outcome == "wrong_identity" and c["truth_label"] == c["prediction_label"]:
            raise ValueError(f"Labels agree: {identity}; use annotation_issue/uncertain if needed")
        rows.append({"case_id": identity, "image": c["image"], "outcome": outcome, "note": r.get("note", "")})
    pending = [c["case_id"] for c in cases if c["review_required"] and c["case_id"] not in seen]
    if pending:
        raise ValueError(f"Review unfinished: {pending}")
    counts = {k: sum(r["outcome"] == k for r in rows) for k in sorted(set(GT_OPTIONS) | set(PRED_OPTIONS))}
    summary = {"reviewed_cases": len(rows), "human_outcomes": counts, "unreviewed_standard_correct": sum(not c["review_required"] and c["case_id"] not in seen for c in cases),
               "review_source": str(review_path.resolve()), "review_sha256": sha(review_path),
               "annotations_changed": False, "models_changed": False, "note": "Manual error taxonomy supplements the original IoU metrics; AP is not recalculated or replaced."}
    (folder / "manual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "manual_decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "image", "outcome", "note"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--apply-review", type=Path)
    args = parser.parse_args()
    folder = args.run / "v4_manual_review"
    if args.apply_review:
        apply_reviews(folder, args.apply_review)
        return
    records = json.loads((args.run / "human_truth.json").read_text(encoding="utf-8"))
    predictions_path = args.run / "yolo_day2_v4/prediction_details.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    config = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    for r in records:
        if sha(Path(r["normalized_image"])) != r["image_sha256"]:
            raise ValueError("Source image changed")
    cases = build_cases(records, predictions, config["display_threshold"])
    folder.mkdir(exist_ok=True)
    (folder / "crops").mkdir(exist_ok=True)
    (folder / "full_truth").mkdir(exist_ok=True)
    case_path = folder / "cases.json"
    if case_path.exists() and json.loads(case_path.read_text(encoding="utf-8")) != cases:
        raise ValueError("Existing review roster changed; create a new review version")
    case_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    by_image = {r["image_name"]: r for r in records}
    for r in records:
        draw_full_truth(r, folder / "full_truth" / r["image_name"])
    for c in cases:
        draw_crop(c, folder / "crops" / f"{c['case_id']}.png", by_image[c["image"]])
    make_html(folder, cases, sha(case_path))
    counts = {kind: sum(c["pairing"] == kind for c in cases) for kind in ("strict_match", "geometry_candidate", "no_candidate", "unmatched_prediction")}
    summary = {"cases": len(cases), "required_review": sum(c["review_required"] for c in cases), "pairing_counts": counts,
               "confidence_threshold": config["display_threshold"], "case_manifest_sha256": sha(case_path),
               "prediction_sha256": sha(predictions_path), "notice": "Geometry candidates require human same-person confirmation; no changed detection metrics."}
    (folder / "preflight.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Open in browser: {folder / 'review.html'}")


if __name__ == "__main__":
    main()
