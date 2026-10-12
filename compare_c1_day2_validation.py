"""Stage 1 — Local Combat Policy: bounded old/new holdout evaluation.

No training, no services, no checkpoint replacement. Preserve original IDs.
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluate_c1_yolo_teammates import match_predictions, draw_review, NAMES
from evaluate_c1_yolo_day1 import benchmark, font

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"c1_yolo_dataset/day2_validation_v1"
WEIGHTS={"old":ROOT/"runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt",
         "new":ROOT/"runs/c1_yolo/day2_yolo26n_1280_v4_real_match-2/weights/best.pt"}
DISPLAY={"old":"旧模型 Day1 V3", "new":"新模型 Day2 V4"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def evaluate(key, records, output, yaml):
    folder=output/key
    rendered=folder/"rendered"
    rendered.mkdir(parents=True)
    model=YOLO(str(WEIGHTS[key]))
    if model.names!={0:"enemy",1:"teammate"}:
        raise ValueError(f"Wrong checkpoint classes: {model.names}")
    model.predict(np.zeros((1080,1920,3),dtype=np.uint8),imgsz=1280,rect=False,
                  device=0,half=True,conf=.05,iou=.5,max_det=100,verbose=False)
    counts={cls:Counter(tp=0,fp=0,fn=0) for cls in NAMES}
    diagnostics=Counter(teammate_as_enemy=0,enemy_as_teammate=0,unmatched_predictions=0,unmatched_targets=0)
    rows,details=[],[]
    for record in records:
        source=Path(record["normalized_image"])
        targets=[{"class":b["class_id"],"box":[(b["xywh"][0]-b["xywh"][2]/2)*1920,
            (b["xywh"][1]-b["xywh"][3]/2)*1080,(b["xywh"][0]+b["xywh"][2]/2)*1920,
            (b["xywh"][1]+b["xywh"][3]/2)*1080]} for b in record["boxes"]]
        result=model.predict(str(source),imgsz=1280,rect=False,device=0,half=True,
                             conf=.05,iou=.5,max_det=100,verbose=False)[0]
        predictions=[{"box":b,"score":s,"class":int(c)} for b,s,c in zip(
            result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
        correct,wrong,missed=match_predictions(predictions,targets)
        identifier=record["preview_id"]
        row=dict(validation_id=identifier,gallery_id=identifier,image_name=source.name,correct=len(correct),
                 wrong_identity=len(wrong),unmatched_predictions=len(predictions)-len(correct)-len(wrong),unmatched_targets=len(missed))
        for cls in NAMES:
            tp=sum(predictions[i]["class"]==cls for i in correct)
            counts[cls].update(tp=tp,fp=sum(p["class"]==cls for p in predictions)-tp,
                               fn=sum(t["class"]==cls for t in targets)-tp)
        for pi,ti in wrong.items():
            diagnostics["teammate_as_enemy" if targets[ti]["class"]==1 else "enemy_as_teammate"]+=1
        for name in ("unmatched_predictions","unmatched_targets"):
            diagnostics[name]+=row[name]
        draw_review(source,rendered/source.name,targets,predictions,correct,wrong,row,.05)
        rows.append(row)
        details.append({**row,"targets":targets,"predictions":predictions,"correct_matches":correct,
                        "wrong_identity_matches":wrong,"unmatched_target_indices":missed})
    latency=benchmark(model)
    class_metrics={NAMES[c]:dict(**v,precision=v["tp"]/max(1,v["tp"]+v["fp"]),
                        recall=v["tp"]/max(1,v["tp"]+v["fn"])) for c,v in counts.items()}
    # AP integrates a confidence sweep; it is not the fixed .05 operating point.
    ap_model=YOLO(str(WEIGHTS[key]))
    val=ap_model.val(data=str(yaml),split="val",imgsz=1280,batch=4,rect=False,
                    device=0,half=True,conf=.001,iou=.5,max_det=100,workers=0,
                    plots=False,save=False,project=str(folder),name="ap_validation",verbose=False)
    summary=dict(checkpoint=str(WEIGHTS[key]),checkpoint_sha256=digest(WEIGHTS[key]),
                 images=len(records),confidence_threshold=.05,matching_iou=.5,
                 class_metrics=class_metrics,identity_diagnostics=dict(diagnostics),latency=latency,
                 map50=float(val.box.map50),map50_95=float(val.box.map),
                 ap_confidence_floor=.001,ap_note="AP sweeps confidence; not precision/recall at .05",
                 training_started=False)
    save_json(folder/"summary.json",summary)
    save_json(folder/"prediction_details.json",details)
    with (folder/"image_review.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary,rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-yolo and CUDA")
    manifest=DATA/"source_manifest.json"
    records=json.loads(manifest.read_text(encoding="utf-8"))
    records=sorted((r for r in records if r["eligible"] and r["scene_type"]=="real_match"),key=lambda r:r["preview_id"])
    if not records or any(r["split"]!="comparison_holdout" for r in records):
        raise ValueError("Invalid holdout")
    training_manifest=ROOT/"c1_yolo_dataset/day1_day2_v4_real_match/source_manifest.json"
    old=json.loads(training_manifest.read_text(encoding="utf-8"))
    old_hashes={r["image_sha256"] for r in old}
    for r in records:
        if digest(Path(r["normalized_image"]))!=r["image_sha256"] or digest(Path(r["normalized_label"]))!=r["label_sha256"]:
            raise ValueError("Holdout content changed")
        if r["image_sha256"] in old_hashes:
            raise ValueError("Holdout overlaps prior training/validation images")
    output=ROOT/"runs/c1_yolo"/("day2_holdout_comparison_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    yaml=output/"holdout.yaml"
    # Ultralytics schema requires train even for val-only; no train() called.
    yaml.write_text(f'path: {DATA.as_posix()}\ntrain: real_match/images\nval: real_match/images\nnames:\n  0: enemy\n  1: teammate\n',encoding="utf-8")
    summaries,rows={},{}
    for key in WEIGHTS:
        print(f"评估 {DISPLAY[key]}",flush=True)
        summaries[key],rows[key]=evaluate(key,records,output,yaml)
    save_json(output/"comparison.json",dict(models=summaries,holdout_manifest_sha256=digest(manifest),
        prior_dataset_manifest_sha256=digest(training_manifest),preview_ids=[r["preview_id"] for r in records],
        training_changed=False,small_or_blurry_targets_removed=False,
        limitation="Single new recording session, correlated frames; not broad independent match benchmark"))
    sheet_links=[]
    for offset in range(0,len(records),8):
        sheet=Image.new("RGB",(7680,4720),"#10161e")
        draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[offset:offset+8]):
            for k,key in enumerate(WEIGHTS):
                x=(j%2*2+k)*1920;y=j//2*1180
                with Image.open(output/key/"rendered"/r["image_name"]) as im:
                    sheet.paste(im.convert("RGB"),(x,y))
                row=rows[key][offset+j]
                draw.text((x+12,y+1095),f"编号{r['preview_id']:02d} {DISPLAY[key]} 正确{row['correct']} 认反{row['wrong_identity']} 未匹配预测{row['unmatched_predictions']}",font=font(34),fill="white")
        path=output/f"对照_{offset//8+1:02d}.jpg"
        sheet.save(path,quality=97,subsampling=0)
        sheet_links.append(f"- [对照 {offset//8+1:02d}]({path.as_posix()})")
    lines=["# 新录制验证图：前后模型对比","","同一图片、阈值0.05、匹配IoU≥0.5、1280、FP16。每组左旧右新；图片编号沿用标注项目。","",
           "| 指标 | 旧 Day1 V3 | 新 Day2 V4 |","|---|---:|---:|"]
    for cls in NAMES.values():
        for name,label in (("tp","正确框"),("fp","误检/多框/定位不合格"),("fn","漏检或认错"),("precision","准确率"),("recall","召回率")):
            values=[summaries[k]["class_metrics"][cls][name] for k in WEIGHTS]
            formatted=[f"{v:.1%}" if name in {"precision","recall"} else str(v) for v in values]
            lines.append(f"| {cls} {label} | {formatted[0]} | {formatted[1]} |")
    for metric in ("map50","map50_95"):
        lines.append(f"| {metric} | {summaries['old'][metric]:.3f} | {summaries['new'][metric]:.3f} |")
    lines+= ["","AP为置信度扫描指标，不是0.05固定阈值的准确率。认反计入预测类别FP及真实类别FN；未匹配预测可能是重复框、框位置偏差或背景误检。模糊小目标未额外排除。","",
        "## 高清对照图（7680×4720）","",*sheet_links,"",
        "青色正确，紫色敌我认反，红色未匹配预测，绿色正确匹配人工框，黄色未正确检出人工框。","",
        "## 单张原尺寸图","",f"[旧模型目录]({(output/'old/rendered').as_posix()}) · [新模型目录]({(output/'new/rendered').as_posix()})"]
    (output/"对比报告.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"结果目录：{output}",flush=True)


if __name__=="__main__":
    main()
