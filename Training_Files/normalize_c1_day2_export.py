"""Stage 1 — Local Combat Policy: normalize human Day2 annotations.

JSON is authoritative. Label Studio YOLO class IDs may include quality/scene
choices, so their IDs must never be copied blindly into the two-class dataset.
No train/val split assigned here; domains and excluded records remain separate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

from PIL import Image

ROOT=Path(__file__).resolve().parent
EXPORT=ROOT/"c1_yolo_export/DAY2/enermy_teammate"
ROSTER=ROOT/"c1_annotations/day2_enemy_teammate_v1/label_studio_tasks.json"
OUTPUT=ROOT/"c1_yolo_dataset/day2_v1_normalized"
IMAGE_ROOT=ROOT/"c1_expansion/day2_preview_v1/images"
SPLIT="unassigned"
CLASSES={"enemy":0,"teammate":1}


def parse_annotation(task):
    annotations=[a for a in task.get("annotations",[]) if not a.get("was_cancelled")]
    if len(annotations)!=1:
        raise ValueError(f"Expected one completed annotation: {task.get('id')}")
    annotation=annotations[0]
    results=annotation["result"]
    choices={}
    for r in results:
        if r.get("type")=="choices":
            key=r["from_name"]
            if key in choices or len(r["value"]["choices"])!=1:
                raise ValueError(f"Ambiguous choices: {task['id']}")
            choices[key]=r["value"]["choices"][0]
    quality=choices.get("frame_quality")
    scene=choices.get("scene_type")
    if quality not in {"usable","uncertain","discard"}:
        raise ValueError("Invalid frame quality")
    if scene not in {"real_match","training_range","uncertain"}:
        raise ValueError("Invalid scene type")
    if quality=="usable" and scene=="uncertain":
        raise ValueError("Usable image requires a resolved scene")
    boxes=[]
    for r in results:
        if r.get("type")!="rectanglelabels":
            continue
        v=r["value"]
        labels=v["rectanglelabels"]
        if len(labels)!=1 or labels[0] not in CLASSES:
            raise ValueError(f"Unsupported box class: {labels}")
        if (r["original_width"],r["original_height"])!=(1920,1080):
            raise ValueError("Box/source dimensions disagree")
        if r.get("image_rotation",0)!=0 or v.get("rotation",0)!=0:
            raise ValueError("Rotated boxes are unsupported")
        x,y,w,h=(float(v[k]) for k in ("x","y","width","height"))
        if not all(math.isfinite(n) for n in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or max(x+w,y+h)>100.000001:
            raise ValueError(f"Invalid box geometry: {task['id']}")
        boxes.append((labels[0],((x+w/2)/100,(y+h/2)/100,w/100,h/100)))
    return quality,scene,boxes,annotation


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing normalized data: {OUTPUT}")
    jsons=list(EXPORT.glob("*.json")); zips=list(EXPORT.glob("*.zip"))
    if len(jsons)!=1 or len(zips)!=1:
        raise ValueError("Expected exactly one JSON and one ZIP export")
    roster=json.loads(ROSTER.read_text(encoding="utf-8"))
    expected={t["meta"]["preview_id"]:t for t in roster}
    tasks=json.loads(jsons[0].read_text(encoding="utf-8-sig"))
    if len(tasks)!=len(expected) or {t["meta"]["preview_id"] for t in tasks}!=set(expected):
        raise ValueError("Export task roster is incomplete or duplicated")
    records=[]
    stats=defaultdict(Counter)
    # Complete read-only validation before writing any normalized output.
    with ZipFile(zips[0]) as archive:
        class_names=archive.read("classes.txt").decode("utf-8-sig").splitlines()
        images=[n for n in archive.namelist() if n.startswith("images/") and not n.endswith("/")]
        if len(images)!=len(tasks):
            raise ValueError("Unexpected ZIP image count")
        for task in sorted(tasks,key=lambda t:t["meta"]["preview_id"]):
            identifier=task["meta"]["preview_id"]
            original=expected[identifier]
            if task["data"]!=original["data"] or task["meta"]!=original["meta"]:
                raise ValueError(f"Source data/metadata changed: {identifier}")
            relative=parse_qs(urlsplit(task["data"]["image"]).query)["d"][0]
            source=(ROOT/relative).resolve()
            source.relative_to(IMAGE_ROOT.resolve())
            with Image.open(source) as im:
                if im.size!=(1920,1080) or hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()!=task["meta"]["pixel_sha256"]:
                    raise ValueError(f"Source pixels changed: {identifier}")
            matches=[n for n in images if Path(n).name.endswith(source.name)]
            if len(matches)!=1 or archive.read(matches[0])!=source.read_bytes():
                raise ValueError(f"ZIP/source mismatch: {identifier}")
            quality,scene,boxes,annotation=parse_annotation(task)
            zip_label="labels/"+Path(matches[0]).stem+".txt"
            exported=[]
            if zip_label in archive.namelist():
                for line in archive.read(zip_label).decode("utf-8-sig").splitlines():
                    fields=line.split()
                    if len(fields)!=5:
                        raise ValueError("Invalid exported YOLO row")
                    cls=int(fields[0])
                    if not 0<=cls<len(class_names) or class_names[cls] not in CLASSES:
                        raise ValueError("Export contains a non-detection box class")
                    exported.append((class_names[cls],tuple(map(float,fields[1:]))))
            if len(exported)!=len(boxes):
                raise ValueError(f"JSON/ZIP box count differs: {identifier}")
            # Match unordered boxes: export formatting may reorder predictions.
            remaining=exported.copy()
            for name,coordinates in boxes:
                match=next((i for i,(n,c) in enumerate(remaining) if n==name
                    and max(abs(a-b) for a,b in zip(c,coordinates))<=1e-5),None)
                if match is None:
                    raise ValueError(f"JSON/ZIP coordinates differ: {identifier}")
                remaining.pop(match)
            bucket=scene if quality=="usable" else quality
            stats[bucket]["images"]+=1
            stats[bucket]["empty_annotation_images"]+=int(not boxes)
            stats[bucket]["negative_images"]+=int(not boxes and quality=="usable")
            for name in CLASSES:
                stats[bucket][f"{name}_boxes"]+=sum(n==name for n,_ in boxes)
            records.append({**task["meta"],"quality":quality,"scene_type":scene,
                "source_image":str(source),"image_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "annotation_id":annotation["id"],"annotation_updated_at":annotation["updated_at"],
                "boxes":[{"class_name":n,"class_id":CLASSES[n],"xywh":list(c)} for n,c in boxes],
                "eligible":quality=="usable","split":SPLIT})
    OUTPUT.mkdir(parents=True)
    for record in records:
        if not record["eligible"]:
            continue
        domain=record["scene_type"]
        image=OUTPUT/domain/"images"/record["image_name"]
        label=OUTPUT/domain/"labels"/Path(record["image_name"]).with_suffix(".txt")
        image.parent.mkdir(parents=True,exist_ok=True)
        label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(record["source_image"],image)
        lines=[str(b["class_id"])+" "+" ".join(f"{v:.10f}" for v in b["xywh"]) for b in record["boxes"]]
        label.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
        record["normalized_image"]=str(image)
        record["normalized_label"]=str(label)
        record["label_sha256"]=hashlib.sha256(label.read_bytes()).hexdigest()
    summary={"stage":"Stage 1 — Local Combat Policy","task":"C1 Enemy / Teammate Detection",
        "annotation_authority":str(jsons[0]),"export_sha256":hashlib.sha256(jsons[0].read_bytes()).hexdigest(),
        "zip_sha256":hashlib.sha256(zips[0].read_bytes()).hexdigest(),"tasks":len(records),
        "statistics":{k:dict(v) for k,v in stats.items()},"normalized_classes":CLASSES,
        "original_zip_classes":class_names,"split":SPLIT,
        "policy":"Only usable records retained. Training range and real match kept separate; uncertain/discard never become negatives.",
        "excluded_preview_ids":[r["preview_id"] for r in records if not r["eligible"]],
        "source_recordings_and_day1_unchanged":True,"training_started":False}
    for name,value in (("dataset_summary.json",summary),("source_manifest.json",records)):
        (OUTPUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
