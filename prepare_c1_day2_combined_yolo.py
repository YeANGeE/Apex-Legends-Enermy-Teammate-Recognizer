"""Mechanical C1 merge; all new real-match frames TRAIN only, old val preserved."""
from __future__ import annotations
import hashlib
import json
import shutil
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"c1_yolo_dataset/day1_v3_enemy_teammate"
DAY2=ROOT/"c1_yolo_dataset/day2_v1_normalized"
REPAIR=ROOT/"c1_annotations/day1_validation_repair_v1/corrected_validation"
OUTPUT=ROOT/"c1_yolo_dataset/day1_day2_v4_real_match"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing dataset: {OUTPUT}")
    old=json.loads((BASE/"source_manifest.json").read_text(encoding="utf-8"))
    new=json.loads((DAY2/"source_manifest.json").read_text(encoding="utf-8"))
    repair=json.loads((REPAIR/"summary.json").read_text(encoding="utf-8"))
    if sha(BASE/"source_manifest.json")!=repair["base_manifest_sha256"]:
        raise ValueError("Corrected validation source manifest changed")
    merged=[]
    for record in old:
        r=record.copy()
        image=BASE/"images"/r["split"]/r["image_name"]
        label=(REPAIR/"labels" if r["split"]=="val" else BASE/"labels"/"train")/image.with_suffix(".txt").name
        if sha(image)!=r["image_sha256"]:
            raise ValueError(f"Day1 image changed: {image}")
        if r["split"]=="val" and sha(label)!=repair["label_sha256"][label.name]:
            raise ValueError(f"Corrected validation changed: {label}")
        r.update(source_image=str(image),source_label=str(label),label_sha256=sha(label),
                 source_domain="day1",validation_annotation="human_repaired_v1" if r["split"]=="val" else None)
        merged.append(r)
    for record in new:
        if not record["eligible"] or record["scene_type"]!="real_match":
            continue
        r=record.copy()
        image=Path(r["normalized_image"]); label=Path(r["normalized_label"])
        if sha(image)!=r["image_sha256"] or sha(label)!=r["label_sha256"]:
            raise ValueError("Day2 normalized content changed")
        r.update(split="train",encounter_id=r["tentative_group"],source_image=str(image),
                 source_label=str(label),source_domain="day2_real_match",
                 annotation_source="day2_completed_manual_annotation")
        merged.append(r)
    groups={s:{r["encounter_id"] for r in merged if r["split"]==s} for s in ("train","val")}
    hashes={s:{r["image_sha256"] for r in merged if r["split"]==s} for s in ("train","val")}
    if groups["train"]&groups["val"] or hashes["train"]&hashes["val"]:
        raise ValueError("Training/validation leakage")
    assert Counter(r["split"] for r in merged)=={"train":830,"val":62}
    assert len({r["image_name"] for r in merged})==len(merged)
    stats=defaultdict(Counter)
    for r in merged:
        split=r["split"]
        image=OUTPUT/"images"/split/r["image_name"]
        label=OUTPUT/"labels"/split/image.with_suffix(".txt").name
        image.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r["source_image"],image); shutil.copy2(r["source_label"],label)
        assert sha(image)==r["image_sha256"] and sha(label)==r["label_sha256"]
        lines=label.read_text().splitlines()
        stats[split]["images"]+=1
        stats[split]["negative_images"]+=int(not lines)
        for cls,name in ((0,"enemy"),(1,"teammate")):
            stats[split][f"{name}_boxes"]+=sum(l.startswith(f"{cls} ") for l in lines)
    summary={"stage":"Stage 1 — Local Combat Policy","task":"C1 Enemy / Teammate Detection",
        "images":{s:stats[s]["images"] for s in stats},"split_statistics":{s:dict(v) for s,v in stats.items()},
        "day1_training_images":717,"day2_real_match_training_images":113,
        "day2_training_range_images_added":0,"test_images_used":0,
        "validation_source":str(REPAIR),"original_validation_image_split_preserved":True,
        "annotation_authority":str(DAY2/"dataset_summary.json"),
        "day2_manifest_sha256":sha(DAY2/"source_manifest.json"),
        "warning":"Day2 is training data, not independent validation/test. Repaired Day1 val was informed by earlier predictions; future fresh-match evaluation required.",
        "training_started":False}
    (OUTPUT/"data.yaml").write_text(f"path: {OUTPUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: enemy\n  1: teammate\n",encoding="utf-8")
    for name,value in (("source_manifest.json",merged),("dataset_summary.json",summary)):
        (OUTPUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
