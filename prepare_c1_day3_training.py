"""Mechanical merge: all usable Day3 images train; old validation untouched."""
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"c1_yolo_dataset/day1_day2_v4_real_match"
NEW=ROOT/"c1_yolo_dataset/day3_v1_normalized"
OUTPUT=ROOT/"c1_yolo_dataset/day3_v4_targeted_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserving prepared training version")
    old=json.loads((BASE/"source_manifest.json").read_text(encoding="utf-8"))
    fresh=json.loads((NEW/"source_manifest.json").read_text(encoding="utf-8"))
    rows=[]
    for r in old:
        if r["split"] not in ("train","val"):
            raise ValueError("Unexpected baseline split")
        image=BASE/"images"/r["split"]/r["image_name"]
        label=BASE/"labels"/r["split"]/(Path(r["image_name"]).stem+".txt")
        if sha(image)!=r["image_sha256"] or sha(label)!=r["label_sha256"]:
            raise ValueError("Baseline data changed")
        rows.append({**r,"source_image":str(image),"source_label":str(label)})
    for r in fresh:
        if not r["eligible"]:
            continue
        image,label=Path(r["normalized_image"]),Path(r["normalized_label"])
        if sha(image)!=r["image_sha256"] or sha(label)!=r["label_sha256"]:
            raise ValueError("Normalized Day3 changed")
        rows.append({**r,"split":"train","encounter_id":"day3_match_boundaries_unconfirmed_all_train",
                     "source_domain":"day3_real_match","source_image":str(image),"source_label":str(label),
                     "scene_type":"real_match","annotation_source":"day3_completed_manual_annotation"})
    if Counter(r["split"] for r in old)!={"train":830,"val":62} or len(rows)!=1446:
        raise ValueError("Unexpected baseline/new roster")
    names=[r["image_name"] for r in rows]
    if len(names)!=len(set(names)):
        raise ValueError("Filename collision")
    hashes={s:{r["image_sha256"] for r in rows if r["split"]==s} for s in ("train","val")}
    pixels={s:{r["pixel_sha256"] for r in rows if r["split"]==s and r.get("pixel_sha256")} for s in ("train","val")}
    groups={s:{r["encounter_id"] for r in rows if r["split"]==s} for s in ("train","val")}
    if hashes["train"]&hashes["val"] or pixels["train"]&pixels["val"] or groups["train"]&groups["val"]:
        raise ValueError("Exact content/source group overlap with validation")
    OUTPUT.mkdir(parents=True)
    stats=defaultdict(Counter)
    for r in rows:
        image=OUTPUT/"images"/r["split"]/r["image_name"]
        label=OUTPUT/"labels"/r["split"]/(Path(r["image_name"]).stem+".txt")
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r["source_image"],image);shutil.copy2(r["source_label"],label)
        if sha(image)!=r["image_sha256"] or sha(label)!=r["label_sha256"]:
            raise ValueError("Copy verification failed")
        lines=label.read_text(encoding="utf-8").splitlines()
        stats[r["split"]].update(images=1,negative_images=int(not lines),
            enemy_boxes=sum(line.startswith("0 ") for line in lines),teammate_boxes=sum(line.startswith("1 ") for line in lines))
        r.update(normalized_image=str(image),normalized_label=str(label))
    summary={"stage":"Stage 1 — Local Combat Policy","images":dict(Counter(r["split"] for r in rows)),
             "split_statistics":{k:dict(v) for k,v in stats.items()},"day3_training_images_added":554,"test_images_used":0,
             "baseline_manifest_sha256":sha(BASE/"source_manifest.json"),"day3_manifest_sha256":sha(NEW/"source_manifest.json"),
             "all_new_usable_images_train":True,"original_validation_preserved":True,"training_started":False,
             "warning":"62 old validation images monitor fitting; no fresh independent Day3 test. Unknown multi-match boundaries do not cross splits because all Day3 is train. New match required for promotion."}
    (OUTPUT/"data.yaml").write_text(f"path: {OUTPUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: enemy\n  1: teammate\n",encoding="utf-8")
    for name,data in (("source_manifest.json",rows),("dataset_summary.json",summary)):
        (OUTPUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
