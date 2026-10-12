"""User-run V4 targeted fine-tuning, Stage 1 — Local Combat Policy. --check-only never trains."""
import argparse
import json
import math
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from normalize_c1_day3_export import sha
from train_c1_yolo_day1_baseline import TRAIN_SETTINGS

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"c1_yolo_dataset/day3_v4_targeted_v1"
WEIGHTS=ROOT/"runs/c1_yolo/day2_yolo26n_1280_v4_real_match-2/weights/best.pt"
SETTINGS={**TRAIN_SETTINGS,"data":str(DATA/"data.yaml"),"batch":4,"lr0":.0001,"epochs":30,"patience":10,
          "resume":False,"name":"day3_yolo26n_1280_v4_targeted_v1"}


def check_configuration(weights=None, initial_classes="fps", expected_sha="5a051aac818c4fb7b9eb0fc3f0f210d1924c1e33f8fdd5b818656e2788f0f5bc", expected_counts=None):
    weights = WEIGHTS if weights is None else weights
    if not torch.cuda.is_available():
        raise RuntimeError("请使用 .venv-yolo GPU解释器")
    get_cfg(overrides={"model":str(weights),"task":"detect",**SETTINGS})
    config=yaml.safe_load((DATA/"data.yaml").read_text(encoding="utf-8"))
    summary=json.loads((DATA/"dataset_summary.json").read_text(encoding="utf-8"))
    rows=json.loads((DATA/"source_manifest.json").read_text(encoding="utf-8"))
    expected_counts={"train":1384,"val":62} if expected_counts is None else expected_counts
    if config.get("names")!={0:"enemy",1:"teammate"} or config.get("test") or summary["images"]!=expected_counts:
        raise ValueError("Dataset configuration differs")
    if sha(weights)!=expected_sha:
        raise ValueError("Initialization checkpoint changed")
    for split in ("train","val"):
        subset=[r for r in rows if r["split"]==split]
        images={p.name for p in (DATA/"images"/split).iterdir() if p.suffix.lower() in (".jpg",".png")}
        labels={p.stem for p in (DATA/"labels"/split).glob("*.txt")}
        if images!={r["image_name"] for r in subset} or labels!={Path(n).stem for n in images}:
            raise ValueError("Image/label/manifest roster differs")
        for r in subset:
            image=DATA/"images"/split/r["image_name"]
            label=DATA/"labels"/split/(image.stem+".txt")
            if sha(image)!=r["image_sha256"] or sha(label)!=r["label_sha256"]:
                raise ValueError("Training data changed")
            for line in label.read_text(encoding="utf-8").splitlines():
                fields=line.split()
                if len(fields)!=5 or fields[0] not in ("0","1"):
                    raise ValueError("Invalid detection label")
                x,y,w,h=map(float,fields[1:])
                if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-8 or max(x+w/2,y+h/2)>1+1e-8:
                    raise ValueError("Invalid normalized coordinates")
    model=YOLO(str(weights))
    if initial_classes=="fps":
        valid=model.names=={0:"enemy",1:"teammate"}
    elif initial_classes=="coco":
        valid=len(model.names)==80 and model.names.get(0)=="person"
    else:
        valid=False
    if not valid or model.task!="detect":
        raise ValueError("Initialization classes differ")
    print(f"检查通过：{expected_counts['train']}张训练，{expected_counts['val']}张旧验证；初始化权重/标签/配置一致。起点={weights}",flush=True)
    print(f"imgsz={SETTINGS['imgsz']} batch={SETTINGS['batch']} {SETTINGS['optimizer']} lr0={SETTINGS['lr0']} 最多{SETTINGS['epochs']}轮 patience={SETTINGS['patience']}；旧验证仅监测训练，独立实战验收另做。",flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check-only",action="store_true");args=parser.parse_args()
    check_configuration()
    if args.check_only:
        print("仅检查，未训练。",flush=True);return
    model=YOLO(str(WEIGHTS))
    model.train(**SETTINGS)
    receipt={"initial_checkpoint":str(WEIGHTS),"initial_checkpoint_sha256":sha(WEIGHTS),
             "manifest_sha256":sha(DATA/"source_manifest.json"),"settings":SETTINGS,
             "promotion_status":"unpromoted; requires fresh match evaluation"}
    (Path(model.trainer.save_dir)/"data_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"候选权重：{model.trainer.best}\n结果：{model.trainer.save_dir}",flush=True)


if __name__=="__main__":
    main()
