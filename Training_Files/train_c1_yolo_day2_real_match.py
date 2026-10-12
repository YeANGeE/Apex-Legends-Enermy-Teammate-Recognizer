"""Stage 1 — Local Combat Policy / C1 visual supervised fine-tuning.

User-run PyCharm entry point. --check-only cannot start training or a service.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from train_c1_yolo_day1_baseline import TRAIN_SETTINGS

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"c1_yolo_dataset/day1_day2_v4_real_match/data.yaml"
WEIGHTS=ROOT/"runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt"
SETTINGS={**TRAIN_SETTINGS,"data":str(DATA),"batch":4,"lr0":0.0003,"epochs":60,
          "patience":20,"resume":False,"name":"day2_yolo26n_1280_v4_real_match"}


def check_configuration():
    if not torch.cuda.is_available() or not WEIGHTS.is_file():
        raise RuntimeError("需要 .venv-yolo GPU 环境及上一轮两类 best.pt。")
    get_cfg(overrides={"model":str(WEIGHTS),"task":"detect",**SETTINGS})
    config=yaml.safe_load(DATA.read_text(encoding="utf-8"))
    if config.get("names")!={0:"enemy",1:"teammate"} or config.get("test"):
        raise ValueError("类别必须为敌人/队友，不能使用封存测试集")
    root=Path(config["path"])
    summary=json.loads((root/"dataset_summary.json").read_text(encoding="utf-8"))
    records=json.loads((root/"source_manifest.json").read_text(encoding="utf-8"))
    assert summary["images"]=={"train":830,"val":62} and summary["test_images_used"]==0
    groups={s:set() for s in ("train","val")}
    hashes={s:set() for s in groups}
    for split in groups:
        expected={r["image_name"]:r for r in records if r["split"]==split}
        images={p.name:p for p in (root/config[split]).glob("*.jpg")}
        labels={p.stem:p for p in (root/"labels"/split).glob("*.txt")}
        if images.keys()!=expected.keys() or {p.stem for p in images.values()}!=labels.keys():
            raise ValueError("图片/标签/来源清单不一致")
        for name,image in images.items():
            r=expected[name]; label=labels[image.stem]
            if hashlib.sha256(image.read_bytes()).hexdigest()!=r["image_sha256"] or hashlib.sha256(label.read_bytes()).hexdigest()!=r["label_sha256"]:
                raise ValueError(f"数据被修改：{name}")
            groups[split].add(r["encounter_id"]); hashes[split].add(r["image_sha256"])
            if r["source_domain"]=="day2_real_match" and (r["quality"]!="usable" or r["scene_type"]!="real_match" or split!="train"):
                raise ValueError("Day2 质量/场景过滤不一致")
            for line in label.read_text().splitlines():
                fields=line.split()
                if len(fields)!=5 or fields[0] not in ("0","1"):
                    raise ValueError("YOLO 类别或行格式错误")
                x,y,w,h=map(float,fields[1:])
                if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-8 or max(x+w/2,y+h/2)>1+1e-8:
                    raise ValueError("YOLO 框坐标无效")
    if groups["train"]&groups["val"] or hashes["train"]&hashes["val"]:
        raise ValueError("训练验证存在交叉")
    model=YOLO(str(WEIGHTS))
    if model.names!={0:"enemy",1:"teammate"}:
        raise ValueError("起点权重不是敌人/队友两类")
    print("检查通过：830张训练（717旧图+113实战新图），62张已修正验证图；封存测试集未使用。")
    print("训练场图未加入本轮；不确定/不保留未作为负样本。")
    print("起点：上一轮两类 best.pt；新一轮微调，非 resume。")
    print("batch=4，imgsz=1280，AdamW，lr0=0.0003，最多60轮，patience=20。")
    print("这是敌人/队友视觉检测训练，不是动作训练或强化学习。")
    print("注意：Day2本批全用于训练；泛化须另用新交战验收。")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    check_configuration()
    if args.check_only:
        print("仅检查，未启动训练或服务。")
        return
    model=YOLO(str(WEIGHTS)); model.train(**SETTINGS)
    print(f"结果目录：{model.trainer.save_dir}")
    print(f"最佳权重：{model.trainer.best}")


if __name__=="__main__":
    main()
