"""Stage 1 — Local Combat Policy: supervised two-class visual fine-tuning.

Run yourself in PyCharm with .venv-yolo; --check-only never starts training.
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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_yolo_dataset/day1_v3_enemy_teammate/data.yaml"
WEIGHTS = ROOT / "runs/c1_yolo/day1_yolo26n_1280_v2_expanded/weights/best.pt"
SETTINGS = {**TRAIN_SETTINGS, "data": str(DATA), "batch": 4, "epochs": 60,
            "patience": 20, "lr0": 0.0003, "resume": False,
            "name": "day1_yolo26n_1280_v3_enemy_teammate"}


def check_configuration() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("请选择 .venv-yolo 的 GPU 解释器。")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    get_cfg(overrides={"model": str(WEIGHTS), "task": "detect", **SETTINGS})
    config = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    root = Path(config["path"])
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if config.get("names") != {0: "enemy", 1: "teammate"} or config.get("test") or summary["test_images_used"] != 0:
        raise ValueError("只允许敌人/队友两类及训练/验证集。")
    expected = {r["image_name"]: r for r in manifest}
    if len(expected) != len(manifest):
        raise ValueError("来源清单存在重复。")
    encounters = {s: set() for s in ("train", "val")}
    for split in encounters:
        images = {p.stem: p for p in (root / config[split]).glob("*.jpg")}
        labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
        if len(images) != summary["images"][split] or images.keys() != labels.keys():
            raise ValueError(f"{split} 图片/标签配对异常。")
        if {p.name for p in images.values()} != {n for n, r in expected.items() if r["split"] == split}:
            raise ValueError("图片划分与来源清单不一致。")
        for stem, image in images.items():
            record = expected[image.name]
            encounters[split].add(record["encounter_id"])
            if hashlib.sha256(image.read_bytes()).hexdigest() != record["image_sha256"]:
                raise ValueError(f"图片被修改：{image}")
            for line in labels[stem].read_text().splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] not in ("0", "1"):
                    raise ValueError(f"类别或标签格式异常：{labels[stem]}")
                x, y, w, h = map(float, fields[1:])
                if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-8 or max(x+w/2, y+h/2) > 1+1e-8:
                    raise ValueError(f"框坐标异常：{labels[stem]}")
        if summary["split_statistics"][split].get("teammate_boxes", 0) == 0:
            raise ValueError(f"{split} 缺少队友标注。")
    if encounters["train"] & encounters["val"]:
        raise ValueError("训练/验证存在交战片段泄漏。")
    print(f"配置通过：训练 {summary['images']['train']} 张；验证 {summary['images']['val']} 张；未使用测试集。")
    print(f"GPU：{torch.cuda.get_device_name(0)}；类别：0=敌人，1=队友。")
    print(f"起点：{WEIGHTS}")
    print("batch=4，imgsz=1280，AdamW，lr0=0.0003，最多60轮，patience=20。")
    print("新建两类模型并迁移兼容权重；不是原训练的 resume，优化器及轮数重新开始。")
    print("验证图片划分保留，但真值增加了队友；两类总体mAP不能直接与旧单类mAP比较。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_configuration()
    if args.check_only:
        print("仅检查配置，没有启动训练。")
        return
    model = YOLO(str(WEIGHTS))
    model.train(**SETTINGS)
    print(f"结果目录：{model.trainer.save_dir}")
    print(f"最佳权重：{model.trainer.best}")


if __name__ == "__main__":
    main()
