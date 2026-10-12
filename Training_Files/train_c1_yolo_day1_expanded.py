"""Stage 1 — Local Combat Policy: supervised C1 fine-tuning, not a resume.

Run yourself in PyCharm with .venv-yolo. --check-only never starts training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from train_c1_yolo_day1_baseline import TRAIN_SETTINGS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_yolo_dataset/day1_v2_expanded/data.yaml"
WEIGHTS = ROOT / "runs/c1_yolo/day1_yolo26n_1280_v1/weights/best.pt"
# Preserve baseline augmentations; use a lower learning rate to refine weights.
SETTINGS = {
    **TRAIN_SETTINGS,
    "data": str(DATA), "batch": 4, "epochs": 60, "patience": 20,
    "lr0": 0.0003, "name": "day1_yolo26n_1280_v2_expanded",
    "resume": False,
}


def check_configuration() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("请选择 .venv-yolo 的 GPU 解释器。")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    get_cfg(overrides={"model": str(WEIGHTS), "task": "detect", **SETTINGS})
    config = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    root = Path(config["path"])
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    if config.get("names") != {0: "enemy"} or config.get("test") or summary["test_images_used"] != 0:
        raise ValueError("只能使用 enemy 单类及训练/验证集。")
    for split in ("train", "val"):
        images = {p.stem for p in (root / config[split]).glob("*.jpg")}
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        if len(images) != summary["images"][split] or images != labels:
            raise ValueError(f"{split} 图片/标签配对异常。")
    frozen = ROOT / "c1_yolo_dataset/day1_v1_verified"
    for folder, pattern in (("images", "*.jpg"), ("labels", "*.txt")):
        for source in (frozen / folder / "val").glob(pattern):
            if source.read_bytes() != (root / folder / "val" / source.name).read_bytes():
                raise ValueError("验证集发生变化。")
    print(f"配置检查通过：训练 {summary['images']['train']} 张，验证 62 张；测试集未使用。")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"起点：第一次训练的最佳权重 {WEIGHTS}")
    print("batch=4，imgsz=1280，AdamW，lr0=0.0003，最多60轮，patience=20。")
    print("这是新数据上的微调：epoch重新计数，优化器重新初始化，保留原模型参数。")


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
    print(f"训练结果目录：{model.trainer.save_dir}")
    print(f"最佳权重：{model.trainer.best}")


if __name__ == "__main__":
    main()
