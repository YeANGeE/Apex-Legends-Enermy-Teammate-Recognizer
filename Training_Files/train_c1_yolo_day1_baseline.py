"""First YOLO26n baseline for C1, Stage 1 — Local Combat Policy.

Run in PyCharm using .venv-yolo. --check-only validates configuration without
downloading weights or starting training. The sealed test set is never used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "c1_yolo_dataset" / "day1_v1_verified" / "data.yaml"

# Explicit settings make the first experiment easy to explain and reproduce.
TRAIN_SETTINGS = {
    "data": str(DATA_YAML),
    "epochs": 60,
    "patience": 20,
    "imgsz": 1280,
    "batch": 32,
    "device": 0,
    "workers": 0,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "amp": True,
    "seed": 20261008,
    "deterministic": True,
    "project": str(ROOT / "runs" / "c1_yolo"),
    "name": "day1_yolo26n_1280_v1",
    "exist_ok": False,
    "val": True,
    "save": True,
    "plots": True,
    # Keep full-frame HUD context and small enemies for the initial baseline.
    # Stronger augmentations can be tested separately after this run.
    "mosaic": 0.0,
    "mixup": 0.0,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.1,
    "fliplr": 0.0,
    "flipud": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.2,
    "hsv_v": 0.2,
}


def check_configuration() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("请在 PyCharm 中选择 .venv-yolo 的 GPU 解释器。")
    get_cfg(overrides={"model": "yolo26n.pt", "task": "detect", **TRAIN_SETTINGS})
    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    if config.get("names") != {0: "enemy"} or config.get("test"):
        raise ValueError("基线必须只有 enemy 类，且不能引用测试集。")
    data_root = Path(config["path"])
    for split, expected in (("train", 478), ("val", 62)):
        images = {p.stem for p in (data_root / config[split]).glob("*.jpg")}
        labels = {p.stem for p in (data_root / "labels" / split).glob("*.txt")}
        if len(images) != expected or images != labels:
            raise ValueError(f"{split} 图片数量或标签配对异常。")
    print("配置检查通过：训练 478 张，验证 62 张，测试集使用 0 张。")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print("模型：YOLO26n，COCO 预训练权重")
    print("输入尺寸：1280；batch：4；最多 60 轮；early stopping：20 轮")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_configuration()
    if args.check_only:
        print("仅检查配置，没有下载权重或启动训练。")
        return

    # The first run downloads official COCO weights into this local directory.
    pretrained_dir = ROOT / "weights" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained_dir / "yolo26n.pt"))
    model.train(**TRAIN_SETTINGS)
    print(f"训练结果目录：{model.trainer.save_dir}")
    print(f"最佳权重：{model.trainer.best}")
    print("下一步：检查验证结果和漏检，再单独测量预热后的推理延迟。")


if __name__ == "__main__":
    main()
