"""Stage 1 — Local Combat Policy: user-run Day3 two-class Faster R-CNN.

--check-only audits data/environment. --smoke-test does one backward pass,
without optimizer updates. Normal execution starts a new isolated run.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights

from c1_detection.dataset import detection_collate
from c1_detection.model import build_c1_detector
from c1_detection.two_class_dataset import NAMES, YoloTwoClassDataset, audit_training_data, digest
from c1_detection.two_class_metrics import evaluate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_yolo_dataset/day1_day2_v4_real_match"
SETTINGS = {"epochs": 40, "patience": 12, "batch": 2, "lr": .0005,
            "momentum": .9, "weight_decay": .0005, "min_size": 720, "max_size": 1280,
            "seed": 20261009, "workers": 0, "dtype": "float32", "warmup_steps": 200,
            "lr_milestones": [24, 32], "lr_gamma": .1,
            "max_detections": 100, "augmentation": "none", "gradient_clip": 10.0}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_model(pretrained=True):
    model = build_c1_detector(pretrained=pretrained, foreground_classes=2)
    model.transform.min_size = (SETTINGS["min_size"],)
    model.transform.max_size = SETTINGS["max_size"]
    model.roi_heads.detections_per_img = SETTINGS["max_detections"]
    model.roi_heads.nms_thresh = .5
    return model


def check_configuration():
    if not torch.cuda.is_available():
        raise RuntimeError("请选择 .venv-yolo/Scripts/python.exe，CUDA当前不可用。")
    audit = audit_training_data(DATA)
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not cache.is_file():
        raise FileNotFoundError(f"COCO预训练权重尚未缓存：{cache}")
    audit["pretrained_sha256"] = digest(cache)
    train, val = YoloTwoClassDataset(DATA, "train"), YoloTwoClassDataset(DATA, "val")
    print(f"检查通过：训练{len(train)}张，验证{len(val)}张；GPU {torch.cuda.get_device_name(0)}", flush=True)
    print("标签映射：YOLO 0/1 → Faster R-CNN 1/2（敌人/队友）；0为背景。")
    print("COCO人物权重初始化两个前景类别；FP32、batch=2、SGD lr=0.0005。")
    print("长边1280（1080p画面缩至720×1280后内部补齐）；最多40轮，12轮无提升早停。")
    print("验证62张用于选权重，已受先前预测辅助修正；独立holdout留待训练后验收。")
    return audit, train, val


def smoke_test(train):
    model = build_model().cuda().train()
    positive = next(i for i, t in enumerate(train.targets) if (t["labels"] == 2).any())
    negative = next(i for i, t in enumerate(train.targets) if not len(t["boxes"]))
    images, targets = detection_collate([train[positive], train[negative]])
    images = [im.cuda() for im in images]
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    losses = model(images, targets)
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"冒烟检查损失异常：{losses}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    print("冒烟检查通过：双类别正样本+空标注负样本，前向/反向均有限；未更新权重。")
    print({k: float(v.detach()) for k, v in losses.items()}, "gradient_norm", float(norm))
    del images, targets, losses, loss
    model.eval()
    with torch.inference_mode():
        output = model([train[positive][0].cuda()])[0]
    if not set(output["labels"].tolist()).issubset(NAMES) or not torch.isfinite(output["boxes"]).all():
        raise RuntimeError("推理输出类别或坐标异常")
    print("推理接口检查通过。")


def write_history(output, history):
    save_json(output / "history.json", history)
    rows = []
    for record in history:
        fixed = record["metrics"]["operating_points"]["0.05"]
        row = {k: record[k] for k in ("epoch", "lr", "train_loss", "epoch_seconds")}
        row.update(map50=record["metrics"]["map50"], map50_95=record["metrics"]["map50_95"],
                   person_recall=fixed["person_recall"], identity_accuracy=fixed["identity_accuracy_on_located"])
        for name, values in fixed["classes"].items():
            row.update({f"{name}_{k}": v for k, v in values.items()})
        row.update(record["loss_components"])
        rows.append(row)
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in rows]
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train loss")
    axes[1].plot(epochs, [r["map50"] for r in rows], label="val mAP50")
    axes[1].plot(epochs, [r["map50_95"] for r in rows], label="val mAP50-95")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "results.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--resume", type=Path, help="显式指定本脚本生成的last.pt")
    args = parser.parse_args()
    random.seed(SETTINGS["seed"])
    np.random.seed(SETTINGS["seed"])
    torch.manual_seed(SETTINGS["seed"])
    torch.cuda.manual_seed_all(SETTINGS["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    audit, train, val = check_configuration()
    if args.check_only:
        model = build_model()
        assert model.roi_heads.box_predictor.cls_score.out_features == 3
        print("模型三类输出检查通过；检查结束。")
        return
    if args.smoke_test:
        smoke_test(train)
        return
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (checkpoint.get("experiment") != "day3_fasterrcnn_enemy_teammate" or
                checkpoint["settings"] != SETTINGS or checkpoint["data_audit"] != audit):
            raise ValueError("续训模型、配置或数据版本不一致")
        output = args.resume.resolve().parent
    else:
        output = ROOT / "runs/c1_detection" / ("day3_fasterrcnn_enemy_teammate_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    model = build_model().cuda()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                               lr=SETTINGS["lr"], momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=SETTINGS["lr_milestones"], gamma=.1)
    generator = torch.Generator().manual_seed(SETTINGS["seed"])
    history, best_map, best_epoch, first_epoch = [], -1.0, 0, 1
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best_map, best_epoch = checkpoint["history"], checkpoint["best_map"], checkpoint["best_epoch"]
        first_epoch = checkpoint["epoch"]+1
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        generator.set_state(checkpoint["rng"]["loader"])
    loader = DataLoader(train, batch_size=SETTINGS["batch"], shuffle=True, num_workers=0,
                        pin_memory=True, collate_fn=detection_collate, generator=generator)
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    if not checkpoint:
        save_json(output / "run_config.json", {"stage": "Stage 1 — Local Combat Policy",
            "experiment": "day3_fasterrcnn_enemy_teammate", "settings": SETTINGS,
            "classes": {0: "background", **NAMES}, "data": str(DATA), "data_audit": audit,
            "initialization": "COCO person classifier/regressor copied to enemy and teammate",
            "torch": torch.__version__, "torchvision": torchvision.__version__, "gpu": torch.cuda.get_device_name(0),
            "comparison_note": "Same YOLO dataset/split; architecture, optimizer and augmentation differ."})
    print(f"结果目录：{output}", flush=True)
    stopped_early = False
    for epoch in range(first_epoch, SETTINGS["epochs"]+1):
        if epoch-best_epoch > SETTINGS["patience"] and best_epoch:
            stopped_early = True
            break
        epoch_start = time.perf_counter()
        model.train()
        totals, samples = {}, 0
        scheduled_lr = optimizer.param_groups[0]["lr"]
        for step, (images, targets) in enumerate(loader, 1):
            if epoch == 1:
                factor = .1+.9*min(step/SETTINGS["warmup_steps"], 1)
                for group in optimizer.param_groups:
                    group["lr"] = scheduled_lr*factor
            images = [im.cuda(non_blocking=True) for im in images]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                save_json(output / "failure.json", {"epoch": epoch, "step": step,
                    "image_ids": [int(t["image_id"]) for t in targets],
                    "losses": {k: str(float(v.detach())) for k, v in losses.items()}})
                raise FloatingPointError("损失非有限，已停止；查看failure.json，不会静默跳过样本。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            size = len(images)
            samples += size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0)+float(value.detach())*size
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f"轮次{epoch:02d}/{SETTINGS['epochs']} 批次{step}/{len(loader)} loss={float(loss.detach()):.4f}", flush=True)
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr
        metrics = evaluate(model, val, torch.device("cuda:0"))
        components = {k: v/samples for k, v in totals.items()}
        history.append({"epoch": epoch, "lr": scheduled_lr, "train_loss": sum(components.values()),
                        "loss_components": components, "metrics": metrics, "epoch_seconds": time.perf_counter()-epoch_start})
        improved = metrics["map50_95"] > best_map
        if improved:
            best_map, best_epoch = metrics["map50_95"], epoch
        scheduler.step()
        payload = {"experiment": "day3_fasterrcnn_enemy_teammate", "stage": "Stage 1 — Local Combat Policy",
            "model_name": "fasterrcnn_mobilenet_v3_large_fpn_small_anchors", "num_classes": 3,
            "classes": {0: "background", **NAMES}, "settings": SETTINGS, "data_audit": audit,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "metrics": metrics, "history": history, "best_map": best_map, "best_epoch": best_epoch,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(), "loader": generator.get_state()}}
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        write_history(output, history)
        fixed = metrics["operating_points"]["0.05"]
        print(f"轮次{epoch:02d} 验证 mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f} 最佳轮次={best_epoch}", flush=True)
        for name, label in (("enemy", "敌人"), ("teammate", "队友")):
            values = fixed["classes"][name]
            print(f"  {label} 阈值0.05 P={values['precision']:.3f} R={values['recall']:.3f}", flush=True)
    save_json(output / "summary.json", {"stage": "Stage 1 — Local Combat Policy", "best_epoch": best_epoch,
        "best_map50_95": best_map, "completed_epochs": history[-1]["epoch"] if history else 0,
        "stopped_early": stopped_early, "test_images_used": 0, "holdout_used": False,
        "best_checkpoint": str(output / "best.pt"), "warning": audit["warning"]})
    print(f"训练结束，最佳权重：{output / 'best.pt'}；下一步在独立holdout上验收。", flush=True)


if __name__ == "__main__":
    main()
