"""Measure the latency/accuracy trade-off from reducing C1 RPN proposals.

The checkpoint is read-only, validation is the only split used, and all trials
keep the current FP32 precision and 800/1333 transform so proposal count is the
only experimental variable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from benchmark_c1_precision_resolution import benchmark_configuration
from c1_detection import C1CocoDetectionDataset, build_c1_detector


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "c1_annotations" / "day1_full_v1" / "normalized"
CHECKPOINT = ROOT / "runs" / "c1_detection" / "baseline_day1_full_v1" / "best.pt"
OUTPUT = ROOT / "runs" / "c1_detection" / "rpn_proposal_benchmark"

# pre-NMS is the number retained before overlap suppression; post-NMS is the
# number passed to ROI Align and the second-stage classifier/regressor.
CONFIGURATIONS = (
    {
        "name": "baseline_1000",
        "min_size": 800,
        "max_size": 1333,
        "dtype": "fp32",
        "rpn_pre_nms": 1000,
        "rpn_post_nms": 1000,
        "detections_per_img": 100,
    },
    {
        "name": "proposals_600_300",
        "min_size": 800,
        "max_size": 1333,
        "dtype": "fp32",
        "rpn_pre_nms": 600,
        "rpn_post_nms": 300,
        "detections_per_img": 50,
    },
    {
        "name": "proposals_300_150",
        "min_size": 800,
        "max_size": 1333,
        "dtype": "fp32",
        "rpn_pre_nms": 300,
        "rpn_post_nms": 150,
        "detections_per_img": 20,
    },
    {
        "name": "proposals_150_100",
        "min_size": 800,
        "max_size": 1333,
        "dtype": "fp32",
        "rpn_pre_nms": 150,
        "rpn_post_nms": 100,
        "detections_per_img": 20,
    },
    {
        "name": "proposals_100_50",
        "min_size": 800,
        "max_size": 1333,
        "dtype": "fp32",
        "rpn_pre_nms": 100,
        "rpn_post_nms": 50,
        "detections_per_img": 10,
    },
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请选择 Python 3.14 + cu132 解释器。")

    device = torch.device("cuda:0")
    dataset = C1CocoDetectionDataset(
        DATA / "images", DATA / "annotations" / "instances_val.json"
    )
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_c1_detector(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation images: {len(dataset)} | test split: unused")
    print("Fixed variables: FP32, min_size=800, max_size=1333\n")

    rows: list[dict] = []
    for config in CONFIGURATIONS:
        print(
            f"[{config['name']}] RPN "
            f"{config['rpn_pre_nms']} -> {config['rpn_post_nms']}"
        )
        row = benchmark_configuration(model, dataset, device, config)
        rows.append(row)
        print(
            f"  median={row['latency_median_ms']:.2f} ms "
            f"p95={row['latency_p95_ms']:.2f} ms "
            f"mAP50={row['mAP_50']:.4f} "
            f"mAP50:95={row['mAP_50_95']:.4f}\n"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": "Stage 1 — Local Combat Policy",
        "task": "C1 RPN proposal latency benchmark",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_images": len(dataset),
        "test_images_used": 0,
        "fixed_precision": "fp32",
        "fixed_transform": {"min_size": 800, "max_size": 1333},
        "results": rows,
    }
    json_path = OUTPUT / "results.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = OUTPUT / "results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
