"""Bounded Label Studio project import; does NOT start a service."""
import json
import prepare_c1_day2_annotation as annotation


if __name__ == "__main__":
    annotation.PREVIEW = annotation.ROOT / "c1_expansion/day2_validation_v1"
    summary = json.loads((annotation.PREVIEW / "summary.json").read_text(encoding="utf-8"))
    annotation.TASK_COUNT = summary["images"]
    annotation.OUTPUT = annotation.ROOT / "c1_annotations/day2_validation_v1"
    annotation.TITLE = f"DAY2 独立对比验证 V1 — {annotation.TASK_COUNT}张"
    annotation.DATASET = "day2_validation_v1"
    annotation.SPLIT = "comparison_holdout"
    annotation.DESCRIPTION = "新录制独立验证图；仅比较新旧模型，不参与训练。完整标记敌人和队友，无模型预标注。"
    annotation.main()
