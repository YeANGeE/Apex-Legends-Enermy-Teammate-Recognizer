"""User-run three-model external diagnosis, from full Label Studio JSON; no training."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from c1_detection.external_screenshots import DATA, ROOT, TAGS, read_annotations, sha
from c1_detection.two_class_metrics import coco_metrics, fixed_metrics
from compare_c1_day3_fasterrcnn_yolo import DEFAULT_FRCNN, DISPLAY, YOLO_WEIGHTS, Predictor, benchmark_predictor, review_image


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_images(records):
    """Exact-file and exact-pixel overlap checks; source independence still needs review."""
    known = set()
    for manifest in (ROOT / "c1_yolo_dataset").glob("*/source_manifest.json"):
        for r in json.loads(manifest.read_text(encoding="utf-8")):
            known.update(v for v in (r.get("image_sha256"), r.get("pixel_sha256")) if v)
    for r in records:
        path = Path(r["normalized_image"])
        path.resolve().relative_to((DATA / "images").resolve())
        if sha(path) != r["image_sha256"]:
            raise ValueError(f"Frozen image changed: {path}")
        with Image.open(path) as im:
            if im.size != (r["width"], r["height"]):
                raise ValueError(f"Image dimensions changed: {path}")
            pixels = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        if pixels != r["pixel_sha256"] or pixels in known or r["image_sha256"] in known:
            raise ValueError(f"Image changed or overlaps previous dataset: {path}")


def object_diagnosis(pred, target, record, threshold):
    keep = pred["scores"] >= threshold
    boxes, labels, scores = pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep]
    overlap = box_iou(boxes, target["boxes"])
    remaining, matches = set(range(len(target["boxes"]))), {}
    for i in scores.argsort(descending=True).tolist():
        candidates = [j for j in remaining if overlap[i, j] >= .5]
        if candidates:
            j = max(candidates, key=lambda j: float(overlap[i, j]))
            remaining.remove(j)
            matches[j] = i
    rows = []
    for j, (box, label) in enumerate(zip(target["boxes"].tolist(), target["labels"].tolist(), strict=True)):
        i = matches.get(j)
        error = "miss_or_localization" if i is None else "correct" if int(labels[i]) == label else "wrong_identity"
        height1280 = (box[3]-box[1])*1280/max(record["width"], record["height"])
        rows.append({"image": record["image_name"], "target_index": j, "truth": "enemy" if label == 1 else "teammate",
                     "error": error, "height_at_long_edge_1280": height1280,
                     "size_band": "small" if height1280 < 32 else "medium" if height1280 < 96 else "large",
                     "scene_tags": ";".join(record["scene_tags"]), "threshold": threshold})
    return rows


def report(output, summaries, records, excluded, all_rows, threshold):
    keys = list(summaries)
    lines = [f"# 外部截图 {len(keys)} 模型诊断", "", f"可用 {len(records)} 张；不确定排除 {len(excluded)} 张。",
             "", "|指标|"+"|".join(DISPLAY[k] for k in keys)+"|", "|---|"+"---:|"*len(keys)]
    def row(title, values):
        lines.append("|"+title+"|"+"|".join("未定义" if v is None or v == -1 else f"{v:.4f}" for v in values)+"|")
    for metric in ("map50", "map50_95"):
        row(metric, [summaries[k][metric] for k in keys])
    for t in ("0.05", "0.25"):
        for cls, name in (("enemy", "敌人"), ("teammate", "队友")):
            for metric, title in (("precision", "精确率"), ("recall", "召回率")):
                row(f"{name}{title} conf={t}", [summaries[k]["operating_points"][t]["classes"][cls][metric] for k in keys])
        for metric, title in (("person_recall", "人物召回率"), ("identity_accuracy_on_located", "已定位人物身份准确率"),
                              ("teammate_as_enemy", "队友认成敌人数量"), ("enemy_as_teammate", "敌人认成队友数量"),
                              ("unmatched_predictions", "未匹配预测数量")):
            row(f"{title} conf={t}", [summaries[k]["operating_points"][t][metric] for k in keys])
    lines += ["", "未匹配预测包括背景误检、重复框和框偏差；漏定位包含IoU不足。没有类别实例或预测时，相应精确率/召回率未定义。",
              "这是选取的网络截图诊断集，来源可能相关，不能推断总体胜率或跟踪质量。用于迭代后不再是封存测试集。",
              "尺寸分组按原图长边缩放至1280后的身体框高度：小<32，中32–96，大≥96像素；不等同于COCO面积分组。", ""]
    diagnosis_key = keys[-1]
    lines.append(f"## {DISPLAY[diagnosis_key]} 待核查补强方向")
    v4 = [r for r in all_rows if r["model"] == diagnosis_key]
    for error, suggestion in (("miss_or_localization", "补充漏定位条件的数据，核查小目标、遮挡和身体框一致性"),
                              ("wrong_identity", "补充敌我难例，复查姓名标记与角色外观的混淆")):
        failed = [r for r in v4 if r["error"] == error]
        ids = sorted({r["image"] for r in failed})
        lines.append(f"- {suggestion}：{len(failed)} 个目标；图片 {', '.join(ids) or '无'}。")
    lines.append("- 背景负例是否需要加强：结合V4逐图CSV中的未匹配预测和原尺寸标框图，人工区分背景、重复框及定位误差后决定。")
    lines.append("- scene_metrics.json列出场景分组及实例数；标签可重叠，小样本差异只作为线索。低阈值改善召回但增加多框时，先判断置信度问题，再决定是否补数据。")
    lines += ["", "|模型|尺寸组|人工人物数|正确|漏定位/框偏差|敌我认反|", "|---|---|---:|---:|---:|---:|"]
    size_groups = {}
    for key in keys:
        size_groups[key] = {}
        for band in ("small", "medium", "large"):
            subset = [r for r in all_rows if r["model"] == key and r["size_band"] == band]
            counts = {error: sum(r["error"] == error for r in subset) for error in ("correct", "miss_or_localization", "wrong_identity")}
            size_groups[key][band] = {"targets": len(subset), **counts}
            lines.append(f"|{DISPLAY[key]}|{band}|{len(subset)}|{counts['correct']}|{counts['miss_or_localization']}|{counts['wrong_identity']}|")
    save(output / "size_metrics.json", size_groups)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 46)
    header_height = 96
    for r in records:
        paths = [output / key / "rendered" / r["image_name"] for key in keys]
        sizes = []
        for path in paths:
            with Image.open(path) as im:
                sizes.append(im.size)
        sheet = Image.new("RGB", (sum(w for w, h in sizes), max(h for w, h in sizes)+header_height), "#111820")
        draw = ImageDraw.Draw(sheet)
        x = 0
        for key, path in zip(keys, paths, strict=True):
            with Image.open(path) as im:
                tile = im.convert("RGB")
                sheet.paste(tile, (x, header_height))
                x_next = x+tile.width
            draw.text((x+16, 16), DISPLAY[key], font=font, fill="white")
            x = x_next
        path = output / f"compare_{r['preview_id']:03d}.png"
        sheet.save(path, compress_level=3)
        print(f"原尺寸拼接：{path.name} {sheet.width}×{sheet.height}", flush=True)
    lines += ["", f"可视化阈值：{threshold}。compare_*.png为原尺寸无损拼接，无缩放；顶部额外添加96像素标题栏。请以100%缩放查看细小人物。"]
    (output / "对比报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(yolo_only=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=DATA / "annotations.json")
    parser.add_argument("--check-only", action="store_true", help="Audit annotation/data/weight paths; no inference")
    parser.add_argument("--benchmark", action="store_true", help="Optional user-run predictor latency benchmark")
    parser.add_argument("--display-threshold", type=float, choices=(.05, .25), default=.05)
    parser.add_argument("--fasterrcnn-weights", type=Path, default=DEFAULT_FRCNN)
    parser.add_argument("--rebuild-visuals", type=Path, help="Rebuild full-resolution composites from a completed run, without inference")
    args = parser.parse_args()
    if args.rebuild_visuals:
        output = args.rebuild_visuals.resolve()
        summaries = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        records = json.loads((output / "human_truth.json").read_text(encoding="utf-8"))
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        with (output / "object_diagnosis.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        report(output, summaries, records, config["excluded"], rows, config["display_threshold"])
        print(f"拼接重建完成：{output}", flush=True)
        return
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 27 or len({r["pixel_sha256"] for r in manifest}) != 27:
        raise ValueError("Unexpected frozen roster")
    audit_images(manifest)
    records, excluded = read_annotations(args.export, manifest)
    weights = dict(YOLO_WEIGHTS) if yolo_only else {"fasterrcnn_day3": args.fasterrcnn_weights, **YOLO_WEIGHTS}
    hashes = {key: sha(path) for key, path in weights.items()}
    print(f"标注检查通过：可用{len(records)}，不确定{len(excluded)}，人物{sum(len(r['boxes']) for r in records)}。", flush=True)
    if args.check_only:
        print(f"图片、完整人工标注与{len(weights)}个权重路径检查通过；尚未运行模型。")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("请使用 .venv-yolo CUDA解释器")
    output = ROOT / "runs/c1_detection" / ("external_screenshots_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    save(output / "run_config.json", {"stage": "Stage 1 — Local Combat Policy", "manifest_sha256": sha(DATA/"manifest.json"),
         "export_sha256": sha(args.export), "weights": {k: str(p) for k, p in weights.items()}, "weight_sha256": hashes,
         "excluded": excluded, "training_started": False, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
         "display_threshold": args.display_threshold, "benchmark": args.benchmark})
    save(output / "human_truth.json", records)
    targets = [{"boxes": torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(r["labels"], dtype=torch.int64)} for r in records]
    sizes = [(r["width"], r["height"]) for r in records]
    summaries, all_rows = {}, []
    for key, path in weights.items():
        print(f"开始：{DISPLAY[key]}", flush=True)
        predictor = Predictor(key, path)
        folder = output / key
        (folder / "rendered").mkdir(parents=True)
        predictions, review = [], []
        for r, target in zip(records, targets, strict=True):
            with Image.open(r["normalized_image"]) as im:
                rgb = np.array(im.convert("RGB"))
            pred = predictor(rgb, .001)
            predictions.append(pred)
            review.append(review_image(r, target, pred, folder/"rendered", args.display_threshold))
            all_rows.extend({"model": key, **row} for row in object_diagnosis(pred, target, r, args.display_threshold))
            print(f"  {r['image_name']}", flush=True)
        summaries[key] = {**coco_metrics(predictions, targets, sizes),
                          "operating_points": {str(t): fixed_metrics(predictions, targets, t) for t in (.05, .25)}}
        for metrics in summaries[key]["operating_points"].values():
            for cls in metrics["classes"].values():
                if cls["tp"]+cls["fp"] == 0:
                    cls["precision"] = None
                if cls["tp"]+cls["fn"] == 0:
                    cls["recall"] = None
        scene = {}
        for tag in sorted(TAGS):
            indices = [i for i, r in enumerate(records) if tag in r["scene_tags"]]
            if indices:
                scene[tag] = {"images": len(indices), **fixed_metrics([predictions[i] for i in indices], [targets[i] for i in indices], args.display_threshold)}
        save(folder / "scene_metrics.json", scene)
        save(folder / "prediction_details.json", [{"image": r["image_name"], **{k: v.tolist() for k, v in p.items()}} for r, p in zip(records, predictions, strict=True)])
        if args.benchmark:
            rgbs = []
            for r in records:
                with Image.open(r["normalized_image"]) as im:
                    rgbs.append(np.array(im.convert("RGB")))
            summaries[key]["predictor_latency"] = benchmark_predictor(predictor, rgbs)
            del rgbs
        save(folder / "summary.json", summaries[key])
        with (folder / "image_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    with (output / "object_diagnosis.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    save(output / "comparison.json", summaries)
    report(output, summaries, records, excluded, all_rows, args.display_threshold)
    print(f"结果：{output}\n报告：{output / '对比报告.md'}", flush=True)


if __name__ == "__main__":
    main()
