"""Evidence-based V4 diagnosis from completed manual review and saved predictions."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from review_c1_external_v4 import DEFAULT_RUN, NAMES, apply_reviews, overlap, sha


def size_band(height):
    return "small" if height < 32 else "medium" if height < 96 else "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run / "v4_manual_review"
    apply_reviews(folder, args.review)
    payload = json.loads(args.review.read_text(encoding="utf-8-sig"))
    cases = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    indexed = {c["case_id"]: c for c in cases}
    predictions = {p["image"]: p for p in json.loads((args.run / "yolo_day2_v4/prediction_details.json").read_text(encoding="utf-8"))}
    outcomes = Counter(r["outcome"] for r in payload["reviews"])
    rows = []
    for review in payload["reviews"]:
        c = indexed[review["case_id"]]
        row = {"case_id": c["case_id"], "image": c["image"], "outcome": review["outcome"],
               "case_type": "target" if c["target_index"] is not None else "extra_prediction",
               "truth_class": NAMES.get(c["truth_label"], ""), "prediction_class": NAMES.get(c["prediction_label"], ""),
               "scene_tags": ";".join(c["scene_tags"]), "height1280": "", "size_band": "",
               "best_saved_iou": "", "best_saved_confidence": "", "best_saved_class": "",
               "low_score_iou50_candidate": False, "note": review.get("note", "")}
        if c["truth_box"] is not None:
            box = c["truth_box"]
            height = (box[3]-box[1])*1280/max(c["width"], c["height"])
            row.update(height1280=height, size_band=size_band(height))
            p = predictions[c["image"]]
            candidates = [(overlap(box, b)[0], s, int(label)) for b, s, label in zip(p["boxes"], p["scores"], p["labels"], strict=True)]
            if candidates:
                iou, score, label = max(candidates)
                row.update(best_saved_iou=iou, best_saved_confidence=score, best_saved_class=NAMES[label])
                row["low_score_iou50_candidate"] = any(i >= .5 and s < .05 for i, s, label in candidates)
        rows.append(row)
    missed = [r for r in rows if r["outcome"] == "missed"]
    support = Counter()
    for c in cases:
        if c["truth_box"] is not None:
            b = c["truth_box"]
            support[size_band((b[3]-b[1])*1280/max(c["width"], c["height"]))] += 1
    groups = {band: {"all_targets": support[band], "human_missed": sum(r["size_band"] == band for r in missed)} for band in ("small", "medium", "large")}
    summary = {"review_source": str(args.review.resolve()), "review_sha256": sha(args.review),
               "threshold": .05, "reviewed_cases": len(rows), "counts": dict(outcomes),
               "missed_classes": dict(Counter(r["truth_class"] for r in missed)), "missed_size_groups": groups,
               "missed_with_low_score_iou50_candidate": sum(r["low_score_iou50_candidate"] for r in missed),
               "note": "Targets and extra predictions are separate units; extra boxes may refer to the same person. Low-score geometry is a hypothesis, not recovered detections."}
    (folder / "v4_diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "v4_evidence.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# V4 人工复核与补强优先级", "", "范围：27张外部截图，44个人工人物目标，conf=0.05；41项疑点已人工复核。",
             "其余22项为标准匹配且身份一致，未额外人工复核。原始AP与匹配阈值保持不变。", "",
             "|复核类别|数量|统计单位|", "|---|---:|---|"]
    for key, title, unit in (("missed", "漏检", "人工目标"), ("wrong_identity", "认反", "人工目标"),
                             ("size_only", "仅框松紧不同、身份正确", "人工目标"), ("background", "背景误检", "额外预测框"),
                             ("existing_person_correct", "已标人物、身份正确", "额外预测框"),
                             ("existing_person_wrong_identity", "已标人物、认反", "额外预测框"),
                             ("annotation_issue", "人工标注问题", "复核项目")):
        lines.append(f"|{title}|{outcomes[key]}|{unit}|")
    lines += ["", "额外预测框可能覆盖已经计数的人物，不能加到人物召回分母或分子，也不能把6个身份正确框当成6个新人物。",
              "是否为重复框需继续依据一人多框关系判断；本次未自动归为重复。", "",
              "|1280长边下身体高度|全部人工目标|人工确认漏检|", "|---|---:|---:|"]
    for band, title in (("small", "<32像素"), ("medium", "32–96像素"), ("large", "≥96像素")):
        lines.append(f"|{title}|{support[band]}|{groups[band]['human_missed']}|")
    low = summary["missed_with_low_score_iou50_candidate"]
    lines += ["", f"16个漏检中，{low}个存在IoU≥0.5、分数低于0.05的保存预测。身份未必正确，且几何候选可能共享人物；这不等于降低阈值即可找回{low}个人。",
              "保存预测只覆盖conf≥0.001及原来的后处理，无法区分更低分数、NMS抑制与完全未产生目标框。", "",
              "## 下一步优先级", "",
              "1. 优先补小到中等身体尺寸的实战人物，覆盖遮挡、开镜和效果干扰。先独立采集新素材；保证同一人物框规则一致，不省略小人物。",
              f"2. 对{low}个有低分几何候选的漏检，分别检查低置信度、类别分数与定位；当前不直接降低部署阈值。需要用独立视频比较召回收益与背景误检增加。",
              f"3. 将{outcomes['background']}个背景误检作为难负例线索，寻找包含相同干扰物的新素材。负例中如还有真实人物，仍完整标人物，不能整图做空标签。",
              "4. 补充明确的敌我混淆难例：同一人物附近可能同时输出不同类别；保留真实角色外观、姓名标记和遮挡关系，不凭缺少ID判敌人。",
              "5. 框松紧问题优先统一可见身体规则，再决定是否需要训练调整。已确认仅大小不同的2个人工目标不作为分类失败。", "",
              "这批截图保留为诊断集。若纳入训练，后续泛化结论必须来自新独立素材；本步骤没有改变训练数据或启动训练。", "",
              "## 逐项证据", "", "|案例|人工结论|身体高度1280|最佳保存IoU|对应置信度|对应类别|", "|---|---|---:|---:|---:|---|"]
    for r in rows:
        if r["case_type"] == "target":
            def cell(v):
                return f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            lines.append(f"|{r['case_id']}|{r['outcome']}|{cell(r['height1280'])}|{cell(r['best_saved_iou'])}|{cell(r['best_saved_confidence'])}|{r['best_saved_class']}|")
    (folder / "V4复核结论与补强计划.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {folder / 'V4复核结论与补强计划.md'}")


if __name__ == "__main__":
    main()
