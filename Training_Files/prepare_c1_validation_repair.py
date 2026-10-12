"""Prepare ten human-reviewed validation tasks, never training examples."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "c1_annotations/day1_validation_repair_v1"
REVIEW = ROOT / "runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/review_two_class_20261008_181006_922722"


def main():
    roster = json.loads((ROOT / "c1_annotations/day1_enemy_teammate_v1/label_studio_tasks.json").read_text(encoding="utf-8"))
    feedback = json.loads((REVIEW / "human_review.json").read_text(encoding="utf-8"))["feedback"]
    data = ROOT / "c1_yolo_dataset/day1_v3_enemy_teammate"
    tasks = []
    for row in feedback:
        task = copy.deepcopy(roster[row["gallery_id"]-1])
        assert task["meta"]["split"] == "val"
        task["meta"].update(gallery_id=row["gallery_id"], validation_id=row["validation_id"],
                            human_feedback=row["user_feedback"], review_status="validation_truth_repair",
                            validation_only=True)
        results = []
        source = data / "labels/val" / Path(task["meta"]["image_name"]).with_suffix(".txt")
        for index, line in enumerate(source.read_text().splitlines()):
            cls, cx, cy, w, h = map(float, line.split())
            results.append({"id": f"existing{index}", "from_name": "label", "to_name": "image",
                            "type": "rectanglelabels", "original_width": 1920,
                            "original_height": 1080, "image_rotation": 0,
                            "value": {"x": (cx-w/2)*100, "y": (cy-h/2)*100,
                                      "width": w*100, "height": h*100, "rotation": 0,
                                      "rectanglelabels": ["enemy" if cls == 0 else "teammate"]}})
        results.append({"id": "quality", "from_name": "frame_quality", "to_name": "image",
                        "type": "choices", "value": {"choices": ["usable"]}})
        task["predictions"] = [{"model_version": "existing_human_validation_labels_not_model_predictions", "result": results}]
        tasks.append(task)
    assert len(tasks) == 10
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("label_studio_tasks.json", tasks), ("review_mapping.json", feedback)):
        path = OUTPUT / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Refusing to overwrite different existing review: {path}")
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(tasks)} validation-only tasks; existing human labels prefilled, model boxes not used as truth.")


if __name__ == "__main__":
    main()
