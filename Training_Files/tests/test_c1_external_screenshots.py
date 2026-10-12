import json

import pytest
from c1_detection.external_screenshots import read_annotations


def _record(preview_id=1, image_name="EXT_001.png"):
    return {
        "preview_id": preview_id,
        "image_name": image_name,
        "image_url": f"/data/local-files/?d=c1_annotations/external_screenshots_v1/images/{image_name}",
        "width": 200,
        "height": 100,
    }


def _task(record, result=None, annotations=None, was_cancelled=False):
    if annotations is None:
        annotations = [{
            "was_cancelled": was_cancelled,
            "result": result if result is not None else _result(),
        }]
    return {
        "data": {"image": record["image_url"]},
        "annotations": annotations,
    }


def _result(box=None, choices=None, original_width=200, original_height=100):
    box = box if box is not None else {"x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0}
    choices = choices if choices is not None else ["usable"]
    return [
        {"from_name": "frame_quality", "type": "choices", "to_name": "image", "value": {"choices": choices}},
        {
            "from_name": "label",
            "type": "rectanglelabels",
            "to_name": "image",
            "original_width": original_width,
            "original_height": original_height,
            "image_rotation": 0,
            "value": {**box, "rectanglelabels": ["enemy"]},
        },
    ]


def _export(tmp_path, tasks):
    export = tmp_path / "export.json"
    export.write_text(json.dumps(tasks), encoding="utf-8")
    return export


def test_acceptance_valid_boxes_and_labels(tmp_path):
    record = _record()
    usable, excluded = read_annotations(_export(tmp_path, [_task(record)]), [_record()])
    assert excluded == []
    assert len(usable) == 1
    assert usable[0]["boxes"] == [[20.0, 20.0, 80.0, 60.0]]
    assert usable[0]["labels"] == [1]


def test_missing_annotation_raises(tmp_path):
    record = _record()
    with pytest.raises(ValueError, match="exactly one completed human annotation"):
        read_annotations(_export(tmp_path, [_task(record, annotations=[])]), [_record()])


def test_uncertain_excluded_and_distinct_usable_remains(tmp_path):
    r1 = _record(1, "EXT_001.png")
    r2 = _record(2, "EXT_002.png")
    tasks = [
        _task(r1, result=_result(choices=["uncertain"])),
        _task(r2, result=_result()),
    ]
    usable, excluded = read_annotations(_export(tmp_path, tasks), [_record(1, "EXT_001.png"), _record(2, "EXT_002.png")])
    assert [r["image_name"] for r in excluded] == ["EXT_001.png"]
    assert [r["image_name"] for r in usable] == ["EXT_002.png"]


def test_usable_empty_frame_retained(tmp_path):
    empty = _record(1, "EXT_001.png")
    positive = _record(2, "EXT_002.png")

    def empty_result():
        return [{"from_name": "frame_quality", "type": "choices", "to_name": "image", "value": {"choices": ["usable"]}}]

    tasks = [
        _task(empty, result=empty_result()),
        _task(positive, result=_result()),
    ]
    usable, excluded = read_annotations(
        _export(tmp_path, tasks), [_record(1, "EXT_001.png"), _record(2, "EXT_002.png")]
    )
    assert excluded == []
    assert [r["image_name"] for r in usable] == ["EXT_001.png", "EXT_002.png"]
    assert usable[0]["boxes"] == []
    assert usable[0]["labels"] == []
    assert usable[1]["boxes"] == [[20.0, 20.0, 80.0, 60.0]]


def test_duplicate_tasks_raises(tmp_path):
    record = _record()
    tasks = [_task(record), _task(record)]
    with pytest.raises(ValueError, match="Unknown/duplicate image"):
        read_annotations(_export(tmp_path, tasks), [_record()])


@pytest.mark.parametrize(
    "box",
    [
        {"x": float("nan"), "y": 20, "width": 30, "height": 40, "rotation": 0},
        {"x": 90, "y": 20, "width": 30, "height": 40, "rotation": 0},
    ],
)
def test_malformed_or_out_of_bounds_box_raises(tmp_path, box):
    record = _record()
    with pytest.raises(ValueError, match="Invalid box coordinates"):
        read_annotations(_export(tmp_path, [_task(record, result=_result(box=box))]), [_record()])


def test_wrong_original_dimensions_raises(tmp_path):
    record = _record()
    with pytest.raises(ValueError, match="Annotation dimensions differ"):
        read_annotations(
            _export(tmp_path, [_task(record, result=_result(original_width=100, original_height=50))]),
            [_record()],
        )


def test_missing_task_raises(tmp_path):
    with pytest.raises(ValueError, match="Unannotated/missing tasks"):
        read_annotations(_export(tmp_path, []), [_record()])


def test_object_diagnosis_separates_identity_from_localization():
    import torch
    from compare_c1_external_screenshots import object_diagnosis
    target = {"boxes": torch.tensor([[0., 0., 10., 10.], [30., 0., 40., 10.]]),
              "labels": torch.tensor([1, 2])}
    pred = {"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([2]),
            "scores": torch.tensor([.9])}
    record = {"image_name": "EXT_001.png", "width": 1280, "height": 720, "scene_tags": ["dark"]}
    rows = object_diagnosis(pred, target, record, .05)
    assert [r["error"] for r in rows] == ["wrong_identity", "miss_or_localization"]
    assert rows[0]["height_at_long_edge_1280"] == 10
    assert rows[0]["size_band"] == "small"


def test_report_handles_absent_class_and_writes_size_evidence(tmp_path):
    import torch
    from PIL import Image
    from c1_detection.two_class_metrics import coco_metrics, fixed_metrics
    from compare_c1_external_screenshots import report, object_diagnosis
    target = {"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([1])}
    pred = {**target, "scores": torch.tensor([.9])}
    records = [{"image_name": "EXT_001.png", "preview_id": 1, "width": 1280, "height": 720, "scene_tags": []}]
    summaries, rows = {}, []
    for key in ("fasterrcnn_day3", "yolo_day1_v3", "yolo_day2_v4"):
        folder = tmp_path / key / "rendered"
        folder.mkdir(parents=True)
        Image.new("RGB", (100, 60)).save(folder / "EXT_001.png")
        summaries[key] = {**coco_metrics([pred], [target], [(1280, 720)]),
                          "operating_points": {str(t): fixed_metrics([pred], [target], t) for t in (.05, .25)}}
        rows.extend({"model": key, **r} for r in object_diagnosis(pred, target, records[0], .05))
    report(tmp_path, summaries, records, [], rows, .05)
    sizes = json.loads((tmp_path / "size_metrics.json").read_text(encoding="utf-8"))
    assert sizes["yolo_day2_v4"]["small"]["correct"] == 1
    with Image.open(tmp_path / "compare_001.png") as im:
        assert im.size == (300, 156)
