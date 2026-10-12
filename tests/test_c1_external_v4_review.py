import json
import pytest
from review_c1_external_v4 import build_cases, apply_reviews, sha


def sample():
    records = [{"image_name": "EXT_001.png", "normalized_image": "unused.png", "width": 100, "height": 100,
                "boxes": [[0, 0, 10, 10]], "labels": [1], "scene_tags": []}]
    predictions = [{"image": "EXT_001.png", "boxes": [[0, 0, 10, 30]], "labels": [1], "scores": [.8]}]
    return records, predictions


def test_box_extent_candidate_is_not_automatically_accepted():
    records, predictions = sample()
    cases = build_cases(records, predictions, .05)
    assert len(cases) == 1
    assert cases[0]["pairing"] == "geometry_candidate"
    assert cases[0]["review_required"]
    assert cases[0]["iou"] == pytest.approx(1/3)


def test_no_prediction_and_unmatched_predictions_remain_separate():
    records, predictions = sample()
    predictions[0]["boxes"] = [[50, 50, 60, 60]]
    cases = build_cases(records, predictions, .05)
    assert [c["pairing"] for c in cases] == ["no_candidate", "unmatched_prediction"]


def test_high_confidence_identity_conflict_requires_review():
    records, predictions = sample()
    predictions[0]["boxes"] = [[0, 0, 10, 10], [0, 0, 10, 10]]
    predictions[0]["labels"] = [2, 1]
    predictions[0]["scores"] = [.8, .2]
    cases = build_cases(records, predictions, .05)
    assert cases[0]["prediction_label"] == 2
    assert cases[0]["review_required"]
    assert cases[1]["pairing"] == "unmatched_prediction"


def test_manual_summary_rejects_unfinished_or_wrong_case_identity(tmp_path):
    records, predictions = sample()
    cases = build_cases(records, predictions, .05)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"manifest_sha256": sha(path), "reviews": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unfinished"):
        apply_reviews(tmp_path, review)
    review.write_text(json.dumps({"manifest_sha256": sha(path), "reviews": [{"case_id": cases[0]["case_id"], "outcome": "size_only"}]}), encoding="utf-8")
    apply_reviews(tmp_path, review)
    assert json.loads((tmp_path / "manual_summary.json").read_text(encoding="utf-8"))["human_outcomes"]["size_only"] == 1
    review.write_text(json.dumps({"manifest_sha256": "wrong", "reviews": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="different frozen"):
        apply_reviews(tmp_path, review)
