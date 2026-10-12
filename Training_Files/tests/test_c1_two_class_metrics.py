import pytest
import torch

from c1_detection.two_class_metrics import coco_metrics, fixed_metrics


def prediction(boxes, labels, scores):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64), "scores": torch.tensor(scores)}


def test_identity_errors_separated_from_person_recall():
    target = prediction([[0, 0, 10, 10], [20, 0, 30, 10]], [1, 2], [])
    pred = prediction([[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 10]], [2, 2, 1], [.9, .8, .1])
    result = fixed_metrics([pred], [target])
    assert result["person_recall"] == 1
    assert result["identity_accuracy_on_located"] == .5
    assert result["enemy_as_teammate"] == 1
    assert result["unmatched_predictions"] == 1
    assert result["classes"]["enemy"]["tp"] == 1
    assert result["classes"]["teammate"]["fp"] == 1


def test_coco_perfect_two_classes_and_no_predictions():
    target = prediction([[0, 0, 10, 10], [20, 0, 30, 10]], [1, 2], [])
    perfect = prediction([[0, 0, 10, 10], [20, 0, 30, 10]], [1, 2], [.9, .8])
    metrics = coco_metrics([perfect], [target], [(100, 50)])
    assert metrics["map50_95"] == pytest.approx(1)
    assert metrics["per_class_ap"]["teammate"]["map50"] == pytest.approx(1)
    empty = prediction([], [], [])
    assert coco_metrics([empty], [target], [(100, 50)])["map50_95"] == 0


def test_coco_wrong_identity_zero_ap():
    target = prediction([[0, 0, 10, 10], [20, 0, 30, 10]], [1, 2], [])
    wrong = prediction([[0, 0, 10, 10], [20, 0, 30, 10]], [2, 1], [.9, .8])
    assert coco_metrics([wrong], [target], [(100, 50)])["map50"] == 0


def test_empty_frame_false_positive_and_threshold():
    target = prediction([], [], [])
    pred = prediction([[0, 0, 10, 10]], [1], [.1])
    assert fixed_metrics([pred], [target], .05)["classes"]["enemy"]["fp"] == 1
    assert fixed_metrics([pred], [target], .25)["classes"]["enemy"]["fp"] == 0
