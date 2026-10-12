import numpy as np
import pytest
import torch

import compare_c1_day3_fasterrcnn_yolo as comparison


def test_prediction_identity_mapping_and_absolute_coordinates():
    boxes = torch.tensor([[1., 2., 20., 30.], [40., 50., 60., 80.]])
    scores = torch.tensor([.8, .2])
    result = comparison.normalize_prediction(boxes, scores, torch.tensor([0., 1.]), yolo=True)
    assert result["labels"].tolist() == [1, 2]
    assert torch.equal(result["boxes"], boxes)
    assert result["boxes"].device.type == "cpu"
    frc = comparison.normalize_prediction(boxes, scores, torch.tensor([1, 2]))
    assert torch.equal(frc["labels"], result["labels"])


def test_empty_predictions_preserve_shapes():
    result = comparison.normalize_prediction(torch.empty(0, 4), torch.empty(0), torch.empty(0), yolo=True)
    assert result["boxes"].shape == (0, 4)
    assert result["labels"].dtype == torch.int64


def test_yolo_predict_uses_quantize16_and_bgr_input():
    from types import SimpleNamespace
    calls = []
    class FakeYOLO:
        def predict(self, source, **kwargs):
            calls.append((source, kwargs))
            boxes = SimpleNamespace(xyxy=torch.tensor([[1., 2., 5., 6.]]),
                                    conf=torch.tensor([.7]), cls=torch.tensor([0.]))
            return [SimpleNamespace(boxes=boxes)]
    predictor = comparison.Predictor.__new__(comparison.Predictor)
    predictor.key = "yolo_day2_v4"
    predictor.model = FakeYOLO()
    rgb = np.full((10, 10, 3), [10, 20, 30], dtype=np.uint8)
    result = predictor(rgb, .05)
    source, kwargs = calls[0]
    assert kwargs["quantize"] == 16 and "half" not in kwargs
    assert source[0, 0].tolist() == [30, 20, 10]
    assert kwargs["conf"] == .05
    assert result["labels"].tolist() == [1]


@pytest.mark.parametrize("labels,scores,boxes", [
    ([.5], [.5], [[0., 0., 10., 10.]]),
    ([3], [.5], [[0., 0., 10., 10.]]),
    ([1], [float("nan")], [[0., 0., 10., 10.]]),
    ([1], [.5], [[0., 0., 0., 10.]]),
    ([1], [1.1], [[0., 0., 10., 10.]]),
])
def test_invalid_predictions_rejected(labels, scores, boxes):
    with pytest.raises(ValueError):
        comparison.normalize_prediction(torch.tensor(boxes), torch.tensor(scores), torch.tensor(labels))


def test_benchmark_uses_identical_input_boundary_and_working_threshold(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = []
    class FakePredictor:
        dtype = "float32"
        def __call__(self, rgb, conf):
            calls.append((rgb, conf))
    frames = [np.zeros((20, 30, 3), dtype=np.uint8), np.ones((20, 30, 3), dtype=np.uint8)]
    result = comparison.benchmark_predictor(FakePredictor(), frames, warmup=2, iterations=3)
    assert len(calls) == 5
    assert all(conf == .05 for _, conf in calls)
    assert calls[0][0] is frames[0] and calls[1][0] is frames[1]
    assert result["iterations"] == 3
    assert "model-forward-only" in result["note"]
