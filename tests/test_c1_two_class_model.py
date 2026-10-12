import torch

from c1_detection.model import build_c1_detector
from train_c1_fasterrcnn_day3_teammates import SETTINGS, build_model, save_checkpoint


def test_legacy_head_and_new_head_dimensions():
    legacy = build_c1_detector(pretrained=False)
    assert legacy.roi_heads.box_predictor.cls_score.out_features == 2
    assert legacy.roi_heads.box_predictor.bbox_pred.out_features == 8
    new = build_model(pretrained=False)
    assert new.roi_heads.box_predictor.cls_score.out_features == 3
    assert new.roi_heads.box_predictor.bbox_pred.out_features == 12
    assert new.transform.min_size == (720,)
    assert new.transform.max_size == 1280


def test_three_class_checkpoint_roundtrip(tmp_path):
    model = build_model(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=SETTINGS["lr"], momentum=.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [24, 32], gamma=.1)
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "num_classes": 3, "settings": SETTINGS})
    assert not path.with_suffix(".tmp").exists()
    restored = torch.load(path, map_location="cpu", weights_only=False)
    clone = build_model(pretrained=False)
    clone.load_state_dict(restored["model"], strict=True)
    assert restored["num_classes"] == 3
    assert torch.equal(clone.roi_heads.box_predictor.cls_score.weight, model.roi_heads.box_predictor.cls_score.weight)
