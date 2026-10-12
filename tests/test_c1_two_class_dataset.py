"""Adapted from the reviewed DeepSeek draft; local fixtures, no GPU/download."""
from pathlib import Path

import pytest
import torch
from PIL import Image

from c1_detection.two_class_dataset import YoloTwoClassDataset


@pytest.fixture
def root(tmp_path: Path):
    (tmp_path / "data.yaml").write_text("names:\n  0: enemy\n  1: teammate\n", encoding="utf-8")
    (tmp_path / "images/train").mkdir(parents=True)
    (tmp_path / "labels/train").mkdir(parents=True)
    Image.new("RGB", (100, 50), (0, 0, 0)).save(tmp_path / "images/train/sample.jpg")
    return tmp_path


def label(root, text):
    (root / "labels/train/sample.txt").write_text(text, encoding="utf-8")


def test_two_classes_exact_absolute_coordinates(root):
    label(root, "0 0.5 0.5 0.4 0.6\n1 0.1 0.2 0.2 0.4\n")
    image, target = YoloTwoClassDataset(root, "train")[0]
    assert image.dtype == torch.float32 and image.shape == (3, 50, 100)
    assert 0 <= float(image.min()) <= float(image.max()) <= 1
    assert target["labels"].tolist() == [1, 2]
    assert target["labels"].dtype == torch.int64
    assert torch.allclose(target["boxes"], torch.tensor([[30., 10., 70., 40.], [0., 0., 20., 20.]]))
    assert target["image_id"].ndim == 0 and target["image_id"].dtype == torch.int64
    assert target["area"].tolist() == [1200., 400.]
    assert target["iscrowd"].tolist() == [0, 0]


def test_empty_label_yields_zero_box_target(root):
    label(root, "")
    _, target = YoloTwoClassDataset(root, "train")[0]
    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    for key in ("labels", "area", "iscrowd"):
        assert target[key].shape == (0,)


def test_grayscale_converted_to_rgb(root):
    Image.new("L", (100, 50), 128).save(root / "images/train/sample.jpg")
    label(root, "")
    image, _ = YoloTwoClassDataset(root, "train")[0]
    assert torch.allclose(image, torch.full((3, 50, 100), 128/255), atol=.001)


@pytest.mark.parametrize("row", [
    "2 .5 .5 .2 .2", "-1 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 inf .2 .2",
    "0 .5 .5 0 .2", "0 .5 .5 .2 0", "0 .5 .5 -.2 .2", "0 .05 .5 .4 .2",
    "0 .5 .5 2 .2", "0 .5 .5 .2 2", "0 .5 .5 .2", "0 .5 .5 .2 .2 extra",
])
def test_invalid_rows_rejected(root, row):
    label(root, row)
    with pytest.raises(ValueError):
        YoloTwoClassDataset(root, "train")


def test_missing_labels_rejected(root):
    with pytest.raises(ValueError):
        YoloTwoClassDataset(root, "train")


def test_mismatched_stems_rejected(root):
    (root / "labels/train/other.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        YoloTwoClassDataset(root, "train")


def test_test_split_rejected(root):
    with pytest.raises(ValueError):
        YoloTwoClassDataset(root, "test")
