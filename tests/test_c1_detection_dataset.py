from pathlib import Path

import torch

from c1_detection.dataset import C1CocoDetectionDataset, detection_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "c1_annotations" / "day1_seed_v1" / "normalized"


def build_dataset() -> C1CocoDetectionDataset:
    return C1CocoDetectionDataset(
        DATASET_ROOT / "images",
        DATASET_ROOT / "annotations" / "instances_train.json",
    )


def test_train_dataset_shape_and_target_types() -> None:
    dataset = build_dataset()
    assert len(dataset) == 99
    image, target = dataset[0]
    assert image.dtype == torch.float32
    assert image.shape == (3, 1080, 1920)
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64


def test_dataset_contains_positive_and_negative_frames() -> None:
    dataset = build_dataset()
    counts = [dataset.box_count(index) for index in range(len(dataset))]
    assert any(count > 0 for count in counts)
    assert any(count == 0 for count in counts)


def test_detection_collate_preserves_variable_targets() -> None:
    dataset = build_dataset()
    positive = next(index for index in range(len(dataset)) if dataset.box_count(index))
    negative = next(index for index in range(len(dataset)) if not dataset.box_count(index))
    images, targets = detection_collate([dataset[positive], dataset[negative]])
    assert len(images) == 2
    assert targets[0]["boxes"].shape[0] > 0
    assert targets[1]["boxes"].shape == (0, 4)
