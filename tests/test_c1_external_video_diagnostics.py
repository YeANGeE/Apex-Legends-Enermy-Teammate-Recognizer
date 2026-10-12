import pytest

from diagnose_c1_external_videos import detection_flags, disagreement
from sample_c1_external_videos import select_indices


def test_sampling_follows_real_irregular_pts():
    pts = [10.0, 10.2, 10.48, 10.53, 11.49, 11.6, 12.7]
    assert select_indices(pts, 1.) == [3, 5, 6]


def test_short_clip_retains_one_frame():
    assert select_indices([0., .1], 1.) == [0]


@pytest.mark.parametrize("interval", [0., -1., float("nan"), float("inf")])
def test_bad_interval_rejected(interval):
    with pytest.raises(ValueError):
        select_indices([0., 1.], interval)


def test_cross_class_overlap_is_diagnostic_not_ground_truth():
    pred = {"boxes": [[0, 0, 50, 50], [0, 0, 50, 50]], "labels": [1, 2], "scores": [.9, .1]}
    flags = detection_flags(pred, (100, 100))
    assert flags["cross_class_overlap_pairs"] == 1
    result = disagreement(pred, {"boxes": [[0, 0, 50, 50]], "labels": [2], "scores": [.9]})
    assert result["a_unmatched"] == 1
    assert result["matched_identity_conflicts"] == 1
    assert "not FP/FN" in result["note"]


def test_empty_predictions():
    empty = {"boxes": [], "labels": [], "scores": []}
    assert detection_flags(empty, (100, 100))["overlapping_pairs"] == 0
    assert disagreement(empty, empty)["a_unmatched"] == 0
