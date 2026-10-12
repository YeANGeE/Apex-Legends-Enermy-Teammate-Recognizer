from fps_recorder.dxcam_capture import select_ring_sequence,copy_capture_image
import numpy as np
import cv2


def test_recording_still_reads_oldest_available():
    assert select_ring_sequence(4,0,4)==(0,0,0)


def test_realtime_reads_latest_without_fabricating_overwrite():
    assert select_ring_sequence(4,0,4,True)==(3,0,3)


def test_real_overwrite_is_separate_from_intentional_skip():
    assert select_ring_sequence(10,0,4,True)==(9,6,3)
    assert select_ring_sequence(10,0,4)==(6,6,0)


def test_single_new_frame_has_no_skip():
    assert select_ring_sequence(10,9,4,True)==(9,0,0)


def test_direct_downscale_matches_copy_then_resize_and_owns_memory():
    source=np.random.default_rng(11).integers(0,256,(216,384,3),dtype=np.uint8)
    expected=cv2.resize(source.copy(),(192,108),interpolation=cv2.INTER_AREA)
    actual=copy_capture_image(source,(192,108))
    assert np.array_equal(actual,expected)
    assert not np.shares_memory(actual,source)
    source[:]=0
    assert np.array_equal(actual,expected)


def test_native_1080p_delivery_still_copies_ring_memory():
    source=np.ones((108,192,3),dtype=np.uint8)
    actual=copy_capture_image(source,(192,108))
    assert not np.shares_memory(actual,source)
    assert np.array_equal(actual,source)


def test_bgra_downscale_then_bgr_matches_original_bgr_path():
    bgra=np.random.default_rng(16).integers(0,256,(2160,3840,4),dtype=np.uint8)
    original=cv2.resize(cv2.cvtColor(bgra,cv2.COLOR_BGRA2BGR),(1920,1080),interpolation=cv2.INTER_AREA)
    candidate=copy_capture_image(bgra,(1920,1080),output_bgr=True)
    assert np.array_equal(candidate,original)
    assert candidate.shape==(1080,1920,3)
    assert not np.shares_memory(candidate,bgra)
