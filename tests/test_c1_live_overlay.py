"""No capture/window launched: verify latest-frame handoff semantics."""
from preview_c1_live_overlay import LatestFrame
import numpy as np
import cv2


def test_overwrite_returns_latest_and_does_not_build_backlog():
    slot = LatestFrame()
    for frame in range(100):
        slot.put(frame)
    assert slot.overwritten == 99
    assert slot.take() == 99
    assert slot.take() is None


def test_consumed_frame_is_not_counted_as_overwritten():
    slot = LatestFrame()
    slot.put('first')
    assert slot.take() == 'first'
    slot.put('next')
    assert slot.take() == 'next'
    assert slot.overwritten == 0


def test_faster_color_conversion_preserves_pixels():
    bgr=np.random.default_rng(7).integers(0,256,(120,160,3),dtype=np.uint8)
    assert np.array_equal(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), bgr[:,:,::-1].copy())


def test_event_wait_consumes_latest_and_resets_signal():
    slot=LatestFrame()
    slot.put('old');slot.put('latest')
    assert slot.ready.is_set()
    assert slot.wait_take(timeout=0)=='latest'
    assert not slot.ready.is_set()
    assert slot.wait_take(timeout=0) is None
    slot.put('next')
    assert slot.wait_take(timeout=0)=='next'
