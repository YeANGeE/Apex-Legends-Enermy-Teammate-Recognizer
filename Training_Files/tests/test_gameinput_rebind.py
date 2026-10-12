import threading
import time

from fps_recorder.config import GameInputConfig
from fps_recorder.gameinput_capture import GameInputCapture


class _FakeGameInput:
    def __init__(self):
        self.current_calls = 0

    def current_reading(self, _device):
        self.current_calls += 1
        return 1 if self.current_calls == 1 else 2

    def next_reading(self, _reference, _device):
        return None

    def reading_device(self, reference):
        return reference + 100

    def reading_timestamp_us(self, reference):
        return reference

    def release_object(self, _object):
        pass


def test_gameinput_rebinds_when_filtered_device_goes_silent():
    capture = GameInputCapture(GameInputConfig(), None, lambda _packet: None)
    capture.native = _FakeGameInput()
    capture.calibration = object()
    emitted = []
    capture._emit = emitted.append
    worker = threading.Thread(target=capture._run)
    worker.start()
    deadline = time.monotonic() + 1.0
    while len(emitted) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    capture._stop.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert emitted == [1, 2]
    assert capture.snapshot_stats().device_rebinds == 1
