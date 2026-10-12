from unittest.mock import Mock
import pytest
from c1_timer_resolution import TimerResolution


def test_balanced_timer_request_and_idempotent_cleanup():
    api=Mock();api.timeBeginPeriod.return_value=0;api.timeEndPeriod.return_value=0
    timer=TimerResolution(api)
    timer.start();timer.start();timer.stop();timer.stop()
    api.timeBeginPeriod.assert_called_once_with(1)
    api.timeEndPeriod.assert_called_once_with(1)
    assert not timer.active


def test_failed_request_has_no_unmatched_cleanup():
    api=Mock();api.timeBeginPeriod.return_value=1
    timer=TimerResolution(api)
    with pytest.raises(RuntimeError):timer.start()
    timer.stop();api.timeEndPeriod.assert_not_called()
