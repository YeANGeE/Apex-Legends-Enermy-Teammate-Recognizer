from types import SimpleNamespace
import pytest
from c1_capture_latency import split_capture_latency


def packet(start=11_000_000,end=13_000_000,ready=14_000_000):
    return SimpleNamespace(timestamp_ns=1_000_000,capture_copy_start_ns=start,
                           capture_copy_end_ns=end,capture_ready_ns=ready)


def test_components_partition_the_same_frame_age():
    result=split_capture_latency(packet(),17_000_000)
    assert result==dict(source_to_capture_copy=10.,capture_copy=2.,capture_dispatch=1.,capture_to_inference=3.)
    assert sum(result.values())==16.


def test_old_packets_without_diagnostics_remain_supported():
    assert split_capture_latency(packet(start=None,end=None,ready=None),17_000_000)=={}


def test_future_or_mixed_clock_measurement_is_rejected():
    with pytest.raises(ValueError):split_capture_latency(packet(),12_000_000)
