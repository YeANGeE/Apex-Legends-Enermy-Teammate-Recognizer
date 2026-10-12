from dataclasses import dataclass

from fps_recorder.ring_buffer import TimestampRing


@dataclass
class Item:
    timestamp_ns: int


def test_timestamp_ring_retains_requested_window():
    ring = TimestampRing[Item](2_000, lambda item: item.timestamp_ns)
    for timestamp in (0, 1_000, 2_000, 3_000):
        ring.append(Item(timestamp))
    assert [item.timestamp_ns for item in ring.snapshot_since(1_000)] == [
        1_000,
        2_000,
        3_000,
    ]
    assert ring.span_ns() == 2_000

