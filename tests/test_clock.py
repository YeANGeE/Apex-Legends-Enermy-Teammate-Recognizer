from fps_recorder.clock import QpcClock


def test_tick_conversion_is_integer_and_monotonic():
    clock = QpcClock()
    first = clock.now_ns()
    second = clock.now_ns()
    assert isinstance(first, int)
    assert second >= first
    assert clock.frequency > 0

