from types import SimpleNamespace

import pytest

from c2_output_diagnostic import DiagnosticLook, readback_row


def test_bounded_smooth_pulses_and_manual_cancellation():
    diagnostic = DiagnosticLook(80)
    assert diagnostic.step(0, True) == [0., 0.]
    assert diagnostic.step(80_000_000, True)[0] == pytest.approx(.4275)
    diagnostic.step(1_000_000_000, True)
    assert diagnostic.step(1_080_000_000, True)[0] == pytest.approx(.0225, abs=.0001)
    diagnostic.step(2_000_000_000, True)
    assert diagnostic.step(2_080_000_000, True)[0] == pytest.approx(-.4275)
    assert diagnostic.step(2_090_000_000, False) == [0., 0.]
    assert diagnostic.step(3_000_000_000, True) == [0., 0.]
    assert diagnostic.step(3_080_000_000, True)[0] > 0


def test_readback_uses_native_values_instead_of_copying_sent_values():
    native = SimpleNamespace(number=7, pad=SimpleNamespace(rx=-32768, ry=16384, buttons=0))
    row = readback_row(10, 'ai', 1, native, dict(rx=.45, ry=0, buttons=256))
    assert row[4] == -1
    assert row[5] == pytest.approx(.5, abs=.0001)
    assert row[6:] == [0, .45, 0, 256]
