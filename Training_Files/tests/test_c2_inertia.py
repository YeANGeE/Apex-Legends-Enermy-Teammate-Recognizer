import numpy as np
import pytest
from c2_inertia import InertialAxes


def test_retarget_preserves_position_and_velocity_including_opposite_command():
    axes = InertialAxes()
    axes.set_target((1,0),0)
    axes.advance(20_000_000)
    position,velocity = axes.position.copy(),axes.velocity.copy()
    axes.set_target((-1,0),20_000_000)
    assert axes.position == position and axes.velocity == velocity
    axes.advance(21_000_000)
    assert axes.position[0] > position[0]  # Old momentum decelerates before reversal.


def test_integration_uses_elapsed_time_not_update_count():
    a,b = InertialAxes(),InertialAxes()
    for axes in (a,b):axes.set_target((.5,-.3),0)
    for t in range(1,81):a.advance(t*1_000_000)
    b.advance(80_000_000)
    assert np.allclose(a.position,b.position,atol=1e-12)
    assert np.allclose(a.position,[.475,-.285],atol=1e-7)


def test_new_target_is_not_applied_before_its_arrival():
    axes = InertialAxes()
    axes.set_target((0,0),0)
    assert axes.set_target((1,0),100_000_000) == [0,0]
    assert axes.advance(101_000_000)[0] > 0


def test_invalid_targets_do_not_mutate_state_and_time_cannot_reverse():
    axes = InertialAxes()
    axes.advance(10)
    with pytest.raises(ValueError):axes.set_target((float('nan'),0),20)
    assert axes.timestamp_ns == 10
    with pytest.raises(ValueError):axes.advance(9)


def test_full_range_step_stays_bounded_and_crosses_zero_smoothly():
    axes = InertialAxes(initial=(-1,0))
    axes.set_target((1,0),0)
    x = np.array([axes.advance(t*1_000_000)[0] for t in range(251)])
    assert np.all((-1 <= x)&(x <= 1))
    assert np.all(np.diff(x)>=0)
    assert np.max(np.diff(x)) < .05
    assert x[-1] > .99
