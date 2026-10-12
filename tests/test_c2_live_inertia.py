from c2_live_inertia import TrajectoryState


def command(source, axes=(1,0)):
    return dict(source_present_ns=source,base_source_present_ns=source,target_id=1,combined_axes=list(axes))


def test_trajectory_receipt_is_causal_and_no_output_jumps_at_retarget():
    s = TrajectoryState()
    s.command(command(0),10_000_000)
    assert s.axes.position == [0,0]
    before = s.step(20_000_000)
    s.command(command(20_000_000,(-1,0)),20_000_000)
    assert s.axes.position == before['axes']
    assert s.axes.velocity == before['velocity']


def test_expired_target_returns_to_neutral_at_exact_deadline_not_late_tick():
    s = TrajectoryState()
    s.command(command(0),0)
    state = s.step(200_000_000)
    assert not state['fresh_target']
    assert s.axes.target == [0,0]
    assert state['axes'][0] > 0  # Return is smooth, not a reset to zero.


def test_stale_or_future_base_cannot_be_used():
    s = TrajectoryState()
    s.command(command(0),200_000_000)
    assert s.axes.target == [0,0]
    s.command(command(300_000_000),200_000_000)
    assert s.axes.target == [0,0]
