from c2_shadow import causal_window


def test_shadow_history_matches_causal_training_offsets():
    assert causal_window([]) is None
    assert causal_window([0,50_000_000]) is None
    assert causal_window([0,55_000_000,110_000_000]) == [0,1,2]
    assert causal_window([0,300_000_000]) is None
