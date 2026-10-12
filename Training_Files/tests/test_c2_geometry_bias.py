import pytest
from c2_geometry_bias import GeometryBias


def test_target_left_up_adds_correct_signed_bias_and_preserves_base():
    r = GeometryBias().apply(dict(rx=.1,ry=-.1),[400,200,560,340])
    assert r['bias_axes'][0] == -.5
    assert r['bias_axes'][1] == .5
    assert r['combined_axes'] == [-.4,.4]
    assert r['base_axes'] == [.1,-.1]


def test_gain_zero_retains_bc_and_center_has_no_correction():
    assert GeometryBias(gain_x=0,gain_y=0).apply(dict(rx=.3,ry=-.4),[400,200,560,340])['combined_axes'] == [.3,-.4]
    assert GeometryBias().apply(dict(rx=.3,ry=-.4),[900,500,1020,580])['bias_axes'] == [0,0]


def test_correction_is_bounded_but_does_not_guarantee_toward_target():
    r = GeometryBias().apply(dict(rx=.9,ry=0),[400,200,560,340])
    assert r['combined_axes'][0] > 0  # Base may overpower a leftward correction.
    r = GeometryBias().apply(dict(rx=1,ry=0),[1400,500,1560,580])
    assert r['combined_axes'][0] == 1 and r['saturated']


def test_stale_base_and_missing_target_are_not_used():
    g = GeometryBias()
    s = dict(source_present_ns=0,action=dict(rx=0,ry=0))
    selection = dict(selected=dict(track_id=1,score=.8,box=[400,200,560,340]))
    assert g.suggest(s,selection,150_000_001) is None
    assert g.suggest(s,dict(selected=None),0) is None
    assert g.suggest(s,selection,-1) is None


def test_invalid_parameters_and_boxes_are_rejected():
    with pytest.raises(ValueError):GeometryBias(gain_x=float('nan'))
    with pytest.raises(ValueError):GeometryBias().apply(dict(rx=0,ry=0),[-5,0,20,10])


def test_geometry_only_baseline_cannot_be_reversed_by_opposing_bc_axes():
    g=GeometryBias(base_weight=0)
    result=g.apply(dict(rx=1,ry=-1),[400,200,560,340])
    assert result['combined_axes']==[-.5,.5]
    assert g.apply(dict(rx=1,ry=-1),[900,500,1020,580])['combined_axes']==[0,0]
