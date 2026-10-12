import pytest

from c2_engagement_buttons import EngagementButtons


def test_fire_confirms_then_sustains_and_immediately_stops_off_target():
    b=EngagementButtons()
    assert b.update(0,1,True,True)==(False,True)
    assert b.update(59_000_000,1,True,True)==(False,True)
    assert b.update(60_000_000,1,True,True)==(True,True)
    assert b.update(500_000_000,1,True,True)==(True,True)
    assert b.update(510_000_000,1,True,False)==(False,True)
    assert b.update(520_000_000,1,True,True)==(False,True)


def test_loss_stops_fire_immediately_but_ads_survives_short_gap():
    b=EngagementButtons()
    b.update(0,1,True,True)
    b.update(60_000_000,1,True,True)
    assert b.update(70_000_000,None,False,False)==(False,True)
    assert b.update(209_000_000,None,False,False)==(False,True)
    assert b.update(210_000_000,None,False,False)==(False,False)


def test_target_change_reconfirms_and_manual_cancels_ads_grace():
    b=EngagementButtons()
    b.update(0,1,True,True)
    assert b.update(60_000_000,1,True,True)==(True,True)
    assert b.update(61_000_000,2,True,True)==(False,True)
    assert b.update(62_000_000,None,False,False,False)==(False,False)
    assert b.update(63_000_000,None,False,False)==(False,False)


def test_backward_time_rejected():
    b=EngagementButtons()
    b.update(20,1,True,True)
    with pytest.raises(ValueError):b.update(10,1,True,True)
