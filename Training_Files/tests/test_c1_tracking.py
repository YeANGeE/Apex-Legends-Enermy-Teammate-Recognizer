import pytest
from c1_tracking import ScreenTracker


def detection(x=100,cls=0):
    return dict(box=[x,100,x+40,200],score=.8,**{'class':cls})


def test_confirm_and_identity_conflict():
    tracker=ScreenTracker()
    assert tracker.update(0,[detection()])[0]['state']=='tentative'
    tracker.update(16_000_000,[detection(101)])
    row=tracker.update(32_000_000,[detection(102)])[0]
    assert row['track_id']==1 and row['class_id']==0
    row=tracker.update(48_000_000,[detection(103,1)])[0]
    assert row['track_id']==1 and row['class_id'] is None


def test_prediction_expiry_and_reconnect():
    tracker=ScreenTracker(confirm_hits=1)
    tracker.update(0,[detection()]);tracker.update(20_000_000,[detection(104)])
    row=tracker.update(60_000_000,[])[0]
    assert row['state']=='predicted' and row['class_id'] is None and row['box'][0]>104
    assert tracker.update(80_000_000,[detection(110)])[0]['track_id']==1
    assert 'reconnected' in tracker.events
    row=tracker.update(281_000_000,[])[0]
    assert row['state']=='remembered' and row['box'] is None
    assert tracker.update(881_000_000,[])==[]


def test_sustained_opposite_changes_identity_not_single_glitch():
    tracker=ScreenTracker()
    for n in range(8):tracker.update(n*20_000_000,[detection()])
    row=tracker.update(160_000_000,[detection(cls=1)])[0]
    assert row['class_id'] is None and row['historical_class_id']==0
    assert tracker.update(180_000_000,[detection()])[0]['class_id'] is None
    assert tracker.update(200_000_000,[detection()])[0]['class_id']==0
    for n in range(11,30):row=tracker.update(n*20_000_000,[detection(cls=1)])[0]
    assert row['class_id']==1 and row['historical_class_id']==1


def test_memory_reconnect_needs_fresh_identity():
    tracker=ScreenTracker()
    for n in range(3):tracker.update(n*20_000_000,[detection()])
    tracker.update(300_000_000,[])
    row=tracker.update(400_000_000,[detection()])[0]
    assert row['track_id']==1 and row['class_id'] is None
    assert 'memory_reconnected' in tracker.events
    assert tracker.update(420_000_000,[detection()])[0]['class_id'] is None
    assert tracker.update(440_000_000,[detection()])[0]['class_id']==0


def test_ambiguous_memory_does_not_force_reconnect():
    tracker=ScreenTracker()
    for n in range(3):tracker.update(n*20_000_000,[detection()])
    tracker.update(300_000_000,[])
    rows=tracker.update(400_000_000,[detection(99),detection(101)])
    assert next(r for r in rows if r['track_id']==1)['state']=='remembered'
    assert 'ambiguous_reconnect' in tracker.events


def test_memory_does_not_steal_live_track():
    tracker=ScreenTracker()
    for n in range(3):tracker.update(n*20_000_000,[detection()])
    tracker.update(300_000_000,[detection(125)]) # New live track, outside stricter dormant gate.
    rows=tracker.update(320_000_000,[detection(125)])
    assert next(r for r in rows if r['track_id']==2)['last_seen_ns']==320_000_000
    assert next(r for r in rows if r['track_id']==1)['state']=='remembered'


def test_two_people_one_to_one():
    tracker=ScreenTracker(confirm_hits=1)
    tracker.update(0,[detection(100),detection(500,1)])
    rows=tracker.update(20_000_000,[detection(502,1),detection(102)])
    assert [(r['track_id'],r['class_id']) for r in rows]==[(1,0),(2,1)]


def test_timestamps_and_cut():
    tracker=ScreenTracker();tracker.update(100,[detection()])
    with pytest.raises(ValueError):tracker.update(100,[])
    tracker.reset()
    assert tracker.update(200,[detection()])[0]['track_id']==2


def test_invalid_box():
    with pytest.raises(ValueError):ScreenTracker().update(0,[dict(box=[0,0,-1,10],score=.5,**{'class':0})])


def test_confirmation_requires_consecutive_observations():
    tracker=ScreenTracker()
    tracker.update(0,[detection()]);tracker.update(20_000_000,[])
    tracker.update(40_000_000,[detection()]);tracker.update(60_000_000,[])
    row=tracker.update(80_000_000,[detection()])[0]
    assert row['hits']==3 and row['state']=='tentative'
