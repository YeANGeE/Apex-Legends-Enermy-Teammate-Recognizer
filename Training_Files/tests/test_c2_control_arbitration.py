from c2_control_arbitration import OwnershipGate, ButtonLatch, merge_report


def physical(**changes):
    return dict(lx=0,ly=0,rx=0,ry=0,lt=.3,rt=.4,buttons=0x4000,**changes)


def test_all_four_axes_take_over_and_return_requires_continuous_one_second():
    for axis in ('lx','ly','rx','ry'):
        g=OwnershipGate();p=physical()
        assert g.update(p,0)=='manual'
        assert g.update(p,1_000_000_000)=='ai'
        p[axis]=-.10001
        assert g.update(p,1_000_000_001)=='manual'
        p[axis]=.1
        assert g.update(p,1_100_000_000)=='manual'
        assert g.update(p,2_099_999_999)=='manual'
        assert g.update(p,2_100_000_000)=='ai'


def test_pause_cancels_quiet_timer_and_manual_passes_full_report():
    g=OwnershipGate();p=physical()
    g.update(p,0);assert g.update(p,1_000_000_000)=='ai'
    assert g.update(p,1_000_000_001,False)=='manual'
    assert g.update(p,2_000_000_000,True)=='manual'
    assert merge_report(p,'manual',(1,1),True,True)==p


def test_ai_owns_only_right_axes_and_shoulders():
    p=physical();p['buttons']=0x4300
    out=merge_report(p,'ai',(-.2,.3),fire=False,aim=True)
    assert out['buttons']==0x4100
    assert out['rx']==-.2 and out['ry']==.3
    assert out['lt']==.3 and out['rt']==.4 and out['lx']==0


def test_button_hysteresis_cannot_hold_fire_after_target_loss():
    b=ButtonLatch()
    assert b.update(.7)
    assert b.update(.5)
    assert not b.update(.3)
    assert b.update(.8)
    assert not b.update(.9,False)
