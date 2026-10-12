import json

import pytest

import c2_hidhide_session as session


ROOT = r'USB\VID_3537&PID_10C6\SERIAL'
FIXED = r'USB\VID_3537&PID_10C6&MI_00\PARENT'
OLD = {ROOT, FIXED, r'HID\VID_3537&PID_10C6&IG_00\OLD', r'USB\VID_3537&PID_10C6&IG_00\OLD'}
NEW = {ROOT, FIXED, r'HID\VID_3537&PID_10C6&IG_03\NEW', r'USB\VID_3537&PID_10C6&IG_03\NEW'}


def test_regenerated_ig_nodes_journaled_before_hide_and_all_restored(tmp_path, monkeypatch):
    lease=tmp_path/'lease.json'
    monkeypatch.setattr(session,'LEASE',lease)
    session.save(dict(status='active',added_devices=sorted(OLD),added_apps=[],before=dict(devices=[],apps=[],cloak='--cloak-off',inverse='--inv-off')))
    hidden=set(OLD)
    def cli(*args):
        if args[0]=='--dev-hide':
            assert args[1] in json.loads(lease.read_text())['added_devices']
            hidden.add(args[1])
        elif args[0]=='--dev-unhide':hidden.remove(args[1])
    monkeypatch.setattr(session,'cli',cli)
    monkeypatch.setattr(session,'state',lambda:dict(devices=sorted(hidden),apps=[],cloak='--cloak-on' if hidden else '--cloak-off',inverse='--inv-off'))
    session.extend_reconnected_devices(OLD,NEW)
    assert hidden==OLD|NEW
    session.restore()
    assert hidden==set()
    assert json.loads(lease.read_text())['status']=='restored'


@pytest.mark.parametrize('current',[
    (NEW-{ROOT})|{r'USB\VID_3537&PID_10C6\OTHER'},
    (NEW-{FIXED})|{r'USB\VID_3537&PID_10C6&MI_01\OTHER'},
    NEW|{r'HID\VID_OTHER\BAD'},
])
def test_identity_changes_rejected_before_mutation(current,monkeypatch):
    monkeypatch.setattr(session,'cli',lambda *args:pytest.fail('Must not mutate'))
    with pytest.raises(RuntimeError,match='nodes changed'):
        session.extend_reconnected_devices(OLD,current)
