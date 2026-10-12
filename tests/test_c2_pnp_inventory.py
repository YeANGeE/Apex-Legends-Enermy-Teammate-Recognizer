import json
from types import SimpleNamespace
import pytest
import c2_hidhide_session as session


def rows():
    return [dict(Class='USB',InstanceId=r'USB\VID_3537&PID_10C6\serial',Status='OK'),
            dict(Class='XnaComposite',InstanceId=r'USB\VID_3537&PID_10C6&MI_00\child',Status='OK'),
            dict(Class='HIDClass',InstanceId=r'HID\VID_3537&PID_10C6&IG_00\child',Status='OK')]


def test_inventory_queries_parent_and_requires_exact_usb_root(tmp_path,monkeypatch):
    monkeypatch.setattr(session,'ROOT',tmp_path)
    monkeypatch.setattr(session,'controller_rows',lambda vid:rows())
    assert len(session.physical_devices())==3


def test_missing_parent_is_not_bypassed_and_preserves_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(session,'ROOT',tmp_path)
    monkeypatch.setattr(session,'controller_rows',lambda vid:rows()[1:])
    with pytest.raises(RuntimeError,match='found 0'):session.physical_devices()
    assert (tmp_path/'diagnostics/c2_controller_session/pnp_inventory_latest.json').is_file()


def test_native_query_transition_is_retryable_and_preserves_failure_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(session,'ROOT',tmp_path)
    def query(vid):
        raise session.InventoryNotReady('Device disappeared during query')
    monkeypatch.setattr(session,'controller_rows',query)
    with pytest.raises(session.PhysicalDeviceNotReady,match='not ready'):
        session.physical_devices()
    diagnostic=json.loads((tmp_path/'diagnostics/c2_controller_session/pnp_inventory_error.json').read_text())
    assert diagnostic['backend']=='cfgmgr32'
