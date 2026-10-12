import pytest
import run_c2_offline_controller as controller


def test_success_after_reconnect_keeps_waiting_in_same_isolation_session(monkeypatch):
    calls=iter([[0],[]])
    monkeypatch.setattr(controller,'unprivileged_slots',lambda:next(calls))
    monkeypatch.setattr(controller,'connected_slots',lambda:[1])
    monkeypatch.setattr(controller.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda key:0)})())
    controller.wait_for_isolation()


def test_f10_cancels_before_any_virtual_device_creation(monkeypatch):
    monkeypatch.setattr(controller,'unprivileged_slots',lambda:[0])
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda key:0x8000)})())
    with pytest.raises(RuntimeError,match='cancelled'):controller.wait_for_isolation()


def test_startup_waits_for_connection_and_device_tree_before_proceeding(monkeypatch):
    slots=iter([[],[0],[0],[0]])
    calls=iter([controller.PhysicalDeviceNotReady('initializing'),{'HID\\controller','USB\\controller'}])
    monkeypatch.setattr(controller,'connected_slots',lambda:next(slots))
    def inventory():
        value=next(calls)
        if isinstance(value,Exception):raise value
        return value
    monkeypatch.setattr(controller,'physical_devices',inventory)
    monkeypatch.setattr(controller.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda key:0)})())
    devices,slot=controller.wait_for_physical_setup()
    assert slot==0 and len(devices)==2


def test_startup_does_not_resolve_ambiguous_controllers_by_guessing(monkeypatch):
    monkeypatch.setattr(controller,'connected_slots',lambda:[0,1])
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda key:0)})())
    with pytest.raises(RuntimeError,match='More than one'):controller.wait_for_physical_setup()


def test_virtual_first_waits_until_all_slots_released(monkeypatch):
    slots=iter([[0],[]])
    monkeypatch.setattr(controller,'connected_slots',lambda:next(slots))
    monkeypatch.setattr(controller,'unprivileged_slots',lambda:[])
    monkeypatch.setattr(controller.time,'sleep',lambda _:None)
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda _:0)})())
    controller.wait_for_unplug()


def test_virtual_first_reconnect_selects_physical_not_virtual(monkeypatch):
    slots=iter([[0],[0,1]])
    monkeypatch.setattr(controller,'connected_slots',lambda:next(slots))
    monkeypatch.setattr(controller,'physical_devices',lambda:{'owned'})
    monkeypatch.setattr(controller,'unprivileged_slots',lambda:[0])
    read=[]
    monkeypatch.setattr(controller,'physical_report',lambda index:read.append(index))
    monkeypatch.setattr(controller.time,'sleep',lambda _:None)
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda _:0)})())
    assert controller.wait_for_replug(0,{'owned'})==1
    assert read==[1]


@pytest.mark.parametrize('devices,visible,message',[
    ({'different'},[0],'nodes changed'),
    ({'owned'},[0,1],'visible outside'),
])
def test_virtual_first_rejects_changed_device_or_failed_isolation(monkeypatch,devices,visible,message):
    monkeypatch.setattr(controller,'connected_slots',lambda:[0,1])
    monkeypatch.setattr(controller,'physical_devices',lambda:devices)
    monkeypatch.setattr(controller,'unprivileged_slots',lambda:visible)
    def reject_changed(expected,current):
        raise RuntimeError('Physical device nodes changed')
    monkeypatch.setattr(controller,'extend_reconnected_devices',reject_changed)
    monkeypatch.setattr(controller,'USER32',type('Keys',(),{'GetAsyncKeyState':staticmethod(lambda _:0)})())
    with pytest.raises(RuntimeError,match=message):controller.wait_for_replug(0,{'owned'})
