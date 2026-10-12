import json
import pytest
import c2_hidhide_session as session


def test_cli_closes_stdin_instead_of_waiting_for_ide_input(tmp_path,monkeypatch):
    executable=tmp_path/'HidHideCLI.exe'
    executable.touch()
    monkeypatch.setattr(session,'CLI',executable)
    def run(*args,**kwargs):
        assert kwargs['input']==''
        return type('Result',(),dict(returncode=0,stdout='--cloak-off\n',stderr=''))()
    monkeypatch.setattr(session.subprocess,'run',run)
    assert session.cli('--cloak-state')=='--cloak-off'


def test_cli_paths_are_literal_not_backslash_decoded():
    assert session.entries('--app-reg "C:\\tools\\python.exe"','--app-reg') == ['C:\\tools\\python.exe']
    with pytest.raises(ValueError):session.entries('unexpected command','--app-reg')


def test_cpp_quoted_paths_decode_only_quote_and_backslash_escapes():
    line='--app-reg '+r'"C:\\tools\\python.exe"'
    assert session.entries(line,'--app-reg') == [r'C:\tools\python.exe']
    assert session.entries('--dev-hide '+r'"HID\\VID_3537&PID_10C6\\physical"','--dev-hide') == [r'HID\VID_3537&PID_10C6\physical']


def test_restore_only_removes_owned_entries_and_preserves_original_active_config(tmp_path,monkeypatch):
    device='HID\\VID_3537&PID_10C6\\physical'
    before=dict(cloak='--cloak-on',inverse='--inv-off',devices=['HID\\OTHER'],apps=['original.exe'])
    now=dict(cloak='--cloak-on',inverse='--inv-off',devices=['HID\\OTHER',device],apps=['original.exe','bridge.exe'])
    journal=tmp_path/'lease.json'
    journal.write_text(json.dumps(dict(status='active',before=before,added_devices=[device],added_apps=['bridge.exe'])),encoding='utf-8')
    monkeypatch.setattr(session,'LEASE',journal)
    monkeypatch.setattr(session,'state',lambda:now)
    def cli(command,value=None):
        if command=='--dev-unhide':now['devices'].remove(value)
        elif command=='--app-unreg':now['apps'].remove(value)
        else:raise AssertionError('Must not disable originally active global hiding')
    monkeypatch.setattr(session,'cli',cli)
    session.restore()
    assert now==before
    assert json.loads(journal.read_text(encoding='utf-8'))['status']=='restored'


def test_recovery_rejects_non_controller_device_before_mutation(tmp_path,monkeypatch):
    journal=tmp_path/'lease.json'
    journal.write_text(json.dumps(dict(status='active',before={},added_devices=['HID\\KEYBOARD'],added_apps=[])),encoding='utf-8')
    monkeypatch.setattr(session,'LEASE',journal)
    monkeypatch.setattr(session,'cli',lambda *args:pytest.fail('Unexpected driver mutation'))
    with pytest.raises(ValueError):session.restore()
