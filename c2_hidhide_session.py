"""Reversible physical-controller-only HidHide lease and parent-exit recovery."""
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from c2_pnp_native import controller_rows, InventoryNotReady

ROOT = Path(__file__).resolve().parent
CLI = Path('C:/Program Files/Nefarius Software Solutions/HidHide/x64/HidHideCLI.exe')
LEASE = ROOT/'diagnostics/c2_controller_session/hidhide_lease.json'
VID_PID = 'VID_3537&PID_10C6'


class PhysicalDeviceNotReady(RuntimeError):
    """No complete current device tree yet; safe to wait without hiding anything."""


def cli(*arguments):
    if not CLI.is_file():
        raise RuntimeError('Missing HidHideCLI: run setup_c2_controller.py and restart Windows first')
    # The CLI treats redirected/inherited IDE input as a script stream; terminate it with EOF.
    result = subprocess.run([str(CLI),*arguments],input='',capture_output=True,text=True,
                            encoding='utf-8',errors='replace',timeout=15,creationflags=0x08000000)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f'HidHide command failed: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


def entries(text, prefix):
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        opening = prefix+' "'
        if not line.startswith(opening) or not line.endswith('"'):
            raise ValueError('Unexpected HidHide export syntax')
        # 1.5.230 emits C++ std::quoted escapes; preserve ordinary path characters.
        result.append(re.sub(r'\\([\\"])',r'\1',line[len(opening):-1]))
    return result


def state():
    cloak,inverse = cli('--cloak-state'),cli('--inv-state')
    if cloak not in ('--cloak-on','--cloak-off') or inverse not in ('--inv-on','--inv-off'):
        raise ValueError('Unexpected HidHide flag state')
    return dict(cloak=cloak,inverse=inverse,apps=entries(cli('--app-list'),'--app-reg'),
                devices=entries(cli('--dev-list'),'--dev-hide'))


def physical_devices():
    """Native current-device inventory; keep identity validation separate."""
    try:
        rows=controller_rows(VID_PID)
    except InventoryNotReady as exc:
        diagnostic=ROOT/'diagnostics/c2_controller_session/pnp_inventory_error.json'
        diagnostic.parent.mkdir(parents=True,exist_ok=True)
        diagnostic.write_text(json.dumps(dict(error=str(exc),backend='cfgmgr32',
            timestamp_ns=time.time_ns(),device=VID_PID),indent=2),encoding='utf-8')
        raise PhysicalDeviceNotReady('Physical native device inventory not ready: '+str(exc)+'. Diagnostic: '+str(diagnostic)) from None
    snapshot = ROOT/'diagnostics/c2_controller_session/pnp_inventory_latest.json'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if not isinstance(rows,list):
        raise ValueError('PnP inventory must be a list')
    if not rows:
        raise PhysicalDeviceNotReady('Physical controller nodes are not present yet; snapshot: '+str(snapshot))
    devices = set()
    for row in rows:
        path = row['InstanceId']
        root_usb = row['Class']=='USB' and re.fullmatch(r'USB\\VID_3537&PID_10C6\\[^\\]+',path,re.IGNORECASE) is not None
        if ((row['Class'] not in ('HIDClass','XnaComposite') and not root_usb) or VID_PID not in path.upper()
                or not path.upper().startswith(('HID\\','USB\\')) or row['Status'] != 'OK'):
            raise ValueError('Unexpected or unhealthy physical controller node')
        devices.add(path)
    roots = [p for p in devices if re.fullmatch(r'USB\\VID_3537&PID_10C6\\[^\\]+',p,re.IGNORECASE)]
    if len(roots) != 1:
        message=(f'Expected one physical USB parent, found {len(roots)}. Keep controller connected; '
                 f'device snapshot: {snapshot}. No devices hidden.')
        if not roots:
            raise PhysicalDeviceNotReady(message)
        raise RuntimeError(message)
    if not any(p.upper().startswith('HID\\') for p in devices) or not any(p.upper().startswith('USB\\') for p in devices):
        raise PhysicalDeviceNotReady('Physical HID/XUSB nodes are still initializing')
    return devices


def save(value):
    LEASE.parent.mkdir(parents=True,exist_ok=True)
    temporary = LEASE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,LEASE)


def extend_reconnected_devices(expected, current):
    """Accept regenerated IG interfaces only under the same physical USB identity."""
    expected,current=set(expected),set(current)
    root_pattern=r'USB\\VID_3537&PID_10C6\\[^\\]+'
    ig_pattern=r'(HID|USB)\\VID_3537&PID_10C6&IG_[0-9A-F]{2}\\[^\\]+'
    roots=lambda nodes:{p for p in nodes if re.fullmatch(root_pattern,p,re.IGNORECASE)}
    fixed=lambda nodes:{p for p in nodes if not re.fullmatch(ig_pattern,p,re.IGNORECASE)}
    if len(roots(expected))!=1 or roots(current)!=roots(expected) or fixed(current)!=fixed(expected):
        raise RuntimeError('Physical device nodes changed outside XInput IG interfaces; restoring session')
    value=json.loads(LEASE.read_text(encoding='utf-8'))
    if value['status']!='active' or not expected.issubset(set(value['added_devices'])):
        raise RuntimeError('Reconnect cannot extend an inactive or unrelated device lease')
    additions=current-set(value['added_devices'])
    if additions & set(value['before']['devices']):
        raise RuntimeError('Reconnected interfaces overlap pre-existing hidden devices')
    value['added_devices']=sorted(set(value['added_devices'])|additions)
    value.setdefault('reconnections',[]).append(dict(previous=sorted(expected),current=sorted(current)))
    save(value)  # Recovery owns every new node before any mutation.
    for device in sorted(additions):
        cli('--dev-hide',device)
    after=state()
    if after['cloak']!='--cloak-on' or not current.issubset(set(after['devices'])):
        raise RuntimeError('Reconnected interface isolation did not persist')


def restore():
    if not LEASE.is_file():
        return
    value = json.loads(LEASE.read_text(encoding='utf-8'))
    if value['status'] == 'restored':
        return
    # Only remove this session's added entries; preserve original/other devices.
    for device in value['added_devices']:
        if VID_PID not in device.upper():
            raise ValueError('Refusing recovery for a non-controller lease entry')
        cli('--dev-unhide',device)
    if value['before']['cloak'] == '--cloak-off':
        cli('--cloak-off')
    for application in value['added_apps']:
        cli('--app-unreg',application)
    current = state()
    original = value['before']
    if (current['cloak'] != original['cloak'] or current['inverse'] != original['inverse']
            or any(d in current['devices'] for d in value['added_devices'])
            or any(a in current['apps'] for a in value['added_apps'])
            or not set(original['devices']).issubset(set(current['devices']))
            or not {a.casefold() for a in original['apps']}.issubset({a.casefold() for a in current['apps']})):
        raise RuntimeError('HidHide recovery verification failed; use restore_c2_controller.py')
    value['status'] = 'restored'
    save(value)


def begin():
    if LEASE.exists() and json.loads(LEASE.read_text(encoding='utf-8'))['status'] != 'restored':
        restore()
    before = state()
    if before['inverse'] != '--inv-off':
        raise RuntimeError('HidHide inverse mode is enabled; resolve existing configuration first')
    devices = physical_devices()
    if not devices:
        raise RuntimeError('Expected physical controller not found in HidHide gaming-device inventory')
    applications = {str(Path(sys.executable).resolve()),str(Path(sys._base_executable).resolve())}
    value = dict(status='prepared',parent_pid=os.getpid(),before=before,
        added_devices=sorted(devices-set(before['devices'])),
        added_apps=sorted(a for a in applications if a.casefold() not in {p.casefold() for p in before['apps']}))
    if set(devices) & set(before['devices']):
        raise RuntimeError('Physical controller was already hidden; cannot promise restore-to-visible on exit')
    save(value)  # Journal and recovery process exist before any hiding mutation.
    guardian = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watch-parent',str(os.getpid())],
                                creationflags=0x08000000)
    try:
        for application in value['added_apps']:
            cli('--app-reg',application)
        for device in value['added_devices']:
            cli('--dev-hide',device)
        cli('--cloak-on')
        after = state()
        if after['cloak'] != '--cloak-on' or not devices.issubset(set(after['devices'])):
            raise RuntimeError('HidHide session setup did not persist')
        value['status'] = 'active';save(value)
        return guardian
    except BaseException:
        restore()
        raise


def watch(parent_pid):
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_bool,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000,False,parent_pid)
    if not handle:
        restore();return
    try:
        while kernel.WaitForSingleObject(handle,1000) == 258:
            if LEASE.is_file() and json.loads(LEASE.read_text(encoding='utf-8'))['status'] == 'restored':
                return
        # Parent target disappears with its process; now unhide physical input.
        time.sleep(.2)
        restore()
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-parent',type=int)
    args = parser.parse_args()
    if args.watch_parent:
        watch(args.watch_parent)
    else:
        restore()
        print('已按会话备份恢复真实手柄可见性。')
