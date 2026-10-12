"""One-shot monitor explicitly authorized by user; never starts/stops training."""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/c1_yolo/day3_yolo26n_1280_joint_fixed2800_v1'
GUARD=ROOT/'diagnostics/fixed2800_shutdown'
MANIFEST_SHA='ef499410b7f86db6a451dca64d159034138b55c382ac73b03741fbce947a451c'
FILES=['weights/best.pt','weights/last.pt','results.csv','args.yaml','data_receipt.json','results.png']


def save_state(status,**extra):
    value=dict(status=status,updated_at=datetime.now(timezone.utc).isoformat(),run_dir=str(RUN),
               backup_dir=str(GUARD/'saved_results'),**extra)
    temp=GUARD/'state.tmp'
    with temp.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
    os.replace(temp,GUARD/'state.json')


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def process_rows():
    command="Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True,timeout=60)
    data=json.loads(result.stdout or '[]');return data if isinstance(data,list) else [data]


def other_training(rows):
    return [r for r in rows if any(s in (r.get('CommandLine') or '').lower() for s in
                                  ('train_c1_','resume_c1_'))]


def validate_saved_run(run):
    import torch
    receipt=json.loads((run/'data_receipt.json').read_text(encoding='utf-8'))
    if receipt['experiment']!=RUN.name or receipt['manifest_sha256']!=MANIFEST_SHA:
        raise ValueError('Wrong experiment/data receipt')
    with (run/'results.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows:raise ValueError('No completed epochs')
    rows=[{k.strip():v.strip() for k,v in row.items()} for row in rows]
    for row in rows:
        if not all(math.isfinite(float(v)) for v in row.values()):raise ValueError('Nonfinite training metrics')
    epochs=[int(float(row['epoch'])) for row in rows]
    if epochs!=list(range(1,len(rows)+1)) or len(rows)>60:raise ValueError('Invalid epoch roster')
    results={}
    for name in ('best.pt','last.pt'):
        path=run/'weights'/name
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        # strip_optimizer is invoked by trainer.final_eval only at normal training end.
        if ckpt.get('epoch')!=-1 or ckpt.get('optimizer') is not None:
            raise ValueError(f'{name} is still an intermediate training checkpoint')
        model=ckpt.get('model')
        if model is None or model.names!={0:'enemy',1:'teammate'}:raise ValueError('Wrong checkpoint classes')
        for tensor in model.state_dict().values():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():raise ValueError('Nonfinite model weights')
        results[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        del model,ckpt
    return dict(completed_epochs=len(rows),final_epoch_metrics=rows[-1],weights=results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',action='store_true')
    parser.add_argument('--pids',nargs='+',type=int,default=[]);args=parser.parse_args()
    if not args.arm:
        print('Dry status only; no shutdown.');print((GUARD/'state.json').read_text(encoding='utf-8') if (GUARD/'state.json').exists() else 'Not armed')
        return
    if not args.pids:raise ValueError('Must bind existing training process IDs')
    GUARD.mkdir(parents=True,exist_ok=True)
    # Exclusive one-shot lock: an older monitor cannot silently rearm after restart.
    lock=os.open(GUARD/'armed.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(lock)
    handles=[];exit_codes={}
    try:
        rows={r['ProcessId']:r for r in process_rows()}
        for pid in args.pids:
            row=rows.get(pid)
            if row is None or str(ROOT/'train_c1_yolo_joint_fixed_2800ms.py').lower() not in (row.get('CommandLine') or '').lower():
                raise ValueError(f'PID {pid} is not the authorized training')
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        for pid in args.pids:
            handle=kernel.OpenProcess(0x1000,False,pid)
            if not handle:raise ctypes.WinError(ctypes.get_last_error())
            handles.append((pid,handle))
        deadline=time.monotonic()+12*3600
        while time.monotonic()<deadline:
            active=[]
            for pid,handle in handles:
                code=ctypes.c_ulong()
                if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
                if code.value==259:active.append(pid)
                else:exit_codes[str(pid)]=code.value
            if any(code!=0 for code in exit_codes.values()):
                save_state('blocked',reason='Training exited abnormally; no shutdown',exit_codes=exit_codes);return
            if not active:break
            save_state('watching',active_pids=active,exit_codes=exit_codes)
            time.sleep(30)
        else:
            save_state('blocked',reason='12-hour monitoring deadline; no shutdown');return
        save_state('waiting_files',exit_codes=exit_codes)
        paths=[RUN/name for name in FILES]
        if any(not p.is_file() or p.stat().st_size==0 for p in paths):raise ValueError('Missing final output files')
        before=[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
        time.sleep(30)
        if before!=[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]:raise ValueError('Output files still changing')
        save_state('verifying',exit_codes=exit_codes)
        verification=validate_saved_run(RUN)
        if other_training(process_rows()):raise ValueError('Another C1 training process is running; no shutdown')
        backup=GUARD/'saved_results';backup.mkdir()
        for name in FILES:
            source=RUN/name;target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            if sha(source)!=sha(target):raise ValueError('Backup hash mismatch')
        report=dict(exit_codes=exit_codes,**verification,backup_files={name:sha(backup/name) for name in FILES})
        (backup/'completion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if other_training(process_rows()):raise ValueError('New training started before shutdown; no shutdown')
        save_state('shutdown_scheduled',**report,shutdown_command='shutdown.exe /s /t 0; no forced app termination')
        # Ordinary shutdown without /f and without a timed implicit force. Unsaved apps may block it.
        result=subprocess.run(['shutdown.exe','/s','/t','0'],capture_output=True,text=True,timeout=30)
        if result.returncode:save_state('shutdown_failed',reason=result.stderr or result.stdout,**report)
    except Exception as exc:
        save_state('blocked',reason=f'{type(exc).__name__}: {exc}',exit_codes=exit_codes)
    finally:
        for _,handle in handles:kernel.CloseHandle(handle)


if __name__=='__main__':main()
