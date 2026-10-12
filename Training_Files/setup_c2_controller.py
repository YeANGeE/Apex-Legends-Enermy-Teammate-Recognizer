"""User-run signed HidHide installer entry; never hides devices or restarts Windows."""
import ctypes
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTALLER = ROOT/'diagnostics/c2_controller_dependencies/HidHide_1.5.230_x64.exe'

if __name__ == '__main__':
    from c2_hidhide_session import CLI
    if CLI.is_file():
        print('HidHide已存在；安装步骤跳过。下一步运行run_c2_offline_controller.py --check-only。')
    else:
        if not INSTALLER.is_file():
            raise RuntimeError('Missing official signed installer')
        check_code = ('& {param($p) $s=Get-AuthenticodeSignature -LiteralPath $p; '
            'if($s.Status -ne "Valid" -or $s.SignerCertificate.Subject -notmatch "Nefarius"){exit 1}} '
            + "'"+str(INSTALLER).replace("'","''")+"'")
        powershell = Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
        environment = dict(os.environ,PSModulePath=str(powershell.parent/'Modules'))
        check = subprocess.run([str(powershell),'-NoProfile','-Command',check_code],capture_output=True,env=environment)
        if check.returncode:
            raise RuntimeError('Installer signature verification failed')
        print('将打开官方HidHide安装器；按提示安装，完成后手动重启Windows。此脚本不隐藏设备、不自动重启。',flush=True)
        execute = ctypes.windll.shell32.ShellExecuteW
        execute.argtypes = [ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.c_int]
        execute.restype = ctypes.c_void_p
        result = execute(None,'runas',str(INSTALLER),'/norestart',str(ROOT),1)
        if (result or 0) <= 32:
            raise RuntimeError('Installer launch was cancelled or failed')
