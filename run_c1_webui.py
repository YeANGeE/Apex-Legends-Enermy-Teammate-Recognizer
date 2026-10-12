"""Desktop launcher: start the local YOLO WebUI silently, log errors, open browser."""
from __future__ import annotations

import ctypes
import json
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from c1_webui import CHECKPOINT, WEBUI_VERSION

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:7860"
EXPECTED_CHECKPOINT = CHECKPOINT


def status():
    try:
        with urllib.request.urlopen(URL + "/api/status", timeout=1) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def matches(info):
    return info and info.get("webui_version") == WEBUI_VERSION and info.get("backend") == "yolo" and Path(info.get("checkpoint", "")).resolve() == EXPECTED_CHECKPOINT.resolve()


def main():
    info = status()
    if info:
        if not matches(info):
            raise RuntimeError("7860 端口上运行的是其他版本。请关闭旧 WebUI 后再次启动。")
        webbrowser.open(URL)
        return
    python = ROOT / ".venv-yolo/Scripts/python.exe"
    if not python.is_file():
        raise FileNotFoundError(python)
    logs = ROOT / "c1_webui_outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / ("startup_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
    with log.open("wb") as handle:
        process = subprocess.Popen([str(python), "-u", str(ROOT / "c1_webui.py"), "--no-browser"],
                                   cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(90):
        if process.poll() is not None:
            raise RuntimeError(f"服务启动失败，请查看日志：{log}")
        info = status()
        if matches(info):
            webbrowser.open(URL)
            return
        time.sleep(1)
    raise RuntimeError(f"启动较慢，稍后访问 {URL}，或查看日志：{log}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        ctypes.windll.user32.MessageBoxW(0, str(error), "YOLO WebUI 启动提示", 0x10)
