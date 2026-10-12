from __future__ import annotations

import subprocess

from .utils import resolve_executable


def read_encoder_utilization() -> float | None:
    try:
        command = [
            resolve_executable("nvidia-smi"),
            "--query-gpu=utilization.encoder",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def read_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            [
                resolve_executable("nvidia-smi"),
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.splitlines()[0].strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        return None

