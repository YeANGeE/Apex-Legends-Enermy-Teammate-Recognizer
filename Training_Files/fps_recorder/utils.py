from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_executable(name_or_path: str) -> str:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return str(candidate.resolve())
    found = shutil.which(name_or_path)
    if found:
        return found
    if os.name == "nt":
        link = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links"
        for suffix in (".exe", ".cmd", ".bat"):
            path = link / f"{name_or_path}{suffix}"
            if path.is_file():
                return str(path)
    raise FileNotFoundError(f"Executable not found: {name_or_path}")


def human_bytes_per_second(mbps: float) -> float:
    return mbps / 8.0

