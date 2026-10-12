"""Launch the isolated Label Studio service from a normal PyCharm run."""

from __future__ import annotations

import os
import secrets
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_STUDIO = (
    PROJECT_ROOT / ".venv-label313" / "Scripts" / "label-studio.exe"
)
DATA_DIR = PROJECT_ROOT / ".label-studio"
SECRET_FILE = DATA_DIR / "secret_key.txt"


def main() -> None:
    if not LABEL_STUDIO.is_file():
        raise FileNotFoundError(
            f"The isolated Label Studio executable was not found: {LABEL_STUDIO}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SECRET_FILE.exists():
        SECRET_FILE.write_text(secrets.token_hex(32), encoding="utf-8")

    environment = os.environ.copy()
    environment["LABEL_STUDIO_BASE_DATA_DIR"] = str(DATA_DIR)
    environment["SECRET_KEY"] = SECRET_FILE.read_text(encoding="utf-8").strip()
    environment["COLLECT_ANALYTICS"] = "0"
    # The C1 formal dataset contains 600 JPEGs (about 226 MB). Label Studio's
    # defaults allow only 100 files and 250 MB per request, so leave headroom
    # for multipart upload overhead while keeping the limit project-local.
    environment["DATA_UPLOAD_MAX_NUMBER_FILES"] = "1000"
    environment["DATA_UPLOAD_MAX_MEMORY_SIZE"] = str(1024 * 1024 * 1024)
    # Label Studio 1.23.x reads the unprefixed names below. Keep the prefixed
    # aliases as well so the launcher remains compatible with other releases.
    environment["LOCAL_FILES_SERVING_ENABLED"] = "true"
    environment["LOCAL_FILES_DOCUMENT_ROOT"] = str(PROJECT_ROOT)
    environment["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    environment["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(PROJECT_ROOT)

    print("Label Studio: http://127.0.0.1:8080")
    print("Upload limit: 1000 files / 1 GB per request")
    print(f"Local-file root: {PROJECT_ROOT}")
    print("Press PyCharm's red Stop button to stop the annotation server.")
    subprocess.run(
        [str(LABEL_STUDIO), "start", "--port", "8080"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
