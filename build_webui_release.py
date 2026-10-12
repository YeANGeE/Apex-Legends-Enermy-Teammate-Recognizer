"""Build an allowlisted WebUI ZIP with model and hashes; never start the app."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "Apex-Recognizer-WebUI-v0.1.0-Windows"
MODEL = Path("runs/c1_yolo/day4_yolo26n_1280_joint_supplement_v1")


def main() -> None:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise RuntimeError("Commit tracked changes before building a release")
    files = {}
    for relative in ["c1_webui.py", "c1_webui_video.py", "c1_gallery.py",
                     "fps_recorder/__init__.py", "fps_recorder/utils.py",
                     "docs/WEBUI.md", "docs/PROJECT_STATUS.md", "docs/PROGRESS_2026-10-11.md"]:
        files[relative] = ROOT / relative
    for path in sorted((ROOT / "c1_webui_static").glob("*")):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    for path in sorted((ROOT / "packaging/webui").iterdir()):
        if path.is_file():
            files[path.name] = path
    for relative in [MODEL / "weights/best.pt", MODEL / "results.csv"]:
        files[relative.as_posix()] = ROOT / relative
    license_path = ROOT / ".venv-yolo/Lib/site-packages/ultralytics-8.4.174.dist-info/licenses/LICENSE"
    files["third_party/ULTRALYTICS-LICENSE.txt"] = license_path
    contents = {name: path.read_bytes() for name, path in files.items()}
    manifest = {"version": "0.1.0", "source_commit": revision,
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in sorted(contents.items())}}
    contents["MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive = output / f"{NAME}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(f"{NAME}/{name}", (2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, data)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert len(handle.namelist()) == len(contents)
        for name, entry in manifest["files"].items():
            assert hashlib.sha256(handle.read(f"{NAME}/{name}")).hexdigest() == entry["sha256"]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = output / f"{NAME}.zip.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest, "files": len(contents), "source_commit": revision}))


if __name__ == "__main__":
    main()
