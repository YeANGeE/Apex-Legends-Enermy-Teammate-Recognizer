"""Numbered teammate screening; immutable roster and versioned human decisions."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parent
TASKS_FILE = ROOT / "c1_annotations/day1_enemy_teammate_v1/label_studio_tasks.json"
STATE_DIR = ROOT / "c1_annotations/day1_enemy_teammate_v1/screening"


class Gallery:
    def __init__(self, state_dir: Path = STATE_DIR):
        self.tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        self.state_dir = state_dir
        self.lock = threading.RLock()
        self.paths = []
        for task in self.tasks:
            relative = parse_qs(urlsplit(task["data"]["image"]).query)["d"][0]
            path = (ROOT / relative).resolve()
            path.relative_to((ROOT / "c1_yolo_dataset/day1_v2_expanded/images").resolve())
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths.append(path)
        self.roster_hash = hashlib.sha256("\n".join(t["meta"]["image_name"] for t in self.tasks).encode()).hexdigest()

    def state(self):
        with self.lock:
            path = self.state_dir / "selection.json"
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if value["roster_hash"] != self.roster_hash:
                    raise ValueError("图片列表变更，不能沿用旧编号选择。")
                return value
            return {"roster_hash": self.roster_hash, "selected_ids": [], "completed": False}

    def validate_ids(self, ids):
        if not isinstance(ids, list) or any(type(i) is not int or not 1 <= i <= len(self.tasks) for i in ids):
            raise ValueError("图片编号无效。")
        return sorted(set(ids))

    def save(self, ids, roster_hash):
        if roster_hash != self.roster_hash:
            raise ValueError("图片编号版本不一致，请刷新页面。")
        ids = self.validate_ids(ids)
        with self.lock:
            previous = self.state()
            # A selection change reopens screening; historical final exports remain.
            value = {"roster_hash": self.roster_hash, "selected_ids": ids,
                     "completed": previous.get("completed", False) and ids == previous["selected_ids"],
                     "saved_at": datetime.now().isoformat(timespec="seconds")}
            if value["completed"]:
                value["final_result"] = previous.get("final_result")
            self.state_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.state_dir / "selection.tmp"
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.state_dir / "selection.json")
            return value

    def listing(self):
        return {"roster_hash": self.roster_hash, "total": len(self.tasks), "state": self.state(),
                "items": [{"id": i+1, "name": task["meta"]["image_name"],
                           "split": task["meta"]["split"], "encounter": task["meta"]["encounter_id"],
                           "thumbnail": f"/api/gallery/image/{i+1}?thumbnail=1",
                           "image": f"/api/gallery/image/{i+1}"}
                          for i, task in enumerate(self.tasks)]}

    def image(self, identifier: int, thumbnail: bool):
        if not 1 <= identifier <= len(self.paths):
            raise ValueError("图片编号不存在。")
        path = self.paths[identifier-1]
        if not thumbnail:
            return path.read_bytes()
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((640, 360))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()

    def finalize(self, ids, roster_hash, reviewed_all):
        if reviewed_all is not True:
            raise ValueError("请先确认已检查全部图片，未选中的确认为无队友。")
        ids = self.validate_ids(ids)
        if not ids:
            raise ValueError("尚未选择图片，请先勾选含队友或身份不确定的图片。")
        with self.lock:
            previous = self.state()
            if previous.get("completed") and ids == previous["selected_ids"] and roster_hash == self.roster_hash:
                return previous["final_result"]
            self.save(ids, roster_hash)
            snapshot = self.state_dir / ("confirmed_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            snapshot.mkdir()
            selected = set(ids)
            subset = []
            for i, task in enumerate(self.tasks, 1):
                if i in selected:
                    copied = deepcopy(task)
                    copied["meta"]["gallery_id"] = i
                    subset.append(copied)
            tasks_file = snapshot / "selected_label_studio_tasks.json"
            tasks_file.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
            decisions = [{"id": i, "image_name": t["meta"]["image_name"], "split": t["meta"]["split"],
                          "status": "needs_teammate_review" if i in selected else "user_screened_no_teammate",
                          "enemy_boxes_preserved": True}
                         for i, t in enumerate(self.tasks, 1)]
            (snapshot / "screening_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
            key = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
            process = subprocess.run([
                str(ROOT / ".venv-label313/Scripts/python.exe"),
                str(ROOT / "prepare_c1_enemy_teammate_annotation.py"),
                "--create-project", "--tasks-json", str(tasks_file),
                "--project-title", f"DAY1 队友精选 {len(ids)}张 {key}",
                "--output", str(snapshot),
            ], cwd=ROOT, capture_output=True, timeout=180)
            (snapshot / "project_creation.log").write_bytes(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"选择已保存，标注项目创建失败。日志：{snapshot / 'project_creation.log'}")
            info = json.loads((snapshot / "project_info.json").read_text(encoding="utf-8"))
            result = {"selected_count": len(ids), "unselected_count": len(self.tasks)-len(ids),
                      "project_url": info["url"], "project_title": info["title"], "snapshot": str(snapshot)}
            value = self.state()
            value.update(completed=True, final_result=result)
            temporary = self.state_dir / "selection.tmp"
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.state_dir / "selection.json")
            return result
