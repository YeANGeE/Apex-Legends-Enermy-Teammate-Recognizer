"""Validate selected-task import with a rolled-back DB transaction, no fake labels."""
import json
from prepare_c1_enemy_teammate_annotation import OUTPUT, create_project

if __name__ == "__main__":
    tasks = json.loads((OUTPUT / "label_studio_tasks.json").read_text(encoding="utf-8"))
    create_project([tasks[0], tasks[-1]], title="C1 gallery QA rollback", dry_run=True)
