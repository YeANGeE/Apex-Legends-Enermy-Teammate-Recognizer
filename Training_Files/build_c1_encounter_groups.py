import csv
import json
from collections import defaultdict
from pathlib import Path

dataset_root = Path(r"D:\FPS_AI_Dataset\Apex_Day1")
input_manifest = Path(r"C:\Users\15207\Documents\ChatGPT\FPS\c1_clip_manifest.csv")
output_manifest = Path(
    r"C:\Users\15207\Documents\ChatGPT\FPS\c1_grouped_manifest.csv"
)

encounter_gap_seconds = 30.0

with input_manifest.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

rows_by_process = defaultdict(list)

for row in rows:
    metadata_path = (
        dataset_root
        / row["clip_id"]
        / "metadata.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    row["process_id"] = str(metadata["process_id"])
    row["episode_start_ns"] = str(metadata["episode_start_ns"])
    row["episode_end_ns"] = str(metadata["episode_end_ns"])

    rows_by_process[row["process_id"]].append(row)

processes = sorted(
    rows_by_process.items(),
    key=lambda item: min(
        int(row["episode_start_ns"])
        for row in item[1]
    ),
)

all_grouped_rows = []

for session_number, (process_id, process_rows) in enumerate(processes, start=1):
    process_rows.sort(key=lambda row: int(row["episode_start_ns"]))

    encounter_number = 0
    current_group_end_ns = None

    for row in process_rows:
        start_ns = int(row["episode_start_ns"])
        end_ns = int(row["episode_end_ns"])

        if current_group_end_ns is None:
            gap_seconds = None
            encounter_number += 1
        else:
            gap_seconds = (start_ns - current_group_end_ns) / 1_000_000_000

            if gap_seconds > encounter_gap_seconds:
                encounter_number += 1

        row["session_id"] = f"session_{session_number:02d}"
        row["gap_from_previous_clip_s"] = (
            "" if gap_seconds is None else round(gap_seconds, 3)
        )
        row["encounter_id"] = (
            f"session_{session_number:02d}_"
            f"encounter_{encounter_number:03d}"
        )

        if current_group_end_ns is None:
            current_group_end_ns = end_ns
        else:
            current_group_end_ns = max(current_group_end_ns, end_ns)

        all_grouped_rows.append(row)

all_grouped_rows.sort(key=lambda row: int(row["clip_id"].split("_")[-1]))

fieldnames = list(all_grouped_rows[0].keys())

with output_manifest.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_grouped_rows)

encounter_ids = {
    row["encounter_id"]
    for row in all_grouped_rows
}

print("Output:", output_manifest)
print("Clips:", len(all_grouped_rows))
print("Sessions:", len(processes))
print("Encounter groups:", len(encounter_ids))
print("Gap threshold (seconds):", encounter_gap_seconds)
