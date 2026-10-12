import csv
from collections import defaultdict
from pathlib import Path

input_manifest = Path(
    r"C:\Users\15207\Documents\ChatGPT\FPS\c1_grouped_manifest.csv"
)
output_csv = Path(
    r"C:\Users\15207\Documents\ChatGPT\FPS\c1_encounter_review.csv"
)

with input_manifest.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

groups = defaultdict(list)

for row in rows:
    groups[row["encounter_id"]].append(row)

review_rows = []

for encounter_id in sorted(groups):
    encounter_rows = sorted(
        groups[encounter_id],
        key=lambda row: int(row["clip_id"].split("_")[-1]),
    )

    review_rows.append(
        {
            "encounter_id": encounter_id,
            "clip_count": len(encounter_rows),
            "first_clip": encounter_rows[0]["clip_id"],
            "last_clip": encounter_rows[-1]["clip_id"],
            "decision": "KEEP",
            "merge_with": "",
            "split_before_clip": "",
            "notes": "",
        }
    )

with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=review_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(review_rows)

print("Review table:", output_csv)
print("Rows:", len(review_rows))