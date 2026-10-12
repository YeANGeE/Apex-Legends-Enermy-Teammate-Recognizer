"""Apply the human-reviewed C1 encounter grouping without touching raw data."""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_MANIFEST = Path(__file__).with_name("c1_grouped_manifest.csv")
OUTPUT_MANIFEST = Path(__file__).with_name("c1_curated_manifest.csv")


# Inclusive episode ranges. Each tuple is one manually reviewed encounter.
INCLUDED_GROUPS = [
    (13, 29),
    (30, 33),
    (34, 34),
    (35, 36),
    (37, 37),
    (38, 38),
    (39, 39),
    (40, 41),
    (42, 43),
    (44, 45),
    (46, 47),
    (48, 49),
    (50, 50),
    (51, 51),
    (52, 53),
    (54, 54),
    (55, 55),
    (56, 56),
    (58, 61),
    (63, 65),
    (66, 67),
    (71, 73),
    (74, 76),
    (77, 77),
    (78, 78),
    (79, 80),
    (81, 82),
    (83, 88),
    (89, 91),
    (92, 94),
    (95, 97),
    (98, 99),
    (102, 103),
    (104, 105),
    (106, 108),
    (109, 111),
    (112, 115),
    (116, 118),
    (119, 119),
    (120, 122),
    (124, 126),
    (127, 135),
    (136, 138),
    (139, 145),
    (147, 150),
    (151, 155),
    (156, 157),
    (158, 158),
    (159, 160),
    (161, 161),
    (162, 163),
    (164, 164),
    (165, 170),
    (171, 172),
    (173, 173),
    (174, 176),
    (177, 181),
    (182, 183),
    (185, 185),
    (186, 186),
    (188, 188),
    (190, 190),
    (191, 194),
    (195, 196),
    (198, 199),
    (200, 200),
    (201, 201),
    (202, 202),
    (203, 203),
    (204, 205),
    (206, 206),
    (207, 207),
    (208, 208),
    (209, 209),
    (210, 210),
]


EXCLUDED_EPISODES = {
    57: "loading_screen",
    62: "loading_screen",
    68: "downed_no_valid_view",
    69: "downed_no_valid_view",
    70: "loading_screen",
    100: "results_screen",
    101: "results_screen",
    123: "loading_screen",
    146: "loading_screen",
    184: "downed_no_valid_view",
    187: "results_screen",
    189: "map_screen",
    197: "map_screen",
}


REVIEW_NOTES = {
    209: "retreat_no_firing; retained for C1 visual review",
}


def episode_number(clip_id: str) -> int:
    return int(clip_id.rsplit("_", 1)[1])


def main() -> None:
    with INPUT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    input_episodes = {episode_number(row["clip_id"]) for row in rows}
    expected_episodes = set(range(13, 211))
    if input_episodes != expected_episodes:
        missing = sorted(expected_episodes - input_episodes)
        unexpected = sorted(input_episodes - expected_episodes)
        raise ValueError(
            f"Unexpected input manifest coverage; missing={missing}, "
            f"unexpected={unexpected}"
        )

    encounter_for_episode: dict[int, str] = {}
    for group_index, (first_episode, last_episode) in enumerate(
        INCLUDED_GROUPS, start=1
    ):
        encounter_id = f"encounter_{group_index:03d}"
        for episode in range(first_episode, last_episode + 1):
            if episode in encounter_for_episode:
                raise ValueError(f"Episode {episode} appears in multiple groups")
            encounter_for_episode[episode] = encounter_id

    reviewed_episodes = set(encounter_for_episode) | set(EXCLUDED_EPISODES)
    if reviewed_episodes != expected_episodes:
        missing = sorted(expected_episodes - reviewed_episodes)
        unexpected = sorted(reviewed_episodes - expected_episodes)
        raise ValueError(
            f"Manual review is incomplete; missing={missing}, "
            f"unexpected={unexpected}"
        )

    overlap = set(encounter_for_episode) & set(EXCLUDED_EPISODES)
    if overlap:
        raise ValueError(f"Episodes are both included and excluded: {sorted(overlap)}")

    curated_rows = []
    for row in sorted(rows, key=lambda item: episode_number(item["clip_id"])):
        episode = episode_number(row["clip_id"])
        curated = dict(row)
        curated["auto_encounter_id"] = curated.pop("encounter_id")
        curated["encounter_id"] = encounter_for_episode.get(episode, "")
        curated["c1_include"] = "1" if episode in encounter_for_episode else "0"
        curated["exclusion_reason"] = EXCLUDED_EPISODES.get(episode, "")
        curated["review_note"] = REVIEW_NOTES.get(episode, "")
        curated_rows.append(curated)

    fieldnames = list(curated_rows[0])
    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curated_rows)

    included_count = sum(row["c1_include"] == "1" for row in curated_rows)
    excluded_count = len(curated_rows) - included_count
    print("Output:", OUTPUT_MANIFEST)
    print("Reviewed clips:", len(curated_rows))
    print("Included clips:", included_count)
    print("Excluded clips:", excluded_count)
    print("Manual encounter groups:", len(INCLUDED_GROUPS))


if __name__ == "__main__":
    main()
