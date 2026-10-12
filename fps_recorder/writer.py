from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .gameinput_capture import BUTTON_COLUMNS
from .models import GamepadPacket, XInputPacket


def write_frames_csv(path: Path, rows: Iterable[dict[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "source_present_ns",
                "video_pts_ns",
                "is_duplicate",
                "dxgi_present_ticks",
                "global_frame_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gamepad_csv(path: Path, rows: Iterable[GamepadPacket]) -> None:
    fields = [
        "timestamp_ns",
        "gameinput_timestamp_us",
        "lx",
        "ly",
        "rx",
        "ry",
        "lt",
        "rt",
        "buttons_raw",
        *BUTTON_COLUMNS.keys(),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values: dict[str, Any] = {
                "timestamp_ns": row.timestamp_ns,
                "gameinput_timestamp_us": row.gameinput_timestamp_us,
                "lx": format(row.lx, ".9g"),
                "ly": format(row.ly, ".9g"),
                "rx": format(row.rx, ".9g"),
                "ry": format(row.ry, ".9g"),
                "lt": format(row.lt, ".9g"),
                "rt": format(row.rt, ".9g"),
                "buttons_raw": row.buttons,
            }
            values.update(
                {name: int(row.pressed(mask)) for name, mask in BUTTON_COLUMNS.items()}
            )
            writer.writerow(values)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_xinput_csv(path: Path, rows: Iterable[XInputPacket]) -> None:
    fields = ["timestamp_ns", "poll_before_ns", "packet_number", "state_changed",
              "buttons_native", "buttons_normalized", "lx", "ly", "rx", "ry",
              "lt", "rt", *BUTTON_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: getattr(row, key) for key in fields if hasattr(row, key)}
            values["state_changed"] = int(row.state_changed)
            values["buttons_normalized"] = row.buttons
            for name in ("lx", "ly", "rx", "ry", "lt", "rt"):
                values[name] = format(getattr(row, name), ".9g")
            values.update({name: int(row.pressed(mask))
                           for name, mask in BUTTON_COLUMNS.items()})
            writer.writerow(values)
