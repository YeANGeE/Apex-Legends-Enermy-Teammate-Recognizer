from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FramePacket:
    global_frame_id: int
    timestamp_ns: int
    dxgi_present_ticks: int
    image: NDArray[np.uint8]
    is_duplicate: bool = False
    # Optional realtime diagnostics; timestamp_ns remains the alignment authority.
    capture_copy_start_ns: int | None = None
    capture_copy_end_ns: int | None = None
    capture_ready_ns: int | None = None


@dataclass(frozen=True, slots=True)
class GamepadPacket:
    timestamp_ns: int
    gameinput_timestamp_us: int
    buttons: int
    lx: float
    ly: float
    rx: float
    ry: float
    lt: float
    rt: float

    def pressed(self, mask: int) -> bool:
        return bool(self.buttons & mask)


@dataclass(frozen=True, slots=True)
class XInputPacket:
    timestamp_ns: int  # QPC AFTER XInputGetState: conservative observed time.
    poll_before_ns: int
    packet_number: int
    state_changed: bool
    buttons_native: int
    buttons: int  # Normalized to the existing GameInput button masks.
    lx: float
    ly: float
    rx: float
    ry: float
    lt: float
    rt: float

    def pressed(self, mask: int) -> bool:
        return bool(self.buttons & mask)
