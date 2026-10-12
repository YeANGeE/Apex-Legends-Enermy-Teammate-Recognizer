from __future__ import annotations

import ctypes
import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from .clock import QpcClock
from .config import XInputConfig
from .models import XInputPacket

LOGGER = logging.getLogger(__name__)
# XInput native masks -> established dataset GameInput masks.
BUTTON_MAP = ((0x10, 1), (0x20, 2), (0x1000, 4), (0x2000, 8),
              (0x4000, 16), (0x8000, 32), (1, 64), (2, 128),
              (4, 256), (8, 512), (0x100, 1024), (0x200, 2048),
              (0x40, 4096), (0x80, 8192))


class NativePad(ctypes.Structure):
    _fields_ = [("buttons", ctypes.c_uint16), ("lt", ctypes.c_uint8),
                ("rt", ctypes.c_uint8), ("lx", ctypes.c_int16),
                ("ly", ctypes.c_int16), ("rx", ctypes.c_int16),
                ("ry", ctypes.c_int16)]


class NativeState(ctypes.Structure):
    _fields_ = [("number", ctypes.c_uint32), ("pad", NativePad)]


def make_packet(state: NativeState, before: int, after: int,
                changed: bool) -> XInputPacket:
    p = state.pad
    return XInputPacket(
        after, before, int(state.number), changed, int(p.buttons),
        sum(target for source, target in BUTTON_MAP if p.buttons & source),
        *(max(-1.0, value / 32767.0) for value in (p.lx, p.ly, p.rx, p.ry)),
        p.lt / 255.0, p.rt / 255.0,
    )


def _worker(config: XInputConfig, stop, output) -> None:
    """Dedicated process: QPC acquisition timestamps survive IPC batching.

    A paced busy poll deliberately trades up to one CPU core for sub-ms
    acquisition. It never changes timer resolution, drivers, or controller
    output. No invented device timestamps and no silent IPC overflow.
    """
    try:
        dll = ctypes.WinDLL("xinput1_4.dll")
        get_state = dll.XInputGetState
        get_state.argtypes = [ctypes.c_uint32, ctypes.POINTER(NativeState)]
        get_state.restype = ctypes.c_uint32
        clock = QpcClock()
        interval = 1_000_000_000 // config.poll_hz
        batch_interval = config.batch_ms * 1_000_000
        slot = config.user_index
        previous = None
        connected = False
        batch = []
        last_batch = clock.now_ns()
        due = last_batch
        output.put(("ready", None), timeout=2)
        while not stop.is_set():
            if slot is None:
                found = []
                for index in range(4):
                    probe = NativeState()
                    if get_state(index, ctypes.byref(probe)) == 0:
                        found.append(index)
                if len(found) > 1:
                    raise RuntimeError("Multiple XInput controllers: set xinput.user_index explicitly")
                if not found:
                    stop.wait(0.25)
                    continue
                slot = found[0]
            now = clock.now_ns()
            if now < due:
                continue
            state = NativeState()
            before = clock.now_ns()
            result = int(get_state(slot, ctypes.byref(state)))
            after = clock.now_ns()
            due = after + interval  # never invent catch-up samples after stalls
            if result == 1167:
                if batch:
                    output.put(("rows", batch), timeout=0.1)
                    batch = []
                if connected:
                    output.put(("disconnect", after), timeout=0.1)
                connected = False
                previous = None
                # Pin auto-selected slot across reconnects; do not switch players.
                stop.wait(0.25)
                continue
            if result:
                raise OSError(result, "XInputGetState failed")
            p = state.pad
            identity = (state.number, p.buttons, p.lx, p.ly, p.rx, p.ry, p.lt, p.rt)
            changed = previous is not None and identity != previous
            previous = identity
            connected = True
            batch.append(make_packet(state, before, after, changed))
            if after - last_batch >= batch_interval:
                output.put(("rows", batch), timeout=0.1)
                batch = []
                last_batch = after
        if batch:
            output.put(("rows", batch), timeout=2)
    except Exception as exc:
        output.put(("error", repr(exc)), timeout=2)
    finally:
        output.put(("done", None), timeout=2)


@dataclass
class XInputStats:
    readings: int = 0
    state_changes: int = 0
    rate_hz: float = 0.0
    change_rate_hz: float = 0.0
    max_poll_gap_ms: float = 0.0
    max_call_width_ms: float = 0.0
    last_timestamp_ns: int = 0
    connected: bool = False
    errors: int = 0
    disconnects: int = 0
    error_message: str | None = None


class XInputCapture:
    def __init__(self, config: XInputConfig,
                 on_reading: Callable[[XInputPacket], None],
                 on_disconnect: Callable[[int], None]) -> None:
        self.config = config
        self.on_reading = on_reading
        self.on_disconnect = on_disconnect
        context = mp.get_context("spawn")
        self._stop = context.Event()
        self._queue = context.Queue(maxsize=256)
        self._process = context.Process(target=_worker,
                                        args=(config, self._stop, self._queue),
                                        name="XInputSampler")
        self._thread = None
        self._lock = threading.Lock()
        self.stats = XInputStats()

    def start(self) -> None:
        self._process.start()
        try:
            kind, payload = self._queue.get(timeout=15)
            if kind != "ready":
                raise RuntimeError(f"XInput startup failed: {payload}")
        except Exception:
            self.stop()
            raise
        self._thread = threading.Thread(target=self._consume,
                                        name="XInputDelivery", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._process.pid is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=3)
                with self._lock:
                    self.stats.errors += 1
                    self.stats.error_message = "XInput worker forced to stop"

    def snapshot_stats(self) -> XInputStats:
        with self._lock:
            return replace(self.stats)

    def _consume(self) -> None:
        rate_start = time.perf_counter()
        count = changes = 0
        try:
            while True:
                try:
                    kind, payload = self._queue.get(timeout=0.25)
                except queue.Empty:
                    if not self._process.is_alive():
                        raise RuntimeError("XInput worker exited without completion message")
                    with self._lock:
                        self.stats.rate_hz = self.stats.change_rate_hz = 0.0
                    continue
                if kind == "done":
                    return
                if kind == "error":
                    raise RuntimeError(payload)
                if kind == "disconnect":
                    with self._lock:
                        self.stats.connected = False
                        self.stats.disconnects += 1
                    self.on_disconnect(payload)
                    continue
                for row in payload:
                    with self._lock:
                        s = self.stats
                        if s.last_timestamp_ns:
                            s.max_poll_gap_ms = max(s.max_poll_gap_ms,
                                (row.timestamp_ns - s.last_timestamp_ns) / 1e6)
                        s.max_call_width_ms = max(s.max_call_width_ms,
                            (row.timestamp_ns - row.poll_before_ns) / 1e6)
                        s.readings += 1
                        s.state_changes += int(row.state_changed)
                        s.last_timestamp_ns = row.timestamp_ns
                        s.connected = True
                    self.on_reading(row)
                    count += 1
                    changes += int(row.state_changed)
                elapsed = time.perf_counter() - rate_start
                if elapsed >= 1:
                    with self._lock:
                        self.stats.rate_hz = count / elapsed
                        self.stats.change_rate_hz = changes / elapsed
                    rate_start = time.perf_counter()
                    count = changes = 0
        except Exception as exc:
            LOGGER.exception("XInput delivery failed")
            self._stop.set()
            with self._lock:
                self.stats.errors += 1
                self.stats.connected = False
                self.stats.error_message = str(exc)
