from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

from .clock import GameInputCalibration, QpcClock, calibrate_gameinput_clock
from .config import GameInputConfig
from .models import GamepadPacket


LOGGER = logging.getLogger(__name__)

GAMEINPUT_KIND_GAMEPAD = 0x00040000
GAMEINPUT_ENABLE_BACKGROUND_INPUT = 0x00000040

BUTTON_MENU = 0x00000001
BUTTON_VIEW = 0x00000002
BUTTON_A = 0x00000004
BUTTON_B = 0x00000008
BUTTON_X = 0x00000010
BUTTON_Y = 0x00000020
BUTTON_DPAD_UP = 0x00000040
BUTTON_DPAD_DOWN = 0x00000080
BUTTON_DPAD_LEFT = 0x00000100
BUTTON_DPAD_RIGHT = 0x00000200
BUTTON_LB = 0x00000400
BUTTON_RB = 0x00000800
BUTTON_LS = 0x00001000
BUTTON_RS = 0x00002000

BUTTON_COLUMNS = {
    "menu": BUTTON_MENU,
    "view": BUTTON_VIEW,
    "a": BUTTON_A,
    "b": BUTTON_B,
    "x": BUTTON_X,
    "y": BUTTON_Y,
    "dpad_up": BUTTON_DPAD_UP,
    "dpad_down": BUTTON_DPAD_DOWN,
    "dpad_left": BUTTON_DPAD_LEFT,
    "dpad_right": BUTTON_DPAD_RIGHT,
    "lb": BUTTON_LB,
    "rb": BUTTON_RB,
    "ls": BUTTON_LS,
    "rs": BUTTON_RS,
}


class GameInputGamepadState(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_uint32),
        ("leftTrigger", ctypes.c_float),
        ("rightTrigger", ctypes.c_float),
        ("leftThumbstickX", ctypes.c_float),
        ("leftThumbstickY", ctypes.c_float),
        ("rightThumbstickX", ctypes.c_float),
        ("rightThumbstickY", ctypes.c_float),
    ]


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


GAMEINPUT_V3_IID = GUID(
    0x20EFC1C7,
    0x5D9A,
    0x43BA,
    (ctypes.c_ubyte * 8)(0xB2, 0x6F, 0xB8, 0x07, 0xFA, 0x48, 0x60, 0x9C),
)


def _succeeded(hresult: int) -> bool:
    return int(hresult) >= 0


def _com_method(
    instance: ctypes.c_void_p,
    index: int,
    restype,
    *argtypes,
):
    if not instance or not instance.value:
        raise RuntimeError("null COM interface")
    vtable_ptr = ctypes.cast(
        instance, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    )
    address = vtable_ptr.contents[index]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(address)


class NativeGameInput:
    """Small ctypes binding to Microsoft GameInput v3 via the redistributable."""

    def __init__(self) -> None:
        self.api_version = "v3"
        self._dll = ctypes.WinDLL("GameInputRedist.dll")
        initialize = self._dll.GameInputInitialize
        initialize.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        initialize.restype = ctypes.c_long
        self.interface = ctypes.c_void_p()
        result = int(
            initialize(ctypes.byref(GAMEINPUT_V3_IID), ctypes.byref(self.interface))
        )
        if not _succeeded(result):
            raise OSError(
                "GameInputInitialize(v3) failed: "
                f"HRESULT 0x{result & 0xFFFFFFFF:08X}"
            )

        # IGameInput v3 indices: IUnknown(0..2), timestamp(3), current(4),
        # next(5). These are taken from Microsoft.GameInput 3.5.283.
        self._get_current_timestamp = _com_method(
            self.interface, 3, ctypes.c_uint64
        )
        self._get_current_reading = _com_method(
            self.interface,
            4,
            ctypes.c_long,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._get_next_reading = _com_method(
            self.interface,
            5,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._release = _com_method(self.interface, 2, wintypes.ULONG)
        # IGameInput::SetFocusPolicy is vtable slot 16 in v3.
        # Microsoft's default only reports input to the focused process; this
        # flag is required for RB events to survive Alt+Tab/browser focus.
        self._set_focus_policy = _com_method(
            self.interface, 16, None, ctypes.c_uint32
        )
        self._set_focus_policy(
            self.interface, GAMEINPUT_ENABLE_BACKGROUND_INPUT
        )

    def current_timestamp_us(self) -> int:
        return int(self._get_current_timestamp(self.interface))

    def current_reading(self, device: ctypes.c_void_p | None) -> ctypes.c_void_p | None:
        result = ctypes.c_void_p()
        hr = int(
            self._get_current_reading(
                self.interface,
                GAMEINPUT_KIND_GAMEPAD,
                device or None,
                ctypes.byref(result),
            )
        )
        return result if _succeeded(hr) and result.value else None

    def next_reading(
        self, reference: ctypes.c_void_p, device: ctypes.c_void_p | None
    ) -> ctypes.c_void_p | None:
        result = ctypes.c_void_p()
        hr = int(
            self._get_next_reading(
                self.interface,
                reference,
                GAMEINPUT_KIND_GAMEPAD,
                device or None,
                ctypes.byref(result),
            )
        )
        return result if _succeeded(hr) and result.value else None

    def reading_timestamp_us(self, reading: ctypes.c_void_p) -> int:
        # IGameInputReading v3: IUnknown(0..2), kind(3), timestamp(4).
        method = _com_method(reading, 4, ctypes.c_uint64)
        return int(method(reading))

    def reading_device(self, reading: ctypes.c_void_p) -> ctypes.c_void_p | None:
        # GetDevice is vtable slot 5 and returns an AddRef'd interface.
        device = ctypes.c_void_p()
        method = _com_method(
            reading, 5, None, ctypes.POINTER(ctypes.c_void_p)
        )
        method(reading, ctypes.byref(device))
        return device if device.value else None

    def reading_gamepad_state(
        self, reading: ctypes.c_void_p,
    ) -> GameInputGamepadState | None:
        # GetGamepadState is vtable slot 18 in Microsoft's v3 header.
        state = GameInputGamepadState()
        method = _com_method(
            reading,
            18,
            ctypes.c_bool,
            ctypes.POINTER(GameInputGamepadState),
        )
        return state if method(reading, ctypes.byref(state)) else None

    @staticmethod
    def release_object(instance: ctypes.c_void_p | None) -> None:
        if instance and instance.value:
            method = _com_method(instance, 2, wintypes.ULONG)
            method(instance)

    def close(self) -> None:
        if self.interface and self.interface.value:
            self._release(self.interface)
            self.interface = ctypes.c_void_p()

    def __enter__(self) -> "NativeGameInput":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


@dataclass
class GameInputStats:
    readings: int = 0
    errors: int = 0
    connected: bool = False
    last_timestamp_ns: int = 0
    rate_hz: float = 0.0
    next_reading_calls: int = 0
    empty_next_reading_calls: int = 0
    max_reading_gap_ms: float = 0.0
    device_rebinds: int = 0


class GameInputCapture:
    def __init__(
        self,
        config: GameInputConfig,
        clock: QpcClock,
        on_reading: Callable[[GamepadPacket], None],
    ) -> None:
        self.config = config
        self.clock = clock
        self.on_reading = on_reading
        self.native: NativeGameInput | None = None
        self.calibration: GameInputCalibration | None = None
        self.stats = GameInputStats()
        self._stats_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.native = NativeGameInput()
        self.calibration = calibrate_gameinput_clock(
            self.clock,
            self.native.current_timestamp_us,
            samples=self.config.calibration_samples,
            keep_best=self.config.calibration_keep_best,
        )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="GameInputCapture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self.native:
            self.native.close()
        self.native = None

    def snapshot_stats(self) -> GameInputStats:
        with self._stats_lock:
            return GameInputStats(**vars(self.stats))

    def _run(self) -> None:
        assert self.native is not None
        assert self.calibration is not None
        reference: ctypes.c_void_p | None = None
        device: ctypes.c_void_p | None = None
        last_reconnect = 0.0
        rate_started = time.perf_counter()
        rate_count = 0
        sleep_seconds = 1.0 / max(1, self.config.poll_hz)
        last_progress = time.perf_counter()
        last_probe = last_progress
        try:
            while not self._stop.is_set():
                if reference is None:
                    now = time.perf_counter()
                    if now - last_reconnect < self.config.reconnect_seconds:
                        time.sleep(min(sleep_seconds, 0.005))
                        continue
                    last_reconnect = now
                    reference = self.native.current_reading(None)
                    if reference is None:
                        with self._stats_lock:
                            self.stats.connected = False
                        continue
                    device = self.native.reading_device(reference)
                    with self._stats_lock:
                        self.stats.connected = True
                    self._emit(reference)
                    rate_count += 1
                    last_progress = time.perf_counter()
                else:
                    with self._stats_lock:
                        self.stats.next_reading_calls += 1
                    next_reading = self.native.next_reading(reference, device)
                    if next_reading is None:
                        with self._stats_lock:
                            self.stats.empty_next_reading_calls += 1
                        # A device-filtered stream can go silent after a
                        # controller reconnects or the active controller
                        # changes. Probe the unfiltered current stream and
                        # rebind only when it has a genuinely newer reading.
                        probe_now = time.perf_counter()
                        if (
                            probe_now - last_progress >= 0.25
                            and probe_now - last_probe >= 0.25
                        ):
                            last_probe = probe_now
                            current = self.native.current_reading(None)
                            if current is None:
                                self.native.release_object(reference)
                                self.native.release_object(device)
                                reference = None
                                device = None
                                with self._stats_lock:
                                    self.stats.connected = False
                            elif (
                                self.native.reading_timestamp_us(current)
                                > self.native.reading_timestamp_us(reference)
                            ):
                                self.native.release_object(reference)
                                self.native.release_object(device)
                                reference = current
                                device = self.native.reading_device(reference)
                                self._emit(reference)
                                rate_count += 1
                                with self._stats_lock:
                                    self.stats.device_rebinds += 1
                                    self.stats.connected = True
                                last_progress = time.perf_counter()
                                LOGGER.warning("GameInput stream rebound to current device")
                            else:
                                self.native.release_object(current)
                        time.sleep(sleep_seconds)
                    else:
                        self.native.release_object(reference)
                        reference = next_reading
                        self._emit(reference)
                        rate_count += 1
                        last_progress = time.perf_counter()

                elapsed = time.perf_counter() - rate_started
                if elapsed >= 1.0:
                    with self._stats_lock:
                        self.stats.rate_hz = rate_count / elapsed
                    rate_started = time.perf_counter()
                    rate_count = 0
        except Exception:
            LOGGER.exception("GameInput capture failed")
            with self._stats_lock:
                self.stats.errors += 1
                self.stats.connected = False
        finally:
            self.native.release_object(reference)
            self.native.release_object(device)

    def _emit(self, reading: ctypes.c_void_p) -> None:
        assert self.native is not None
        assert self.calibration is not None
        state = self.native.reading_gamepad_state(reading)
        if state is None:
            return
        native_us = self.native.reading_timestamp_us(reading)
        packet = GamepadPacket(
            timestamp_ns=self.calibration.map_to_qpc_ns(native_us),
            gameinput_timestamp_us=native_us,
            buttons=int(state.buttons),
            lx=float(state.leftThumbstickX),
            ly=float(state.leftThumbstickY),
            rx=float(state.rightThumbstickX),
            ry=float(state.rightThumbstickY),
            lt=float(state.leftTrigger),
            rt=float(state.rightTrigger),
        )
        with self._stats_lock:
            if self.stats.last_timestamp_ns:
                gap_ms = (packet.timestamp_ns - self.stats.last_timestamp_ns) / 1_000_000
                self.stats.max_reading_gap_ms = max(self.stats.max_reading_gap_ms, gap_ms)
            self.stats.readings += 1
            self.stats.last_timestamp_ns = packet.timestamp_ns
        self.on_reading(packet)
