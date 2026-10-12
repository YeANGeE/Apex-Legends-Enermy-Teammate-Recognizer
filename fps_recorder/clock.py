from __future__ import annotations

import ctypes
import statistics
from dataclasses import asdict, dataclass
from ctypes import wintypes
from typing import Callable


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_query_counter = _kernel32.QueryPerformanceCounter
_query_counter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
_query_counter.restype = wintypes.BOOL
_query_frequency = _kernel32.QueryPerformanceFrequency
_query_frequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
_query_frequency.restype = wintypes.BOOL


class QpcClock:
    """Exact integer conversion between Windows QPC ticks and nanoseconds."""

    def __init__(self) -> None:
        value = ctypes.c_longlong()
        if not _query_frequency(ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.frequency = int(value.value)

    def ticks(self) -> int:
        value = ctypes.c_longlong()
        if not _query_counter(ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value.value)

    def ticks_to_ns(self, ticks: int) -> int:
        return (int(ticks) * 1_000_000_000) // self.frequency

    def now_ns(self) -> int:
        return self.ticks_to_ns(self.ticks())


@dataclass(frozen=True)
class GameInputCalibration:
    offset_ns: int
    scale_ns_per_us: int
    qpc_frequency: int
    samples: int
    kept_samples: int
    median_call_width_ns: int
    max_kept_call_width_ns: int
    offset_mad_ns: int

    def map_to_qpc_ns(self, gameinput_timestamp_us: int) -> int:
        return int(gameinput_timestamp_us) * self.scale_ns_per_us + self.offset_ns

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def calibrate_gameinput_clock(
    clock: QpcClock,
    gameinput_now_us: Callable[[], int],
    *,
    samples: int = 128,
    keep_best: int = 24,
) -> GameInputCalibration:
    """Map GameInput microseconds to QPC ns using low-call-latency samples.

    GameInput documents its reading timestamps and GetCurrentTimestamp on the
    same microsecond timeline. Bracketing the latter with QPC gives an offset
    estimate without pretending that Python receipt time is the event time.
    """

    observations: list[tuple[int, int]] = []
    for _ in range(samples):
        before = clock.now_ns()
        gameinput_us = int(gameinput_now_us())
        after = clock.now_ns()
        midpoint = (before + after) // 2
        observations.append((after - before, midpoint - gameinput_us * 1_000))
    observations.sort(key=lambda item: item[0])
    kept = observations[: max(1, min(keep_best, len(observations)))]
    offsets = [offset for _, offset in kept]
    offset_ns = int(statistics.median(offsets))
    deviations = [abs(value - offset_ns) for value in offsets]
    return GameInputCalibration(
        offset_ns=offset_ns,
        scale_ns_per_us=1_000,
        qpc_frequency=clock.frequency,
        samples=len(observations),
        kept_samples=len(kept),
        median_call_width_ns=int(statistics.median(width for width, _ in kept)),
        max_kept_call_width_ns=max(width for width, _ in kept),
        offset_mad_ns=int(statistics.median(deviations)),
    )

