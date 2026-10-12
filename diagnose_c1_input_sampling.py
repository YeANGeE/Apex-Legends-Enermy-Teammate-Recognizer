"""Read-only input-path comparison; no frames, training, or controller output.

Use independent processes to avoid a shared Python GIL masking high-rate input.
XInput has no device timestamp: its intervals are host-observed state changes,
not hardware report timestamps. GameInput native timestamps are kept separate.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from fps_recorder.clock import QpcClock
from fps_recorder.config import GameInputConfig
from fps_recorder.gameinput_capture import GAMEINPUT_KIND_GAMEPAD, GameInputCapture, NativeGameInput


class XGamepad(ctypes.Structure):
    _fields_ = [("buttons", ctypes.c_uint16), ("lt", ctypes.c_uint8), ("rt", ctypes.c_uint8),
                ("lx", ctypes.c_int16), ("ly", ctypes.c_int16), ("rx", ctypes.c_int16), ("ry", ctypes.c_int16)]


class XState(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("gamepad", XGamepad)]


def interval_summary(times, scale):
    gaps = [(b-a)*scale for a, b in zip(times, times[1:]) if b > a]
    ordered = sorted(gaps)
    return {"count": len(times), "median_ms": statistics.median(gaps) if gaps else None,
            "p95_ms": ordered[min(len(ordered)-1, int((len(ordered)-1)*.95))] if gaps else None,
            "max_ms": max(gaps) if gaps else None, "gaps_over_100ms": sum(g > 100 for g in gaps)}


def worker(api, seconds, event, ready, output):
    native = capture = reference = device = None
    try:
        count = Counter()
        native_times, changed_times = [], []
        last_state = None
        axes = {name: set() for name in ("lx", "ly", "rx", "ry")}
        def observe(state, stamp):
            nonlocal last_state
            if state != last_state:
                changed_times.append(stamp)
                count["state_changes"] += 1
                for name, value in zip(axes, state[-4:]):
                    axes[name].add(value)
                last_state = state
        if api == "xinput":
            dll = ctypes.WinDLL("xinput1_4.dll")
            get_state = dll.XInputGetState
            get_state.argtypes = [ctypes.c_uint32, ctypes.POINTER(XState)]
            get_state.restype = ctypes.c_uint32
            available = []
            for index in range(4):
                probe = XState()
                if get_state(index, ctypes.byref(probe)) == 0:
                    available.append(index)
            if len(available) != 1:
                raise RuntimeError(f"Expected one XInput device, connected slots: {available}")
            state = XState()
            slot = available[0]
            ready.put(api)
            if not event.wait(20):
                raise TimeoutError("Start synchronization timeout")
            start = time.perf_counter()
            while time.perf_counter()-start < seconds:
                status = int(get_state(slot, ctypes.byref(state)))
                count["calls"] += 1
                if status:
                    count[f"error_{status}"] += 1
                    continue
                count["successful_calls"] += 1
                g = state.gamepad
                observe((g.buttons, g.lt, g.rt, g.lx, g.ly, g.rx, g.ry), time.perf_counter_ns())
            timing_note = "Host-observed XInput state changes; no native hardware timestamp"
        elif api == "gameinput_direct":
            native = NativeGameInput()
            warmup_deadline = time.perf_counter()+3
            while not reference and time.perf_counter() < warmup_deadline:
                reference = native.current_reading(None)
                if not reference:
                    time.sleep(.01)
            if not reference:
                raise RuntimeError("No GameInput gamepad reading")
            device = native.reading_device(reference)
            ready.put(api)
            if not event.wait(20):
                raise TimeoutError("Start synchronization timeout")
            # The preparation countdown exceeds GameInput's history retention;
            # never use that pre-countdown reference as the test starting point.
            native.release_object(reference)
            native.release_object(device)
            reference = native.current_reading(None)
            device = None
            if not reference:
                raise RuntimeError("GameInput disconnected before measurement")
            device = native.reading_device(reference)
            start = time.perf_counter()
            while time.perf_counter()-start < seconds:
                reading = ctypes.c_void_p()
                hr = int(native._get_next_reading(native.interface, reference, GAMEINPUT_KIND_GAMEPAD,
                                                 device, ctypes.byref(reading)))
                count["calls"] += 1
                count[f"HRESULT_0x{hr & 0xFFFFFFFF:08X}"] += 1
                if hr < 0 or not reading.value:
                    if reading.value:
                        native.release_object(reading)
                    continue
                native.release_object(reference)
                reference = reading
                stamp = native.reading_timestamp_us(reading)
                native_times.append(stamp)
                g = native.reading_gamepad_state(reading)
                if g is None:
                    count["invalid_gamepad_state"] += 1
                    continue
                observe((g.buttons, g.leftTrigger, g.rightTrigger, g.leftThumbstickX,
                         g.leftThumbstickY, g.rightThumbstickX, g.rightThumbstickY), time.perf_counter_ns())
            timing_note = "Direct GetNextReading, no sleep; native timestamps separate from arrival times"
        else:
            def on_packet(p):
                if not event.is_set():
                    return
                native_times.append(p.gameinput_timestamp_us)
                observe((p.buttons, p.lt, p.rt, p.lx, p.ly, p.rx, p.ry), time.perf_counter_ns())
            capture = GameInputCapture(GameInputConfig(poll_hz=2000), QpcClock(), on_packet)
            capture.start()
            ready.put(api)
            if not event.wait(20):
                raise TimeoutError("Start synchronization timeout")
            baseline = capture.snapshot_stats()
            start = time.perf_counter()
            time.sleep(seconds)
            stats = capture.snapshot_stats()
            capture.stop()
            capture = None
            count.update(calls=stats.next_reading_calls-baseline.next_reading_calls,
                         empty_calls=stats.empty_next_reading_calls-baseline.empty_next_reading_calls,
                         errors=stats.errors-baseline.errors,
                         device_rebinds=stats.device_rebinds-baseline.device_rebinds)
            timing_note = "Unmodified recorder input module, without capture/encoding/episode manager"
        elapsed = time.perf_counter()-start
        output.put({"api": api, "duration_s": elapsed, "counts": dict(count),
                    "calls_per_s": count["calls"]/elapsed, "state_changes_per_s": count["state_changes"]/elapsed,
                    "native_readings_per_s": len(native_times)/elapsed if native_times else None,
                    "native_intervals": interval_summary(native_times, .001),
                    "host_change_intervals": interval_summary(changed_times, .000001),
                    "unique_axis_values": {k: len(v) for k, v in axes.items()}, "timing_note": timing_note,
                    "movement_verified": len(axes["lx"]) > 30 and len(axes["ly"]) > 30})
    except Exception as error:
        ready.put(api)
        output.put({"api": api, "error": repr(error)})
    finally:
        if capture:
            capture.stop()
        if native:
            native.release_object(reference)
            native.release_object(device)
            native.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    if args.check_only:
        assert ctypes.sizeof(XGamepad) == 12 and ctypes.sizeof(XState) == 16
        ctypes.WinDLL("xinput1_4.dll")
        with NativeGameInput() as api:
            print(f"检查通过：XInput结构与DLL；GameInput {api.api_version}初始化。没有测量、录制或训练。")
        return
    ctx = mp.get_context("spawn")
    event, ready, output = ctx.Event(), ctx.Queue(), ctx.Queue()
    processes = [ctx.Process(target=worker, args=(api, args.seconds, event, ready, output))
                 for api in ("xinput", "gameinput_direct", "recorder_input_only")]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            ready.get(timeout=15)
        print("请持续转动左摇杆；3秒后开始，同时测试三条路径。无需按RB。", flush=True)
        time.sleep(3)
        print(f"开始测量 {args.seconds} 秒，请持续转动左摇杆。", flush=True)
        event.set()
        results = [output.get(timeout=args.seconds+15) for _ in processes]
        for process in processes:
            process.join(timeout=5)
    finally:
        event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
    root = Path(__file__).resolve().parent / "diagnostics/input_sampling"
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    report = {"stage": "Stage 1 — Local Combat Policy", "read_only": True,
              "no_video_capture": True, "no_controller_output": True,
              "note": "Separate processes. Requires one connected controller and continuous left-stick rotation; state changes are not raw HID report counts.",
              "results": sorted(results, key=lambda r: r["api"])}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"诊断报告：{path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
