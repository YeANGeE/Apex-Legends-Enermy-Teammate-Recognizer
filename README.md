# FPS Recorder

## Current checkpoint (2026-10-11)

The project is paused at Stage 1 — Local Combat Policy. C1 enemy/teammate
detection candidates, recording, tracking, and offline controller tooling are
implemented. C2 active following and sustained effective firing remain
unresolved; the current controller policy is not a completed result.

- [Project status](docs/PROJECT_STATUS.md)
- [Paused progress checkpoint](docs/PROGRESS_2026-10-11.md)
- [Architecture and roadmap](docs/ARCHITECTURE_AND_ROADMAP.md)

This repository stores source code, configuration, tests, and progress documents.
Local recordings, annotation datasets, model weights, exported engines, and
machine-specific diagnostic logs are excluded. A fresh clone needs separately
supplied datasets and weights to reproduce training or model demonstrations.
The checkpoint lists retained model paths; it does not include the model files.

## Dual-input update (2026-10-08)

The current `config.yaml` enables continuous GameInput + XInput acquisition.
XInput alone drives RB episode edges; GameInput remains an independent audit
stream. Idle means rolling buffers are active, NOT that capture has stopped.
Press -> previous 2 s + entire hold -> release + 2 s. Press again within those
2 s (including the boundary) to keep the same episode. Saving the previous
episode never stops the rolling devices. Explicit Ctrl+C still shuts down
cleanly; disk exhaustion/device failures cannot guarantee lossless recording.

New version-3 episodes add `xinput.csv`; old `gamepad.csv` columns are unchanged.
XInput timestamps are QPC after the API read, with `poll_before_ns` retained
as the read bracket; they are not native device timestamps. Match to video
`source_present_ns` by the latest input timestamp not later than the frame.
Existing GameInput dataset joins remain unchanged; select XInput explicitly
when building the new action dataset. Never use row-index matching.

`XInput: ... polls/s, ... changes/s` distinguishes repeated successful reads
from changed states. The configured 4000 polls/s costs up to one CPU core in
an isolated process and is not a promise of 4000 hardware updates/s. Its CSV
is streamed by a background writer instead of accumulating full episodes in
RAM. Multiple controllers require `xinput.user_index: 0..3`; null auto-selects
exactly one and pins that slot across reconnects. Reconnects do not fabricate
RB presses. Input errors/disconnects are exposed in metrics and metadata.

Run the existing entry point in PyCharm as module `fps_recorder.main`, with
parameters `--config config.yaml` and the FPS repository as working directory.
No duration limit is used for normal recording. Wait for `Prebuffer` to reach
about 2 s before the first press: nothing can recover footage from before
startup. Use the same interpreter as before; no new dependency is needed.

The older single-input description below describes version-2 recordings.

> **Project status:** development is currently in
> **Stage 1 — Local Combat Policy**. The canonical four-stage architecture,
> terminology, data-alignment contract, and ordered research plan are defined in
> [FPS AI Architecture and Research Roadmap](docs/ARCHITECTURE_AND_ROADMAP.md).
> That document takes precedence over older notes or plans.

Windows-only continuous data logger for imitation learning. It keeps DXGI and
Microsoft GameInput running for the lifetime of the process, detects RB edges,
and writes independent episodes containing two seconds of pre-roll, all firing
bursts, and two seconds after the last RB release.

The synchronization authority is native timestamps:

- video: `DXGI_OUTDUPL_FRAME_INFO.LastPresentTime` exposed as raw ticks by
  DXcam 0.3.0;
- controller: `IGameInputReading::GetTimestamp` from Microsoft GameInput;
- common timeline: integer nanoseconds on the Windows QPC timeline, calibrated
  by bracketing `IGameInput::GetCurrentTimestamp` with QPC reads.

Python receipt time and `frame_id / output_fps` are never used as synchronization
truth.

For every decoded video frame at zero-based index `i`, read row `i` from
`frames.csv`. Use `source_present_ns` to join it to the GameInput QPC
`timestamp_ns` in `gamepad.csv`. `video_pts_ns` is only the nominal 60 FPS
playback position; Matroska may quantize the actual container PTS. Never use
the MKV PTS or `frame_id / 60` to infer when an action occurred. A row with
`is_duplicate=1` repeats a DXGI presentation; its source timestamp remains
unchanged. The first frame of an episode can also be marked duplicate if its
source presentation was already seen in the rolling capture before the episode.

The first `FPSDataset` must use a causal backward as-of join: for each frame
timestamp `t`, choose the latest GameInput reading whose `timestamp_ns <= t`
(zero-order hold). Never pair `frame[i]` with `gamepad[i]`, and do not select a
future controller sample by default. If no earlier controller reading exists,
drop or explicitly invalidate that frame instead of introducing future leakage.

## Environment

The repository has a local `.venv`. In PyCharm, select:

```text
C:\Users\15207\Documents\ChatGPT\FPS\.venv\Scripts\python.exe
```

Dependencies and system components installed during setup:

- Python 3.14.6
- DXcam 0.3.0
- Microsoft GameInput redistributable 3.5.283
- FFmpeg 9.0.2 full build with `av1_nvenc`
- NVIDIA RTX 5090, driver 617.42

Recreate the Python environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

GameInput and FFmpeg can be provisioned with:

```powershell
winget install Microsoft.GameInput
winget install Gyan.FFmpeg
```

## Run

Review `config.yaml`, connect the controller, then run:

```powershell
.\.venv\Scripts\python.exe -m fps_recorder.main --config config.yaml
```

Press Ctrl+C once to stop. The recorder rejects new episodes, completes the
active post-roll, flushes FFmpeg and CSV/JSON, then releases GameInput and
DXcam.

The playable default is 60 FPS capture and 60 FPS storage. A 32-minute real-game
120 FPS capture run had no ring loss or encoder backlog, but the player still
felt intermittent stutter. In a same-game 60 FPS capture comparison, the player
reported clearly smoother gameplay. Use `--capture-fps 120` only for a temporary
A/B check; the override is recorded in each episode's metadata. Restoring 120
FPS for routine recording requires another validated capture implementation.

The default configuration captures the complete DXGI surface at 60 FPS and
stores each capture frame in the 60 FPS encoded video. At 120 FPS capture,
every other global frame is selected. All
GameInput readings remain available. On the current
3840x2160 desktop, the isolated FFmpeg worker scales the full frame to 1080p
on the CPU before AV1 NVENC, so the HUD and peripheral view remain present.
The `full_frame_gpu_downscale` configuration name is historical and does not
describe the current implementation. Set
`resolution_mode: center_crop` only as a fallback if a long stress test shows
that full-frame capture disrupts gameplay. At 4K that fallback region is
`(960, 540, 2880, 1620)`.

### Gameplay-stutter isolation (2026-10-06/07)

The player reported no stutter with the recorder off, slight stutter with
4K/60 DXGI capture alone (no FFmpeg or file writing), essentially no stutter
with FFmpeg/NVENC encoding alone from preloaded 4K frames at 60 FPS (no DXGI),
and worse stutter with the full recorder during complex effects. These are
subjective A/B observations in live gameplay, not a controlled frame-time
trace. They prioritize the DXGI staging readback/CPU conversion path; the
full pipeline adds contention. `benchmark_capture_stages --fps 60 --seconds
120` captured 7,199 frames without ring loss and measured approximately
0.94 ms per staging Map and 3.36 ms per 4K BGR conversion. At 120 FPS these
per-frame costs occur nearly twice as often. A 7,200-frame paced FFmpeg-only
test completed at 59.92 FPS with all packets present. Moving only FFmpeg's
scale filter to CUDA was slower in a 720-frame 4K raw-pipe benchmark (about
146 vs 167 FPS) because 4K BGR frames still crossed the CPU pipe. A true
GPU-surface-to-NVENC prototype remains unverified and is the next candidate
for restoring 120 FPS capture without disrupting gameplay.

An FFmpeg GPU-surface smoke test showed that `ddagrab` can feed 4K D3D11
frames directly into `av1_nvenc` on this machine. It is not a replacement for
the recorder: the tested build failed to initialize `scale_d3d11` for
1920x1080 output (`0x80070057`), and a D3D11-to-CUDA `hwmap` attempt returned
"Function not implemented". The direct 4K test also emitted non-monotonic DTS
warnings and has no per-frame GameInput/QPC mapping. A production GPU path
still needs full-frame 1080p GPU resize, native presentation timestamps,
duplicate accounting, and episode cut/validation.

## Output

```text
recordings/
  episode_000001/
    video.mkv
    frames.csv
    gamepad.csv
    metadata.json
```

`frames.csv` columns are `frame_id`, `source_present_ns`, `video_pts_ns`,
`is_duplicate`, `dxgi_present_ticks`, and `global_frame_id`. Older episodes
use `timestamp_ns` in place of `source_present_ns` and do not have the two new
columns. Read both schemas when combining old and new recordings.

An episode is first written as `episode_NNNNNN.inprogress` and renamed only
after the encoder and metadata are complete. An `.inprogress` directory after a
crash is deliberately retained for recovery/diagnosis.

The full 4K DXGI surface is captured as BGR24, scaled to 1080p on the CPU by the isolated
FFmpeg worker, and encoded by NVENC. BGR24 reduced the raw-memory and pipe load
by 25% versus BGRA in local testing without reducing DXcam's capture rate.

DXcam 0.3's NumPy/Cython color conversion normally uses OpenMP on 4K frames.
On the local 32-logical-core machine, that path consumed about 74% of total
CPU while idle. `capture.numpy_parallel_threshold_pixels: 1000000000` selects
the serial conversion path for 4K frames. In 12-second capture checks this
reduced recorder CPU use to about 1.6% of total CPU while maintaining about
120 capture FPS and zero ring overwrites. The setting is applied through the
DXcam 0.3 extension at startup; retest it after upgrading DXcam or changing
capture resolution/hardware.

The default AV1 NVENC profile targets 12 Mbps and caps at 14 Mbps (CQ 29).
At 14 Mbps, one hour of video is approximately 6.3 GB before container
overhead, under the 8 GB/hour storage goal. The target is about 5.4 GB/hour.
The 6.6-second validation clip produced 398 frames, 60 FPS and 12.12 MB;
its short-clip average was 14.62 Mbps, or 6.58 GB/hour if sustained.
Actual bitrate is recorded per episode; acceptable visual quality must be
validated on representative combat footage.

## Encoding benchmark

Use a representative 30-60 second 1080p60 combat sample:

```powershell
.\.venv\Scripts\python.exe -m fps_recorder.benchmark_encoder `
  C:\path\to\combat_sample.mkv
```

It tests 6/8/10/12/14 Mbps and CQ variants, producing `results.csv`,
`results.json`, encoded samples, bitrate, MB/s, projected sizes, encoder FPS,
drop count and sampled NVENC utilization. Inspect small enemies, crosshair,
fast pans, recoil and background separation before choosing a lower setting.

To isolate raw pipe, CPU scaling, pixel conversion, and NVENC throughput on
this machine, run:

```powershell
.\.venv\Scripts\python.exe -m fps_recorder.benchmark_raw_pipeline `
  D:\FPS_AI_Dataset\Apex\episode_000012\video.mkv --frames 720
```

This repeats eight decoded frames and tests both 1080p BGR and upscaled 4K BGR.
It does not include DXcam, game rendering, or changing live combat frames;
the result is a component benchmark, not a 120 FPS live capture certificate.

## Live acceptance test

Each run writes `session_YYYYMMDD_HHMMSS_metrics.csv` beside the episodes.
It records one row per status interval with both lifetime-average and interval
capture FPS, interval source-cadence gap/duplicate counts, GameInput rates,
encoder queue depth, overflow, process-tree CPU/RAM, NVENC utilization,
storage rate, and episode counts. Per-episode metadata additionally reports
queue depth after pre-roll and at close, startup time, pipe-write time,
queue-wait time, encoder flush time, and `encoder_pipe_write_fps` (frames divided
by cumulative pipe-write time, excluding waits for new frames). The old `encoder_fps` remains
frame count divided by complete worker lifetime; it includes startup and
flush and is not a steady-state encoding rate.

Run 30 minutes in the real game, include a 2-5 minute continuous/rapid-fire
episode, repeated short bursts, and an Alt+Tab/background-input trial. Check
that the queue does not rise through sustained recording, no overflows or
ring overwrites occur, and video frame count equals CSV row count. Aim for
at least 75 stored frames per second of spare component throughput (25%
margin over 60) before collecting many hours. `encoder_fps` includes real-time
waiting for frames and cannot establish this margin by itself; use the queue
trend and the raw-pipeline benchmark together. A queue peak just after an
episode starts includes the two-second pre-roll burst and alone does not prove
a steady-state backlog.

GameInput currently polls for new readings at up to 2000 Hz and consumes
successive `GetNextReading` results; it does not save only every Nth reading.
`gameinput_next_calls`, `gameinput_empty_next_calls`, and
`gameinput_max_gap_ms` help distinguish a device reporting near 60 Hz from
the capture thread failing to keep up. Software cannot create new native
readings beyond the device's report rate.

## Important engineering behavior

- RB is rising/falling-edge detected; holding it cannot retrigger.
- GameInput explicitly enables its official background-input focus policy, so
  controller readings continue while the game/recorder is not foreground.
- A new RB-down inside post-roll cancels the planned end and stays in the same
  episode.
- DXcam and GameInput are never restarted at episode boundaries.
- Encoding and file finalization run on independent workers.
- Capture callbacks use non-blocking bounded queues. Any queue overflow is
  counted rather than silently hidden.
- `frames.csv` is the authoritative mapping from encoded frame index to native
  presentation time. The MKV is CFR for encoder compatibility, but downstream
  training/alignment must use the CSV timestamps.
- When the desktop does not present a new surface, DXcam video mode duplicates
  the last image to keep the configured capture timeline. Selected rows intentionally
  retain the same native `LastPresentTime` and are counted as duplicate
  timestamps; the recorder never fabricates a new DXGI presentation time.
- The stored 60 FPS stream contains every frame at the default 60 FPS capture
  setting, or every even global frame when capture is overridden to 120 FPS.
  When live capture averages below its configured target, source
  time and nominal video playback time can drift; downstream training must use
  `source_present_ns`. The CSV is a 60 FPS observation subset, while GameInput
  retains its native reading cadence.
