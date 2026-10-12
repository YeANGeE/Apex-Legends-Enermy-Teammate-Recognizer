# FPS AI Architecture and Research Roadmap

Status: current project baseline, effective 2026-10-07.

Execution status: paused at the user's request on 2026-10-11. Resume only on
an explicit user request; do not automatically train, test, optimize, or start
services. Current checkpoint: `docs/PROJECT_STATUS.md` and
`docs/PROGRESS_2026-10-11.md`. Stage 1 remains the current stage.

This document defines the canonical project scope, terminology, architecture,
and development order. If an older README, design note, TODO, source comment,
or experiment plan conflicts with it, this document takes precedence.

## Project objective

The project is a layered FPS embodied agent, not a single end-to-end model that
is expected to perform perception, combat, navigation, tactics, and language
reasoning at controller frequency.

The formal development stages are:

1. **Stage 1 — Local Combat Policy**
2. **Stage 2 — World Action Model + Planning / Navigation**
3. **Stage 3 — Tactical Intelligence / Model-Based Tactical Planning**
4. **Stage 4 — Language-Conditioned Communication & Cooperation**

The methodological progression is:

```text
Imitation Learning
  -> Temporal Visual Control
  -> World Action Modeling
  -> Model-Based Planning
  -> Tactical Decision Making
  -> Language-Conditioned Cooperation
```

The long-term outcome is a world-model-driven embodied agent. Combat,
navigation, tactics, and language are trained as distinct capabilities and
integrated incrementally. Low-level control runs at high frequency; high-level
reasoning runs at lower frequency.

## Stage 1 — Local Combat Policy

Stage 1 is the current implementation stage. The recorder and the first
datasets exist primarily to support it.

### Responsibilities

The Local Combat Policy handles behavior within a local engagement:

- enemy acquisition and continuous target tracking;
- crosshair control and aim tracking;
- fire and cease-fire decisions;
- ADS versus hip-fire selection;
- recoil correction;
- local combat movement, including strafing, peeking, and small forward/backward
  adjustments;
- maintaining a useful engagement distance; and
- short-horizon continuity when an enemy briefly leaves view.

It does **not** handle cross-map path finding, building traversal, stairs,
long-range route planning, active objective seeking, or the high-level decision
of whether an engagement should occur.

### Stage 1 latency budget

The warmed, steady-state C1 detector targets a p95 model-forward latency of no
more than 15 ms on the current RTX 5090 development system. Latency must be
reported as a distribution after warm-up rather than inferred from one isolated
WebUI request, because an idle GPU can produce a slower cold sample.

The approximately 150 ms upper bound for human-like reaction is a behavioral
envelope, not a compute budget. Perception, tracking, and control should finish
as quickly and consistently as practical; any deliberate acquisition delay,
smoothing, or reaction-time variation is applied after perception and must not
block the continuous tracking loop. End-to-end capture-to-action latency and
model-forward latency are measured separately.

### Initial learning problem

The initial observation may be an image history

```text
I[t-k:t]
```

optionally accompanied by action history

```text
A[t-k:t-1].
```

The output is the current controller action `A[t]`. The action schema must be
able to represent at least:

```text
LX, LY, RX, RY, LT, RT, buttons
```

Development may progress through these narrower tasks:

1. aim and target tracking;
2. fire decision;
3. ADS versus hip-fire;
4. combat movement; and
5. a joint Local Combat Policy.

The first training method is imitation learning / behavior cloning. The first
model should predict `RX`, `RY`, `RT`, and `LT`; `LX`, `LY`, buttons, temporal
context, and the complete joint policy follow after the basic pipeline is
validated.

Controller binding override confirmed 2026-10-10: this user's current Apex
profile binds ADS/aim to `LB` and fire to `RB`. The current C2 action head and
supervision therefore use `RX`, `RY`, `RB` (fire), `LB` (aim), with binary button
targets; do not treat `RT`/`LT` as fire/aim for this profile. The full recorder
schema retains axes, triggers and buttons for other bindings and later tasks.

## Recorder data contract

Each complete episode contains:

```text
episode_NNNNNN/
  video.mkv
  frames.csv
  gamepad.csv
  metadata.json
```

One validated reference clip had the following observed properties:

- 1920 x 1080 at 60 FPS;
- approximately 8.53 seconds;
- 512 video frames;
- 553 GameInput readings, approximately 65 Hz;
- zero encoder queue overflows;
- zero DXcam ring overwrites;
- every video frame decodable; and
- no actual duplicate video frames detected.

These figures describe a validated clip, not a guaranteed fixed sampling rate
or a replacement for per-episode validation.

### Dual-input recorder extension (2026-10-08)

Format version 3 adds `xinput.csv` without changing the GameInput meaning or
columns of `gamepad.csv`. Both input paths and video run continuously, including
while idle and while a previous episode is being finalized. In dual mode only
XInput RB edges trigger episodes: two seconds before press, the entire hold,
and two seconds after the last release. A press at or before that post-roll
deadline merges into the existing episode. Starting/reconnecting with RB held
does not manufacture a press. The initial two seconds are a warm-up, not
recoverable footage from before the process started.

XInput is polled in an independent process. Each successful poll records
`poll_before_ns` and `timestamp_ns` (QPC after XInputGetState), packet number,
native buttons, normalized buttons, axes, and `state_changed`. Its timestamp
means host observation, NOT a device event timestamp. Poll rate and state-change
rate are reported separately; repeated states are intentional observations,
not invented hardware reports. Data is streamed to disk through a bounded
queue; overflow/errors must be exposed, never silently treated as valid data.

`gamepad.csv` retains calibrated GameInput reading times; video retains DXGI
presentation times. All three are in the same QPC coordinate system. Do not
shift XInput onto the lower-rate GameInput state transitions, and do not infer
device latency from one matched transition. Native GameInput and high-frequency
XInput are retained separately for audit. The original default GameInput join
below remains compatible; a dataset builder selecting the recommended new
`xinput.csv` stream must do so explicitly and use the SAME causal backward join
on `timestamp_ns`. Do not mix or silently fall back between the input streams.
One predecessor input before the clip boundary may be included to seed the
first frame's causal action. Video/source gaps and truncated pre/post-roll still
require per-episode quality checks; high poll rate does not certify capture
quality or game-processing latency.

### Timestamp-based frame/action alignment

Array-index alignment is forbidden:

```text
frame[i] -> gamepad[i]  # invalid
```

Video and GameInput run at different and potentially varying rates, so index
alignment accumulates drift. Alignment must use the native timestamps recorded
in `frames.csv` and `gamepad.csv`.

The first dataset uses causal zero-order hold. For a frame with source timestamp
`t`, its action label is the newest GameInput reading that is not later than
the frame:

```text
a_t = gamepad[argmax_j timestamp_j such that timestamp_j <= t]
```

In implementation terms, this is an as-of join in the backward direction.
Future samples must not be selected by default because that introduces future
leakage. A frame with no earlier GameInput sample is not silently paired with a
future sample; it should be dropped or explicitly marked invalid by the dataset
builder.

For current episodes, `t` is `frames.csv:source_present_ns` and controller time
is `gamepad.csv:timestamp_ns`. Older frame schemas may use `timestamp_ns` in
place of `source_present_ns`. `video_pts_ns`, MKV PTS, frame index, and
`frame_id / FPS` are not synchronization authorities.

## Stage 2 — World Action Model + Planning / Navigation

Stage 2 is not called "VLA navigation." Its core system is a **World Action
Model (WAM) + Planner + Navigation Controller**.

The WAM answers:

> What will happen if I take this action?

Its basic prediction problem is:

```text
(current observation/state, candidate action)
  -> predicted future state/observation
```

The Stage 2 control flow is:

```text
Observation
  -> WAM rollout
  -> candidate future trajectories
  -> scorer / planner
  -> selected action / waypoint / skill
  -> Navigation Controller
  -> Gamepad
```

Capabilities are introduced progressively: free-space and obstacle
understanding, door/corridor/stairs understanding, movement affordance, path
prediction, local navigation, waypoint following, obstacle avoidance,
long-range navigation, building traversal, and active exploration.

For candidate action sequences `a^(1), ..., a^(N)`, the WAM predicts horizon
rollouts

```text
s_hat[t+1:t+H]^(i)
```

and a planner selects

```text
a* = argmax_i J(a^(i)).
```

Possible planning methods include sampling, beam search, the cross-entropy
method (CEM), model-predictive control (MPC), a learned value function, and a
learned reward model. These are future options, not current implementation
requirements.

## Stage 3 — Tactical Intelligence / Model-Based Tactical Planning

Stage 3 answers the high-level question:

> What should the agent do now?

It is not an aim-improvement stage. A tactical policy considers state such as
health, ammunition, enemies, teammates, objectives, and available cover, then
emits an intent or skill. The vocabulary can include:

```text
ENGAGE, DISENGAGE, RETREAT, HOLD, PUSH, FLANK, ROTATE,
TAKE_COVER, REVIVE, FOLLOW, PROTECT_TEAMMATE, LOOT,
RELOAD, REPOSITION
```

The tactical policy should normally run at approximately 1-5 Hz, not produce
60 Hz gamepad values directly. Its output dispatches to an appropriate
lower-level capability, for example:

```text
ENGAGE     -> Local Combat Policy
RETREAT    -> navigation capability
TAKE_COVER -> WAM Planner + Navigation Controller
REVIVE     -> navigation + interaction
```

Later, WAM rollouts can predict the consequences of tactical alternatives,
such as leaving cover, holding, or retreating. This develops the stage from a
reactive tactical policy into **Model-Based Tactical Planning**.

## Stage 4 — Language-Conditioned Communication & Cooperation

Stage 4 accepts voice, text, teammate pings, game markers, and contextual
commands. Its language component maps communication to a structured intent,
skill, and target:

```text
"come back; stop fighting" -> DISENGAGE + FOLLOW(teammate)
"hold this door for me"    -> HOLD + WATCH_DIRECTION(door)
"go to my marker"          -> MOVE_TO(PING)
"I am low; cover me"       -> PROTECT_TEAMMATE
```

The language component does not predict `RX`, `RY`, `LX`, or `LY` every
16.7 ms. It selects high-level tasks and invokes existing skills.

## WAM, language, policy, and controller terminology

These responsibilities must remain distinct:

| Component | Primary question | Output |
| --- | --- | --- |
| VLM/VLA or language understanding | What should I do? | task, intent, skill, target |
| World Action Model | What will happen if I do this? | predicted future trajectory/state |
| Planner / scorer | Which predicted option best serves the goal? | action sequence, waypoint, or skill |
| Policy / controller | What exact action should I execute now? | low-level gamepad action |

A VLM/VLA may extract `MOVE_TO(second floor, teammate)` from a language goal.
The WAM and planner determine a viable way to get there, and the Navigation
Controller converts the chosen waypoint or local plan into gamepad actions.

## Target layered architecture

```text
Voice / Text / Ping
        |
        v
Communication / Language Understanding
        |
        v
Tactical Policy / Tactical Planner                 low frequency (about 1-5 Hz)
        |
        v
Intent / Skill
        |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
Local Combat Policy      WAM + Planner       Interaction
30-60 Hz                 low/mid frequency   event-driven
                             |
                             v
                     Navigation Controller
                         10-60 Hz
        |                    |                    |
        +--------------------+--------------------+
                             |
                             v
                          Gamepad
```

Module boundaries should leave room for these later stages, but current code
must not over-engineer or prematurely implement them.

## Current development priorities

Work is formally in Stage 1. The ordered near-term plan is:

1. freeze and document the Recorder data format;
2. implement timestamp-based frame/action alignment;
3. build a PyTorch `FPSDataset`;
4. produce `observation_t -> action_t` examples from a clip;
5. make a small model clearly overfit one short clip;
6. verify the full `video -> dataset -> DataLoader -> model -> loss -> backprop`
   pipeline;
7. increase the amount and diversity of recorded data;
8. initially predict `RX`, `RY`, `RT`, and `LT`;
9. add `LX`, `LY`, and buttons later; and
10. add temporal context and develop the complete Local Combat Policy.

Stages 2-4 are architectural direction and research scope. They are not the
current implementation backlog.

## Extension constraints for current code

Current interfaces should preserve these distinctions without building unused
frameworks:

- observations and action labels must have explicit timestamp semantics;
- controller-action schemas should be extensible from the initial four axes to
  all axes and buttons;
- temporal context should be addable without changing synchronization truth;
- future high-level modules should communicate through structured intents,
  skills, targets, waypoints, or candidate action sequences rather than hidden
  direct dependencies; and
- training, evaluation, and dataset code should state which stage and which
  subtask they serve.

## Long-term research formulation

The common predictive objective can be written as

```text
p(s[t+1:t+n] | s[t], a[t:t+n]).
```

Predicting future state conditioned on the current state and planned actions
can ultimately support navigation, cover selection, engagement prediction,
escape and retreat, teammate assistance, and tactical planning.
