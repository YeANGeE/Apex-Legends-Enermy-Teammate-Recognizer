# Repository development instructions

Before changing architecture, datasets, training code, module names, TODOs, or
research plans, read `docs/ARCHITECTURE_AND_ROADMAP.md`. It is the canonical
project baseline and overrides older conflicting notes and comments.

Use these formal stage names exactly:

1. Stage 1 — Local Combat Policy
2. Stage 2 — World Action Model + Planning / Navigation
3. Stage 3 — Tactical Intelligence / Model-Based Tactical Planning
4. Stage 4 — Language-Conditioned Communication & Cooperation

The project is currently implementing Stage 1. Preserve extension boundaries
for later stages, but do not implement or scaffold Stage 2-4 unless a task
explicitly requires it.

All frame/action dataset alignment must be timestamp-based. The default label
for frame time `t` is the latest GameInput reading with `timestamp_ns <= t`
(causal backward as-of join / zero-order hold). Never align video frames and
controller readings by row index, and never select a future action by default.

Keep language, tactical planning, world-action prediction, navigation control,
and high-frequency combat control as distinct responsibilities. In particular,
do not refer to Stage 2 as "VLA navigation."

When the user clearly confirms a step with wording such as "已通过", "已检查",
"完成了", or an equivalent completion statement, treat that step as accepted
and automatically continue with the already established next step. Do not ask
the user to repeat an instruction to proceed. Stop only when the next action
requires a new material choice, authorization, or information that has not
already been provided.

Use a semi-teaching workflow for all substantive project steps, not only
training: briefly explain why, the basic principle, what will change, and how
to interpret the output before giving 1-2 PyCharm operations. When a step has
no reported issue, continue the established next step without asking again.
Codex directly handles basic mechanical work: dependency installation,
model export/conversion, data organization, configuration fixes, and bounded
code diagnostics/tests. Explain relevant changes briefly without requiring
the user to repeat these operations. User executes substantive model training,
model validation/performance benchmarks, and manual judgment/annotation in
PyCharm. Service/application startup remains user-managed as specified below.
Do not skip unresolved failures or assume an unreported run has completed.

For model-training work, use a teaching-first workflow suitable for the user's
PhD learning. Explain the purpose, theory, data flow, important code, expected
outputs, and diagnostic interpretation for each substantive training step.
Unless a step is merely mechanical or the user explicitly asks Codex to run it,
guide the user to execute training themselves in PyCharm rather than running it
for them. Codex may handle repetitive data conversion, manifests, export,
visualization, and other non-instructive plumbing. Do not silently perform a
new substantive training run before teaching the step and giving the user the
PyCharm procedure.

## User-managed service startup

All services and persistent processes must be started by the user in PyCharm.
This includes Label Studio, WebUI servers, recorders, and other long-running
application services. Codex prepares or edits the scripts and configuration,
provides the PyCharm entry point, interpreter, working directory, and arguments,
then waits for the user to start them. Do not launch, host, or leave these
services running through Codex tools, background helpers, or hidden processes.
An explicit current request to have Codex start a particular service is required
to override this rule. Read-only checks of user-started services and bounded
mechanical data preparation remain allowed. Never stop or restart user-started
services without the user's authorization.

Starting with the Day2 annotation workflow, use Chinese user-facing labels in
Label Studio and model demonstration UIs. Keep stable internal identifiers in
English for dataset/code compatibility: display `敌人` for `enemy`, `队友` for
`teammate`, `可用` for `usable`, and `不确定` for `uncertain`.
