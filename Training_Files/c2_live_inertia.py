"""User-started shadow trajectory scheduling. No virtual-controller/device API."""
import csv
import threading
from pathlib import Path

import numpy as np

from c2_inertia import InertialAxes


class TrajectoryState:
    def __init__(self, response_ms=80.):
        self.axes = InertialAxes(response_ms)
        self.deadline = None
        self.source_ns = None
        self.target_id = None

    def step(self, now_ns):
        if self.deadline is not None and now_ns >= self.deadline:
            self.axes.set_target((0,0),self.deadline)
            self.deadline = None
            self.target_id = None
        position = self.axes.advance(now_ns)
        return dict(sample_ns=now_ns,source_present_ns=self.source_ns,
                    target_id=self.target_id,fresh_target=self.deadline is not None,
                    target_axes=self.axes.target.copy(),axes=position,
                    velocity=self.axes.velocity.copy(),controller_output=False)

    def command(self, correction, arrival_ns):
        self.step(arrival_ns)
        source = correction['source_present_ns'] if correction else None
        base_source = correction.get('base_source_present_ns',source) if correction else None
        valid = (correction is not None and max(source,base_source) <= arrival_ns
                 and 0 <= arrival_ns-min(source,base_source) < 150_000_000)
        if not valid:
            self.axes.set_target((0,0),arrival_ns)
            self.deadline = None
            self.target_id = None
            return
        self.axes.set_target(correction['combined_axes'],arrival_ns)
        self.source_ns = source
        self.deadline = min(source,base_source)+150_000_000
        self.target_id = correction['target_id']


class LiveShadowInertia:
    def __init__(self, clock, stop, output, response_ms=80.):
        self.clock,self.stop,self.output = clock,stop,Path(output)
        self.state = TrajectoryState(response_ms)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.intervals = []
        self.steps = []
        self.source_ages = []
        self.samples = 0
        self.publications = 0
        self.thread = None
        self.period_ns = 1_000_000

    def publish(self, correction):
        with self.lock:
            # Actual receipt after lock acquisition, never the earlier source/finish time.
            self.state.command(correction,int(self.clock.now_ns()))
            self.publications += 1

    def snapshot(self):
        with self.lock:
            return (dict(self.latest) if self.latest else None),self.error

    def start(self):
        self.thread = threading.Thread(target=self._run,name='C2ShadowInertia',daemon=True)
        self.thread.start()

    def join(self):
        if self.thread:
            self.thread.join(timeout=5)
        if self.thread and self.thread.is_alive():
            self.error = self.error or 'Shadow trajectory worker did not stop'

    def _run(self):
        previous = None
        try:
            with self.output.open('w',newline='',encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['sample_ns','source_present_ns','target_id','fresh_target',
                                 'target_rx','target_ry','inertial_rx','inertial_ry','velocity_rx','velocity_ry'])
                while not self.stop.is_set():
                    with self.lock:
                        now = int(self.clock.now_ns())
                        result = self.state.step(now)
                        self.latest = result
                    if previous:
                        self.intervals.append((now-previous['sample_ns'])/1e6)
                        self.steps.append(np.abs(np.array(result['axes'])-previous['axes']).tolist())
                    if result['fresh_target']:
                        self.source_ages.append((now-result['source_present_ns'])/1e6)
                    writer.writerow([now,result['source_present_ns'],result['target_id'],int(result['fresh_target']),
                                     *result['target_axes'],*result['axes'],*result['velocity']])
                    self.samples += 1
                    previous = result
                    # No catch-up burst of stale reports after scheduler delays.
                    remaining = (now+self.period_ns-self.clock.now_ns())/1e9
                    if remaining > 0:
                        self.stop.wait(remaining)
        except Exception as exc:
            with self.lock:
                self.error = repr(exc)

    def report(self):
        intervals = np.asarray(self.intervals)
        return dict(samples=self.samples,publications=self.publications,nominal_hz=1000,
            measured_mean_hz=float(1000/intervals.mean()) if len(intervals) else None,
            interval_median_ms=float(np.median(intervals)) if len(intervals) else None,
            interval_p95_ms=float(np.quantile(intervals,.95)) if len(intervals) else None,
            interval_max_ms=float(intervals.max()) if len(intervals) else None,
            gaps_over_5ms=int(np.sum(intervals>5)),gaps_over_20ms=int(np.sum(intervals>20)),
            max_axis_step=np.max(self.steps,axis=0).tolist() if self.steps else None,
            boundary_hits=self.state.axes.boundary_hits,error=self.error,
            worker_stopped=self.thread is None or not self.thread.is_alive(),controller_output=False,
            note='Measured shadow-state samples, not USB/device/game reports. Fresh target ages are source-to-shadow age, '
                 'not capture-to-action latency. A delayed sample may show larger displacement despite continuous internal trajectory.')
