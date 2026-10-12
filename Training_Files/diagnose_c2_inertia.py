"""Finite offline trajectory replay on a 1ms grid, not a real 1000Hz report test."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from c2_inertia import InertialAxes
from normalize_c1_day3_export import sha

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT/'runs/c1_live_overlay/20261010_232952_350522'


def synthetic(response_ms):
    trajectory = InertialAxes(response_ms,initial=(-1,0))
    trajectory.set_target((1,0),0)
    times = np.arange(0,401_000_000,1_000_000,dtype=np.int64)
    positions = np.array([trajectory.advance(int(t)) for t in times])
    velocity = np.diff(positions[:,0])/.001
    arrival = np.flatnonzero(np.abs(positions[:,0]-1)<=.01)
    zero = np.flatnonzero(positions[:,0]>=0)
    return times/1e6,positions[:,0],velocity,dict(response95_ms=response_ms,
        zero_crossing_ms=float(times[zero[0]]/1e6) if len(zero) else None,
        within_0_01_of_target_ms=float(times[arrival[0]]/1e6) if len(arrival) else None,
        peak_observed_speed_units_per_second=float(np.max(np.abs(velocity))),
        boundary_hits=trajectory.boundary_hits)


def load_events(run):
    events = []
    for line in (run/'frames.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        # Suggestions were unavailable at source presentation; preserve measured processing delay.
        ready = row['source_present_ns']+int(round(row['measurements']['source_age_at_finish']*1e6))
        correction = row.get('geometry_bias')
        axes = correction['combined_axes'] if correction else [0.,0.]
        events.append(dict(ready=ready,source=row['source_present_ns'],axes=axes,
                           valid=correction is not None))
    if not events or any(b['ready']<a['ready'] for a,b in zip(events,events[1:])):
        raise ValueError('Missing or unordered processing completion timeline')
    return events


def replay(events,response_ms):
    trajectory = InertialAxes(response_ms)
    times = np.arange(events[0]['ready'],events[-1]['ready']+251_000_000,1_000_000,dtype=np.int64)
    rows = []
    index = -1
    last_target = None
    for raw_time in times:
        t = int(raw_time)
        while index+1<len(events) and events[index+1]['ready']<=t:
            index += 1
        event = events[index]
        fresh = event['valid'] and 0 <= t-event['source'] <= 150_000_000
        target = event['axes'] if fresh else [0.,0.]
        # Every target switch is causal and starts from the actual current trajectory state.
        if target != last_target:
            position = trajectory.set_target(target,t)
            last_target = target.copy()
        else:
            position = trajectory.advance(t)
        rows.append([t, *target, *position,int(fresh)])
    array = np.asarray(rows)
    speeds = np.abs(np.diff(array[:,3:5],axis=0))/.001
    error = np.abs(array[:,3:5]-array[:,1:3])
    valid = array[:,5] == 1
    return array,dict(response95_ms=response_ms,nominal_grid_hz=1000,
        max_axis_step=(speeds.max(axis=0)*.001).tolist(),
        speed_max=speeds.max(axis=0).tolist(),
        speed_p95=np.quantile(speeds,.95,axis=0).tolist(),
        mean_abs_difference_from_unsmoothed=error.mean(axis=0).tolist(),
        opposing_direction_fraction=[float(np.mean(array[valid,i+1]*array[valid,i+3]<0)) if valid.any() else None for i in range(2)],
        boundary_hits=trajectory.boundary_hits,
        note='Difference is trajectory lag/distortion, not human action error. Direction reversal can temporarily preserve old velocity.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    if summary.get('error') or summary['capture_stats']['errors']:
        raise ValueError('Cannot use failed live run')
    events = load_events(args.run)
    out = ROOT/'runs/c2_inertia_review'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    fig,plots = plt.subplots(4,1,figsize=(13,12))
    tests = []
    replays = []
    for response_ms in (40.,80.,120.):
        t,x,v,case = synthetic(response_ms)
        tests.append(case)
        plots[0].plot(t,x,label=f'{response_ms:g} ms to 95%')
        plots[1].plot(t[1:],v,label=f'{response_ms:g} ms')
        replay_array,metrics = replay(events,response_ms)
        replays.append(metrics)
        if response_ms == 80:
            with (out/'replay_1000hz.csv').open('w',newline='',encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['simulated_report_ns','target_rx','target_ry','inertial_rx','inertial_ry','fresh_target'])
                for row in replay_array:
                    writer.writerow([int(row[0]),*row[1:5].tolist(),int(row[5])])
            # Show the most active contiguous 5s section, keeping the full replay in CSV.
            activity = np.abs(np.diff(replay_array[:,1:3],axis=0)).sum(axis=1)
            window = min(5000,len(activity))
            cumulative = np.r_[0.,np.cumsum(activity)]
            segment = int(np.argmax(cumulative[window:]-cumulative[:-window]))
            section = replay_array[segment:segment+5000]
            seconds = (section[:,0]-replay_array[0,0])/1e9
            for axis in range(2):
                plots[axis+2].plot(seconds,section[:,axis+1],label='BC + bias target',alpha=.7)
                plots[axis+2].plot(seconds,section[:,axis+3],label='Inertial trajectory (80ms)')
    plots[0].axhline(1,color='black',linestyle='--',alpha=.4)
    plots[0].set(title='Synthetic RX: -1 to +1 from rest',ylabel='RX',xlabel='Time (ms)')
    plots[1].set(title='Synthetic movement speed',ylabel='Axis units / second',xlabel='Time (ms)')
    plots[2].set(title='Recorded geometry suggestions: RX (selected 5s section)',ylabel='RX',xlabel='Replay time (s)')
    plots[3].set(title='Recorded geometry suggestions: RY (same section)',ylabel='RY',xlabel='Replay time (s)')
    for plot in plots:
        plot.grid(alpha=.2);plot.legend()
    fig.suptitle('Offline 1ms-grid replay: NO controller output, NO learned inertia',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/'trajectories.png',dpi=140);plt.close(fig)
    report = dict(controller_output=False,trained_with_rl=False,inertia_learned_from_human=False,
        source_run=str(args.run.resolve()),source_frames_sha256=sha(args.run/'frames.jsonl'),
        synthetic_cases=tests,replay_candidates=replays,
        parameter_note='40/80/120ms are experimental rest-to-step 95% settling times, not finite exact arrival deadlines. '
                       'Current small human calibration cannot identify intended-command inertia or full-range maximum speed.',
        scope='Offline simulated report grid, not Windows scheduling, USB reports, game reception or closed-loop performance. '
              'Target loss/stale input softly returns to zero; RB/LB and device emergency-stop behavior not implemented here.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(tests,ensure_ascii=False,indent=2))
    print('离线轨迹图与报告：',out)


if __name__ == '__main__':
    main()
