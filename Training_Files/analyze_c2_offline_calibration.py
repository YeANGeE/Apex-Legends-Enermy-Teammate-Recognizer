"""Read-only physical-input calibration diagnostics; no RL or output policy updates."""
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = {'hip': ROOT/'runs/c1_live_overlay/20261010_233630_558325',
        'ads': ROOT/'runs/c1_live_overlay/20261010_233815_642962'}


def backward_resample(times, values, grid):
    indices = np.searchsorted(times,grid,side='right')-1
    if np.any(indices < 0):
        raise ValueError('Grid has no causal input predecessor')
    return values[indices]


def response_pairs(observations, times, axes, buttons, left_axes, axis, mode, lag_ms):
    """~100ms same-ID intervals; inputs only from within/before observation times."""
    result = []
    for end in range(len(observations)):
        final = observations[end]
        begin = end
        while begin > 0 and final['time']-observations[begin]['time'] < 100_000_000:
            begin -= 1
        first = observations[begin]
        elapsed = (final['time']-first['time'])/1e9
        if not .08 <= elapsed <= .15:
            continue
        interval = observations[begin:end+1]
        if any(r['id'] != final['id'] or r['score'] < .25 for r in interval):
            continue
        if any(b['time']-a['time'] > 60_000_000 for a,b in zip(interval,interval[1:])):
            continue
        left = first['time']-int(lag_ms*1e6)
        right = final['time']-int(lag_ms*1e6)
        if left < times[0] or right > times[-1]:
            continue
        grid = np.arange(left,right,1_000_000,dtype=np.int64)
        index = np.searchsorted(times,grid,side='right')-1
        if np.any(grid-times[index] > 5_000_000):
            continue
        action = axes[index]
        if not np.all(buttons[index,1] == 0) or not np.all(buttons[index,0] == (mode == 'ads')):
            continue
        if np.any(np.abs(left_axes[index]) > .1):
            continue
        primary = float(action[:,axis].mean())
        secondary = float(np.abs(action[:,1-axis]).mean())
        if abs(primary) < .05 or secondary > abs(primary)/2:
            continue
        result.append((primary,(final['center'][axis]-first['center'][axis])/elapsed))
    if len(result) < 3:
        return dict(samples=len(result),correlation=None,linear_slope_px_per_second_per_axis=None,
                    usable_for_provisional_linear_fit=False)
    pairs = np.asarray(result)
    correlation = float(np.corrcoef(pairs.T)[0,1]) if np.std(pairs[:,0]) > 1e-8 and np.std(pairs[:,1]) > 1e-8 else None
    slope,intercept = np.linalg.lstsq(np.c_[pairs[:,0],np.ones(len(pairs))],pairs[:,1],rcond=None)[0]
    expected = -1 if axis == 0 else 1
    useful = (len(pairs) >= 50 and correlation is not None and expected*correlation >= .5
              and np.any(pairs[:,0] < -.05) and np.any(pairs[:,0] > .05))
    return dict(samples=len(result),correlation=correlation,
                linear_slope_px_per_second_per_axis=float(slope),intercept_px_per_second=float(intercept),
                usable_for_provisional_linear_fit=bool(useful),
                note='Overlapping intervals are not independent trials. Target center jitter, nonlinear game response and lag confound fit.')


def analyze(mode, folder):
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
    physical = summary['physical_input_stats']
    if summary['status'] != 'stopped' or physical['errors'] or physical['disconnects']:
        raise ValueError('Run failed or controller disconnected')
    if config['physical_input_recording']['aim_mode_label'] != mode:
        raise ValueError('Calibration mode differs')
    with (folder/'xinput.csv').open(newline='',encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([int(r['timestamp_ns']) for r in rows],np.int64)
    axes = np.array([[float(r['rx']),float(r['ry'])] for r in rows])
    buttons = np.array([[int(r['lb']),int(r['rb'])] for r in rows])
    left_axes = np.array([[float(r['lx']),float(r['ly'])] for r in rows])
    if len(times) != physical['readings'] or np.any(np.diff(times) <= 0) or not np.isfinite(axes).all():
        raise ValueError('Input count, time or axis integrity failed')
    grid = np.arange(times[0],times[-1]+1,1_000_000,dtype=np.int64)
    resampled = backward_resample(times,axes,grid)
    sampled_indices = np.searchsorted(times,grid,side='right')-1
    ages = grid-times[sampled_indices]
    valid = ages <= 5_000_000
    speeds = {}
    for step in (1,10):
        accepted = valid[step:] & valid[:-step]
        speed = np.abs(resampled[step:]-resampled[:-step])[accepted]/(step/1000)
        speeds[f'{step}ms_difference'] = dict(p50=np.quantile(speed,.5,axis=0).tolist(),
            p95=np.quantile(speed,.95,axis=0).tolist(),p99=np.quantile(speed,.99,axis=0).tolist(),
            max=speed.max(axis=0).tolist(),note='Observed signal changes, includes noise/quantization; not a learned inertia or hardware maximum.')
    observations = []
    last = None
    for line in (folder/'frames.jsonl').open(encoding='utf-8'):
        frame = json.loads(line)
        t = frame['source_present_ns']
        if last is not None and t <= last:
            raise ValueError('Perception source time not increasing')
        last = t
        selected = frame.get('target_selection',{}).get('selected')
        if selected:
            box = selected['box']
            observations.append(dict(time=t,id=selected['track_id'],score=selected['score'],
                                     center=[(box[0]+box[2])/2,(box[1]+box[3])/2]))
        else:
            observations.append(dict(time=t,id=None,score=0,center=[0,0]))
    response = {f'past_input_offset_{lag}ms':{
        axis_name:response_pairs(observations,times,axes,buttons,left_axes,axis,mode,lag)
        for axis,axis_name in enumerate(('rx','ry'))} for lag in (0,25,50,75,100,150)}
    return dict(mode=mode,run=str(folder),actual_recorded_seconds=float((times[-1]-times[0])/1e9),
        requested_seconds=config['seconds'],readings=len(rows),median_poll_ms=float(np.median(np.diff(times))/1e6),
        lb_fraction=float(buttons[:,0].mean()),rb_fraction=float(buttons[:,1].mean()),
        axis_min=axes.min(axis=0).tolist(),axis_max=axes.max(axis=0).tolist(),
        source_gap_count=summary['capture_stats']['source_cadence_gaps'],
        speeds_units_per_second=speeds,response_fits=response)


def main():
    report = dict(stage='Stage 1 — Local Combat Policy',controller_output=False,
        trained_with_rl=False,policy_parameters_updated=False,
        scope='Human-operated static-target calibration; causal backward input resampling. Offset sweep is exploratory '
              'and cannot be called measured hardware latency, closed-loop reward or validated gain.',
        modes={mode:analyze(mode,folder) for mode,folder in RUNS.items()})
    out = ROOT/'diagnostics/c2_calibration'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for mode,r in report['modes'].items():
        print(mode,':',r['readings'],'条，',round(r['actual_recorded_seconds'],2),'秒，LB比例',round(r['lb_fraction'],3))
        print('过去50ms偏移拟合：',r['response_fits']['past_input_offset_50ms'])
    print('分析结果：',out)


if __name__ == '__main__':
    main()
