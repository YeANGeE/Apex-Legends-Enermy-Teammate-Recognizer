"""Read-only analysis of saved DAY4 logs; no inference, training or controller."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EPISODES = [1, 3, 6, 7, 9]


def analyze(run):
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    if config['controller_output'] or not config['target_selection_enabled']:
        raise ValueError('Expected offline target-selection run')
    result = []
    for episode in EPISODES:
        clip = run / f'episode_{episode:06d}'
        summary = json.loads((clip / 'summary.json').read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in (clip / 'tracks.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows) != summary['frames'] or [r['frame_id'] for r in rows] != list(range(len(rows))):
            raise ValueError('Incomplete frame log')
        if any(b['source_present_ns'] < a['source_present_ns'] for a, b in zip(rows, rows[1:])):
            raise ValueError('Nonmonotonic native time')
        unique = [r for r in rows if not r['reused_timestamp']]
        events = []
        invalid = []
        for r in unique:
            selection = r['target_selection']
            target = selection['target_id']
            if target is not None:
                matches = [t for t in r['tracks'] if t['track_id'] == target]
                if len(matches) != 1 or not all((matches[0]['class_id'] == 0,
                    matches[0]['state'] == 'observed', matches[0]['confirmed'],
                    matches[0]['position_valid'], matches[0]['association_valid'])):
                    invalid.append(r['frame_id'])
            if selection['reason'] in ('lost', 'switch', 'acquire'):
                events.append(dict(frame=r['frame_id'], elapsed_s=round(
                    (r['source_present_ns'] - rows[0]['source_present_ns']) / 1e9, 3),
                    reason=selection['reason'], previous=selection['previous_target_id'], target=target))
        # Closest lost -> acquire transitions are review candidates, not proven ID switches.
        candidates = []
        last_lost = None
        for event in events:
            if event['reason'] == 'lost':
                last_lost = event
            elif event['reason'] == 'acquire' and last_lost is not None:
                gap = round(event['elapsed_s'] - last_lost['elapsed_s'], 3)
                if 0 <= gap <= .5:
                    candidates.append(dict(start_frame=last_lost['frame'], end_frame=event['frame'],
                        gap_s=gap, previous_id=last_lost['previous'], new_id=event['target']))
                last_lost = None
        candidates.sort(key=lambda c: (c['previous_id'] == c['new_id'], -c['gap_s'], c['start_frame']))
        unassociated = sum(any(t['state'] == 'unassociated' for t in r['tracks']) for r in unique)
        result.append(dict(episode=episode, frames=len(rows), unique_frames=len(unique),
            unassociated_frames=unassociated, unassociated_percent=round(100*unassociated/len(unique), 2),
            target_event_counts={k:sum(e['reason'] == k for e in events) for k in ('acquire','lost','switch')},
            selection_invariant_violations=invalid, review_candidates=candidates[:3],
            predict_p95_ms=summary['prediction_call_p95_ms'],
            processing_p95_ms=summary['phase_timings_ms']['detection_and_tracking']['p95'],
            video=str(clip/'跟随演示.mp4')))
    return dict(run=str(run), weights_sha256=config['weights_sha256'],
        logged_confidence=config['confidence'], threshold_note='旧日志阈值写死，不能证明实际运行阈值；用户报告已改0.10。', clips=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'runs/c1_tracking/20261010_172436_488771')
    args = parser.parse_args()
    report = analyze(args.run.resolve())
    output = ROOT/'diagnostics/c1_day4_tracking_review'/args.run.name
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# DAY4跟踪复核', '', f"结果来源：{report['run']}", '', report['threshold_note'], '',
        '未关联比例只统计非重复源时间戳帧。获得/丢失/切换为算法事件，不能直接当作人物ID错误。', '',
        '|片段|未关联帧比例|获得/丢失/切换|predict p95(ms)|检测+外观+跟踪 p95(ms)|',
        '|---|---:|---|---:|---:|']
    for c in report['clips']:
        counts=c['target_event_counts']
        lines.append(f"|E{c['episode']}|{c['unassociated_percent']:.2f}%|{counts['acquire']}/{counts['lost']}/{counts['switch']}|{c['predict_p95_ms']:.2f}|{c['processing_p95_ms']:.2f}|")
    lines += ['', 'predict包含预处理/传输/后处理，并非纯模型forward。整体处理排除解码、绘图和编码；不能作为捕获到动作延迟，也不能将各阶段p95相加。', '',
        '## 人工复核位置', '', '打开各片段的跟随演示.mp4，按页眉帧号检查下列位置及前后约半秒。确认：目标是否仍可见、是否为同一个人、是否合理转火、是否锁到背景/队友。短暂丢失再获得只是候选问题，不能自动判定为错误。', '']
    for c in report['clips']:
        lines.append(f"- E{c['episode']}：" + ('；'.join(f"帧{x['start_frame']}–{x['end_frame']}，ID{x['previous_id']}→{x['new_id']}，间隔{x['gap_s']:.3f}s" for x in c['review_candidates']) or '没有0.5秒内丢失再获得事件，可抽看整段。'))
    violations=sum(len(c['selection_invariant_violations']) for c in report['clips'])
    lines += ['', f'目标选择状态约束违反数：{violations}。此检查只验证算法选择了已确认、当前观测、关联有效的enemy轨迹，不能证明敌我分类本身正确。', '',
        '反馈格式：E编号 + 帧号 + 同人编号跳变/合理转火/漏检/背景误锁/队友误锁。所有候选都合理且其余画面满意时，可反馈“跟踪复核通过”；再进行不含绘图编码的独立延迟测试。']
    (output/'复核报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('复核报告：', output/'复核报告.md')


if __name__ == '__main__':
    main()
