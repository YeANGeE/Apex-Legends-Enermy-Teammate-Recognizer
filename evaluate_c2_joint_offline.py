"""User-run action diagnostics on existing validation caches; no controller output."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from c2_policy import ACTION_ORDER, ACTION_SCHEMA, CombatPolicy, FPSDataset
from normalize_c1_day3_export import sha
from prepare_c2_recording_manifest import OUTPUT

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs/c2_bc/joint_20261010_222327_686748/best.pt'


def metrics(prediction, target, button_threshold=.5, active_axis_threshold=.1):
    error = np.abs(prediction - target)
    result = dict(samples=len(target), action_mae=error.mean(axis=0).tolist(),
                  action_abs_error_p95=np.quantile(error, .95, axis=0).tolist())
    result['axes'] = {}
    for i, name in enumerate(ACTION_ORDER[:2]):
        active = np.abs(target[:, i]) >= active_axis_threshold
        same = np.sign(prediction[active, i]) == np.sign(target[active, i])
        result['axes'][name] = dict(active_samples=int(active.sum()),
            same_direction_fraction=float(same.mean()) if active.any() else None,
            active_mae=float(error[active, i].mean()) if active.any() else None)
    result['buttons'] = {}
    for i, name in enumerate(ACTION_ORDER[2:], 2):
        predicted = prediction[:, i] >= button_threshold
        actual = target[:, i] == 1
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        result['buttons'][name] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
            precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,
            false_press_rate=fp/(fp+tn) if fp+tn else None,
            actual_press_fraction=float(actual.mean()),
            predicted_press_fraction=float(predicted.mean()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if checkpoint['architecture'] != 'CombatPolicy_rb_lb_v2' or config['mode'] != 'joint':
        raise ValueError('Requires the current joint C2 architecture')
    if config['action_schema'] != ACTION_SCHEMA or config['output_order'] != ACTION_ORDER:
        raise ValueError('Wrong action schema or button mapping')
    manifest = OUTPUT / 'recording_manifest.json'
    digest = sha(manifest)
    if config['manifest_sha256'] != digest:
        raise ValueError('Training manifest changed')
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    validation = [r for r in rows if r['split'] == 'val']
    if [r['key'] for r in validation] != config['validation_clips']:
        raise ValueError('Validation roster changed')
    if set(config['train_clips']) & {r['key'] for r in validation}:
        raise ValueError('Training/validation overlap')
    folders = []
    for row in validation:
        folder = OUTPUT / 'cache' / row['key']
        receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
        provenance = receipt['provenance']
        if (provenance['manifest_sha256'] != digest
                or provenance['action_schema'] != ACTION_SCHEMA
                or provenance['action_order'] != ACTION_ORDER
                or provenance['input_stream'] != 'gamepad.csv'
                or provenance['key'] != row['key']):
            raise ValueError('Validation cache provenance differs')
        for name in ('images.npy', 'features.npy', 'actions.npy', 'timestamps.npy'):
            if sha(folder / name) != receipt['cache_files_sha256'][name]:
                raise ValueError(f'Validation cache changed: {folder / name}')
        folders.append(folder)
    dataset = FPSDataset(folders)
    model = CombatPolicy()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'离线动作诊断：第{checkpoint["epoch"]}轮权重，{len(validation)}段验证录像，'
          f'{len(dataset)}个因果历史样本；不训练、不输出手柄。', flush=True)
    if args.check_only:
        print('权重、映射、缓存哈希及验证分组检查通过；尚未运行模型验证。')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    model = model.to('cuda').eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    predictions = []
    targets = []
    with torch.inference_mode():
        for images, features, target in loader:
            predictions.append(model(images.to('cuda'), features.to('cuda')).cpu().numpy())
            targets.append(target.numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite policy output')
    measured = metrics(prediction, target)
    if not np.allclose(measured['action_mae'], checkpoint['action_mae'], atol=1e-5, rtol=1e-4):
        raise ValueError('Recomputed validation MAE differs from selected checkpoint')
    clip_ids = np.array([ci for ci, _, _ in dataset.indices])
    per_clip = {r['key']: metrics(prediction[clip_ids == ci], target[clip_ids == ci])
                for ci, r in enumerate(validation) if np.any(clip_ids == ci)}
    report = dict(stage='Stage 1 — Local Combat Policy', weights=str(args.weights.resolve()),
        weights_sha256=sha(args.weights), manifest_sha256=digest, selected_epoch=checkpoint['epoch'],
        action_order=ACTION_ORDER, input_stream='gamepad.csv', split='val',
        button_threshold=.5, active_axis_threshold=.1, aggregate=measured, per_clip=per_clip,
        training_constant_baseline_mae=checkpoint['constant_baseline_mae'],
        controller_output=False, smoothing_applied=False,
        scope='Existing DAY1 validation used for checkpoint selection; not untouched test, '
              'closed-loop hit rate, high-frequency XInput imitation, or latency measurement.')
    out = ROOT / 'runs/c2_offline_review' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['clip', 'frame_id', 'source_present_ns']
                        + ['human_'+a for a in ACTION_ORDER] + ['predicted_'+a for a in ACTION_ORDER])
        for n, (ci, i, _) in enumerate(dataset.indices):
            sample = validation[ci]['samples'][i]
            writer.writerow([validation[ci]['key'], sample['frame_id'], sample['timestamp_ns']]
                            + target[n].tolist() + prediction[n].tolist())
    print(json.dumps(measured, ensure_ascii=False, indent=2))
    print('结果：', out)


if __name__ == '__main__':
    main()
