"""Causal live C2 observations and read-only predictions; no input APIs."""
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from c2_policy import ACTION_ORDER, ACTION_SCHEMA, CombatPolicy, detection_features, history_indices
from normalize_c1_day3_export import sha
from prepare_c2_recording_manifest import OUTPUT

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs/c2_bc/joint_20261010_222327_686748/best.pt'


def causal_window(times):
    """Same history lookup as training, with the newest sample as current."""
    return history_indices(times, len(times)-1) if times else None


class ShadowPolicy:
    def __init__(self, weights=WEIGHTS):
        self.weights = Path(weights).resolve()
        checkpoint = torch.load(self.weights, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if (checkpoint['architecture'] != 'CombatPolicy_rb_lb_v2' or config['mode'] != 'joint'
                or config['action_schema'] != ACTION_SCHEMA or config['output_order'] != ACTION_ORDER):
            raise ValueError('Requires joint RX/RY/RB/LB policy')
        manifest = OUTPUT / 'recording_manifest.json'
        if config['manifest_sha256'] != sha(manifest):
            raise ValueError('Policy training manifest changed')
        rows = json.loads(manifest.read_text(encoding='utf-8'))
        if ([r['key'] for r in rows if r['split'] == 'train'] != config['train_clips']
                or [r['key'] for r in rows if r['split'] == 'val'] != config['validation_clips']):
            raise ValueError('Policy roster changed')
        self.model = CombatPolicy().eval()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.info = dict(weights=str(self.weights), weights_sha256=sha(self.weights),
            selected_epoch=checkpoint['epoch'], action_order=ACTION_ORDER,
            action_schema=ACTION_SCHEMA, training_input_stream='gamepad.csv',
            controller_output=False, smoothing_applied=False, sample_interval_ms=50,
            history_ms=[-100,-50,0], button_display_threshold=.5,
            note='Read-only action suggestions, uncalibrated button scores; not hit rate or 1000Hz output.')
        self.frames = deque(maxlen=12)
        self.latest = None
        self.last_source = None

    def start(self):
        self.model = self.model.to('cuda')
        with torch.inference_mode():
            for _ in range(10):
                self.model(torch.zeros(1,9,160,288,device='cuda'), torch.zeros(1,58,device='cuda'))
        torch.cuda.synchronize()

    def update(self, image, predictions, source_ns):
        source_ns = int(source_ns)
        if self.last_source is not None and source_ns <= self.last_source:
            return self.latest
        self.last_source = source_ns
        if self.frames and source_ns-self.frames[-1][0] > 200_000_000:
            self.frames.clear()
            self.latest = None
        if self.frames and source_ns-self.frames[-1][0] < 50_000_000:
            return self.latest
        small = cv2.cvtColor(cv2.resize(image, (288,160), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.frames.append((source_ns, small))
        times = [t for t, _ in self.frames]
        hs = causal_window(times)
        if hs is None:
            return None
        images = np.stack([self.frames[j][1] for j in hs]).transpose(0,3,1,2).reshape(9,160,288).astype(np.float32)/255
        dt = np.array([(source_ns-times[hs[0]])/1e9, (source_ns-times[hs[1]])/1e9], np.float32)
        features = np.r_[detection_features(predictions), dt]
        with torch.inference_mode():
            action = self.model(torch.from_numpy(images)[None].to('cuda'),
                                torch.from_numpy(features)[None].to('cuda')).cpu().numpy()[0]
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite shadow prediction')
        self.latest = dict(source_present_ns=source_ns, history_source_ns=[times[j] for j in hs],
                           action=dict(zip(ACTION_ORDER, map(float, action))), controller_output=False)
        return self.latest
