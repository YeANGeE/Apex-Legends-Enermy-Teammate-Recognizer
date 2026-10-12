"""Pure manual/AI ownership and report merging; no device access."""
import math


class OwnershipGate:
    def __init__(self, threshold=.1, quiet_seconds=1.):
        self.threshold,self.quiet_ns = threshold,int(quiet_seconds*1e9)
        self.quiet_since = None
        self.mode = 'manual'
        self.last_ns = None

    def update(self, physical, now_ns, enabled=True):
        if self.last_ns is not None and now_ns < self.last_ns:
            raise ValueError('Ownership time moved backward')
        self.last_ns = now_ns
        values = [float(physical[k]) for k in ('lx','ly','rx','ry')]
        if not all(math.isfinite(v) and abs(v)<=1 for v in values):
            raise ValueError('Invalid physical axes')
        if not enabled or max(map(abs,values)) > self.threshold:
            self.mode = 'manual'
            self.quiet_since = None
        else:
            if self.quiet_since is None:
                self.quiet_since = now_ns
            if now_ns-self.quiet_since >= self.quiet_ns:
                self.mode = 'ai'
        return self.mode


class ButtonLatch:
    def __init__(self, on=.65, off=.35):
        self.on,self.off,self.pressed = on,off,False

    def update(self, score, allowed=True):
        if not allowed:
            self.pressed = False
        elif self.pressed:
            self.pressed = score > self.off
        else:
            self.pressed = score >= self.on
        return self.pressed


def merge_report(physical, mode, ai_axes=(0,0), fire=False, aim=False):
    result = dict(physical)
    if mode == 'manual':
        return result
    if mode != 'ai':
        raise ValueError('Unknown ownership')
    result['rx'],result['ry'] = map(float,ai_axes)
    result['buttons'] = (int(physical['buttons']) & ~0x300) | (0x200 if fire else 0) | (0x100 if aim else 0)
    return result
