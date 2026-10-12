"""Readable summaries of unmodified shadow suggestions; never generates control."""
from collections import deque


class ShadowDisplay:
    def __init__(self):
        self.samples = deque(maxlen=64)
        self.last_ns = None
        self.fire_since_ns = None

    def observe(self, suggestion):
        if not suggestion:
            return
        t = suggestion['source_present_ns']
        if self.last_ns is not None and t <= self.last_ns:
            return
        if self.last_ns is not None and t-self.last_ns > 200_000_000:
            self.samples.clear()
            self.fire_since_ns = None
        a = suggestion['action']
        if a['rb'] >= .5:
            if self.fire_since_ns is None:
                self.fire_since_ns = t
        else:
            self.fire_since_ns = None
        self.last_ns = t
        self.samples.append(suggestion)
        while self.samples and t-self.samples[0]['source_present_ns'] > 1_000_000_000:
            self.samples.popleft()

    def text(self):
        if not self.samples:
            return '等待因果历史'
        a = self.samples[-1]['action']
        count = len(self.samples)
        values = [s['action'] for s in self.samples]
        fire_switches = sum((b['rb'] >= .5) != (c['rb'] >= .5) for b, c in zip(values, values[1:]))
        duration = (self.last_ns-self.fire_since_ns)/1e6 if self.fire_since_ns is not None else 0
        fire_fraction = sum(v['rb'] >= .5 for v in values)/count
        aim_fraction = sum(v['lb'] >= .5 for v in values)/count
        return (f"最新快照：RX {a['rx']:+.2f}  RY {a['ry']:+.2f}\n"
                f"开火分数 {a['rb']:.2f}：{'建议按下' if a['rb'] >= .5 else '建议释放'}\n"
                f"当前连续建议开火 {duration:.0f}ms\n"
                f"最近≤1秒：开火按下样本 {fire_fraction:.0%}，切换 {fire_switches}次\n"
                f"瞄准分数 {a['lb']:.2f}，按下样本 {aim_fraction:.0%}\n"
                f"RX范围 {min(v['rx'] for v in values):+.2f}～{max(v['rx'] for v in values):+.2f}")
