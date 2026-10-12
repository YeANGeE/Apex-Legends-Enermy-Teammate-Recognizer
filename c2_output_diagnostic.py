"""Bounded offline diagnostic; does not access devices or infer actions."""
from c2_inertia import InertialAxes


class DiagnosticLook:
    def __init__(self, response_ms=80):
        self.response_ms = response_ms
        self.axes = InertialAxes(response_ms)
        self.start_ns = None

    def step(self, now_ns, enabled):
        if not enabled:
            self.start_ns = None
            self.axes = InertialAxes(self.response_ms)
            return [0., 0.]
        if self.start_ns is None:
            self.start_ns = now_ns
        phase = int((now_ns-self.start_ns)//1_000_000_000) % 4
        target = (.45, 0., -.45, 0.)[phase]
        self.axes.set_target((target, 0.), now_ns)
        return self.axes.position.copy()

def readback_row(now_ns, mode, slot, native, previous):
    p = native.pad
    normalize = lambda value: max(-1., value/32767.)
    return [now_ns, mode, slot, native.number, normalize(p.rx), normalize(p.ry),
            int(p.buttons), previous['rx'], previous['ry'], previous['buttons']]
