"""Causal second-order joystick trajectory candidate. No device/report APIs.

The state is position plus velocity. Changing targets preserves both, so an
ongoing movement decelerates before it reverses. Parameters are not learned.
"""
import math


class InertialAxes:
    def __init__(self, response95_ms=80., initial=(0.,0.)):
        if not math.isfinite(response95_ms) or not 10 <= response95_ms <= 1000:
            raise ValueError('response95_ms must be finite, between 10 and 1000')
        self.position = self._axes(initial)
        self.velocity = [0.,0.]
        self.target = self.position.copy()
        self.timestamp_ns = None
        # (1+z)*exp(-z)=0.05; exact critically-damped rest-to-step response.
        self.omega = 4.743864518 / (response95_ms/1000)
        self.boundary_hits = 0

    @staticmethod
    def _axes(values):
        if len(values) != 2 or not all(math.isfinite(float(v)) and -1 <= float(v) <= 1 for v in values):
            raise ValueError('Two finite normalized axes required')
        return [float(v) for v in values]

    def advance(self, timestamp_ns):
        if not isinstance(timestamp_ns,int) or timestamp_ns < 0:
            raise ValueError('Integer nonnegative timestamp required')
        if self.timestamp_ns is None:
            self.timestamp_ns = timestamp_ns
            return self.position.copy()
        if timestamp_ns < self.timestamp_ns:
            raise ValueError('Trajectory time cannot move backward')
        dt = (timestamp_ns-self.timestamp_ns)/1e9
        decay = math.exp(-self.omega*dt)
        for axis in range(2):
            error = self.position[axis]-self.target[axis]
            coefficient = self.velocity[axis]+self.omega*error
            position = self.target[axis]+(error+coefficient*dt)*decay
            velocity = (self.velocity[axis]-self.omega*coefficient*dt)*decay
            if position < -1 or position > 1:
                position = max(-1.,min(1.,position))
                velocity = 0.
                self.boundary_hits += 1
            self.position[axis],self.velocity[axis] = position,velocity
        self.timestamp_ns = timestamp_ns
        return self.position.copy()

    def set_target(self, axes, timestamp_ns):
        target = self._axes(axes)
        # Advance under the OLD target up to the command arrival, then retarget.
        self.advance(timestamp_ns)
        self.target = target
        return self.position.copy()
