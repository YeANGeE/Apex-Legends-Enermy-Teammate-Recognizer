"""Additive target-position correction for read-only offline policy diagnostics."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class GeometryBias:
    gain_x: float = 1.0
    gain_y: float = 1.0
    base_weight: float = 1.0
    correction_limit: float = .5
    invert_y: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) and 0 <= v <= 10 for v in (self.gain_x,self.gain_y,self.base_weight)):
            raise ValueError('Gains/base weight must be finite and between 0 and 10')
        if not math.isfinite(self.correction_limit) or not 0 <= self.correction_limit <= 1:
            raise ValueError('Correction limit must be between 0 and 1')

    def config(self):
        return dict(**asdict(self), trained_with_rl=False, controller_output=False,
                    error_reference='Selected box center relative to screen center; positive RX right, positive RY up assumed')

    def apply(self, base_action, box, width=1920, height=1080):
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Invalid target box')
        if not all(math.isfinite(float(v)) and v > 0 for v in (width,height)):
            raise ValueError('Invalid image size')
        x1,y1,x2,y2 = map(float,box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Target box must be nonempty and inside image')
        rx,ry = float(base_action['rx']),float(base_action['ry'])
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in (rx,ry)):
            raise ValueError('Base axes outside normalized range')
        dx,dy = (x1+x2)/2-width/2,(y1+y2)/2-height/2
        ex,ey = dx/(width/2),-dy/(height/2)
        if self.invert_y:
            ey = -ey
        clip = lambda v,limit: max(-limit,min(limit,v))
        bx,by = clip(self.gain_x*ex,self.correction_limit),clip(self.gain_y*ey,self.correction_limit)
        before_clip = [self.base_weight*rx+bx,self.base_weight*ry+by]
        return dict(error_pixels=[dx,dy],error_axes=[ex,ey],bias_axes=[bx,by],
                    base_axes=[rx,ry],combined_axes=[clip(v,1) for v in before_clip],
                    saturated=any(abs(v)>1 for v in before_clip),
                    controller_output=False,trained_with_rl=False)

    def suggest(self, suggestion, selection, source_ns):
        if not suggestion or not selection.get('selected'):
            return None
        age = source_ns-suggestion['source_present_ns']
        if age < 0 or age > 150_000_000:
            return None
        selected = selection['selected']
        if selected['score'] < .25:
            return None
        x1,y1,x2,y2 = selected['box']
        box = [max(0,min(1920,x1)),max(0,min(1080,y1)),
               max(0,min(1920,x2)),max(0,min(1080,y2))]
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        result = self.apply(suggestion['action'],box)
        result.update(source_present_ns=source_ns,base_source_present_ns=suggestion['source_present_ns'],
                      target_id=selected['track_id'])
        return result
