"""Stage 1 deterministic target selection only; no aiming/fire/controller access."""
import math

REASONS={"acquire":"首次选择","hold":"保持当前目标","switch":"原目标失效，立即转火",
         "lost":"原目标失效，无有效敌人","idle":"无有效敌人"}


class EnemyTargetSelector:
    def __init__(self,min_confidence=.05,high_confidence=.25,large_area_ratio=.01):
        if not 0<=min_confidence<=high_confidence<=1 or not 0<large_area_ratio<=1:
            raise ValueError("Invalid target thresholds")
        self.min_confidence=min_confidence;self.high_confidence=high_confidence
        self.large_area_ratio=large_area_ratio;self.target_id=None;self.timestamp=None

    def update(self,tracks,timestamp_ns,width,height):
        if self.timestamp is not None and timestamp_ns<=self.timestamp:
            raise ValueError("Target selection requires increasing source timestamps")
        if width<=0 or height<=0:raise ValueError("Invalid frame size")
        self.timestamp=timestamp_ns
        candidates=[]
        for tr in tracks:
            if (tr.get('track_id') is None or tr.get('class_id')!=0 or tr.get('state')!='observed'
                    or not tr.get('confirmed') or not tr.get('position_valid')
                    or not tr.get('association_valid',True)):
                continue
            box=tr.get('last_observed_box',tr.get('box'))
            score=tr.get('score')
            if box is None or len(box)!=4 or score is None or not all(math.isfinite(v) for v in (*box,score)):
                continue
            x1,y1,x2,y2=box
            area=max(0,min(width,x2)-max(0,x1))*max(0,min(height,y2)-max(0,y1))
            if area<=0 or not self.min_confidence<=score<=1:continue
            ratio=area/(width*height)
            priority=(0 if score>=self.high_confidence else 2)+(0 if ratio>=self.large_area_ratio else 1)
            candidates.append({'track_id':tr['track_id'],'box':list(box),'score':score,'area_pixels':area,
                               'area_ratio':ratio,'priority':priority+1})
        candidates.sort(key=lambda c:(c['priority'],-c['score'],-c['area_pixels'],c['track_id']))
        old=self.target_id
        selected=next((c for c in candidates if c['track_id']==old),None)
        if selected is not None:reason='hold'
        elif candidates:
            selected=candidates[0];reason='acquire' if old is None else 'switch'
        else:reason='idle' if old is None else 'lost'
        self.target_id=selected['track_id'] if selected else None
        return {'target_id':self.target_id,'previous_target_id':old,'reason':reason,
                'timestamp_ns':timestamp_ns,'candidate_count':len(candidates),'selected':selected,
                'ranked_candidates':candidates,'controller_output':False}
