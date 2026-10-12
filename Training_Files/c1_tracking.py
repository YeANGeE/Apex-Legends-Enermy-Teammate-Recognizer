"""Stage 1 — Local Combat Policy: causal screen-space tracking, no control.

Simple greedy association baseline, NOT appearance re-identification or world
motion compensation. Camera motion/crossings can cause ID swaps. Predicted
positions never become detections or confident enemy evidence.
"""
from dataclasses import dataclass, field
from math import hypot, isfinite, exp


def overlap(a, b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda c:(c[2]-c[0])*(c[3]-c[1])
    return intersection/max(1e-9,area(a)+area(b)-intersection)


@dataclass
class Track:
    identifier: int
    box: list
    last_seen_ns: int
    velocity: list=field(default_factory=lambda:[0.0]*4)
    hits: int=1
    streak: int=1
    votes: list=field(default_factory=lambda:[0.0,0.0])
    last_class: int=0
    score: float=0.0
    confirmed: bool=False
    identity_history: list=field(default_factory=list)
    stable_class: int | None=None
    identity_blocked: bool=False
    identity_streak: int=0
    identity_candidate: int | None=None
    reconfirming: bool=False


    def predict(self,t):
        dt=max(0,(t-self.last_seen_ns)/1e9)
        box=[x+v*dt for x,v in zip(self.box,self.velocity)]
        box[2]=max(box[0]+2,box[2]);box[3]=max(box[1]+2,box[3])
        return box


class ScreenTracker:
    def __init__(self,max_lost_s=.2,confirm_hits=3,memory_s=.8):
        if not 0<max_lost_s<=memory_s<=2 or confirm_hits<1:
            raise ValueError("Invalid tracker settings")
        self.max_lost_ns=round(max_lost_s*1e9)
        self.memory_ns=round(memory_s*1e9)
        self.confirm_hits=confirm_hits
        self.tracks={}
        self.next_id=1
        self.timestamp=None
        self.events=[]

    def reset(self):
        self.tracks.clear()  # Keep monotonic IDs within a clip across cuts.

    def update_identity(self,tr,t,cls,score,reconfirm=False):
        """Conflict immediately blocks output; sustained evidence can change it.

        All evidence is past/current and time-windowed, with real timestamps.
        Stored identity is historical information, NOT proof of current class.
        """
        if reconfirm:
            tr.identity_history=[];tr.identity_streak=0;tr.identity_candidate=None
            tr.identity_blocked=True
            tr.reconfirming=True
        tr.identity_history=[r for r in tr.identity_history if t-r[0]<=400_000_000]
        tr.identity_history.append((t,cls,score))
        tr.identity_streak=tr.identity_streak+1 if tr.identity_candidate==cls else 1
        tr.identity_candidate=cls
        if tr.stable_class is not None and cls!=tr.stable_class:
            if not tr.identity_blocked:self.events.append('identity_conflict')
            tr.identity_blocked=True
        votes=[0.,0.]
        for stamp,c,weight in tr.identity_history:
            votes[c]+=weight*exp(-(t-stamp)/120_000_000)
        tr.votes=votes
        share=votes[cls]/max(1e-9,sum(votes))
        required=self.confirm_hits if tr.stable_class is None or tr.reconfirming else (2 if cls==tr.stable_class else 3)
        support=sum(c==cls for _,c,_ in tr.identity_history)
        if tr.identity_streak>=required and support>=required and share>=.75:
            if tr.stable_class is not None and tr.stable_class!=cls:self.events.append('identity_changed')
            tr.stable_class=cls;tr.identity_blocked=False
            tr.reconfirming=False

    def association_score(self,track,d,t,dormant):
        # Do not extrapolate coordinates past the short prediction horizon.
        stamp=min(t,track.last_seen_ns+self.max_lost_ns)
        p=track.predict(stamp);b=d['box'];iou=overlap(p,b)
        distance=hypot((p[0]+p[2]-b[0]-b[2])/2,(p[1]+p[3]-b[1]-b[3])/2)
        diagonal=max(20,hypot(p[2]-p[0],p[3]-p[1]))
        ratio=((b[2]-b[0])*(b[3]-b[1]))/max(1,(p[2]-p[0])*(p[3]-p[1]))
        if dormant:
            valid=.5<=ratio<=2 and iou>=.3 and distance/diagonal<=.3
        else:
            valid=.25<=ratio<=4 and (iou>=.1 or distance/diagonal<=.5)
        return iou-.25*distance/diagonal if valid else None

    def select_pairs(self,t,detections):
        """Baseline greedy matching; advanced tracker overrides this seam."""
        pairs=[];memory_pairs=[]
        for identifier,track in self.tracks.items():
            dormant=t-track.last_seen_ns>self.max_lost_ns
            for j,d in enumerate(detections):
                score=self.association_score(track,d,t,dormant)
                if score is not None:(memory_pairs if dormant else pairs).append((score,identifier,j))
        active_detections={j for _,_,j in pairs}
        memory_pairs=[p for p in memory_pairs if p[2] not in active_detections]
        accepted_memory=[]
        for score,identifier,j in memory_pairs:
            rivals=[s for s,i,k in memory_pairs if (i==identifier and k!=j) or (k==j and i!=identifier)]
            if rivals and score-max(rivals)<.15:
                self.events.append('ambiguous_reconnect');continue
            accepted_memory.append((score,identifier,j))
        return sorted(pairs,reverse=True)+sorted(accepted_memory,reverse=True)

    def on_detection(self,tr,d,t,dormant=False):
        """Optional auxiliary appearance update; never changes class evidence."""
        pass

    def apply_detection_identity(self,tr,d,t,dormant=False):
        """Default class evidence; advanced grouping may explicitly withhold it."""
        self.update_identity(tr,t,d['class'],d['score'],reconfirm=dormant)

    def should_create(self,d,t):
        """Advanced lifecycle may defer ambiguous births, not detections."""
        return True

    @staticmethod
    def validate_detections(detections):
        for d in detections:
            b=d['box']
            if len(b)!=4 or not all(isfinite(v) for v in b) or b[2]<=b[0] or b[3]<=b[1] or d['class'] not in (0,1) or not 0<=d['score']<=1:
                raise ValueError("Invalid detection")

    def update(self,t,detections):
        if self.timestamp is not None and t<=self.timestamp:
            raise ValueError("Tracker requires strictly increasing native timestamps")
        self.validate_detections(detections)
        self.timestamp=t
        self.events=[]
        for identifier in list(self.tracks):
            tr=self.tracks[identifier]
            horizon=self.memory_ns if tr.confirmed else self.max_lost_ns
            if t-tr.last_seen_ns>horizon:
                del self.tracks[identifier]
                self.events.append('expired')
        pairs=self.select_pairs(t,detections)
        used_tracks,used_detections=set(),set()
        for _,identifier,j in pairs:
            if identifier in used_tracks or j in used_detections:
                continue
            tr=self.tracks[identifier]; d=detections[j]
            dt=(t-tr.last_seen_ns)/1e9
            dormant=t-tr.last_seen_ns>self.max_lost_ns
            if dormant:
                tr.velocity=[0.0]*4
            elif dt>0:
                measured=[(b-a)/dt for a,b in zip(tr.box,d['box'])]
                tr.velocity=[max(-2500,min(2500,.5*a+.5*b)) for a,b in zip(tr.velocity,measured)]
            was_missing=tr.streak==0
            tr.streak=tr.streak+1 if not was_missing else 1
            tr.hits+=1
            tr.confirmed |= tr.streak>=self.confirm_hits
            tr.box=list(d['box']);tr.last_seen_ns=t;tr.last_class=d['class'];tr.score=d['score']
            self.apply_detection_identity(tr,d,t,dormant)
            self.on_detection(tr,d,t,dormant)
            used_tracks.add(identifier);used_detections.add(j)
            if was_missing:self.events.append('reconnected')
            if dormant:self.events.append('memory_reconnected')
        for identifier,tr in self.tracks.items():
            if identifier not in used_tracks:tr.streak=0
        for j,d in enumerate(detections):
            if j in used_detections:continue
            if not self.should_create(d,t):continue
            tr=Track(self.next_id,list(d['box']),t,last_class=d['class'],score=d['score'])
            tr.confirmed=self.confirm_hits==1
            self.apply_detection_identity(tr,d,t)
            self.on_detection(tr,d,t)
            self.tracks[self.next_id]=tr;self.next_id+=1
            self.events.append('created')
        return self.snapshot(t)

    def snapshot(self,t):
        result=[]
        for tr in self.tracks.values():
            age=t-tr.last_seen_ns
            horizon=self.memory_ns if tr.confirmed else self.max_lost_ns
            if age<0 or age>horizon:continue
            observed=age==0
            confirmed=tr.confirmed
            # A contradictory current observation immediately clears identity.
            identity=tr.stable_class if observed and confirmed and not tr.identity_blocked and tr.last_class==tr.stable_class else None
            state='observed' if observed and confirmed else 'tentative' if observed else 'predicted' if age<=self.max_lost_ns else 'remembered'
            result.append(dict(track_id=tr.identifier,box=tr.box.copy() if observed else tr.predict(t) if state=='predicted' else None,
                class_id=identity,state=state,position_valid=observed,confirmed=confirmed,
                historical_class_id=tr.stable_class,last_observed_box=tr.box.copy(),identity_status='confirmed' if identity is not None else 'uncertain',
                last_observed_class=tr.last_class,score=tr.score,age_ms=age/1e6,hits=tr.hits,
                timestamp_ns=t,last_seen_ns=tr.last_seen_ns))
        return result
