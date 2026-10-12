"""Stage 1: timestamp-native camera/appearance-assisted screen tracking.

Not a BoT-SORT wrapper: preserves the existing causal time and identity contract.
No action output, world coordinates, training, or guaranteed person identity.
"""
from collections import deque
from math import hypot
from time import perf_counter

import cv2
import numpy as np

from c1_tracking import ScreenTracker, overlap

# reference reproduces the 32-thread setting observed on this development host.
# threads keeps identical camera size/point budget; fast is opt-in until reviewed.
CAMERA_PROFILES={
    'reference':dict(width=640,max_corners=350,opencv_threads=32),
    'threads':dict(width=640,max_corners=350,opencv_threads=4),
    'fast':dict(width=480,max_corners=200,opencv_threads=4),
}


def global_assignment(scores):
    """Maximum-score rectangular Hungarian assignment with private dummy columns.

    None forbids a pair. Every row may remain unmatched at score zero. No scipy
    dependency, no greedy order dependence, and no forced invalid assignments.
    """
    n=len(scores)
    if not n:return []
    m=len(scores[0])
    if not m:return []
    costs=[[1e6 if x is None else -float(x) for x in row]+[0.]*n for row in scores]
    width=m+n
    u=[0.]*(n+1);v=[0.]*(width+1);p=[0]*(width+1);way=[0]*(width+1)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=[float('inf')]*(width+1);used=[False]*(width+1)
        while True:
            used[j0]=True;i0=p[j0];delta=float('inf');j1=0
            for j in range(1,width+1):
                if used[j]:continue
                cur=costs[i0-1][j-1]-u[i0]-v[j]
                if cur<minimum[j]:minimum[j]=cur;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(width+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    return [(p[j]-1,j-1) for j in range(1,m+1)
            if p[j] and scores[p[j]-1][j-1] is not None and scores[p[j]-1][j-1]>0]


class CameraCompensator:
    """Small grayscale background flow + forward/backward check + affine RANSAC.

    The affine approximation is local: perspective/parallax, flash, fast turns,
    and low texture can invalidate it. A rejected estimate never clears IDs.
    """
    def __init__(self,width=640,max_corners=350):
        if not 240<=width<=1280 or not 30<=max_corners<=1000:
            raise ValueError('Invalid camera work size/corner count')
        self.width=width;self.previous=None;self.mask=None;self.timestamp=None
        self.max_corners=max_corners;self.phase_times={}

    def reset(self):
        self.previous=None;self.mask=None;self.timestamp=None

    def update(self,rgb,detections,t):
        self.phase_times={name:0. for name in ('prepare','corners','flow_forward','flow_backward','filter','ransac','total')}
        before=perf_counter()
        matrix,info=self._update(rgb,detections,t)
        self.phase_times['total']=(perf_counter()-before)*1000
        info['timings_ms']=self.phase_times.copy()
        info['work_width']=self.width;info['max_corners']=self.max_corners
        return matrix,info

    def _update(self,rgb,detections,t):
        before=perf_counter()
        height,width=rgb.shape[:2];small_h=round(height*self.width/width)
        gray=cv2.cvtColor(cv2.resize(rgb,(self.width,small_h)),cv2.COLOR_RGB2GRAY)
        sx=self.width/width;sy=small_h/height
        mask=np.full(gray.shape,255,np.uint8)
        # Apex screen HUD and weapon region are not moving-world evidence.
        mask[:int(.08*small_h)]=0;mask[int(.83*small_h):]=0
        mask[:int(.30*small_h),:int(.23*self.width)]=0
        mask[int(.55*small_h):,int(.65*self.width):]=0
        cv2.circle(mask,(self.width//2,small_h//2),round(20*self.width/640),0,-1)
        padding=max(1,round(6*self.width/640))
        for d in detections:
            x1,y1,x2,y2=d['box']
            a=(max(0,int(x1*sx)-padding),max(0,int(y1*sy)-padding))
            b=(min(self.width-1,int(x2*sx)+padding),min(small_h-1,int(y2*sy)+padding))
            cv2.rectangle(mask,a,b,0,-1)
        old,old_mask,old_time=self.previous,self.mask,self.timestamp
        self.previous=gray;self.mask=mask;self.timestamp=t
        self.phase_times['prepare']=(perf_counter()-before)*1000
        info=dict(status='initial',inliers=0,ratio=0.,matrix=None)
        if old is None:return None,info
        if t<=old_time:raise ValueError('Camera requires increasing source timestamps')
        if t-old_time>100_000_000:
            info['status']='timestamp_gap';return None,info
        info['status']='rejected'
        before=perf_counter()
        points=cv2.goodFeaturesToTrack(old,maxCorners=self.max_corners,qualityLevel=.015,
                                     minDistance=max(3,round(8*self.width/640)),mask=old_mask)
        self.phase_times['corners']=(perf_counter()-before)*1000
        if points is None or len(points)<15:return None,info
        before=perf_counter()
        nxt,status,_=cv2.calcOpticalFlowPyrLK(old,gray,points,None,winSize=(21,21),maxLevel=3)
        self.phase_times['flow_forward']=(perf_counter()-before)*1000
        if nxt is None:return None,info
        before=perf_counter()
        back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,old,nxt,None,winSize=(21,21),maxLevel=3)
        self.phase_times['flow_backward']=(perf_counter()-before)*1000
        if back is None:return None,info
        before=perf_counter()
        a=points.reshape(-1,2);b=nxt.reshape(-1,2)
        good=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back.reshape(-1,2)-a,axis=1)<1.5*self.width/640)
        good&=np.isfinite(b).all(axis=1)&(b[:,0]>=0)&(b[:,0]<self.width)&(b[:,1]>=0)&(b[:,1]<small_h)
        indices=np.flatnonzero(good)
        # Vectorized mask lookup: same floor-to-pixel semantics as the old loop.
        if len(indices):
            pixels=b[indices].astype(np.int32)
            indices=indices[mask[pixels[:,1],pixels[:,0]]!=0]
        self.phase_times['filter']=(perf_counter()-before)*1000
        if len(indices)<15:return None,info
        a=a[indices];b=b[indices]
        before=perf_counter()
        transform,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,
            ransacReprojThreshold=2.5*self.width/640,maxIters=1000,confidence=.99)
        self.phase_times['ransac']=(perf_counter()-before)*1000
        if transform is None or inliers is None or not np.isfinite(transform).all():return None,info
        keep=inliers.ravel().astype(bool);count=int(keep.sum());ratio=count/len(keep)
        info.update(inliers=count,ratio=ratio)
        scale=hypot(transform[0,0],transform[1,0])
        spread=np.ptp(a[keep],axis=0) if count else [0,0]
        if count<12 or ratio<.5 or not .7<=scale<=1.4 or abs(transform[0,2])>.45*self.width or abs(transform[1,2])>.45*small_h or spread[0]<.2*self.width or spread[1]<.15*small_h:
            return None,info
        # Convert small-image affine coordinates to original screen coordinates.
        down=np.diag([sx,sy,1.]);up=np.diag([1/sx,1/sy,1.])
        full=(up@np.vstack([transform,[0,0,1]])@down)[:2]
        info.update(status='accepted',matrix=full.tolist())
        return full,info


class AppearanceEncoder:
    """Pretrained classification-backbone descriptors, NOT game-trained ReID.

    Loaded only by the user's demo launch. BGR crops are batched on CUDA.
    Small/invalid crops have no feature; no color-only fake identity fallback.
    """
    def __init__(self,weights,device=0):
        from ultralytics import YOLO
        self.model_path=str(weights)
        self.model=YOLO(str(weights));self.device=device
        self.quantize=32 if str(device)=='cpu' else 16
        self.model.embed(np.zeros((224,224,3),np.uint8),imgsz=224,device=device,quantize=self.quantize,verbose=False)

    def extract(self,rgb,detections):
        features=[None]*len(detections);crops=[];indices=[];h,w=rgb.shape[:2]
        for i,d in enumerate(detections):
            x1,y1,x2,y2=d['box']
            x1=max(0,min(w,int(x1)));x2=max(0,min(w,int(x2)))
            y1=max(0,min(h,int(y1)));y2=max(0,min(h,int(y2)))
            if x2-x1<10 or y2-y1<20:continue
            crops.append(rgb[y1:y2,x1:x2,::-1].copy());indices.append(i)
        if crops:
            vectors=self.model.embed(crops,imgsz=224,device=self.device,quantize=self.quantize,verbose=False)
            if len(vectors)!=len(crops):raise RuntimeError('Appearance embedding count mismatch')
            for i,vector in zip(indices,vectors):
                features[i]=normalize(vector.detach().float().cpu().numpy())
        return features


def normalize(vector):
    if vector is None:return None
    value=np.asarray(vector,dtype=np.float32).reshape(-1)
    norm=float(np.linalg.norm(value))
    return value/norm if value.size and np.isfinite(value).all() and norm>1e-9 else None


def group_duplicate_candidates(detections,features,protected=None):
    """Conservative complete-link grouping, not guaranteed person deduplication.

    Never merge two candidates clearly explained by distinct confirmed tracks.
    Retain original member indices and class conflict; no new confident class.
    """
    protected=protected or {};vectors=[normalize(v) for v in features]
    area=lambda b:max(1e-9,(b[2]-b[0])*(b[3]-b[1]))
    def compatible(i,j):
        if i in protected and j in protected and protected[i]!=protected[j]:return False
        a=detections[i]['box'];b=detections[j]['box'];iou=overlap(a,b)
        va,vb=vectors[i],vectors[j]
        if va is not None and vb is not None and float(va@vb)<.85:return False
        if iou>=.90:return True
        # Full/partial duplicates: both need compatible appearance, aligned
        # horizontal centers and tops, and strong containment. Avoid a generic
        # class-agnostic NMS that could suppress crossing/occluded people.
        if va is None or vb is None:return False
        small,big=(a,b) if area(a)<=area(b) else (b,a)
        intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        width=big[2]-big[0];height=big[3]-big[1]
        return (intersection/area(small)>=.95 and area(small)/area(big)>=.25
                and (small[2]-small[0])/max(1,width)>=.45
                and abs((small[0]+small[2]-big[0]-big[2])/2)<=.2*width
                and abs(small[1]-big[1])<=.15*height)
    groups=[]
    for i in sorted(range(len(detections)),key=lambda k:(-detections[k]['score'],k)):
        target=next((g for g in groups if all(compatible(i,j) for j in g)),None)
        if target is None:groups.append([i])
        else:target.append(i)
    groups.sort(key=min)
    candidates=[];candidate_features=[];audit=[]
    for members in groups:
        # A contained partial box should not replace the available full box.
        representative=max(members,key=lambda i:(area(detections[i]['box']),detections[i]['score'],-i))
        conflict=len({detections[i]['class'] for i in members})>1
        candidate=dict(detections[representative],_class_conflict=conflict,_member_indices=members)
        candidates.append(candidate);candidate_features.append(features[representative])
        audit.append(dict(member_indices=members,representative_index=representative,class_conflict=conflict))
    return candidates,candidate_features,audit


class AdvancedScreenTracker(ScreenTracker):
    def __init__(self,*args,appearance=True,camera_width=640,max_corners=350,stable_geometry=False,group_duplicates=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.appearance=appearance;self.gallery={};self.observed_boxes={}
        self.stable_geometry=stable_geometry
        self.group_duplicates=group_duplicates
        self.camera=CameraCompensator(width=camera_width,max_corners=max_corners);self.diagnostics={};self.assignment_audit=[]
        self.deferred={};self.unassociated=[];self.gate_counts={}

    def reset(self):
        super().reset();self.gallery.clear();self.observed_boxes.clear();self.camera.reset()
        self.deferred={};self.unassociated=[];self.gate_counts={}

    def compensate(self,matrix):
        for tr in self.tracks.values():
            x1,y1,x2,y2=tr.box
            corners=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]],dtype=float)@matrix.T
            low=corners.min(axis=0);high=corners.max(axis=0)
            tr.box=[*low,*high]
            # Transport object velocity to current camera axes; do not turn the
            # camera displacement itself into persistent object velocity.
            velocity=np.asarray(tr.velocity).reshape(2,2)@matrix[:,:2].T
            tr.velocity=velocity.reshape(-1).tolist()

    def similarity(self,tr,d):
        vector=normalize(d.get('feature'))
        gallery=self.gallery.get(tr.identifier)
        if not self.appearance or vector is None or not gallery:return None
        return max(float(np.clip(old@vector,-1,1)) for old in gallery)

    def evaluate_pair(self,tr,d,t,dormant):
        p=tr.predict(min(t,tr.last_seen_ns+self.max_lost_ns));b=d['box']
        iou=overlap(p,b)
        distance=hypot((p[0]+p[2]-b[0]-b[2])/2,(p[1]+p[3]-b[1]-b[3])/2)
        diagonal=max(20,hypot(p[2]-p[0],p[3]-p[1]));relative=distance/diagonal
        # Corner-velocity prediction can shrink a box near zero during recoil /
        # partial occlusion. For recent confirmed tracks only, use the most
        # recent measured box (already transported by camera compensation) as
        # the area reference. Do not modify prediction, geometry, appearance,
        # ambiguity gates, tentative tracks, or dormant reconnection rules.
        area_reference=tr.box if self.stable_geometry and tr.confirmed and not dormant else p
        ratio=((b[2]-b[0])*(b[3]-b[1]))/max(1,(area_reference[2]-area_reference[0])*(area_reference[3]-area_reference[1]))
        similarity=self.similarity(tr,d)
        if not .25<=ratio<=4:return None,'area_gate'
        if similarity is None:
            # Without an appearance cue never claim extended re-identification.
            if dormant:return None,'memory_missing_feature'
            if iou<.1 and relative>.5:return None,'position_gate'
            return max(.001,.65*iou+.35*max(0,1-relative)),'accepted_position_only'
        # Strong appearance disagreement forbids positional ID stealing.
        if similarity<.65:return None,'appearance_disagreement'
        if dormant:
            if similarity<.90:return None,'memory_appearance_gate'
            if relative>2 or not .5<=ratio<=2:return None,'memory_position_gate'
        elif not (iou>=.1 or relative<=.5 or (similarity>=.86 and relative<=1.25)):
            return None,'position_gate'
        geometry=.65*iou+.35*max(0,1-relative/2)
        return .55*similarity+.45*geometry-(.05 if dormant else 0),'accepted_appearance'

    def association_score(self,tr,d,t,dormant):
        return self.evaluate_pair(tr,d,t,dormant)[0]

    def solve_group(self,identifiers,indices,all_scores):
        """One solve, then reject genuinely near-tied alternatives within group.

        Do not re-solve after rejection: that could force the runner-up. Entries
        belonging to tentative/other lifecycle groups cannot veto stable IDs.
        """
        scores=[[all_scores[k][j] for j in indices] for k in identifiers]
        selected=global_assignment(scores);accepted=[]
        for i,j in selected:
            value=scores[i][j]
            rivals=[s for k,s in enumerate(scores[i]) if k!=j and s is not None]
            rivals += [row[j] for k,row in enumerate(scores) if k!=i and row[j] is not None]
            if rivals and any(abs(value-rival)<.04 for rival in rivals):
                self.events.append('ambiguous_assignment');continue
            accepted.append((value,identifiers[i],indices[j]))
        return accepted

    def select_pairs(self,t,detections):
        self.gate_counts={};self.deferred={};self.unassociated=[]
        scores={}
        for k,tr in self.tracks.items():
            scores[k]=[]
            for d in detections:
                score,reason=self.evaluate_pair(tr,d,t,t-tr.last_seen_ns>self.max_lost_ns)
                scores[k].append(score)
                self.gate_counts[reason]=self.gate_counts.get(reason,0)+1
        recent=[k for k,tr in self.tracks.items() if tr.confirmed and t-tr.last_seen_ns<=self.max_lost_ns]
        memory=[k for k,tr in self.tracks.items() if tr.confirmed and t-tr.last_seen_ns>self.max_lost_ns]
        tentative=[k for k,tr in self.tracks.items() if not tr.confirmed]
        available=set(range(len(detections)));selected=[]
        # Stable current IDs, then strong memory reconnection, then provisional
        # tracks. Each group gets a global assignment, not greedy per-person.
        for group in (recent,memory):
            accepted=self.solve_group(group,sorted(available-set(self.deferred)),scores)
            selected+=accepted;available.difference_update(j for _,_,j in accepted)
            for j in available:
                if any(scores[k][j] is not None for k in group):
                    self.deferred[j]='confirmed_candidate_unresolved'
        # Withheld detections remain visible but have no inherited ID. A failed
        # feature comparison at virtually the old location is not proof of a
        # new person either (flash/partial occlusion may alter the crop).
        for j in available:
            b=detections[j]['box']
            for k in recent:
                tr=self.tracks[k];p=tr.predict(min(t,tr.last_seen_ns+self.max_lost_ns))
                if overlap(p,b)>=.4:
                    self.deferred.setdefault(j,'near_confirmed_unresolved')
        accepted=self.solve_group(tentative,sorted(available-set(self.deferred)),scores)
        selected+=accepted;available.difference_update(j for _,_,j in accepted)
        for j in available:
            if any(scores[k][j] is not None for k in tentative):
                self.deferred.setdefault(j,'tentative_candidate_unresolved')
        for _,_,j in selected:self.deferred.pop(j,None)
        self.assignment_audit=[dict(track_id=k,detection_index=j,score=score,
                                    appearance_similarity=self.similarity(self.tracks[k],detections[j]))
                               for score,k,j in selected]
        return selected

    def should_create(self,d,t):
        j=d['_detection_index']
        if j not in self.deferred:return True
        self.events.append('birth_deferred')
        self.unassociated.append(dict(track_id=None,box=list(d['box']),class_id=None,state='unassociated',
            position_valid=True,association_valid=False,confirmed=False,historical_class_id=None,
            last_observed_box=list(d['box']),identity_status='uncertain',last_observed_class=d['class'],
            score=d['score'],age_ms=0.,hits=1,timestamp_ns=t,last_seen_ns=t,
            appearance_available=normalize(d.get('feature')) is not None,detection_index=j,
            association_reason=self.deferred[j]))
        return False

    def on_detection(self,tr,d,t,dormant=False):
        self.observed_boxes[tr.identifier]=list(d['box'])
        vector=normalize(d.get('feature'))
        # Avoid permanently contaminating the gallery with a conflicting class
        # crop. This does not prove the remaining accepted matches are correct.
        if self.appearance and vector is not None and not tr.identity_blocked:
            self.gallery.setdefault(tr.identifier,deque(maxlen=5)).append(vector)

    def apply_detection_identity(self,tr,d,t,dormant=False):
        if d.get('_class_conflict'):
            # A group has both enemy and teammate hypotheses. Never turn its
            # representative's arbitrary class into current class evidence.
            tr.identity_history=[];tr.votes=[0.,0.];tr.identity_blocked=True
            tr.identity_streak=0;tr.identity_candidate=None;tr.reconfirming=True
            self.events.append('duplicate_class_conflict')
        else:super().apply_detection_identity(tr,d,t,dormant)

    def snapshot(self,t):
        rows=super().snapshot(t)
        for row in rows:
            row['last_observed_box']=self.observed_boxes.get(row['track_id'],row['last_observed_box']).copy()
            row['appearance_available']=bool(self.gallery.get(row['track_id']))
            row['association_valid']=row['position_valid'] and row['confirmed']
        return rows+[dict(row) for row in self.unassociated if row['timestamp_ns']==t]

    def update_frame(self,t,detections,rgb,features=None):
        if self.timestamp is not None and t<=self.timestamp:
            raise ValueError('Tracker requires strictly increasing native timestamps')
        self.validate_detections(detections)
        if features is not None and len(features)!=len(detections):raise ValueError('Feature count mismatch')
        before=perf_counter()
        transform,camera_info=self.camera.update(rgb,detections,t)
        camera_ms=(perf_counter()-before)*1000
        before=perf_counter()
        if transform is not None:self.compensate(transform)
        group_audit=[];input_count=len(detections)
        if self.group_duplicates:
            features=features if features is not None else [None]*len(detections)
            protected={}
            for j,d in enumerate(detections):
                ranked=sorted([(overlap(tr.predict(min(t,tr.last_seen_ns+self.max_lost_ns)),d['box']),k)
                               for k,tr in self.tracks.items() if tr.confirmed and t-tr.last_seen_ns<=self.max_lost_ns],reverse=True)
                if ranked and ranked[0][0]>=.5 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.1):
                    protected[j]=ranked[0][1]
            detections,features,group_audit=group_duplicate_candidates(detections,features,protected)
        expired=[dict(track_id=k,age_ms=(t-tr.last_seen_ns)/1e6,confirmed=tr.confirmed)
                 for k,tr in self.tracks.items()
                 if t-tr.last_seen_ns>(self.memory_ns if tr.confirmed else self.max_lost_ns)]
        enriched=[dict(d,feature=features[i] if features is not None else None,_detection_index=i) for i,d in enumerate(detections)]
        rows=super().update(t,enriched)
        for group in group_audit:
            if len(group['member_indices'])>1:self.events.append('duplicate_grouped')
        association_ms=(perf_counter()-before)*1000
        self.gallery={k:v for k,v in self.gallery.items() if k in self.tracks}
        self.observed_boxes={k:v for k,v in self.observed_boxes.items() if k in self.tracks}
        self.diagnostics=dict(camera=camera_info,assignments=self.assignment_audit,pair_gate_counts=self.gate_counts,
            duplicate_groups=group_audit,raw_detection_count=input_count,association_candidate_count=len(detections),
            tracking_timings_ms=dict(camera=camera_ms,association_and_state=association_ms),
            expired_tracks=expired,
            deferred_births=[dict(detection_index=j,reason=reason) for j,reason in sorted(self.deferred.items())],
            lifecycle=dict(recent_confirmed=sum(tr.confirmed and t-tr.last_seen_ns<=self.max_lost_ns for tr in self.tracks.values()),
                           remembered_confirmed=sum(tr.confirmed and t-tr.last_seen_ns>self.max_lost_ns for tr in self.tracks.values()),
                           tentative=sum(not tr.confirmed for tr in self.tracks.values())))
        self.events.append('camera_'+camera_info['status'])
        return rows
