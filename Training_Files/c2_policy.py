"""Stage 1 action learning helpers. No live capture or controller output."""
import bisect
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

CONTROLLER_BINDINGS=json.loads((Path(__file__).resolve().parent/'apex_controller_bindings.json').read_text(encoding='utf-8'))
ACTION_ORDER=CONTROLLER_BINDINGS['c2_output_order']
ACTION_SCHEMA=CONTROLLER_BINDINGS['c2_action_schema']


def action_values(row):
    values=[float(row['rx']),float(row['ry']),float(row['rb']),float(row['lb'])]
    if not np.isfinite(values).all() or any(abs(v)>1.001 for v in values[:2]) or any(v not in (0.,1.) for v in values[2:]):
        raise ValueError('Expected normalized RX/RY and binary RB/LB')
    return values


def backward_index(times,t):
    return bisect.bisect_right(times,t)-1


def detection_features(detections):
    # Detector hypotheses are observations, never human action truth.
    features=np.zeros((8,7),dtype=np.float32)
    for i,d in enumerate(sorted(detections,key=lambda d:-d['score'])[:8]):
        x1,y1,x2,y2=d['box']
        features[i]=[(x1+x2)/3840,(y1+y2)/2160,(x2-x1)/1920,(y2-y1)/1080,
                     d['score'],float(d['class']==0),float(d['class']==1)]
    return features.reshape(-1)


def history_indices(times,index):
    t=int(times[index])
    indices=[backward_index(times,t-100_000_000),backward_index(times,t-50_000_000),index]
    if min(indices)<0:return None
    # No fabricated frames across large source gaps.
    if any(t-int(times[j])>200_000_000 for j in indices):return None
    return indices


class CombatPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(9,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,5)),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64*3*5+56+2,128),nn.ReLU(),nn.Linear(128,4))

    def forward(self,images,features):
        raw=self.head(torch.cat([self.visual(images),features],dim=1))
        return torch.cat([torch.tanh(raw[:,:2]),torch.sigmoid(raw[:,2:])],dim=1)


class FPSDataset(torch.utils.data.Dataset):
    def __init__(self,files):
        self.clips=[];self.indices=[]
        for file in files:
            from pathlib import Path
            if Path(file).is_dir():
                clip={k:np.load(Path(file)/(k+'.npy'),mmap_mode='r',allow_pickle=False)
                      for k in ['images','features','actions','timestamps']}
            else:
                with np.load(file,allow_pickle=False) as z:
                    clip={k:z[k].copy() for k in ['images','features','actions','timestamps']}
            ci=len(self.clips);self.clips.append(clip)
            for i in range(len(clip['timestamps'])):
                hs=history_indices(clip['timestamps'].tolist(),i)
                if hs is not None:self.indices.append((ci,i,hs))
        if not self.indices:raise ValueError('No causal history/action samples')

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        ci,i,hs=self.indices[index];clip=self.clips[ci];times=clip['timestamps']
        images=clip['images'][hs].transpose(0,3,1,2).reshape(9,160,288).astype(np.float32)/255
        dt=np.array([(int(times[i])-int(times[hs[0]]))/1e9,(int(times[i])-int(times[hs[1]]))/1e9],np.float32)
        return torch.from_numpy(images),torch.from_numpy(np.r_[clip['features'][i],dt]),torch.from_numpy(clip['actions'][i].copy())
