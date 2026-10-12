"""User-run Stage 1 first action learner; default one-clip pipeline test only."""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from c2_policy import CombatPolicy,FPSDataset,ACTION_ORDER,ACTION_SCHEMA
from prepare_c2_recording_manifest import OUTPUT
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent


def score(model,loader,device):
    total=np.zeros(4);count=0
    model.eval()
    with torch.no_grad():
        for images,features,target in loader:
            prediction=model(images.to(device),features.to(device))
            total+=(prediction-target.to(device)).abs().sum(dim=0).cpu().numpy()
            count+=len(target)
    return (total/count).tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true',help='Full grouped training after one-clip pipeline acceptance')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    manifest=OUTPUT/'recording_manifest.json'
    rows=json.loads(manifest.read_text(encoding='utf-8'))
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    chosen=[r for r in rows if r['split']=='train'] if args.all else [r for r in rows if r['key']==summary['smoke_clip']]
    validation=[r for r in rows if r['split']=='val'] if args.all else []
    files={}
    for r in chosen+validation:
        folder=OUTPUT/'cache'/r['key']
        receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
        if (receipt['provenance'].get('action_schema')!=ACTION_SCHEMA
                or receipt['provenance'].get('action_order')!=ACTION_ORDER):
            raise ValueError('Wrong action cache: this run requires RX/RY/RB/LB, not trigger labels')
        if receipt['provenance']['manifest_sha256']!=sha(manifest):raise ValueError('Prepared manifest differs')
        for name,digest in receipt['cache_files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Prepared cache changed')
        files[r['key']]=folder
    dataset=FPSDataset([files[r['key']] for r in chosen])
    valset=FPSDataset([files[r['key']] for r in validation]) if validation else None
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    epochs=args.epochs or (30 if args.all else 150)
    if epochs<=0:raise ValueError('epochs must be positive')
    print(f'{"全部数据" if args.all else "单片段拟合检查"}：{len(dataset)}训练样本，{len(valset) if valset else 0}验证样本；输出RX/RY/RB(开火)/LB(瞄准)，按钮为按下概率，无游戏控制。',flush=True)
    if args.check_only:print('检查完成，未训练。');return
    torch.manual_seed(20261010);np.random.seed(20261010)
    torch.backends.cudnn.benchmark=False
    device='cuda';model=CombatPolicy().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0)
    evaluate=torch.utils.data.DataLoader(valset or dataset,batch_size=32,num_workers=0)
    out=ROOT/'runs/c2_bc'/(("joint_" if args.all else "overfit_")+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    receipt=dict(stage='Stage 1 — Local Combat Policy',manifest_sha256=sha(manifest),
        mode='joint' if args.all else 'one_clip_pipeline_check',train_clips=[r['key'] for r in chosen],
        validation_clips=[r['key'] for r in validation],output_order=ACTION_ORDER,action_schema=ACTION_SCHEMA,
        button_mapping={'fire':'rb','aim':'lb'},
        epochs=epochs,controller_output=False,image_history_ms=[-100,-50,0],
        note='Behavior-cloning error is not hit rate. One-clip fit is not generalization or closed-loop acceptance.')
    (out/'run_config.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    # A constant predictor is a meaningful benchmark for an imbalanced action dataset.
    y=np.asarray([dataset.clips[ci]['actions'][i] for ci,i,_ in dataset.indices])
    constant=y.mean(axis=0)
    evalset=valset or dataset
    yy=np.asarray([evalset.clips[ci]['actions'][i] for ci,i,_ in evalset.indices])
    baseline=np.abs(yy-constant).mean(axis=0).tolist()
    best=float('inf');logs=[]
    for epoch in range(1,epochs+1):
        model.train();running=0.;count=0
        for images,features,target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction=model(images.to(device),features.to(device))
            loss=torch.nn.functional.smooth_l1_loss(prediction,target.to(device))
            loss.backward();optimizer.step()
            running+=loss.item()*len(target);count+=len(target)
        mae=score(model,evaluate,device);metric=float(np.mean(mae))
        row=dict(epoch=epoch,train_loss=running/count,action_mae=mae,mean_mae=metric);logs.append(row)
        checkpoint=dict(state_dict=model.state_dict(),config=receipt,epoch=epoch,action_mae=mae,
                        constant_baseline_mae=baseline,architecture='CombatPolicy_rb_lb_v2')
        torch.save(checkpoint,out/'last.pt')
        if metric<best:best=metric;torch.save(checkpoint,out/'best.pt')
        if args.all or epoch==1 or epoch%10==0:
            label='验证' if args.all else '拟合样本'
            print(f'epoch{epoch}: 训练loss={row["train_loss"]:.5f} {label}MAE RX/RY/RB/LB={mae}',flush=True)
        (out/'results.json').write_text(json.dumps(dict(baseline_mae=baseline,epochs=logs),indent=2),encoding='utf-8')
    print('结果：',out,'\n候选权重：',out/'best.pt')


if __name__=='__main__':main()
