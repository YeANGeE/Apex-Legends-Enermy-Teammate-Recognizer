"""User-started full C2 preparation then training, preserving held-out clips."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import prepare_c2_policy_cache as cache
import train_c2_behavior_cloning as training
from prepare_c2_recording_manifest import OUTPUT
from c2_policy import ACTION_SCHEMA


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.epochs<=0:parser.error('epochs must be positive')
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    rows=json.loads((OUTPUT/'recording_manifest.json').read_text(encoding='utf-8'))
    if summary.get('action_schema')!=ACTION_SCHEMA:raise ValueError('Requires corrected RB fire/LB aim labels')
    selected=[r for r in rows if r['split'] in ['train','val']]
    missing=[r for r in selected if not (OUTPUT/'cache'/r['key']/'receipt.json').is_file()]
    needed=sum(len(r['samples'])*160*288*3 for r in missing)+2*1024**3
    free=shutil.disk_usage(OUTPUT).free
    if free<needed:raise RuntimeError(f'Insufficient cache disk space: need about {needed/1024**3:.1f}GiB, free {free/1024**3:.1f}GiB')
    print('C2全量联合训练：DAY1–4，GameInput因果对齐，RX/RY/RB开火/LB瞄准。',flush=True)
    print(f'阶段1准备缓存：{len(selected)}段训练/验证录像，{len(missing)}段尚需完整解码及冻结C1特征。已有有效缓存复用。',flush=True)
    print('测试与诊断录像不加入；不会操作游戏。C1权重冻结，C2从随机初始化开始联合学习，不加载单段拟合权重。',flush=True)
    original=sys.argv
    try:
        sys.argv=[str(Path(cache.__file__)),'--all',*(['--check-only'] if args.check_only else [])]
        cache.main()
        if args.check_only and missing:
            print('入口及缓存准备配置检查完成；完整训练检查需等剩余缓存生成，当前未训练。',flush=True)
            return
        print(f'阶段2：缓存通过后训练{args.epochs}轮；在独立DAY1交战组上报告验证动作误差，best.pt按验证均值误差选择。',flush=True)
        sys.argv=[str(Path(training.__file__)),'--all','--epochs',str(args.epochs),*(['--check-only'] if args.check_only else [])]
        training.main()
    finally:
        sys.argv=original


if __name__=='__main__':main()
