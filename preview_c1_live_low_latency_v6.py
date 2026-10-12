"""V4 plus direct capture-ring downscale; original detection pixels retained."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile','--event-wakeup']):
        raise ValueError('V6 fixes the V4 comparison settings')
    print('低延迟V6：在采集环内直接生成1080p独立图像，省去4K全图复制；保留V4参数并记录采集耗时。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120','--ui-poll-ms','2',
              '--latest-capture','--tracking-profile','fast','--high-resolution-timer','--capture-timing',
              '--capture-downscale',*sys.argv[1:]]
    shared.main()
