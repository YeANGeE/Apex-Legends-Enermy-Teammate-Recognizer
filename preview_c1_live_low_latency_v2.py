"""Only newest native capture copy; compare against TRT realtime v1."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms']):
        raise ValueError('V2 fixes the same detector/tracker/capture/UI settings as V1')
    print('低延迟V2：仅增加采集环最新帧读取及重复画面免拷贝；相机/模型/阈值保持。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120',
              '--ui-poll-ms','2','--latest-capture',*sys.argv[1:]]
    shared.main()
