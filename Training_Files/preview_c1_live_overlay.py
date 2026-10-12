"""User-started read-only desktop overlay. No input injection or game hooks."""
import argparse
import csv
import ctypes
from ctypes import wintypes
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from c1_tracking_advanced import AdvancedScreenTracker, AppearanceEncoder, CAMERA_PROFILES
from c1_target_selection import EnemyTargetSelector
from demo_c1_video_tracking import prediction_rows, APPEARANCE_WEIGHTS
from demo_c1_day4_supplement_tracking import WEIGHTS, EXPECTED_SHA
from normalize_c1_day3_export import sha
from fps_recorder.clock import QpcClock
from fps_recorder.config import CaptureConfig, XInputConfig
from fps_recorder.dxcam_capture import DXcamCapture
from c1_timer_resolution import TimerResolution
from c1_capture_latency import split_capture_latency

ROOT = Path(__file__).resolve().parent
CONFIDENCE = .10
user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]


class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.packet = None
        self.overwritten = 0
        self.ready = threading.Event()

    def put(self, packet):
        with self.lock:
            if self.packet is not None:
                self.overwritten += 1
            self.packet = packet
            self.ready.set()

    def take(self):
        with self.lock:
            packet, self.packet = self.packet, None
            self.ready.clear()
            return packet

    def wait_take(self, timeout=.02):
        if not self.ready.wait(timeout):return None
        return self.take()


def distribution(values):
    return dict(count=len(values), median_ms=float(np.median(values)) if values else None,
                p95_ms=float(np.percentile(values,95)) if values else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-index', type=int, default=0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--detector', type=Path, default=WEIGHTS)
    parser.add_argument('--capture-fps', type=int, choices=[60,120], default=60)
    parser.add_argument('--ui-poll-ms', type=int, choices=[2,8], default=8)
    parser.add_argument('--latest-capture', action='store_true',help='Skip stale ring entries and duplicate image copies; realtime only')
    parser.add_argument('--tracking-profile',choices=list(CAMERA_PROFILES),default='threads')
    parser.add_argument('--game-fps-label',type=int,help='Manual experiment label only; does not set or measure game FPS')
    parser.add_argument('--high-resolution-timer',action='store_true')
    parser.add_argument('--event-wakeup',action='store_true',help='Wake inference on capture callback instead of polling every 2ms')
    parser.add_argument('--capture-timing',action='store_true',help='Measure native source -> ring copy -> handoff -> inference')
    parser.add_argument('--capture-downscale',action='store_true',help='Produce owned 1080p output in ring reader, avoiding full 4K copy')
    parser.add_argument('--capture-color',choices=['BGR','BGRA'],default='BGR')
    parser.add_argument('--tracker', choices=['advanced','motion'], default='advanced',
                        help='motion: diagnostic comparison without appearance embeddings')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--c2-weights',type=Path,help='Optional read-only action suggestions, no controller output')
    parser.add_argument('--c2-geometry-bias',action='store_true')
    parser.add_argument('--geometry-gain-x',type=float,default=1.)
    parser.add_argument('--geometry-gain-y',type=float,default=1.)
    parser.add_argument('--geometry-base-weight',type=float,default=1.)
    parser.add_argument('--geometry-correction-limit',type=float,default=.5)
    parser.add_argument('--geometry-invert-y',action='store_true')
    parser.add_argument('--geometry-display-ms',type=int,default=1000,help='Combined-action snapshot display interval only')
    parser.add_argument('--c2-inertia-ms',type=float,help='Read-only second-order trajectory response, no controller')
    parser.add_argument('--record-xinput',action='store_true',help='Read-only physical input recording for offline calibration')
    parser.add_argument('--xinput-user-index',type=int,choices=range(4))
    parser.add_argument('--calibration-aim-mode',choices=['hip','ads'],help='Manual session label, not a control command')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    if args.c2_geometry_bias and not args.c2_weights:
        parser.error('Geometry correction requires the BC base prediction')
    if args.c2_inertia_ms is not None:
        if not args.c2_geometry_bias:
            parser.error('Inertia preview requires the additive geometry candidate')
        from c2_inertia import InertialAxes
        InertialAxes(args.c2_inertia_ms)
    if not 100 <= args.geometry_display_ms <= 5000:
        parser.error('geometry-display-ms must be between 100 and 5000')
    if args.calibration_aim_mode and not args.record_xinput:
        parser.error('Calibration label requires physical input recording')
    if args.game_fps_label is not None and args.game_fps_label <= 0:
        parser.error('game-fps-label must be positive')
    if args.capture_downscale and not args.latest_capture:
        parser.error('capture-downscale requires latest-capture')
    if args.capture_color=='BGRA' and not args.capture_downscale:
        parser.error('BGRA candidate requires capture-downscale to deliver BGR to the detector')
    if sha(WEIGHTS) != EXPECTED_SHA or (args.tracker=='advanced' and not APPEARANCE_WEIGHTS.is_file()) or not torch.cuda.is_available():
        raise RuntimeError('Missing/changed weights, appearance model or CUDA')
    detector = args.detector.resolve()
    if detector.suffix == '.engine':
        from benchmark_c1_tensorrt import verify_engine
        manifest = verify_engine(detector, source_weights=WEIGHTS)
        if manifest['export_args']['quantize'] != 16:
            raise ValueError('Live candidate requires FP16 engine')
    elif detector != WEIGHTS.resolve():
        raise ValueError('Use the current PT model or its verified engine')
    import importlib.metadata
    if importlib.metadata.version('dxcam') != '0.3.0':
        raise RuntimeError('Capture requires the audited DXcam 0.3.0 runtime')
    import tkinter as tk
    policy = None
    if args.c2_weights:
        from c2_shadow import ShadowPolicy
        from c2_shadow_display import ShadowDisplay
        policy = ShadowPolicy(args.c2_weights)
    action_display = ShadowDisplay() if policy else None
    geometry = None
    if args.c2_geometry_bias:
        from c2_geometry_bias import GeometryBias
        geometry = GeometryBias(args.geometry_gain_x,args.geometry_gain_y,args.geometry_base_weight,
                                args.geometry_correction_limit,args.geometry_invert_y)
    if args.check_only:
        print('检查通过：权重、CUDA、DXcam 0.3.0、Tk；未启动采集/窗口/推理。')
        return
    out = ROOT/'runs/c1_live_overlay'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    clock = QpcClock()
    slot = LatestFrame()
    stop = threading.Event()
    lock = threading.Lock()
    shared = dict(result=None, error=None, finished=False)
    timings = {k:[] for k in ('prepare','predict','appearance','tracking','processing','source_age_at_start','source_age_at_finish','source_age_at_ui_submit','ui_poll_interval')}
    if args.capture_timing:
        timings.update({k:[] for k in ('source_to_capture_copy','capture_copy','capture_dispatch','capture_to_inference')})
    if policy:
        timings['c2_shadow_update'] = []
    config = dict(weights=str(detector), weights_sha256=sha(detector),source_weights_sha256=EXPECTED_SHA, confidence=CONFIDENCE,
        detector_backend='tensorrt' if detector.suffix=='.engine' else 'pytorch',
        capture_fps=args.capture_fps,ui_poll_ms=args.ui_poll_ms,
        latest_capture=args.latest_capture,
        stable_geometry=True, group_duplicates=False, tracker=args.tracker, tracking_profile=args.tracking_profile,
        camera_profile=CAMERA_PROFILES[args.tracking_profile],game_fps_label=args.game_fps_label,
        game_fps_label_note='User-supplied cap label, not observed FPS',
        timer_resolution_request_ms=1 if args.high_resolution_timer else None,
        inference_wakeup='capture_event' if args.event_wakeup else '2ms_poll',
        capture_timing_enabled=args.capture_timing,
        capture_downscale=args.capture_downscale,
        capture_color=args.capture_color,
        capture_output=args.output_index, capture_device=args.device_index, seconds=args.seconds,
        controller_output=False, mouse_keyboard_injection=False, source_timestamp='DXGI LastPresentTime mapped to QPC ns',
        resize=('ring reader -> owned 1920x1080 INTER_AREA, without full-resolution copy' if args.capture_downscale
                else 'full captured output -> 1920x1080 using CPU INTER_AREA if needed'),
        timing_note='Processing includes resize, detector predict, appearance, tracking and selection. UI submit age is software submission, not physical display or capture-to-action latency.')
    if policy:
        config['timing_note'] += ' Processing also includes optional C2 shadow updates; c2_shadow_update includes skipped updates and CPU preparation, not pure model forward.'
    if geometry:
        config['geometry_bias'] = geometry.config()
        config['geometry_display'] = dict(refresh_ms=args.geometry_display_ms,raw_panel_visible=False,
            graphical_combined_axes=True,predictions_modified=False)
        timings['geometry_bias'] = []
    if args.record_xinput:
        config['physical_input_recording'] = dict(stream='xinput.csv',requested_poll_hz=4000,
            user_index=args.xinput_user_index,aim_mode_label=args.calibration_aim_mode,
            timestamp='QPC after XInputGetState; host observation, not hardware event',
            controller_output=False,executed_by='human',note='Input and perception retained separately, no future/asynchronous pairing.')
    capture = DXcamCapture(CaptureConfig(device_idx=args.device_index, output_idx=args.output_index,
        fps=args.capture_fps, dxcam_buffer_frames=4,output_color=args.capture_color), clock, slot.put,latest_only=args.latest_capture,
        measure_latency=args.capture_timing,downscale_for_preview=args.capture_downscale)
    root = None
    worker = None
    timer = TimerResolution()
    input_capture = None
    input_handle = None
    inertia = None
    started = time.monotonic()
    try:
        if args.high_resolution_timer:timer.start()
        print(f'加载模型及预热；F8隐藏/显示，F10退出；运行{args.seconds:g}秒。不控制游戏。', flush=True)
        profile=CAMERA_PROFILES[args.tracking_profile]
        cv2.setNumThreads(profile['opencv_threads'])
        model = YOLO(str(detector),task='detect')
        for _ in range(10):
            model.predict(np.zeros((1080,1920,3),np.uint8),imgsz=1280,rect=False,device=0,
                          quantize=16,conf=CONFIDENCE,iou=.5,max_det=100,verbose=False,nms=None)
        if model.predictor.model.end2end:
            raise RuntimeError('Expected end2end=False runtime')
        if model.names != {0:'enemy',1:'teammate'}:
            raise RuntimeError('Detector classes differ')
        if policy:
            policy.start()
            config['c2_shadow'] = policy.info
            config['c2_display'] = dict(refresh_ms=500, summary_window_ms=1000,
                predictions_modified=False, note='Snapshot display only; sample fractions are not time fractions.')
        if args.c2_inertia_ms is not None:
            from c2_live_inertia import LiveShadowInertia
            inertia = LiveShadowInertia(clock,stop,out/'inertia_samples.csv',args.c2_inertia_ms)
            config['inertia'] = dict(response95_ms=args.c2_inertia_ms,requested_period_ms=1,
                controller_output=False,trained=False,graph_refresh_ms=16,numeric_refresh_ms=args.geometry_display_ms)
        appearance = AppearanceEncoder(APPEARANCE_WEIGHTS) if args.tracker=='advanced' else None
        tracker = AdvancedScreenTracker(max_lost_s=.2,memory_s=.8,stable_geometry=True,
            group_duplicates=False,camera_width=profile['width'],max_corners=profile['max_corners'],appearance=args.tracker=='advanced')
        selector = EnemyTargetSelector()
        if args.record_xinput:
            from fps_recorder.xinput_capture import XInputCapture
            input_handle = (out/'xinput.csv').open('w',newline='',encoding='utf-8')
            fields = ['timestamp_ns','poll_before_ns','packet_number','state_changed','rx','ry','lx','ly','rb','lb']
            input_writer = csv.DictWriter(input_handle,fieldnames=fields)
            input_writer.writeheader()
            def record_physical_input(row):
                values = {k:getattr(row,k) for k in fields if hasattr(row,k)}
                values.update(state_changed=int(row.state_changed),rb=int(bool(row.buttons_native & 0x200)),
                              lb=int(bool(row.buttons_native & 0x100)))
                input_writer.writerow(values)
            input_capture = XInputCapture(XInputConfig(enabled=True,poll_hz=4000,user_index=args.xinput_user_index),
                record_physical_input,lambda timestamp:None)
            input_capture.start()
        capture.start()
        desc = capture.camera._output.desc.DesktopCoordinates
        left,top = int(desc.left),int(desc.top)
        width,height = int(desc.right-desc.left),int(desc.bottom-desc.top)
        if capture.camera.rotation_angle != 0:
            raise RuntimeError('Initial overlay requires an unrotated output')
        root = tk.Tk()
        root.withdraw()
        window = tk.Toplevel(root)
        window.withdraw()
        window.overrideredirect(True)
        window.configure(bg='#010203')
        window.attributes('-transparentcolor','#010203')
        window.attributes('-topmost',True)
        canvas = tk.Canvas(window,width=width,height=height,bg='#010203',highlightthickness=0)
        canvas.pack()
        window.update_idletasks()
        hwnd = user32.GetAncestor(window.winfo_id(),2)
        style = user32.GetWindowLongPtrW(hwnd,-20)
        user32.SetWindowLongPtrW(hwnd,-20,style | 0x80000 | 0x20 | 0x08000000 | 0x80)
        affinity = wintypes.DWORD()
        if not user32.SetWindowDisplayAffinity(hwnd,0x11) or not user32.GetWindowDisplayAffinity(hwnd,ctypes.byref(affinity)) or affinity.value != 0x11:
            raise RuntimeError('Cannot exclude overlay from capture; stopping to prevent feedback')
        window.deiconify()
        window.update_idletasks()
        # Tk mapping can rewrite the wrapper styles. Apply native styles AFTER map.
        hwnd = user32.GetAncestor(window.winfo_id(),2)
        style = user32.GetWindowLongPtrW(hwnd,-20)
        user32.SetWindowLongPtrW(hwnd,-20,style | 0x80000 | 0x20 | 0x08000000 | 0x80)
        if (not user32.SetWindowDisplayAffinity(hwnd,0x11)
                or not user32.GetWindowDisplayAffinity(hwnd,ctypes.byref(affinity)) or affinity.value != 0x11):
            raise RuntimeError('Mapped overlay capture exclusion failed')
        if not user32.SetWindowPos(hwnd,wintypes.HWND(-1),left,top,width,height,0x10 | 0x40):
            raise ctypes.WinError(ctypes.get_last_error())
        user32.ShowWindow(hwnd,4)
        canvas.create_rectangle(12,12,610,68,outline='#00ff88',width=3,tags='diagnostic')
        hud = canvas.create_text(24,40,anchor='w',fill='#00ff88',font=('Microsoft YaHei',12,'bold'),
            text='只读叠加层已显示 | 等待画面 | F8隐藏 F10退出',tags='diagnostic')
        action_hud = None
        geometry_hud = None
        if policy and not geometry:
            canvas.create_rectangle(12,76,900,360,outline='#ffdd66',width=2,tags='diagnostic')
            canvas.create_text(24,96,anchor='w',fill='#ffdd66',font=('Microsoft YaHei',14,'bold'),
                text='C2只读快照：每0.5秒刷新 | 不发送手柄',tags='diagnostic')
            canvas.create_oval(35,155,155,275,outline='#ffdd66',width=2,tags='diagnostic')
            canvas.create_line(35,215,155,215,fill='#777777',tags='diagnostic')
            canvas.create_line(95,155,95,275,fill='#777777',tags='diagnostic')
            canvas.create_text(95,305,text='右摇杆原始建议\n正RY向上示意',fill='#ffdd66',tags='diagnostic')
            action_dot = canvas.create_oval(89,209,101,221,fill='#ffdd66',tags='diagnostic')
            action_hud = canvas.create_text(195,130,anchor='nw',fill='#ffdd66',font=('Microsoft YaHei',14,'bold'),
                text='等待因果历史',tags='diagnostic')
        if geometry:
            canvas.create_rectangle(12,76,870,440,outline='#a5b4fc',width=3,tags='diagnostic')
            canvas.create_text(30,102,anchor='w',fill='#a5b4fc',font=('Microsoft YaHei',16,'bold'),
                text=(f'惯性后摇杆轨迹｜{args.c2_inertia_ms:g}ms响应｜只读' if inertia else
                      f'合成摇杆建议（C2＋bias）｜每{args.geometry_display_ms/1000:g}秒快照｜只读'),tags='diagnostic')
            canvas.create_oval(55,145,245,335,outline='#a5b4fc',width=3,tags='diagnostic')
            canvas.create_line(55,240,245,240,fill='#777777',tags='diagnostic')
            canvas.create_line(150,145,150,335,fill='#777777',tags='diagnostic')
            for x,y,label in [(32,240,'左'),(268,240,'右'),(150,132,'上'),(150,353,'下')]:
                canvas.create_text(x,y,text=label,fill='#e0e7ff',font=('Microsoft YaHei',12),tags='diagnostic')
            geometry_arrow = canvas.create_line(150,240,150,240,fill='#a5b4fc',width=6,arrow='last',state='hidden',tags='diagnostic')
            geometry_dot = canvas.create_oval(142,232,158,248,fill='#a5b4fc',outline='white',state='hidden',tags='diagnostic')
            geometry_hud = canvas.create_text(305,148,anchor='nw',fill='#e0e7ff',font=('Microsoft YaHei',18,'bold'),
                text='等待有效目标',tags='diagnostic')
            geometry_buttons = []
            for y,label in ([] if inertia else [(315,'开火 RB'),(360,'瞄准 LB')]):
                canvas.create_text(305,y,anchor='w',text=label,fill='#e0e7ff',font=('Microsoft YaHei',13),tags='diagnostic')
                canvas.create_rectangle(405,y-12,755,y+12,outline='#a5b4fc',tags='diagnostic')
                geometry_buttons.append(canvas.create_rectangle(405,y-12,405,y+12,fill='#a5b4fc',outline='',tags='diagnostic'))
            if inertia:
                canvas.create_text(305,330,anchor='w',text='本轮只检查摇杆轨迹\n开火／瞄准未加入惯性处理',
                    fill='#a5b4fc',font=('Microsoft YaHei',12),tags='diagnostic')
            canvas.create_text(30,410,anchor='w',text=('箭头连续显示惯性轨迹；数字定时刷新。未发送手柄信号。' if inertia else
                '箭头表示合成方向与幅度；条形表示按钮分数。未发送手柄信号。'),
                fill='#a5b4fc',font=('Microsoft YaHei',12),tags='diagnostic')
        config.update(screen_rect=[left,top,width,height],capture_exclusion_api_verified=True,
            capture_exclusion_visual_verified=False,dxcam_version='0.3.0',appearance_sha256=sha(APPEARANCE_WEIGHTS) if appearance else None,
            overlay_revision=2,topmost_refresh_ms=500,always_visible_diagnostic=True)
        (out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        print('叠加层已设置采集排除；请确认框与人物对齐，且没有自反馈。输出：',out,flush=True)
        started = time.monotonic()

        def infer():
            last_t = None
            count = 0
            try:
                with (out/'frames.jsonl').open('w',encoding='utf-8') as log:
                    while not stop.is_set():
                        packet = slot.wait_take() if args.event_wakeup else slot.take()
                        if packet is None:
                            if not args.event_wakeup:stop.wait(.002)
                            continue
                        if last_t is not None and packet.timestamp_ns <= last_t:
                            continue
                        last_t = packet.timestamp_ns
                        before = clock.now_ns()
                        if before-packet.timestamp_ns > 150_000_000:
                            with lock:
                                shared['result'] = None
                            continue
                        image = packet.image
                        if image.shape[:2] != (1080,1920):
                            image = cv2.resize(image,(1920,1080),interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        t0 = clock.now_ns()
                        predictions = prediction_rows(model.predict(image,imgsz=1280,rect=False,device=0,
                            quantize=16,conf=CONFIDENCE,iou=.5,max_det=100,verbose=False,nms=None)[0])
                        t1 = clock.now_ns()
                        features = appearance.extract(rgb,predictions) if appearance else None
                        t2 = clock.now_ns()
                        tracks = tracker.update_frame(packet.timestamp_ns,predictions,rgb,features)
                        selection = selector.update(tracks,packet.timestamp_ns,1920,1080)
                        tracking_finish = clock.now_ns()
                        suggestion = policy.update(image,predictions,packet.timestamp_ns) if policy else None
                        policy_finish = clock.now_ns()
                        corrected = geometry.suggest(suggestion,selection,packet.timestamp_ns) if geometry else None
                        if inertia:
                            inertia.publish(corrected)
                        finish = clock.now_ns()
                        count += 1
                        measurements = dict(prepare=(t0-before)/1e6,source_age_at_start=(before-packet.timestamp_ns)/1e6,
                            predict=(t1-t0)/1e6,appearance=(t2-t1)/1e6,
                            tracking=(tracking_finish-t2)/1e6,processing=(finish-before)/1e6,
                            source_age_at_finish=(finish-packet.timestamp_ns)/1e6)
                        if policy:
                            measurements['c2_shadow_update'] = (policy_finish-tracking_finish)/1e6
                        if geometry:
                            measurements['geometry_bias'] = (finish-policy_finish)/1e6
                        if args.capture_timing:
                            measurements.update(split_capture_latency(packet,before))
                        if count > 30:
                            for key,value in measurements.items():
                                timings[key].append(value)
                        camera_info=tracker.diagnostics.get('camera',{})
                        result = dict(sequence=count,source_present_ns=packet.timestamp_ns,tracks=tracks,
                            camera_status=camera_info.get('status'),camera_matrix=camera_info.get('matrix'),
                            target=selection['target_id'],measurements=measurements)
                        if policy:
                            result['c2_shadow'] = suggestion
                        if geometry:
                            result['geometry_bias'] = corrected
                        with lock:
                            shared['result'] = result
                        log.write(json.dumps(dict(**result,detections=predictions,target_selection=selection),ensure_ascii=False)+'\n')
                        if count == 60:
                            cv2.imwrite(str(out/'采集原图_不含叠加框.png'),image)
            except Exception as exc:
                with lock:
                    shared['error'] = repr(exc)
            finally:
                with lock:
                    shared['finished'] = True

        worker = threading.Thread(target=infer,name='ReadOnlyC1',daemon=True)
        if inertia:
            inertia.start()
        worker.start()
        shown = None
        hidden = False
        old_f8 = False
        last_raise = 0.0
        last_hud = 0.0
        last_action_hud = 0.0
        last_geometry_hud = 0.0
        last_inertia_graph = 0.0
        topmost_refreshes = 0
        last_tick_ns = None

        def tick():
            nonlocal shown,hidden,old_f8,last_raise,last_hud,last_action_hud,last_geometry_hud,last_inertia_graph,topmost_refreshes,last_tick_ns
            tick_ns=clock.now_ns()
            f8 = bool(user32.GetAsyncKeyState(0x77)&0x8000)
            if f8 and not old_f8:
                hidden = not hidden
                user32.ShowWindow(hwnd,0 if hidden else 4)
            old_f8 = f8
            with lock:
                result,error,finished = shared['result'],shared['error'],shared['finished']
            inertia_result = None
            if inertia:
                inertia_result,inertia_error = inertia.snapshot()
                if inertia_error:
                    with lock:
                        shared['error'] = inertia_error
                    error = inertia_error
            if action_display is not None and result is not None:
                action_display.observe(result.get('c2_shadow'))
            if last_tick_ns is not None and result is not None and result['sequence']>30:
                timings['ui_poll_interval'].append((tick_ns-last_tick_ns)/1e6)
            last_tick_ns=tick_ns
            stats = capture.snapshot_stats()
            input_stats = input_capture.snapshot_stats() if input_capture else None
            if input_stats and (input_stats.errors or input_stats.disconnects):
                with lock:
                    shared['error'] = input_stats.error_message or 'Physical controller disconnected during calibration'
                error = shared['error']
            if error or stats.errors or finished or user32.GetAsyncKeyState(0x79)&0x8000 or time.monotonic()-started >= args.seconds:
                stop.set()
                root.quit()
                return
            now = time.monotonic()
            if not hidden and now-last_raise >= .5:
                # Restore Z-order after game activation without taking keyboard focus.
                if not user32.SetWindowPos(hwnd,wintypes.HWND(-1),left,top,width,height,0x10 | 0x40):
                    with lock:
                        shared['error'] = 'Overlay SetWindowPos failed'
                    stop.set();root.quit();return
                last_raise = now
                topmost_refreshes += 1
            age = (clock.now_ns()-result['source_present_ns'])/1e6 if result else float('inf')
            if inertia_result and not hidden and now-last_inertia_graph >= .016:
                u = inertia_result['axes']
                scale = max(1.,float(np.hypot(*u)))
                px,py = 150+95*u[0]/scale,240-95*u[1]/scale
                canvas.coords(geometry_arrow,150,240,px,py)
                canvas.coords(geometry_dot,px-8,py-8,px+8,py+8)
                canvas.itemconfigure(geometry_arrow,state='normal')
                canvas.itemconfigure(geometry_dot,state='normal')
                last_inertia_graph = now
            if now-last_hud >= .1:
                message = '等待画面' if result is None else f"处理{result['measurements']['processing']:.0f}ms | 画面年龄{age:.0f}ms"
                if result is not None and age > 150:
                    message += ' | 过期框已清除'
                canvas.itemconfigure(hud,text=f'只读叠加 | {message} | F8隐藏 F10退出')
                if action_hud is not None:
                    suggestion = result.get('c2_shadow') if result else None
                    suggestion_age = (clock.now_ns()-suggestion['source_present_ns'])/1e6 if suggestion else float('inf')
                    if suggestion_age > 150:
                        canvas.itemconfigure(action_hud,text='等待因果历史／建议已过期\n不显示旧动作')
                        canvas.itemconfigure(action_dot,state='hidden')
                        last_action_hud = 0.0
                    elif now-last_action_hud >= .5:
                        a = suggestion['action']
                        canvas.itemconfigure(action_hud,text=action_display.text())
                        dx,dy = 95+60*a['rx'],215-60*a['ry']
                        canvas.coords(action_dot,dx-6,dy-6,dx+6,dy+6)
                        canvas.itemconfigure(action_dot,state='normal')
                        last_action_hud = now
                if geometry_hud is not None:
                    correction = result.get('geometry_bias') if result else None
                    if inertia_result is not None:
                        if now-last_geometry_hud >= args.geometry_display_ms/1000:
                            u = inertia_result['axes']
                            state_text = f"目标 #{inertia_result['target_id']}" if inertia_result['fresh_target'] else '无有效目标，平滑回中'
                            canvas.itemconfigure(geometry_hud,text=f"{state_text}\n惯性 RX {u[0]:+.2f}\n惯性 RY {u[1]:+.2f}\n数字为定时快照")
                            last_geometry_hud = now
                        for bar in geometry_buttons:
                            canvas.itemconfigure(bar,state='hidden')
                    elif not correction or age > 150:
                        canvas.itemconfigure(geometry_hud,text='无有效目标／建议过期\n等待合成建议')
                        canvas.itemconfigure(geometry_arrow,state='hidden')
                        canvas.itemconfigure(geometry_dot,state='hidden')
                        for bar in geometry_buttons:
                            canvas.itemconfigure(bar,state='hidden')
                        last_geometry_hud = 0.0
                    elif now-last_geometry_hud >= args.geometry_display_ms/1000:
                        u = correction['combined_axes']
                        horizontal = '右' if u[0]>.05 else '左' if u[0]<-.05 else ''
                        vertical = '上' if u[1]>.05 else '下' if u[1]<-.05 else ''
                        direction = horizontal+vertical or '居中'
                        canvas.itemconfigure(geometry_hud,text=(f"目标 #{correction['target_id']}  ·  {direction}\n"
                            f"合成 RX {u[0]:+.2f}\n合成 RY {u[1]:+.2f}\n"
                            f"{'已达幅度上限' if correction['saturated'] else '当前快照'}"))
                        # Radial projection is for the circular illustration only; numeric/logged axes stay exact.
                        radius_scale = max(1.,float(np.hypot(*u)))
                        px,py = 150+95*u[0]/radius_scale,240-95*u[1]/radius_scale
                        canvas.coords(geometry_arrow,150,240,px,py)
                        canvas.coords(geometry_dot,px-8,py-8,px+8,py+8)
                        canvas.itemconfigure(geometry_arrow,state='normal')
                        canvas.itemconfigure(geometry_dot,state='normal')
                        button_action = result['c2_shadow']['action']
                        for bar,y,key in zip(geometry_buttons,(315,360),('rb','lb')):
                            canvas.coords(bar,405,y-12,405+350*button_action[key],y+12)
                            canvas.itemconfigure(bar,state='normal',fill='#86efac' if button_action[key]>=.5 else '#64748b')
                        last_geometry_hud = now
                last_hud = now
            if not hidden and (result is None or age > 150):
                canvas.delete('track')
                shown = None
            elif not hidden and result['sequence'] != shown:
                canvas.delete('track')
                sx,sy = width/1920,height/1080
                for tr in result['tracks']:
                    if tr['state'] not in ('observed','tentative','unassociated') or tr['box'] is None:
                        continue
                    color = '#ff6868' if tr['class_id']==0 else '#50bfff' if tr['class_id']==1 else '#aaaaaa'
                    if tr['track_id'] is not None and tr['track_id']==result['target']:
                        color = '#00ff88'
                    x1,y1,x2,y2 = tr['box']
                    canvas.create_rectangle(x1*sx,y1*sy,x2*sx,y2*sy,outline=color,width=3,tags='track')
                    label = {0:'敌人',1:'队友',None:'不确定'}[tr['class_id']]
                    canvas.create_text(x1*sx,max(16,y1*sy-14),anchor='w',text=f"#{tr['track_id']} {label}",fill=color,font=('Microsoft YaHei',12,'bold'),tags='track')
                canvas.tag_raise('diagnostic')
                if result['sequence'] > 30:
                    timings['source_age_at_ui_submit'].append((clock.now_ns()-result['source_present_ns'])/1e6)
                shown = result['sequence']
            root.after(args.ui_poll_ms,tick)

        root.after(args.ui_poll_ms,tick)
        root.mainloop()
    except Exception as exc:
        with lock:
            shared['error'] = repr(exc)
    finally:
        stop.set()
        capture.stop()
        if worker is not None:
            worker.join(timeout=15)
        if inertia:
            inertia.join()
        if input_capture is not None:
            input_capture.stop()
        if input_handle is not None:
            input_handle.flush()
            input_handle.close()
        if root is not None:
            root.destroy()
        timer.stop()
        with lock:
            error = shared['error']
        report = dict(status='failed' if error or capture.snapshot_stats().errors else 'stopped',error=error,
            timings_ms={k:distribution(v) for k,v in timings.items()},capture_stats=vars(capture.snapshot_stats()),
            latest_slot_overwrites=slot.overwritten,worker_stopped=worker is None or not worker.is_alive(),
            controller_output=False, note=config['timing_note'])
        if input_capture is not None:
            physical_stats = vars(input_capture.snapshot_stats())
            report['physical_input_stats'] = physical_stats
            if physical_stats['errors'] or physical_stats['disconnects'] or not physical_stats['readings']:
                report['status'] = 'failed'
                report['error'] = report['error'] or physical_stats['error_message'] or 'Missing/disconnected physical input'
                error = report['error']
        if inertia:
            report['inertia'] = inertia.report()
            if report['inertia']['error']:
                report['status'] = 'failed'
                error = report['error'] = report['error'] or report['inertia']['error']
        (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('结果：',out,flush=True)
        if error:
            raise RuntimeError(error)


if __name__ == '__main__':
    main()
