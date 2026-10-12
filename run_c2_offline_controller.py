"""User-run offline range only: one merged virtual pad, reversible physical hiding."""
import argparse
import csv
import ctypes
import importlib.util
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from benchmark_c1_day4_engine import ENGINE
from benchmark_c1_tensorrt import verify_engine
from c1_target_selection import EnemyTargetSelector
from c1_timer_resolution import TimerResolution
from c1_tracking_advanced import AdvancedScreenTracker, CAMERA_PROFILES
from c2_control_arbitration import OwnershipGate, ButtonLatch, merge_report
from c2_geometry_bias import GeometryBias
from c2_live_inertia import TrajectoryState
from c2_shadow import ShadowPolicy
from c2_output_diagnostic import DiagnosticLook, readback_row
from c2_engagement_buttons import EngagementButtons
from demo_c1_day4_supplement_tracking import WEIGHTS
from demo_c1_video_tracking import prediction_rows
from fps_recorder.clock import QpcClock
from fps_recorder.config import CaptureConfig
from fps_recorder.dxcam_capture import DXcamCapture
from fps_recorder.xinput_capture import NativeState
from preview_c1_live_overlay import LatestFrame
from c2_hidhide_session import CLI, LEASE, begin, restore, physical_devices, PhysicalDeviceNotReady, extend_reconnected_devices, state as hide_state

ROOT = Path(__file__).resolve().parent
PROBE_EXE = Path('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
USER32 = ctypes.WinDLL('user32')
READ = ctypes.WinDLL('xinput1_4.dll').XInputGetState
READ.argtypes=[ctypes.c_uint32,ctypes.POINTER(NativeState)]
READ.restype=ctypes.c_uint32


def connected_slots():
    return [i for i in range(4) if READ(i,ctypes.byref(NativeState())) == 0]


def physical_report(slot):
    native = NativeState()
    if READ(slot,ctypes.byref(native)):
        raise RuntimeError('Physical controller disconnected/read failed; stop output and restore device')
    p = native.pad
    return dict(lx=max(-1,p.lx/32767),ly=max(-1,p.ly/32767),
                rx=max(-1,p.rx/32767),ry=max(-1,p.ry/32767),
                lt=p.lt/255,rt=p.rt/255,buttons=int(p.buttons))


def unprivileged_slots():
    # Different non-allowlisted executable verifies physical XInput is hidden.
    script = r'''Add-Type -TypeDefinition '
using System;using System.Collections.Generic;using System.Runtime.InteropServices;
public class C2Probe {
[StructLayout(LayoutKind.Sequential)]public struct Pad {public ushort b;public byte l,r;public short x,y,u,v;}
[StructLayout(LayoutKind.Sequential)]public struct State {public uint n;public Pad p;}
[DllImport("xinput1_4.dll")]public static extern uint XInputGetState(uint i,out State s);
public static int[] Slots(){var a=new List<int>();for(uint i=0;i<4;i++){State s;if(XInputGetState(i,out s)==0)a.Add((int)i);}return a.ToArray();}}
';ConvertTo-Json -InputObject @([C2Probe]::Slots()) -Compress'''
    result = subprocess.run([str(PROBE_EXE),'-NoProfile','-Command',script],capture_output=True,
                            text=True,encoding='utf-8',errors='replace',timeout=20,creationflags=0x08000000,
                            env=dict(os.environ,PSModulePath=str(PROBE_EXE.parent/'Modules')))
    if result.returncode:
        raise RuntimeError('Independent XInput visibility probe failed: '+result.stderr)
    slots = json.loads(result.stdout)
    if not isinstance(slots,list):
        raise ValueError('Unexpected visibility probe')
    return slots


def send(pad, report):
    pad.report.wButtons = report['buttons']
    pad.left_joystick_float(x_value_float=report['lx'],y_value_float=report['ly'])
    pad.right_joystick_float(x_value_float=report['rx'],y_value_float=report['ry'])
    pad.left_trigger_float(value_float=report['lt'])
    pad.right_trigger_float(value_float=report['rt'])
    pad.update()


def wait_for_isolation(timeout_seconds=45):
    if not unprivileged_slots():
        if len(connected_slots())!=1:
            raise RuntimeError('Merger cannot read exactly one physical input after hiding')
        return
    print('隔离尚未生效：请现在拔下并重新插入真实手柄（同一USB口）。程序保持隔离，等待最多45秒；尚未创建虚拟手柄。',flush=True)
    deadline=time.monotonic()+timeout_seconds
    while time.monotonic()<deadline:
        if USER32.GetAsyncKeyState(0x79)&0x8000:
            raise RuntimeError('Isolation setup cancelled by F10; restoring physical controller')
        time.sleep(.5)
        if not unprivileged_slots() and len(connected_slots())==1:
            print('物理隔离已通过：合并器能读真实手柄，独立进程看不到真实手柄。',flush=True)
            return
    raise RuntimeError('Physical isolation still failed after reconnect window; restore and stop, no virtual output started')


def wait_for_physical_setup(timeout_seconds=45):
    deadline=time.monotonic()+timeout_seconds
    announced=False
    last_error='Physical XInput controller not connected'
    while time.monotonic()<deadline:
        if USER32.GetAsyncKeyState(0x79)&0x8000:
            raise RuntimeError('Physical connection setup cancelled by F10; no device hiding applied')
        slots=connected_slots()
        if len(slots)>1:
            raise RuntimeError('More than one XInput controller connected; cannot select physical input safely')
        if len(slots)==1:
            try:
                devices=physical_devices()
                current=connected_slots()
                if current==slots:
                    return devices,slots[0]
                last_error='Physical controller changed while enumerating; waiting for stable connection'
            except PhysicalDeviceNotReady as exc:
                last_error=str(exc)
        if not announced:
            print('启动连接等待：请先插好真实手柄并保持连接。尚未隐藏设备；稍后只有提示“隔离尚未生效”时才拔插。',flush=True)
            announced=True
        time.sleep(.5)
    raise RuntimeError('Physical startup did not become ready within 45 seconds: '+last_error)


def wait_for_unplug(timeout_seconds=90):
    print('槽位0准备：请现在拔下真实手柄，暂时不要插回。程序等待最多90秒，F10取消。',flush=True)
    deadline=time.monotonic()+timeout_seconds
    while time.monotonic()<deadline:
        if USER32.GetAsyncKeyState(0x79)&0x8000:
            raise RuntimeError('Virtual-first setup cancelled; restore physical device')
        if not connected_slots() and not unprivileged_slots():
            return
        time.sleep(.5)
    raise RuntimeError('Physical controller did not disconnect in time; restore and stop')


def wait_for_replug(virtual_slot, expected_devices, timeout_seconds=90):
    print('虚拟手柄已占槽位0：现在把真实手柄插回同一USB口，等待合并完成。',flush=True)
    deadline=time.monotonic()+timeout_seconds
    while time.monotonic()<deadline:
        if USER32.GetAsyncKeyState(0x79)&0x8000:
            raise RuntimeError('Virtual-first reconnect cancelled; restore physical device')
        slots=connected_slots()
        if virtual_slot not in slots or len(slots)>2:
            raise RuntimeError('Unexpected controller change during virtual-first reconnect')
        other=[value for value in slots if value!=virtual_slot]
        if len(other)==1:
            try:
                devices=physical_devices()
            except PhysicalDeviceNotReady:
                time.sleep(.5);continue
            if devices!=set(expected_devices):
                extend_reconnected_devices(expected_devices,devices)
                print('同一真实手柄的XInput接口路径已更新；新接口已加入本轮隔离及退出恢复记录。',flush=True)
            if unprivileged_slots()!=[virtual_slot]:
                raise RuntimeError('Reconnected physical controller is visible outside merger; restore and stop')
            physical_report(other[0])
            return other[0]
        time.sleep(.5)
    raise RuntimeError('Physical controller reconnect timed out; restore and stop')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--isolation-only',action='store_true',help='Test hide/one neutral virtual pad/restore without model inference or gameplay')
    parser.add_argument('--response95-ms',type=float,default=80)
    parser.add_argument('--diagnostic-look',action='store_true',help='Offline smooth left/right input-path test; no model actions or AI buttons')
    parser.add_argument('--start-paused',action='store_true',help='Keep AI paused initially; press F9 after offline game starts')
    parser.add_argument('--virtual-first',action='store_true',help='Guided unplug/create slot0/replug setup; requires offline game closed before starting')
    parser.add_argument('--button-policy',choices=['bc','geometry'],default='bc',help='bc: learned scores; geometry: offline target/alignment sustained-button baseline')
    parser.add_argument('--geometry-base-weight',type=float,default=1.,help='Weight of BC RX/RY in additive geometry control; 0 is an offline geometry-only comparison')
    parser.add_argument('--geometry-gain-x',type=float,default=1.)
    parser.add_argument('--geometry-gain-y',type=float,default=1.)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:
        parser.error('seconds must be positive')
    TrajectoryState(args.response95_ms)
    geometry=GeometryBias(base_weight=args.geometry_base_weight,gain_x=args.geometry_gain_x,gain_y=args.geometry_gain_y)
    if args.diagnostic_look and args.isolation_only:
        parser.error('diagnostic-look and isolation-only are separate tests')
    verify_engine(ENGINE,source_weights=WEIGHTS)
    policy = ShadowPolicy()
    if importlib.util.find_spec('vgamepad') is None:
        raise RuntimeError('Missing vgamepad')
    if not CLI.is_file():
        raise RuntimeError('HidHide missing. First run setup_c2_controller.py in PyCharm, install, then restart Windows.')
    before = hide_state()
    device_nodes,physical_slot = wait_for_physical_setup()
    if any(Path(a).name.lower() in ('powershell.exe','pwsh.exe') for a in before['apps']):
        raise RuntimeError('Visibility probe is allowlisted in existing HidHide settings; cannot verify single-device isolation')
    if args.check_only:
        print(f'静态检查通过：模型、映射、vgamepad、HidHide、{len(device_nodes)}个真实手柄节点、唯一真实手柄；未隐藏、未创建虚拟手柄、未采集。')
        return
    out=ROOT/'runs/c2_closed_loop'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    config=dict(stage='Stage 1 — Local Combat Policy',scope='User-authorized local offline range',
        isolation_only=args.isolation_only,
        diagnostic_look=args.diagnostic_look,
        start_paused=args.start_paused,
        virtual_first=args.virtual_first,
        button_policy=args.button_policy,geometry_button_confirm_ms=60,geometry_aim_release_ms=150,
        physical_slot=physical_slot,threshold=.1,threshold_axes=['lx','ly','rx','ry'],quiet_seconds=1,
        response95_ms=args.response95_ms,geometry=geometry.config(),
        fire_on=.65,fire_off=.35,aim_on=.6,aim_off=.4,fire_requires_crosshair_inside_selected_box=True,
        passthrough=['lx','ly','lt','rt','all buttons except AI-owned RB/LB'],
        focus_gating=False,focus_note='User requested no foreground gating; F9 manually pauses AI.',
        requested_report_hz=1000,controller_output=True,seconds=args.seconds,
        trained_with_rl=False,policy=policy.info,cleanup_lease=str(LEASE))
    (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    stop=threading.Event();lock=threading.Lock();shared=dict(packet=None,error=None)
    clock=QpcClock();slot=LatestFrame();pad=None;worker=None;guardian=None;capture=None
    timer=TimerResolution();error=None;intervals=[];sources=[];mode_counts={'manual':0,'ai':0};transitions=[];virtual_created=False
    readback_samples=[];readback_errors=0;readback_handle=None
    try:
        timer.start()
        cv2.setNumThreads(4)
        if not args.isolation_only and not args.diagnostic_look:
            detector=YOLO(str(ENGINE),task='detect')
            predict=dict(imgsz=1280,rect=False,device=0,quantize=16,conf=.1,iou=.5,max_det=100,verbose=False,nms=None)
            for _ in range(10):detector.predict(np.zeros((1080,1920,3),np.uint8),**predict)
            if detector.predictor.model.end2end or detector.names != {0:'enemy',1:'teammate'}:
                raise RuntimeError('Detector branch/classes differ')
            policy.start()
        tracker=AdvancedScreenTracker(max_lost_s=.2,memory_s=.8,stable_geometry=True,
            group_duplicates=False,camera_width=480,max_corners=200,appearance=False)
        selector=EnemyTargetSelector()
        guardian=begin()
        wait_for_isolation()
        physical_slot=connected_slots()[0]
        config['physical_slot']=physical_slot
        (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        physical_report(physical_slot)  # Allowlisted merger must still read real input.
        if args.virtual_first:
            wait_for_unplug()
        import vgamepad as vg
        pad=vg.VX360Gamepad()  # Neutral report only at construction.
        virtual_created=True
        time.sleep(.25)
        visible_slots=unprivileged_slots()
        if args.virtual_first:
            if visible_slots != [0] or connected_slots()!=[0]:
                raise RuntimeError('Virtual controller did not obtain XInput slot0; restore and stop')
            physical_slot=wait_for_replug(0,device_nodes)
            config['physical_slot']=physical_slot
        if len(visible_slots) != 1 or len(connected_slots()) != 2 or visible_slots[0]==physical_slot:
            raise RuntimeError('Expected exactly one game-visible virtual controller and one hidden physical input')
        virtual_slot=visible_slots[0]
        config['virtual_slot']=virtual_slot
        (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        if args.isolation_only:
            print('隔离测试通过：外部仅见1个中立虚拟手柄，程序能读隐藏的真实手柄。现在销毁虚拟设备并恢复真实手柄。',flush=True)
            return
        if not args.diagnostic_look:
            capture=DXcamCapture(CaptureConfig(device_idx=0,output_idx=0,fps=120,dxcam_buffer_frames=4,
            output_color='BGR'),clock,slot.put,latest_only=True,measure_latency=True,downscale_for_preview=True)
            capture.start()
        def infer():
            last=None
            try:
                with (out/'perception.jsonl').open('w',encoding='utf-8') as log:
                    while not stop.is_set():
                        frame=slot.take()
                        if frame is None:
                            stop.wait(.002);continue
                        if last is not None and frame.timestamp_ns<=last:continue
                        last=frame.timestamp_ns
                        if clock.now_ns()-frame.timestamp_ns>150_000_000:
                            with lock:shared['packet']=None
                            continue
                        image=frame.image
                        if image.shape[:2]!=(1080,1920):image=cv2.resize(image,(1920,1080),interpolation=cv2.INTER_AREA)
                        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        detections=prediction_rows(detector.predict(image,**predict)[0])
                        tracks=tracker.update_frame(frame.timestamp_ns,detections,rgb,None)
                        selection=selector.update(tracks,frame.timestamp_ns,1920,1080)
                        base=policy.update(image,detections,frame.timestamp_ns)
                        correction=geometry.suggest(base,selection,frame.timestamp_ns)
                        packet=dict(source_present_ns=frame.timestamp_ns,ready_ns=clock.now_ns(),
                                    selection=selection,base=base,correction=correction)
                        with lock:shared['packet']=packet
                        enemy_tracks=[tr for tr in tracks if tr.get('class_id')==0]
                        target_diagnostics=dict(
                            detected_enemy=sum(d['class']==0 for d in detections),
                            detected_enemy_score_ge025=sum(d['class']==0 and d['score']>=.25 for d in detections),
                            observed_enemy_tracks=sum(tr.get('state')=='observed' for tr in enemy_tracks),
                            confirmed_observed_enemy_tracks=sum(tr.get('state')=='observed' and bool(tr.get('confirmed')) for tr in enemy_tracks),
                            eligible_enemy_tracks=sum(tr.get('state')=='observed' and bool(tr.get('confirmed')) and bool(tr.get('position_valid')) and bool(tr.get('association_valid',True)) for tr in enemy_tracks),
                            selected=selection['selected'] is not None,correction_valid=correction is not None)
                        track_audit=[dict((key,tr.get(key)) for key in (
                            'track_id','state','class_id','last_observed_class','identity_status',
                            'confirmed','position_valid','association_valid','association_reason','score')) for tr in tracks]
                        log.write(json.dumps(dict(**packet,detections=detections,target_diagnostics=target_diagnostics,
                            track_audit=track_audit,tracker_audit=tracker.diagnostics),ensure_ascii=False)+'\n')
            except BaseException as exc:
                with lock:shared['error']=repr(exc)
        if not args.diagnostic_look:
            worker=threading.Thread(target=infer,name='C2OfflinePerception',daemon=True);worker.start()
        gate=OwnershipGate();trajectory=TrajectoryState(args.response95_ms)
        fire=ButtonLatch();aim=ButtonLatch(.6,.4);enabled=not args.start_paused;previous_mode='manual';previous_packet=None
        engagement=EngagementButtons()
        old_f9=False;last_tick=None;start=time.monotonic()
        diagnostic=DiagnosticLook(args.response95_ms)
        last_readback=None;previous_sent=None;last_status=0
        readback_handle=(out/'virtual_readback.csv').open('w',newline='',encoding='utf-8')
        readback_writer=csv.writer(readback_handle)
        readback_writer.writerow(['read_ns','mode','virtual_slot','packet_number','read_rx','read_ry','read_buttons','previous_sent_rx','previous_sent_ry','previous_sent_buttons'])
        print('单一虚拟手柄已合并：四轴>10%立即人工接管；回中1秒返回AI。F9暂停/恢复AI，F10退出。',flush=True)
        print(f'设备槽位：真实={physical_slot}（隐藏），虚拟={virtual_slot}（游戏可见）。',flush=True)
        if args.start_paused:
            print('AI初始暂停；虚拟手柄保持连接，真实操作透传。可现在启动离线游戏，进入靶场后按F9开始。',flush=True)
        if args.button_policy=='geometry' and not args.diagnostic_look:
            print('离线按钮基线：LB随有效目标持续开镜（失目标150ms释放）；RB准心在目标框内连续60ms后持续开火，离框/失目标立即释放。视角模型保持。',flush=True)
        if not args.diagnostic_look:
            print(f'视角合成：C2摇杆权重={geometry.base_weight:g}；几何增益X/Y={geometry.gain_x:g}/{geometry.gain_y:g}，惯性{args.response95_ms:g}ms。',flush=True)
        if args.diagnostic_look:
            print('输入通路诊断：AI模式平滑右转1秒→回中1秒→左转1秒→回中1秒循环，幅度0.45，不由AI瞄准或开火。请进入离线靶场观察。',flush=True)
        with (out/'executed_actions.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle)
            writer.writerow(['input_observed_ns','decision_ns','send_end_ns','mode','fresh_target','source_present_ns',
                'physical_lx','physical_ly','physical_rx','physical_ry','physical_buttons',
                'sent_lx','sent_ly','sent_rx','sent_ry','sent_lt','sent_rt','sent_buttons'])
            while time.monotonic()-start<args.seconds and not USER32.GetAsyncKeyState(0x79)&0x8000:
                tick=clock.now_ns()
                if last_tick is not None:intervals.append((tick-last_tick)/1e6)
                last_tick=tick
                physical=physical_report(physical_slot);observed=clock.now_ns()
                if previous_sent is not None and (last_readback is None or observed-last_readback>=50_000_000):
                    native=NativeState()
                    result=READ(virtual_slot,ctypes.byref(native))
                    if result:
                        readback_errors+=1
                        raise RuntimeError('Virtual XInput readback failed; stop and restore')
                    row=readback_row(observed,previous_mode,virtual_slot,native,previous_sent)
                    readback_writer.writerow(row);readback_samples.append(row)
                    last_readback=observed
                f9=bool(USER32.GetAsyncKeyState(0x78)&0x8000)
                if f9 and not old_f9:enabled=not enabled
                old_f9=f9
                with lock:packet,problem=shared['packet'],shared['error']
                if problem:raise RuntimeError(problem)
                if capture is not None and capture.snapshot_stats().errors:raise RuntimeError('Capture failure')
                now=clock.now_ns();mode=gate.update(physical,now,enabled)
                if mode != previous_mode:
                    transitions.append(dict(timestamp_ns=now,mode=mode))
                    print('接管状态：','AI' if mode=='ai' else '人工',flush=True)
                    previous_packet=None
                if mode=='manual':
                    engagement.update(now,None,False,False,False)
                    diagnostic.step(now,False)
                    trajectory=TrajectoryState(args.response95_ms)
                    trajectory.axes.position=[physical['rx'],physical['ry']]
                    trajectory.axes.target=trajectory.axes.position.copy()
                    trajectory.axes.advance(now)
                    fire.update(0,False);aim.update(0,False)
                    merged=merge_report(physical,'manual');fresh=False;source=None
                elif args.diagnostic_look:
                    fresh=False;source=None
                    merged=merge_report(physical,'ai',diagnostic.step(now,True),False,False)
                else:
                    if packet is not previous_packet:
                        trajectory.command(packet['correction'] if packet else None,now)
                        previous_packet=packet
                    state=trajectory.step(now);fresh=state['fresh_target'];source=state['source_present_ns']
                    base=packet['base'] if packet else None
                    selected=packet['selection']['selected'] if packet else None
                    allowed=fresh and base is not None and selected is not None
                    aligned=allowed and selected['box'][0]<=960<=selected['box'][2] and selected['box'][1]<=540<=selected['box'][3]
                    if args.button_policy=='geometry':
                        pressed_fire,pressed_aim=engagement.update(now,selected['track_id'] if selected else None,allowed,aligned)
                    else:
                        pressed_fire=fire.update(base['action']['rb'] if base else 0,aligned)
                        pressed_aim=aim.update(base['action']['lb'] if base else 0,allowed)
                    merged=merge_report(physical,'ai',state['axes'],pressed_fire,pressed_aim)
                send(pad,merged);end=clock.now_ns()
                previous_sent=dict(merged)
                if now-last_status>=1_000_000_000:
                    read_axes=readback_samples[-1][4:6] if readback_samples else [0.,0.]
                    target_status='目标识别=不适用（输入诊断）' if args.diagnostic_look else f'有效目标={int(fresh)}'
                    print(f"输出诊断：{'AI' if mode=='ai' else '人工'} {target_status} 发送RX/RY={merged['rx']:+.3f}/{merged['ry']:+.3f} 虚拟回读={read_axes[0]:+.3f}/{read_axes[1]:+.3f}",flush=True)
                    last_status=now
                if mode=='ai' and fresh:sources.append((end-source)/1e6)
                mode_counts[mode]+=1
                writer.writerow([observed,now,end,mode,int(fresh),source,
                    *[physical[k] for k in ('lx','ly','rx','ry','buttons')],
                    *[merged[k] for k in ('lx','ly','rx','ry','lt','rt','buttons')]])
                previous_mode=mode
                remaining=(tick+1_000_000-clock.now_ns())/1e9
                if remaining>0:stop.wait(remaining)
    except BaseException as exc:
        error=repr(exc)
    finally:
        stop.set()
        if readback_handle is not None:readback_handle.close()
        if pad is not None:
            try:pad.reset();pad.update()
            except Exception as exc:error=error or 'Neutral report failed: '+repr(exc)
            del pad;pad=None
        if capture is not None:capture.stop()
        if worker is not None:worker.join(timeout=15)
        try:
            if guardian is not None:restore()
        except Exception as exc:error=(error or '')+' HidHide restore failed: '+repr(exc)
        if guardian is not None:
            try:guardian.wait(timeout=3)
            except subprocess.TimeoutExpired:error=(error or '')+' Recovery guardian still running'
        timer.stop()
        report=dict(status='failed' if error else 'stopped',error=error,controller_output=bool(sum(mode_counts.values())),
            virtual_device_created=virtual_created,
            virtual_readback_samples=len(readback_samples),virtual_readback_errors=readback_errors,
            virtual_readback_max_abs_rx_ry=[max(abs(row[i]) for row in readback_samples) for i in (4,5)] if readback_samples else None,
            mode_samples=mode_counts,transitions=transitions,
            requested_hz=1000,measured_mean_hz=float(1000/np.mean(intervals)) if intervals else None,
            period_p95_ms=float(np.quantile(intervals,.95)) if intervals else None,
            period_max_ms=max(intervals) if intervals else None,
            source_to_send_api_p95_ms=float(np.quantile(sources,.95)) if sources else None,
            worker_stopped=worker is None or not worker.is_alive(),
            note='Actual driver update API calls, not game reception/physical reaction. Human hit rate reported separately; no RL training.')
        (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('结果：',out)
        if error:raise RuntimeError(error)


if __name__ == '__main__':
    main()
