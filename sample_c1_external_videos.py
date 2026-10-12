import argparse
import bisect
import hashlib
import json
import math
import subprocess
import pathlib
from PIL import Image, ImageDraw, ImageFont

BIN = pathlib.Path('C:/Users/15207/AppData/Local/Microsoft/WinGet/Links')
FFMPEG = BIN / 'ffmpeg.EXE'
FFPROBE = BIN / 'ffprobe.EXE'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def probe_frames(video):
    cmd = [str(FFPROBE), '-v', 'error', '-select_streams', 'v:0',
           '-show_frames', '-show_entries',
           'frame=best_effort_timestamp_time,width,height', '-of', 'json',
           str(video)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError('ffprobe failed: ' + res.stderr.strip())
    data = json.loads(res.stdout)
    pts, dims = [], []
    for fr in data.get('frames', []):
        t = fr.get('best_effort_timestamp_time')
        if t is None:
            raise RuntimeError('missing timestamp in ' + str(video))
        t = float(t)
        if not math.isfinite(t):
            raise RuntimeError('nonfinite timestamp in ' + str(video))
        if pts and t < pts[-1]:
            raise RuntimeError('decreasing timestamps in ' + str(video))
        pts.append(t)
        dims.append((int(fr['width']), int(fr['height'])))
    if not pts:
        raise RuntimeError('no frames in ' + str(video))
    return pts, dims


def select_indices(pts, interval):
    if not pts or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Nonempty PTS and positive finite interval required')
    first, last = pts[0], pts[-1]
    span = last - first
    out = []
    k = 0
    while True:
        desired = 0.5 + interval * k
        if desired > span:
            break
        idx = bisect.bisect_left(pts, first + desired)
        if idx >= len(pts):
            break
        if not out or idx != out[-1]:
            out.append(idx)
        k += 1
    return out or [0]


def extract(video, indices, outdir):
    (outdir / 'frames').mkdir(parents=True, exist_ok=True)
    sel = '+'.join('eq(n\\,%d)' % i for i in indices)
    cmd = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-threads', '2',
           '-i', str(video), '-vf', 'select=' + sel, '-fps_mode', 'vfr',
           '-frames:v', str(len(indices)), '-threads', '2', '-n',
           str(outdir / 'frames' / '%06d.png')]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError('ffmpeg failed: ' + res.stderr.strip())


def font():
    try:
        return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    except Exception:
        return ImageFont.load_default()


def contact(records, outpath):
    tw, th = 480, 310
    iw, ih = 480, 270
    cols, rows = 4, 4
    sheet = Image.new('RGB', (tw * cols, th * rows), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    fnt = font()
    for n, rec in enumerate(records):
        r, c = divmod(n, cols)
        if r >= rows:
            break
        im = Image.open(rec['image_path'])
        im = im.convert('RGB')
        im.thumbnail((iw, ih))
        x = c * tw + (iw - im.width) // 2
        y = r * th
        sheet.paste(im, (x, y))
        draw.text((c * tw + 4, y + ih + 4),
                  '%s  %.3fs' % (rec['id'], rec['elapsed_s']), fill=(255, 255, 255), font=fnt)
    sheet.save(outpath, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', action='append', required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--interval', type=float, default=1.0)
    args = ap.parse_args()
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise SystemExit('ffmpeg/ffprobe not found at fixed path')
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error('--interval must be positive and finite')
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('output already exists: ' + str(out))
    out.mkdir(parents=True)
    sources, all_frames = [], []
    for vi, vpath in enumerate(args.video):
        clip_id = 'C%02d' % (vi + 1)
        video = pathlib.Path(vpath).resolve()
        print('[%s] probing %s' % (clip_id, video), flush=True)
        pts, dims = probe_frames(video)
        idxs = select_indices(pts, args.interval)
        clipdir = out / ('clip_' + clip_id)
        extract(video, idxs, clipdir)
        frames = sorted((clipdir / 'frames').glob('*.png'))
        if len(frames) != len(idxs):
            raise RuntimeError('frame count mismatch %d != %d' % (len(frames), len(idxs)))
        records = []
        for n, (idx, fp) in enumerate(zip(idxs, frames)):
            with Image.open(fp) as im:
                w, h = im.size
            if (w, h) != dims[idx]:
                raise RuntimeError('dimension mismatch at %s' % fp.name)
            media_pts = pts[idx]
            rid = '%s_%06d' % (clip_id, n + 1)
            records.append({
                'id': rid, 'clip_id': clip_id, 'sequence': n + 1,
                'decode_index': idx, 'media_pts_s': media_pts,
                'elapsed_s': media_pts - pts[0],
                'image_path': str(fp.resolve()), 'width': w, 'height': h,
                'image_sha256': sha(fp)})
        sources.append({
            'clip_id': clip_id, 'source_path': str(video),
            'source_sha256': sha(video), 'first_pts_s': pts[0],
            'last_pts_s': pts[-1], 'decoded_frames': len(pts),
            'sampled_frames': len(records)})
        all_frames.extend(records)
        sheets = [records[i:i + 16] for i in range(0, len(records), 16)]
        for si, chunk in enumerate(sheets):
            contact(chunk, clipdir / ('contact_%02d.jpg' % (si + 1)))
        print('[%s] %d/%d sampled' % (clip_id, len(records), len(pts)), flush=True)
    manifest = {
        'stage': 'Stage 1 — Local Combat Policy',
        'purpose': 'Unlabelled diagnostic sampling; not train/val',
        'interval_s': args.interval, 'sources': sources, 'frames': all_frames,
        'training_started': False, 'action_labels_available': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('wrote manifest with %d frames' % len(all_frames), flush=True)


if __name__ == '__main__':
    main()
