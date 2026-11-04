"""Render the 20s No Way short: frames -> ffmpeg (video only). Audio is made by audio.py."""
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import (W, H, FPS, NFRAMES, font, to_img, from_img, cam2d, shake, grade, lerp, smoothstep,
                    ease_in_out, ease_out, ease_out_back, hash2)
from street import Street, draw_sprites, draw_spray, draw_beam, EL_W, EL_Y
import tank
import paper
import aerial
import photo

# narration placement (seconds) — must match audio.py
NARR = {"l1": 0.15, "l2": 3.55, "l3": 9.35, "l4": 16.35}
AMBER = (255, 184, 64)
RED = (255, 98, 60)

# ---------------------------------------------------------------- captions
# (text, segment start/end within the clip as measured from the voice track)
CAPTIONS = [
    ("l1", [(0.0, 2.28, [["IN", "1919,"], ["BOSTON", "WAS", "HIT"], ["BY", "A", "DEADLY", "WAVE…"]]),
            (2.56, 3.14, [["OF", "MOLASSES."]])]),
    ("l2", [(0.0, 1.56, [["A", "MASSIVE", "STORAGE"], ["TANK", "BURST,"]]),
            (1.78, 4.28, [["SENDING", "MILLIONS"], ["OF", "GALLONS"], ["FLOODING", "THROUGH"], ["THE", "NEIGHBORHOOD."]])]),
    ("l3", [(0.0, 1.9, [["THE", "DISASTER"], ["KILLED", "21", "PEOPLE,"]]),
            (2.02, 3.45, [["AND", "INJURED", "150."]])]),
    ("l4", [(0.0, 0.62, [["AND", "YES…"]]), (0.8, 1.85, [["THIS", "ACTUALLY"], ["HAPPENED."]])]),
]
SPOKEN = {"1919,": "nineteen nineteen", "21": "twenty one", "150.": "one hundred fifty"}
KEYWORDS = {"MOLASSES.", "DEADLY", "BURST,", "MILLIONS", "21", "150.", "KILLED", "INJURED", "ACTUALLY", "WAVE…"}


def build_caption_chunks():
    chunks = []
    for line, segs in CAPTIONS:
        off = NARR[line]
        for s0, s1, groups in segs:
            words = [w for g in groups for w in g]
            wts = [len(SPOKEN.get(w, w)) + 2 for w in words]
            tot = sum(wts)
            t = off + s0
            times = []
            for wt in wts:
                d = (s1 - s0) * wt / tot
                times.append((t, t + d))
                t += d
            i = 0
            for g in groups:
                ws = []
                for w in g:
                    ws.append((w, times[i][0], times[i][1]))
                    i += 1
                chunks.append(ws)
    # chunk display windows
    out = []
    for k, ws in enumerate(chunks):
        start = ws[0][1] - 0.04
        end = ws[-1][2] + 0.35
        if k + 1 < len(chunks):
            end = min(end, chunks[k + 1][0][1] - 0.04)
        out.append((start, end, ws))
    return out


CHUNKS = build_caption_chunks()
F_CAP = None


def draw_text_stroke(d, xy, text, f, fill, stroke=9, shadow=True):
    d.text(xy, text, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))


def caption_layer(t):
    """Bold, word-synced captions in the lower-middle safe zone."""
    for start, end, ws in CHUNKS:
        if start <= t < end:
            break
    else:
        return None
    f = font("anton", 112)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    space = 26
    widths = [d.textlength(w, font=f) for w, _, _ in ws]
    total = sum(widths) + space * (len(ws) - 1)
    lines = [list(range(len(ws)))]
    if total > W - 140:
        mid = len(ws) // 2 + (len(ws) % 2)
        lines = [list(range(mid)), list(range(mid, len(ws)))]
    pop = ease_out_back((t - start) / 0.16, 2.2)
    sc = 0.7 + 0.3 * pop
    y0 = 1300 - (len(lines) - 1) * 70
    for li, idxs in enumerate(lines):
        lw = sum(widths[i] for i in idxs) + space * (len(idxs) - 1)
        x = (W - lw) / 2
        for i in idxs:
            w, ws0, ws1 = ws[i]
            active = ws0 <= t < ws1 + 0.05
            col = AMBER if (w in KEYWORDS or active) else (255, 255, 255)
            if active and w in KEYWORDS:
                col = (255, 214, 120)
            d.text((x, y0 + li * 140), w, font=f, fill=col + (255,), stroke_width=10, stroke_fill=(0, 0, 0, 255))
            x += widths[i] + space
    # soft shadow + pop scale
    sh = layer.split()[3].filter(ImageFilter.GaussianBlur(14))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(sh.point(lambda v: int(v * 0.7)))
    layer = Image.alpha_composite(shadow, layer)
    if abs(sc - 1) > 1e-3:
        layer = cam2d(layer, sc, W / 2, 1330, 0, 0, 1330 - H / 2, fill=(0, 0, 0, 0))
    return layer


def typewriter(d, text, x, y, t, t0, cps=26, size=46, fill=(240, 230, 210)):
    n = int(max(0, (t - t0) * cps))
    if n <= 0:
        return
    s = text[:n]
    f = font("type", size)
    full = d.textlength(text, font=f)
    d.rectangle([x - 18, y - 10, x + full + 18, y + size + 14], fill=(0, 0, 0, 150))
    d.text((x + 3, y + 3), s, font=f, fill=(0, 0, 0, 200))
    d.text((x, y), s, font=f, fill=fill + (255,))
    if n < len(text) and int(t * 6) % 2 == 0:
        cw = d.textlength(s, font=f)
        d.rectangle([x + cw + 4, y + 8, x + cw + 26, y + size], fill=fill + (255,))


def overlay_layer(t):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    used = False
    # ---- HOOK title
    if t < 3.05:
        used = True
        fade = 1 - smoothstep(2.82, 3.02, t)
        typewriter(d, "BOSTON  ·  JANUARY 15, 1919", 120, 250, t, 0.05, cps=40, size=48)
        for i, (txt, col, t0, y) in enumerate([("A DEADLY WAVE", (255, 255, 255), 0.08, 330), ("OF MOLASSES?", AMBER, 0.42, 480)]):
            if t < t0:
                continue
            k = ease_out((t - t0) / 0.18, 3)
            size = int(150 * (1.5 - 0.5 * k))
            f = font("anton", size)
            tw = d.textlength(txt, font=f)
            jx = (np.sin(t * 60 + i) * 6 * (1 - k))
            a = int(255 * min(1, k * 1.5) * fade)
            sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sub)
            sd.text(((W - tw) / 2 + jx, y - (size - 150) * 0.5), txt, font=f, fill=col + (a,), stroke_width=11, stroke_fill=(0, 0, 0, a))
            layer = Image.alpha_composite(layer, sub)
            d = ImageDraw.Draw(layer)
    # ---- documentary labels
    if 3.25 < t < 5.05:
        used = True
        typewriter(d, "PURITY DISTILLING CO. TANK", 90, 250, t, 3.3, cps=34, size=46)
        typewriter(d, "50 FT TALL · 2.3 MILLION GALLONS", 90, 320, t, 3.95, cps=34, size=46)
    if 6.15 < t < 8.8:
        used = True
        typewriter(d, "WAVE: UP TO 25 FT HIGH", 90, 250, t, 6.2, cps=30, size=44)
        typewriter(d, "SPEED: ~35 MPH", 90, 325, t, 7.05, cps=30, size=44)
    # ---- stats over the aerial
    if 13.85 < t < 16.25:
        used = True
        out = 1 - smoothstep(15.95, 16.2, t)
        rows = [(21, "KILLED", 13.9, RED), (150, "INJURED", 14.35, AMBER)]
        for i, (num, lab, t0, col) in enumerate(rows):
            if t < t0:
                continue
            k = ease_out((t - t0) / 0.35, 3)
            val = int(round(num * min(1, (t - t0) / 0.55)))
            a = int(255 * k * out)
            x = 110 - (1 - k) * 160
            y = 300 + i * 270
            fN = font("anton", 230)
            d.text((x, y), str(val), font=fN, fill=col + (a,), stroke_width=10, stroke_fill=(0, 0, 0, a))
            nw = d.textlength(str(num), font=fN)
            d.text((x + nw + 34, y + 96), lab, font=font("anton", 100), fill=(255, 255, 255, a), stroke_width=8, stroke_fill=(0, 0, 0, a))
        if t > 14.85:
            k = ease_out((t - 14.85) / 0.4)
            a = int(255 * k * out)
            typewriter(d, "STREETS KNEE-DEEP IN MOLASSES", 112, 870, t, 14.85, cps=36, size=48)
    if 16.6 < t < 18.3:
        used = True
        typewriter(d, "BOSTON, 1919", 110, 250, t, 16.7, cps=14, size=44, fill=(225, 225, 220))
    return layer if used else None


# ---------------------------------------------------------------- end card
def end_card(t, idx):
    a = t - 18.45
    img = Image.new("RGB", (W, H), (8, 7, 6))
    d = ImageDraw.Draw(img)
    f = font("anton", 210)
    txt = "NO WAY"
    tw = d.textlength(txt, font=f)
    scale = 1.0 + 0.04 * a
    title = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(title)
    td.text(((W - tw) / 2, 760), txt, font=f, fill=(245, 240, 230, 255))
    bw = 520 * ease_out(a / 0.6)
    td.rectangle([W / 2 - bw / 2, 1035, W / 2 + bw / 2, 1047], fill=AMBER + (255,))
    fs = font("oswaldm", 46)
    sub = "TRUE STORIES  ·  TOO STRANGE TO MAKE UP"
    sw = td.textlength(sub, font=fs)
    al = int(255 * smoothstep(0.35, 0.8, a))
    td.text(((W - sw) / 2, 1080), sub, font=fs, fill=(190, 182, 168, al))
    title = cam2d(title, scale, W / 2, 960, 0, 0, 0, fill=(0, 0, 0, 0))
    arr = np.asarray(title, np.float32) / 255.0
    base = np.asarray(img, np.float32) / 255.0
    # glitch intensity: strong at the start, then subtle periodic bursts
    rng = np.random.default_rng(idx * 13 + 1)
    g = max(0.0, 1 - a / 0.45) + (0.6 if (0.95 < a < 1.05 or 1.4 < a < 1.46) else 0.0) + (0.12 if rng.random() < 0.15 else 0)
    rgb = arr[..., :3] * arr[..., 3:4]
    al = arr[..., 3]
    if g > 0.02:
        off = int(18 * g)
        r = np.roll(rgb[..., 0], off, 1)
        b = np.roll(rgb[..., 2], -off, 1)
        rgb = np.stack([np.maximum(r, rgb[..., 0] * 0.4), rgb[..., 1], np.maximum(b, rgb[..., 2] * 0.4)], -1)
        al = np.maximum(al, np.maximum(np.roll(al, off, 1), np.roll(al, -off, 1)))
        # slice displacement
        for _ in range(int(3 + 9 * g)):
            y0 = rng.integers(600, 1200)
            hh = rng.integers(6, 40)
            sh = int(rng.normal(0, 70 * g))
            rgb[y0:y0 + hh] = np.roll(rgb[y0:y0 + hh], sh, 1)
            al[y0:y0 + hh] = np.roll(al[y0:y0 + hh], sh, 1)
    out = base * (1 - al[..., None]) + rgb
    # scanlines + faint noise
    out[::4] *= 0.85
    out += (rng.random((H, W, 1)).astype(np.float32) - 0.5) * (0.05 + 0.25 * g)
    fade_in = smoothstep(0.0, 0.08, a)
    return np.clip(out * fade_in, 0, 1)


# ---------------------------------------------------------------- scenes
HOOK_SPRITES = [dict(kind=0, X=-19, Z=48, h=5.8), dict(kind=1, X=-21, Z=62, h=5.6), dict(kind="cart", X=-6, Z=74, h=6.5),
                dict(kind=2, X=7, Z=40, h=5.8), dict(kind=0, X=20, Z=95, h=5.8), dict(kind=1, X=-20, Z=120, h=5.6)]


def scene_hook(t):
    wz = 125 * (1 - (t / 2.95) ** 1.35) + 2
    st = Street(f=1000, yh=1300, ch=4.5, camX=-4, camZ=-t * 4, time=t, wave_z=wz, wave_h=32)
    img = st.render()
    sp = [dict(s, Z=s["Z"] - t * 6 * (s["X"] < 0)) for s in HOOK_SPRITES]  # some flee
    img = draw_sprites(img, st, sp)
    img = draw_spray(img, st)
    amp = 3 + 22 * smoothstep(1.2, 2.9, t)
    sx, sy, sr = shake(t, amp, 11, seed=1)
    z = 1.04 + 0.05 * t / 3
    return from_img(cam2d(to_img(img), z, W / 2, H / 2, sr, sx, sy))


def scene_tank(t):
    p = ease_in_out((t - 3.0) / 2.1)
    z = lerp(0.9, 1.08, p)
    cyw = lerp(820, 1180, p)
    img = tank.render(t, zoom=z, cyw=cyw)
    amp = 2 + 5 * smoothstep(3.8, 5.0, t) + (40 * np.exp(-(t - 5.1) * 4) if t >= 5.1 else 0)
    sx, sy, sr = shake(t, amp, 14, seed=2)
    out = from_img(cam2d(to_img(img), 1.03, W / 2, H / 2, sr, sx, sy))
    if t >= 5.1:  # burst flash
        fl = np.exp(-(t - 5.1) * 9)
        out = out * (1 - fl) + np.array([1.0, 0.95, 0.85]) * fl
    return out


FLOOD_SPRITES = [dict(kind=2, X=-8, Z=60, h=5.8), dict(kind=0, X=4, Z=75, h=5.8), dict(kind=1, X=-18, Z=55, h=5.6)]
PLANKS = [dict(kind="plank", X=x, Z=z, h=6, ang=a) for x, z, a in [(-6, 70, 0.4), (3, 95, -0.3), (9, 120, 0.9), (-12, 140, 0.1), (0, 165, -0.6)]]


def scene_flood(t):
    a = t - 5.7
    front = 95 - a * 30 - a * a * 2.5
    camZ = -a * 5
    st = Street(f=900, yh=1080, ch=4.0, camX=1.5, camZ=camZ, time=t, flood_h=3.0, flood_front=front + 1,
                flood_speed=40, wave_z=front, wave_h=9.0, fog=330)
    img = st.render()
    sp = []
    for s in FLOOD_SPRITES:
        z = s["Z"] - a * 9
        if z < front - 0.5:
            sp.append(dict(s, Z=z))
    for s in PLANKS:
        z = s["Z"] - a * 40
        if z > front:
            sp.append(dict(s, Z=z, Y=2.6, ang=s["ang"] + a))
    img = draw_sprites(img, st, sp)
    img = draw_spray(img, st, n=90, seed=5)
    sx, sy, sr = shake(t, 10 + 14 * smoothstep(7.5, 8.9, t), 12, seed=3)
    out = from_img(cam2d(to_img(img), 1.05, W / 2, H / 2, sr, sx, sy))
    if a < 0.18:  # flash from the burst
        fl = 1 - a / 0.18
        out = out * (1 - fl) + np.array([1.0, 0.95, 0.85]) * fl
    return out


def damaged_street(t, camX=-2.0):
    st = Street(f=950, yh=1050, ch=6, camX=camX, camZ=(t - 11.8) * 2, time=t, flood_h=1.4, flood_speed=1.5,
                el_gap=(55, 90), collapsed=[(1, 2)], fog=260)
    img = st.render()
    img = draw_beam(img, st, (st.el_c - EL_W, EL_Y, 55, 6), (st.el_c - EL_W + 3, 0.5, 80), 1.3)
    img = draw_beam(img, st, (st.el_c + EL_W, EL_Y + 2, 90, 4), (st.el_c + 3, 0.5, 70), 1.3)
    sp = [dict(kind=2, X=-6, Z=30, h=5.8, Y=-1.4), dict(kind=0, X=-9, Z=36, h=5.8, Y=-1.4), dict(kind=2, X=2, Z=48, h=5.8, Y=-1.4),
          dict(kind="plank", X=-3, Z=22, h=7, ang=0.3, Y=1.2), dict(kind="plank", X=5, Z=40, h=8, ang=-0.5, Y=1.2),
          dict(kind="cart", X=-14, Z=65, h=6.5, Y=-1.0)]
    return draw_sprites(img, st, sp)


def scene_damage(t):
    p = (t - 11.75) / 2.2
    img = damaged_street(t, camX=lerp(-5, 2, p))
    sx, sy, sr = shake(t, 3, 4, seed=4)
    return from_img(cam2d(to_img(img), 1.04 + 0.05 * p, W / 2, H / 2, sr, sx, sy))


PAPERS = None
AERIAL = None
PHOTO = None


def init_assets():
    global PAPERS, AERIAL, PHOTO
    ph1 = to_img(damaged_street(12.0, -2))
    ph2 = to_img(scene_tank(3.2))
    PAPERS = paper.Papers(ph1, ph2)
    AERIAL = aerial.Aerial()
    PHOTO = photo.Photo()


def scene_paper(t):
    base = PAPERS.desk.copy()
    a = t - 8.95
    # page 1 spins in, then a slow push toward the headline
    k = ease_out(a / 0.45, 3)
    rot = lerp(-28, -3, k) + a * 0.6
    zoom = lerp(0.4, 0.66, k) + a * 0.03
    out = PAPERS.place(base, 0, zoom, 750, lerp(700, 560, ease_in_out(a / 2)), rot, lerp(-600, 0, k), lerp(900, 0, k), focus_y=H * 0.45)
    b = t - 10.55
    if b > 0:
        k2 = ease_out(b / 0.38, 3)
        rot2 = lerp(24, 4, k2) - b * 0.5
        zoom2 = lerp(0.45, 0.68, k2) + b * 0.04
        out = PAPERS.place(out, 1, zoom2, 750, 520 + b * 30, rot2, lerp(900, 0, k2), lerp(-500, 0, k2), focus_y=H * 0.42)
    sx, sy, sr = shake(t, 4 * np.exp(-max(0, a) * 3) + 6 * (np.exp(-b * 6) if b > 0 else 0), 10, seed=6)
    return from_img(cam2d(to_img(out), 1.02, W / 2, H / 2, sr, sx, sy))


def scene_aerial(t):
    p = (t - 13.6) / 2.6
    return AERIAL.render(p, t)


def scene_photo(t):
    p = ease_in_out((t - 15.9) / 2.6) * 0.85 + 0.15 * (t - 15.9) / 2.6
    g = PHOTO.render(p, t, W, H)
    # photo print border + edge wear, visible at the start, zooms out of frame
    return np.repeat(g[..., None], 3, 2)


def mix(a, b, k):
    return a * (1 - k) + b * k


def frame(idx):
    t = idx / FPS
    bw = 0.0
    keep = 0.32
    if t < 2.95:
        img = scene_hook(t)
        if t > 2.75:  # wave engulfs the lens
            img = img * (1 - smoothstep(2.75, 2.95, t))
    elif t < 5.7:
        img = scene_tank(t)
        img = img * smoothstep(2.95, 3.12, t)
    elif t < 8.95:
        img = scene_flood(t)
        if t > 8.7:
            img = img * (1 - smoothstep(8.7, 8.95, t))
    elif t < 11.75:
        img = scene_paper(t)
    elif t < 13.6:
        img = scene_damage(t)
        if t < 11.95:
            img = mix(scene_paper(t), img, smoothstep(11.75, 11.95, t))
    elif t < 15.9:
        img = scene_aerial(t)
        if t < 13.85:
            img = mix(scene_damage(t), img, smoothstep(13.6, 13.85, t))
    elif t < 18.45:
        img = scene_photo(t)
        k = smoothstep(15.9, 16.35, t)
        if t < 16.35:
            img = mix(scene_aerial(t), img, k)
        bw = k
        keep = 0.32 * (1 - k)
    else:
        img = None
    if img is not None:
        out = grade(img, idx, t, bw=bw, keep_color=keep, grain=0.06 + 0.03 * bw)
        if 18.25 < t < 18.45:  # glitch into the end card
            rng = np.random.default_rng(idx)
            gk = smoothstep(18.25, 18.45, t)
            for _ in range(int(4 + 10 * gk)):
                y0 = rng.integers(0, H - 60)
                hh = rng.integers(8, 60)
                out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(rng.normal(0, 120 * gk)), 1)
            out[..., 0] = np.roll(out[..., 0], int(20 * gk), 1)
            out = out * (1 - gk * 0.6)
    else:
        out = end_card(t, idx)
    pil = to_img(out).convert("RGBA")
    if t < 18.45:
        ov = overlay_layer(t)
        if ov is not None:
            pil = Image.alpha_composite(pil, ov)
        cap = caption_layer(t)
        if cap is not None:
            pil = Image.alpha_composite(pil, cap)
    return np.asarray(pil.convert("RGB")).tobytes()


def _init():
    init_assets()


if __name__ == "__main__":
    out = sys.argv[1]
    frames = range(NFRAMES)
    if len(sys.argv) > 2:  # preview specific frames as PNG
        _init()
        for fi in map(int, sys.argv[2:]):
            Image.frombytes("RGB", (W, H), frame(fi)).save(f"{out}_{fi:03d}.png")
        sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool(int(os.environ.get("JOBS", 4)), initializer=_init) as pool:
        for i, fb in enumerate(pool.imap(frame, frames, chunksize=4)):
            ff.stdin.write(fb)
            if i % 30 == 0:
                print("frame", i, flush=True)
    ff.stdin.close()
    ff.wait()
