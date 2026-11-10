"""Render the Lamborghini Urus reel frame by frame.
usage: render.py out.mp4 audio.wav        (full render)
       render.py --preview t1 t2 ...       (writes prev_<t>.jpg)
"""
import cv2, numpy as np, sys, subprocess, math, os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SDR = os.path.join(HERE, 'sdr')
W, H, FPS, DUR = 1080, 1920, 30, 30.0
FONT = '/usr/share/fonts/opentype/inter/'
MONO = '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf'
YELLOW = (0, 225, 255)      # BGR
LIME = (59, 255, 182)

# ---------------- easing ----------------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lin(t, a, b): return clamp((t - a) / (b - a)) if b > a else float(t >= a)
def ease_out(x): return 1 - (1 - x) ** 3
def ease_in(x): return x ** 3
def ease_io(x): return 3 * x * x - 2 * x ** 3
def back(x, s=2.2):
    x -= 1
    return x * x * ((s + 1) * x + s) + 1
def mix(a, b, x): return a + (b - a) * x

# ---------------- source frames ----------------
_caps, _cache = {}, {}
def src(clip, t):
    idx = int(round(t * FPS))
    key = (clip, idx)
    if key in _cache: return _cache[key]
    cap = _caps.get(clip)
    if cap is None:
        cap = _caps[clip] = cv2.VideoCapture(f'{SDR}/IMG_{clip}.mp4')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, min(n - 1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok: f = np.zeros((H, W, 3), np.uint8)
    if len(_cache) > 64: _cache.clear()
    _cache[key] = f
    return f

def xform(img, zoom=1.0, c=(540, 960), to=None, rot=0.0, dx=0.0, dy=0.0):
    """Scale/rotate about point c in the source, landing it at `to` (+dx,dy)."""
    to = to or c
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), rot, zoom)
    M[0, 2] += to[0] - c[0] + dx
    M[1, 2] += to[1] - c[1] + dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), M

def mpt(M, p):
    return (M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2])

def shake(t, amt, seed=0):
    return (amt * (math.sin(t * 71 + seed) + 0.5 * math.sin(t * 133 + 2 * seed)),
            amt * (math.cos(t * 59 + seed) + 0.5 * math.sin(t * 97 + 3 * seed)))

# ---------------- grading ----------------
x = np.arange(256) / 255.0
LUT_BEFORE = np.clip((0.05 + 0.86 * x ** 1.04) * 255, 0, 255).astype(np.uint8)
s = np.clip((x - 0.025) / 0.975, 0, 1)
s = s + 0.22 * (s - 0.5) * (1 - np.abs(2 * s - 1)) * 2   # gentle S-curve
LUT_AFTER = np.clip(s * 255, 0, 255).astype(np.uint8)
yy, xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.32 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4)
VIG = np.clip(VIG, 0.55, 1).astype(np.float32)[..., None]

def saturate(img, k):
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, k, g, 1 - k, 0)

def grade_before(img, sat=0.74):
    img = cv2.LUT(img, LUT_BEFORE)
    img = saturate(img, sat)
    # slight warm/dingy cast
    return cv2.addWeighted(img, 0.94, np.full_like(img, (60, 80, 90)), 0.06, 0)

def grade_after(img):
    img = cv2.LUT(img, LUT_AFTER)
    img = saturate(img, 1.08)
    small = cv2.resize(img, (W // 4, H // 4))
    hi = cv2.threshold(small, 200, 255, cv2.THRESH_TOZERO)[1]
    bloom = cv2.resize(cv2.GaussianBlur(hi, (0, 0), 9), (W, H))
    img = cv2.add(img, (bloom * 0.22).astype(np.uint8))
    return (img.astype(np.float32) * VIG).astype(np.uint8)

# ---------------- effects ----------------
def flash(img, a, col=255):
    if a <= 0: return img
    return cv2.addWeighted(img, 1 - a, np.full_like(img, col), a, 0)

def mblur(img, k, vertical=False):
    k = int(k)
    if k < 3: return img
    return cv2.blur(img, (1, k) if vertical else (k, 1))

def chroma(img, px):
    px = int(px)
    if px == 0: return img
    b, g, r = cv2.split(img)
    b = np.roll(b, -px, 1); r = np.roll(r, px, 1)
    return cv2.merge([b, g, r])

def grad_top(img, h=1100, a=0.5):
    g = np.linspace(a, 0, h, dtype=np.float32)[:, None, None]
    img = img.copy()
    img[:h] = (img[:h].astype(np.float32) * (1 - g)).astype(np.uint8)
    return img

# ---------------- text ----------------
_tcache = {}
def text_img(txt, font, size, fill=(255, 255, 255), stroke=0, sfill=(0, 0, 0), track=0):
    key = (txt, font, size, fill, stroke, sfill, track)
    if key in _tcache: return _tcache[key]
    f = ImageFont.truetype(font, size)
    widths = [f.getlength(ch) for ch in txt]
    tw = int(sum(widths) + track * (len(txt) - 1)) + 2 * stroke + 20
    asc, desc = f.getmetrics()
    th = asc + desc + 2 * stroke + 20
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    xx_ = stroke + 10
    for ch, w_ in zip(txt, widths):
        d.text((xx_, stroke + 10), ch, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=sfill + (255,))
        xx_ += w_ + track
    a = np.array(im)
    a = a[..., [2, 1, 0, 3]]  # RGBA -> BGRA
    _tcache[key] = a
    return a

def overlay(img, rgba, cx, cy, scale=1.0, rot=0.0, alpha=1.0, blur=0):
    if alpha <= 0.01 or scale <= 0.01: return img
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    # bounding box to limit work
    pts = np.array([mpt(M, p) for p in [(0, 0), (w, 0), (0, h), (w, h)]])
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0: return img
    M[0, 2] -= x0; M[1, 2] -= y0
    layer = cv2.warpAffine(rgba, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    if blur: layer = mblur(layer, blur)
    a = layer[..., 3:4].astype(np.float32) / 255 * alpha
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + layer[..., :3].astype(np.float32) * a).astype(np.uint8)
    return img

def tracked_line(img, txt, font, size, cx, cy, track, t, t0, per=0.035, fill=(255, 255, 255)):
    """Letters fade/slide in one at a time, centered at cx."""
    f = ImageFont.truetype(font, size)
    widths = [f.getlength(ch) for ch in txt]
    total = sum(widths) + track * (len(txt) - 1)
    xx_ = cx - total / 2
    for i, (ch, w_) in enumerate(zip(txt, widths)):
        p = ease_out(lin(t, t0 + i * per, t0 + i * per + 0.35))
        if ch != ' ' and p > 0:
            g = text_img(ch, font, size, fill)
            overlay(img, g, xx_ + w_ / 2, cy + (1 - p) * 30, 1.0, 0, p, blur=int((1 - p) * 25))
        xx_ += w_ + track
    return img

# ---------------- graphic accents ----------------
def scribble_circle(img, c, r, p, seed=0, col=YELLOW, th=7):
    """Hand-drawn circle that draws itself (p: 0..1)."""
    if p <= 0: return img
    n = int(80 * p) + 2
    ang = np.linspace(0, 2 * np.pi * 1.12 * p, n) + seed
    rad = r * (1 + 0.07 * np.sin(ang * 3 + seed) + 0.04 * ang / (2 * np.pi))
    pts = np.stack([c[0] + rad * np.cos(ang), c[1] + rad * 0.92 * np.sin(ang)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 0), th + 6, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
    return img

def googly_eyes(img, cx, cy, r, t, scale=1.0, look=None):
    if scale <= 0.01: return img
    r = int(r * scale)
    for k, ox in enumerate((-1.08 * r, 1.08 * r)):
        ex, ey = int(cx + ox), int(cy)
        cv2.ellipse(img, (ex, ey), (int(r * 0.92), r), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.ellipse(img, (ex, ey), (int(r * 0.80), int(r * 0.88)), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
        if look is None:
            lx = math.sin(t * 9) * 0.45 + 0.15 * math.sin(t * 23 + k)
            ly = 0.25 * math.cos(t * 7 + k)
        else:
            lx, ly = look
        px, py = int(ex + lx * r * 0.42), int(ey + ly * r * 0.45)
        cv2.circle(img, (px, py), int(r * 0.40), (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, (px - int(r * 0.13), py - int(r * 0.15)), int(r * 0.11), (255, 255, 255), -1, cv2.LINE_AA)
    return img

_spark_pos = {}
def sparkle(img, key, t, t0, dur=0.45, region=None, size=70):
    p = lin(t, t0, t0 + dur)
    if p <= 0 or p >= 1: return img
    if key not in _spark_pos:
        x0, y0, x1, y1 = region or (100, 300, 980, 1600)
        g = cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (0, 0), 6)
        _, _, _, mx = cv2.minMaxLoc(g)
        _spark_pos[key] = (mx[0] + x0, mx[1] + y0)
    cx, cy = _spark_pos[key]
    s = math.sin(p * math.pi) * size
    rot = p * 60
    layer = np.zeros_like(img)
    for a, sc in ((rot, 1.0), (rot + 90, 1.0), (rot + 45, 0.45), (rot + 135, 0.45)):
        ra = math.radians(a)
        L, w_ = s * sc, max(2.0, s * 0.09)
        dx, dy = math.cos(ra), math.sin(ra)
        pts = np.array([(cx + dx * L, cy + dy * L), (cx - dy * w_, cy + dx * w_),
                        (cx - dx * L, cy - dy * L), (cx + dy * w_, cy - dx * w_)], np.int32)
        cv2.fillConvexPoly(layer, pts, (255, 255, 255), cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 10)
    img = cv2.add(img, glow)
    return cv2.add(img, layer)

def check_mark(img, cx, cy, r, p, col=LIME):
    if p <= 0: return img
    pc = ease_out(lin(p, 0, 0.5))
    cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360 * pc, col, 5, cv2.LINE_AA)
    pk = ease_out(lin(p, 0.35, 1))
    a = (cx - r * 0.45, cy + r * 0.02); b = (cx - r * 0.1, cy + r * 0.38); c = (cx + r * 0.5, cy - r * 0.35)
    if pk > 0:
        l1 = min(pk / 0.4, 1)
        p1 = (a[0] + (b[0] - a[0]) * l1, a[1] + (b[1] - a[1]) * l1)
        cv2.line(img, tuple(map(int, a)), tuple(map(int, p1)), col, 7, cv2.LINE_AA)
        if pk > 0.4:
            l2 = (pk - 0.4) / 0.6
            p2 = (b[0] + (c[0] - b[0]) * l2, b[1] + (c[1] - b[1]) * l2)
            cv2.line(img, tuple(map(int, b)), tuple(map(int, p2)), col, 7, cv2.LINE_AA)
    return img

def polaroid(frame_bg, card_src, t, t0):
    """Freeze frame that pops out into a tilted snapshot card."""
    p = lin(t, t0, t0 + 0.14)
    sc = mix(1.0, 0.74, back(p, 1.6)) if p < 1 else 0.74
    rot = mix(0, -5, ease_out(p))
    bg = cv2.GaussianBlur(cv2.resize(frame_bg, (W // 4, H // 4)), (0, 0), 6)
    bg = (cv2.resize(bg, (W, H)) * 0.45).astype(np.uint8)
    bw, bb = 26, 110
    card = cv2.copyMakeBorder(card_src, bw, bb, bw, bw, cv2.BORDER_CONSTANT, value=(245, 245, 242))
    rgba = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
    shadow = np.zeros_like(rgba); shadow[..., 3] = 150
    out = bg.copy()
    k = clamp(p * 3)
    out = cv2.addWeighted(out, k, frame_bg, 1 - k, 0)
    overlay(out, shadow, 540 + 18, 960 + 40 + 24, sc, rot, k * 0.8)
    overlay(out, rgba, 540, 960 + 42 * k, sc, rot)
    return out, sc, rot

# ---------------- timeline ----------------
BUGS = [(897, 1119)]
BADGE_HOOD = (517, 1612)
BOTTLE = (285, 1065)
LEAVES = [(765, 750), (560, 1060)]
BADGE_DAY = (900, 975)      # 2582 @ 2.6
BADGE_NIGHT = (660, 1260)   # 2592 @ 2.6

def split_panel(img, cy):
    y0 = int(clamp(cy - 478, 0, H - 956))
    return img[y0:y0 + 956]

def frame_at(t):
    before = t < 11.0
    acc = []   # deferred overlays: functions(img)->img applied after grading

    # ---------- HOOK ----------
    if t < 1.0:
        u = t
        if u < 0.55:
            st = 0.1 + u
        else:   # rewind during record scratch
            st = 0.65 - (u - 0.55) * 1.2 + 0.08 * math.sin((u - 0.55) * 40)
        f = src(2585, st)
        z = mix(1.0, 1.38, ease_out(lin(u, 0, 0.14)))
        rot = 4 * math.sin((u - 0.55) * 30) * (u > 0.55)
        dx, dy = shake(t, 6 * (1 - lin(u, 0, 0.3)) + 8 * (u > 0.55))
        img, _ = xform(f, z, BOTTLE, (420, 1180), rot, dx, dy)
        img = grade_before(img); before = 'done'
        img = flash(img, 0.7 * (1 - lin(u, 0, 0.1)))
        img = grad_top(img)
        def tx(img, u=u):
            p = lin(u, 0.04, 0.24)
            wob = 7 * math.sin((u - 0.55) * 28) if u > 0.55 else 0
            return overlay(img, text_img('WAIT…', FONT + 'InterDisplay-Black.otf', 220, (255, 255, 255), 14),
                           540, 560, back(p) if p < 1 else 1.0, -6 + wob)
        acc.append(tx)
    elif t < 2.5:
        u = t - 1.0
        f = src(2578, 1.4 + u)
        z = mix(1.35, 1.0, ease_out(lin(u, 0, 0.5)))
        dx, dy = shake(t, 10 * (1 - lin(u, 0, 0.25)))
        img, _ = xform(f, z, (540, 1100), None, 0, dx, dy)
        img = grade_before(img); before = 'done'
        img = flash(img, 0.55 * (1 - lin(u, 0, 0.12)))
        img = chroma(img, 10 * (1 - lin(u, 0, 0.1)))
        img = grad_top(img, 1150, 0.55)
    elif t < 3.0:
        u = t - 2.5
        f = src(2578, 2.9).copy()
        for i, b in enumerate(BUGS):
            scribble_circle(f, b, 70, ease_out(lin(u, 0.05 + 0.06 * i, 0.2 + 0.06 * i)), seed=i * 1.7)
        scribble_circle(f, BADGE_HOOD, 125, ease_out(lin(u, 0.2, 0.42)), seed=0.5, col=LIME, th=8)
        z = mix(1.0, 1.06, ease_out(lin(u, 0, 0.5)))
        img, _ = xform(f, z, (540, 1000), None, mix(0, 1.5, ease_out(lin(u, 0, 0.1))))
        img = grade_before(img); before = 'done'
        img = flash(img, 0.9 * (1 - lin(u, 0, 0.08)))
        img = grad_top(img, 1150, 0.55)
    # ---------- BEFORE ----------
    elif t < 3.5:
        u = t - 3.0
        img, _ = xform(src(2579, 0.2 + u * 1.2), mix(1.25, 1.12, ease_out(lin(u, 0, 0.3))), (400, 1000), (540, 1000))
    elif t < 4.0:
        u = t - 3.5
        f = src(2580, 1.8 + u)
        img, _ = xform(f, mix(1.0, 1.2, ease_out(lin(u, 0, 0.2))), (540, 1300), None, mix(-4, 0, ease_out(lin(u, 0, 0.2))))
        img = mblur(img, 60 * (1 - lin(u, 0, 0.1)), True)
    elif t < 4.5:
        u = t - 4.0
        img, _ = xform(src(2581, u), mix(1.0, 1.1, u * 2), (660, 860))
    elif t < 5.0:
        u = t - 4.5
        card = src(2581, 0.5).copy()
        card, _ = xform(card, 1.1, (660, 860))
        for i, l in enumerate(LEAVES):
            scribble_circle(card, l, 95, ease_out(lin(u, 0.15 + 0.08 * i, 0.32 + 0.08 * i)), seed=i + 2, th=9)
        card = grade_before(card)
        img, sc, rot = polaroid(card, card, t, 4.5)
        img = flash(img, 0.85 * (1 - lin(u, 0, 0.07)))
        before = 'done'
    elif t < 5.5:
        u = t - 5.0
        img, _ = xform(src(2582, 0.1 + u), 1.15, (500, 1050), None, 0, mix(700, 0, ease_out(lin(u, 0, 0.1))))
        img = mblur(img, 120 * (1 - lin(u, 0, 0.1)))
    elif t < 6.0:
        u = t - 5.5
        img, _ = xform(src(2583, 2.1 + u * 0.6), mix(1.05, 1.5, ease_io(lin(u, 0, 0.5))), (820, 1180), (640, 1060))
    elif t < 7.0:
        u = t - 6.0
        k = min(int(u / 0.25), 3)
        lu = u - k * 0.25
        zs = [1.0, 1.18, 1.38, 1.62][k]
        to = (mix(420, 540, k / 3), mix(1180, 960, k / 3))
        img, _ = xform(src(2585, 0.05 + lu), zs, BOTTLE, to, [-3, 3, -4, 5][k])
        if k == 3: img = chroma(img, 12 * (1 - lin(lu, 0, 0.12)))
        img = flash(img, 0.25 * (1 - lin(lu, 0, 0.06)))
    elif t < 7.5:
        u = t - 7.0
        img, _ = xform(src(2584, 0.8 + u * 1.6), 1.1, (540, 1100))
    elif t < 9.0:
        u = t - 7.5
        top = xform(src(2580, 2.6 + u * 0.8), 1.25, (560, 1250))[0]
        bot = xform(src(2586, 1.4 + u * 0.8), 1.1, (520, 1000))[0]
        top, bot = split_panel(top, 1150), split_panel(bot, 1000)
        img = np.zeros((H, W, 3), np.uint8)
        ox1 = int(mix(-W, 0, back(lin(u, 0, 0.18), 1.4)))
        ox2 = int(mix(W, 0, back(lin(u, 0.08, 0.26), 1.4)))
        def paste(dst, panel, y, ox):
            x0, x1 = max(0, ox), min(W, W + ox)
            if x1 > x0: dst[y:y + 956, x0:x1] = panel[:, x0 - ox:x1 - ox]
        bg = cv2.GaussianBlur(cv2.resize(src(2585, 1.0), (W // 8, H // 8)), (0, 0), 3)
        img[:] = (cv2.resize(bg, (W, H)) * 0.5).astype(np.uint8)
        paste(img, top, 0, ox1)
        paste(img, bot, 964, ox2)
        img[956:964] = 235
    elif t < 10.0:
        u = t - 9.0
        st = 0.5 + (2.0 * u if u < 0.45 else 0.9 + 0.35 * (u - 0.45))
        img, _ = xform(src(2587, st), mix(1.12, 1.04, u), (540, 1150), None, mix(2, -1, u))
        img = mblur(img, 50 * (1 - lin(u, 0, 0.4)))
    elif t < 11.0:
        u = t - 10.0
        st = 2.0 + u * 0.75
        z = mix(1.0, 2.0, ease_in(lin(u, 0, 1.0)))
        to = (mix(BADGE_DAY[0], 540, ease_io(lin(u, 0, 0.9))), mix(BADGE_DAY[1], 960, ease_io(lin(u, 0, 0.9))))
        dx, dy = shake(t, 9 * lin(u, 0.3, 0.9))
        whip = lin(u, 0.86, 1.0)
        img, _ = xform(src(2582, st), z, BADGE_DAY, to, 0, dx - 650 * ease_in(whip), dy)
        img = mblur(img, 260 * whip)
    # ---------- TRANSFORMATION ----------
    elif t < 12.0:
        u = t - 11.0
        st = 2.6 + u * 0.7
        z = mix(2.0, 1.15, ease_out(lin(u, 0, 0.8)))
        to = (540 + mix(650, 0, ease_out(lin(u, 0, 0.12))), mix(960, 1080, ease_io(lin(u, 0, 1))))
        img, _ = xform(src(2592, st), z, BADGE_NIGHT, to)
        img = mblur(img, 240 * (1 - lin(u, 0, 0.13)))
    elif t < 13.0:
        u = t - 12.0
        b = grade_before(xform(src(2583, 1.7 + u), 1.0)[0])
        a = grade_after(xform(src(2593, 1.3 + u), 1.05, (450, 645), (630, 600))[0])
        lx = int(mix(-10, W + 10, ease_io(lin(u, 0.08, 0.8))))
        img = b.copy()
        if lx > 0: img[:, :min(lx, W)] = a[:, :min(lx, W)]
        line = np.zeros_like(img)
        cv2.line(line, (lx, 0), (lx, H), (255, 255, 255), 8)
        img = cv2.add(img, cv2.GaussianBlur(line, (0, 0), 14))
        img = cv2.add(img, line)
        before = 'done'
    # ---------- AFTER ----------
    elif t < 15.0:
        u = t - 13.0
        st = 0.5 + (1.5 * u if u < 0.6 else 0.9 + 0.55 * (u - 0.6))
        img, _ = xform(src(2591, st), mix(1.0, 1.08, ease_io(u / 2)), (560, 1000))
        img = mblur(img, 70 * (1 - lin(u, 0, 0.12)), True)
    elif t < 17.0:
        u = t - 15.0
        img, _ = xform(src(2589, 0.2 + 0.8 * u), mix(1.08, 1.0, ease_io(u / 2)), (540, 1000))
    elif t < 18.5:
        u = t - 17.0
        img, _ = xform(src(2593, 2.2 + 0.85 * u), mix(1.0, 1.07, ease_io(u / 1.5)), (600, 900))
    elif t < 19.5:
        u = t - 18.5
        img, _ = xform(src(2592, 2.9 + 0.9 * u), 1.05, (560, 1000), None, mix(-2, 0, ease_out(u)))
    elif t < 20.0:
        u = t - 19.5
        img, _ = xform(src(2592, 2.55 + 0.3 * u), mix(1.0, 1.4, ease_out(lin(u, 0, 0.5))), BADGE_NIGHT, (540, 1050))
        img = flash(img, 0.35 * (1 - lin(u, 0, 0.08)))
    elif t < 22.5:
        u = t - 20.0
        st = 2.4 + 0.65 * u
        img, _ = xform(src(2591, st), mix(1.0, 1.07, ease_io(u / 2.5)), (600, 1100))
    # ---------- FINALE ----------
    else:
        u = t - 22.5
        st = min(0.0 + u * 0.765, 5.75)
        img, _ = xform(src(2594, st), mix(1.08, 1.0, ease_out(lin(u, 0, 4))), (540, 1150), None, 0,
                       mix(-600, 0, ease_out(lin(u, 0, 0.13))))
        img = mblur(img, 220 * (1 - lin(u, 0, 0.14)))

    # ---------- grade ----------
    if before is True:
        img = grade_before(img, sat=mix(0.74, 0.45, lin(t, 10.0, 10.9)) if t >= 10 else 0.74)
    elif before is False:
        img = grade_after(img)
    for fn in acc: img = fn(img)

    # ---------- post accents ----------
    if 11.0 <= t < 11.4:
        img = flash(img, 0.95 * (1 - lin(t, 11.0, 11.25)))
        img = chroma(img, 14 * (1 - lin(t, 11.0, 11.12)))
    if 22.5 <= t < 22.7:
        img = flash(img, 0.4 * (1 - lin(t, 22.5, 22.65)))
    img = sparkle(img, 'a', t, 11.25, 0.5, (300, 700, 800, 1300), 80)
    img = sparkle(img, 'b', t, 13.7, 0.45)
    img = sparkle(img, 'c', t, 15.9, 0.45)
    img = sparkle(img, 'd', t, 17.6, 0.45)
    img = sparkle(img, 'e', t, 19.55, 0.45, (400, 850, 700, 1250), 95)
    img = sparkle(img, 'f', t, 21.0, 0.45)
    img = sparkle(img, 'g', t, 24.2, 0.5, (100, 1100, 1000, 1500), 85)

    # hook typography (1.0 .. 3.0)
    if 1.0 <= t < 3.0:
        out = ease_in(lin(t, 2.9, 3.0))
        sW = mix(1.0, 0.55, ease_out(lin(t, 1.0, 1.15)))
        overlay(img, text_img('WAIT…', FONT + 'InterDisplay-Black.otf', 220, (255, 255, 255), 14),
                540, mix(560, 300, ease_out(lin(t, 1.0, 1.15))), sW * (1 - out), -6)
        p1 = lin(t, 1.02, 1.2)
        overlay(img, text_img('THIS IS A', FONT + 'InterDisplay-Black.otf', 110, (255, 255, 255), 10),
                540, 520, (back(p1) if p1 < 1 else 1) * (1 - out), 3)
        p2 = lin(t, 1.5, 1.7)
        bump = 1 + 0.06 * math.sin(max(0, t - 2.5) * 30) * (t > 2.5) * (1 - lin(t, 2.5, 2.8))
        overlay(img, text_img('LAMBO?', FONT + 'InterDisplay-Black.otf', 205, (255, 225, 0), 16),
                540, 720, (back(p2) if p2 < 1 else 1) * bump * (1 - out), -3)
        pe = lin(t, 1.75, 1.95)
        look = (0.0, 0.9) if t >= 2.5 else None
        googly_eyes(img, 540, 955, 80, t, (back(pe) if pe < 1 else 1) * (1 - out), look)

    # BEFORE tag (camcorder style)
    if 3.0 <= t < 10.9 and not (4.5 <= t < 5.0):
        if int(t * 2) % 2 == 0: cv2.circle(img, (78, 128), 13, (40, 40, 235), -1, cv2.LINE_AA)
        overlay(img, text_img('BEFORE', MONO, 46, (255, 255, 255), 3), 210, 130, 1.0)
    # AFTER tag
    if 11.05 <= t < 13.6:
        a = lin(t, 11.05, 11.3) * (1 - lin(t, 13.2, 13.6))
        overlay(img, text_img('AFTER', FONT + 'InterDisplay-Light.otf', 52, (255, 255, 255), 0, track=14), 190, 130, 1.0, 0, a)

    # finale typography
    if t >= 25.0:
        fade = 1 - lin(t, 29.35, 29.95)
        if t >= 29.0:
            img = np.zeros_like(img)   # beat-synced cut to black, type stays
        tl = grad_top(img, 950, 0.65 * ease_out(lin(t, 25.0, 25.6)))
        tr = mix(34, 16, ease_out(lin(t, 25.0, 26.2)))
        tracked_line(tl, 'LAMBORGHINI', FONT + 'InterDisplay-SemiBold.otf', 92, 540, 250, tr, t, 25.0, 0.03)
        tracked_line(tl, 'URUS', FONT + 'InterDisplay-Light.otf', 92, 540, 360, tr + 26, t, 25.35, 0.06)
        lw = int(260 * ease_out(lin(t, 25.8, 26.3)))
        if lw > 0: cv2.line(tl, (540 - lw, 445), (540 + lw, 445), (235, 235, 235), 2, cv2.LINE_AA)
        p3 = lin(t, 26.2, 26.6)
        overlay(tl, text_img('DETAIL COMPLETE', FONT + 'InterDisplay-Medium.otf', 46, (255, 255, 255), 0, track=9),
                500, 510 + (1 - ease_out(p3)) * 20, 1.0, 0, p3)
        check_mark(tl, 848, 510, 28, lin(t, 27.0, 27.3))
        img = cv2.addWeighted(tl, fade, img, 1 - fade, 0)
        if t >= 29.0: img = (img * fade).astype(np.uint8)
    return img


def main():
    if sys.argv[1] == '--preview':
        for ts in sys.argv[2:]:
            cv2.imwrite(os.path.join(HERE, f'prev_{ts}.jpg'), cv2.resize(frame_at(float(ts)), (540, 960)))
        return
    out, audio = sys.argv[1], sys.argv[2]
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-i', audio, '-map', '0:v', '-map', '1:a',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        p.stdin.write(frame_at(i / FPS).tobytes())
        if i % 90 == 0: print(f'{i}/{n}', flush=True)
    p.stdin.close(); p.wait()
    print('done', p.returncode)


if __name__ == '__main__':
    main()
