"""Shared helpers: procedural noise, texture sampling, fonts, film grade."""
import os
import numpy as np
from PIL import Image, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DUR = 20.0
NFRAMES = int(FPS * DUR)

FONT_DIR = os.environ.get("OT_FONTS", os.path.join(os.path.dirname(__file__), "assets", "fonts"))


def font(name, size):
    files = {
        "anton": "anton-latin-400-normal.woff",
        "black": "unifrakturmaguntia-latin-400-normal.woff",
        "old": "old-standard-tt-latin-400-normal.woff",
        "oldb": "old-standard-tt-latin-700-normal.woff",
        "oldi": "old-standard-tt-latin-400-italic.woff",
        "type": "special-elite-latin-400-normal.woff",
        "play": "playfair-display-latin-900-normal.woff",
        "playi": "playfair-display-latin-700-italic.woff",
        "oswald": "oswald-latin-700-normal.woff",
        "oswaldm": "oswald-latin-500-normal.woff",
    }
    return ImageFont.truetype(os.path.join(FONT_DIR, files[name]), size)


# ---------------------------------------------------------------- noise
def fractal_noise(n, octaves=6, seed=0, base=4, persistence=0.55):
    """Tileable fractal value noise in [0,1], shape (n, n)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        g = base * (2 ** o)
        if g > n:
            break
        grid = rng.random((g, g)).astype(np.float32)
        # tile-safe upscale: pad by wrapping, resize, crop
        pad = np.pad(grid, 1, mode="wrap")
        img = Image.fromarray(pad).resize(((g + 2) * n // g, (g + 2) * n // g), Image.BICUBIC)
        a = np.asarray(img)[n // g: n // g + n, n // g: n // g + n]
        out += a * amp
        tot += amp
        amp *= persistence
    out /= tot
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out


NOISE = fractal_noise(512, seed=1)
NOISE2 = fractal_noise(512, seed=7, base=8)
NOISE_FINE = fractal_noise(512, seed=3, base=32, octaves=4)


def tex(t, u, v):
    """Bilinear sample of tileable 2D texture t at float texel coords u (x), v (y)."""
    n0, n1 = t.shape
    u0 = np.floor(u)
    v0 = np.floor(v)
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)
    iu = u0.astype(np.int64) % n1
    iv = v0.astype(np.int64) % n0
    iu1 = (iu + 1) % n1
    iv1 = (iv + 1) % n0
    a = t[iv, iu] * (1 - fu) + t[iv, iu1] * fu
    b = t[iv1, iu] * (1 - fu) + t[iv1, iu1] * fu
    return a * (1 - fv) + b * fv


def hash2(a, b, seed=0):
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    h = (a * 374761393 + b * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float32) / 65535.0


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ease_in_out(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** p


def ease_out_back(x, s=1.7):
    x = np.clip(x, 0, 1) - 1
    return x * x * ((s + 1) * x + s) + 1


def lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------------------- 2D camera
def to_img(arr):
    return Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8))


def from_img(img):
    return np.asarray(img.convert("RGB"), np.float32) / 255.0


def cam2d(img, zoom=1.0, cx=None, cy=None, rot=0.0, dx=0.0, dy=0.0, out=(W, H), resample=Image.BICUBIC, fill=(0, 0, 0)):
    """View into `img` (PIL): centered at (cx,cy) in source px, zoom (output px per source px),
    rotation in degrees, then screen offset dx,dy."""
    sw, sh = img.size
    cx = sw / 2 if cx is None else cx
    cy = sh / 2 if cy is None else cy
    ow, oh = out
    a = np.deg2rad(rot)
    c, s = np.cos(a) / zoom, np.sin(a) / zoom
    # output (x,y) -> source
    ox, oy = ow / 2 + dx, oh / 2 + dy
    coeffs = (c, -s, cx - c * ox + s * oy, s, c, cy - s * ox - c * oy)
    return img.transform(out, Image.AFFINE, coeffs, resample=resample, fillcolor=fill)


def shake(t, amp, freq=9.0, seed=0):
    """Smooth pseudo-random handheld/impact shake offsets."""
    k = np.arange(1, 4)
    ph = hash2(k, seed) * 6.28
    ph2 = hash2(k, seed + 5) * 6.28
    sx = np.sum(np.sin(t * freq * k * 1.37 + ph) / k)
    sy = np.sum(np.sin(t * freq * k * 1.11 + ph2) / k)
    sr = np.sum(np.sin(t * freq * k * 0.83 + ph * 2) / k)
    return sx * amp, sy * amp, sr * amp * 0.03


# ------------------------------------------------------------- film look
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H / 2) / (H * 0.58)) ** 2)
VIGNETTE = (1 - smoothstep(0.55, 1.25, _r) * 0.78).astype(np.float32)[..., None]

_grain_rng = np.random.default_rng(11)
GRAIN = []
for i in range(12):
    g = _grain_rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    g = np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR))
    GRAIN.append(g)

SEPIA_DARK = np.array([0.045, 0.03, 0.02], np.float32)
SEPIA_MID = np.array([0.62, 0.5, 0.38], np.float32)
SEPIA_HI = np.array([1.0, 0.95, 0.85], np.float32)
BW_DARK = np.array([0.03, 0.03, 0.035], np.float32)
BW_MID = np.array([0.5, 0.5, 0.5], np.float32)
BW_HI = np.array([0.97, 0.97, 0.95], np.float32)


def tone(lum, dark, mid, hi):
    lum = lum[..., None]
    lo = np.clip(lum * 2, 0, 1)
    up = np.clip(lum * 2 - 1, 0, 1)
    return np.where(lum < 0.5, dark + (mid - dark) * lo, mid + (hi - mid) * up)


def grade(img, frame_idx, t, bw=0.0, keep_color=0.32, contrast=1.12, grain=0.06, flicker=0.035, dust=True):
    """Vintage film grade: duotone toning that keeps a hint of the molasses amber,
    contrast curve, vignette, flicker, grain, dust and scratches."""
    lum = img[..., 0] * 0.3 + img[..., 1] * 0.59 + img[..., 2] * 0.11
    lum = np.clip((lum - 0.5) * contrast + 0.5, 0, 1)
    sep = tone(lum, SEPIA_DARK, SEPIA_MID, SEPIA_HI)
    if bw > 0:
        sep = sep * (1 - bw) + tone(lum, BW_DARK, BW_MID, BW_HI) * bw
    kc = keep_color * (1 - bw)
    col = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
    out = sep * (1 - kc) + col * kc
    rng = np.random.default_rng(frame_idx * 7 + 3)
    fl = 1 + (rng.random() - 0.5) * 2 * flicker
    out = out * VIGNETTE * fl
    g = GRAIN[rng.integers(len(GRAIN))]
    # grain strongest in mid-tones like real film
    out = out + g[..., None] * grain * (0.35 + lum[..., None] * (1 - lum[..., None]) * 2.6)
    if dust:
        out = add_dust(out, rng)
    return np.clip(out, 0, 1)


def add_dust(out, rng):
    n = rng.integers(2, 9)
    for _ in range(n):
        x, y = rng.integers(0, W), rng.integers(0, H)
        r = rng.integers(1, 5)
        val = 0.0 if rng.random() < 0.7 else 0.9
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r), max(0, x - r), min(W, x + r)
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * 0.3 + val * 0.7
    if rng.random() < 0.18:  # hairline
        x, y = rng.integers(0, W), rng.integers(0, H)
        L = rng.integers(20, 70)
        ys = np.clip(y + np.arange(L), 0, H - 1)
        xs = np.clip(x + (np.sin(np.arange(L) * 0.15) * 6).astype(int), 0, W - 1)
        out[ys, xs] = out[ys, xs] * 0.3
    if rng.random() < 0.35:  # vertical scratch
        x = rng.integers(40, W - 40)
        out[:, x:x + 2] = out[:, x:x + 2] * 0.75 + 0.2
    return out


def blur(arr, r):
    return from_img(to_img(arr).filter(ImageFilter.GaussianBlur(r)))
