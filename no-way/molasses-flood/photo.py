"""A 'found' black-and-white photograph of Boston's waterfront, built in parallax layers."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import NOISE, NOISE2, NOISE_FINE, tex

PW, PH = 1300, 2310
HZ = 1250  # waterline


def _sky():
    yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
    g = np.clip(yy / HZ, 0, 1)
    v = 0.86 - 0.42 * (1 - g) ** 1.5
    c = tex(NOISE2, xx * 0.3, yy * 0.6)
    v = v * (0.85 + 0.25 * c)
    sun = np.exp(-(((xx - 880) / 160) ** 2 + ((yy - 560) / 160) ** 2)) * 0.25
    v = v + sun
    return np.clip(v, 0, 1)


def _skyline(rng):
    lay = Image.new("LA", (PW, PH), (0, 0))
    d = ImageDraw.Draw(lay)
    base = HZ - 130
    # far skyline
    x = -20
    while x < PW:
        w = rng.integers(40, 110)
        h = rng.integers(60, 230)
        tone = 90 + rng.integers(0, 30)
        d.rectangle([x, base - h, x + w, base + 10], fill=(tone, 255))
        for wy in range(base - h + 12, base, 18):
            for wx in range(x + 6, x + w - 6, 14):
                if rng.random() < 0.5:
                    d.rectangle([wx, wy, wx + 5, wy + 8], fill=(tone - 30, 255))
        if rng.random() < 0.35:
            cx = x + w // 2
            d.rectangle([cx - 6, base - h - 40, cx + 6, base - h], fill=(tone - 10, 255))
        x += w
    # Custom House tower
    cx = 860
    d.rectangle([cx - 55, base - 620, cx + 55, base], fill=(78, 255))
    d.rectangle([cx - 70, base - 640, cx + 70, base - 610], fill=(70, 255))
    d.ellipse([cx - 32, base - 600, cx + 32, base - 536], fill=(150, 255))
    d.ellipse([cx - 26, base - 594, cx + 26, base - 542], fill=(110, 255))
    d.polygon([(cx - 60, base - 640), (cx + 60, base - 640), (cx, base - 760)], fill=(70, 255))
    for wy in range(base - 520, base - 40, 28):
        for wx in (cx - 38, cx - 14, cx + 10, cx + 30):
            d.rectangle([wx, wy, wx + 8, wy + 15], fill=(55, 255))
    # Old North Church steeple (North End)
    cx = 300
    d.rectangle([cx - 45, base - 260, cx + 45, base], fill=(150, 255))
    d.rectangle([cx - 32, base - 360, cx + 32, base - 260], fill=(160, 255))
    d.rectangle([cx - 24, base - 440, cx + 24, base - 360], fill=(170, 255))
    d.rectangle([cx - 16, base - 500, cx + 16, base - 440], fill=(175, 255))
    d.polygon([(cx - 14, base - 500), (cx + 14, base - 500), (cx, base - 660)], fill=(180, 255))
    d.line([(cx, base - 660), (cx, base - 700)], fill=(60, 255), width=3)
    for wy in (base - 330, base - 410, base - 480):
        d.rectangle([cx - 6, wy, cx + 6, wy + 22], fill=(80, 255))
    # chimney smoke
    sm = Image.new("LA", (PW, PH), (0, 0))
    sd = ImageDraw.Draw(sm)
    for sx in (520, 610, 1080):
        for k in range(30):
            r = 10 + k * 3.5
            sd.ellipse([sx + k * 9 - r, base - 260 - k * 16 - r, sx + k * 9 + r, base - 260 - k * 16 + r], fill=(200, max(0, 60 - k * 2)))
    sm = sm.filter(ImageFilter.GaussianBlur(12))
    return Image.alpha_composite(lay.convert("RGBA"), sm.convert("RGBA")).convert("LA")


def _wharf(rng):
    lay = Image.new("LA", (PW, PH), (0, 0))
    d = ImageDraw.Draw(lay)
    base = HZ + 30
    # warehouses on the wharf
    x = -10
    while x < PW:
        w = rng.integers(120, 220)
        h = rng.integers(90, 170)
        tone = 55 + rng.integers(0, 25)
        d.rectangle([x, base - h, x + w, base], fill=(tone, 255))
        d.polygon([(x - 5, base - h), (x + w + 5, base - h), (x + w // 2, base - h - 45)], fill=(tone - 15, 255))
        for wx in range(x + 12, x + w - 16, 26):
            d.rectangle([wx, base - h + 25, wx + 12, base - h + 48], fill=(25, 255))
            d.rectangle([wx, base - h + 70, wx + 12, base - h + 93], fill=(25, 255))
        x += w + rng.integers(0, 30)
    d.rectangle([0, base, PW, base + 30], fill=(40, 255))
    for px in range(0, PW, 22):
        d.rectangle([px, base + 30, px + 8, base + 85], fill=(30, 255))
    # sailing ships: masts + rigging
    for sx, mh in ((640, 640), (1130, 700)):
        hull_y = base + 70
        d.polygon([(sx - 170, hull_y), (sx + 190, hull_y), (sx + 150, hull_y + 70), (sx - 140, hull_y + 70)], fill=(22, 255))
        for k, off in enumerate((-90, 60)):
            mx = sx + off
            top = hull_y - mh + abs(off) * 0.6
            d.line([(mx, hull_y), (mx, top)], fill=(25, 255), width=7)
            for yard in range(3):
                yy = top + 70 + yard * 110
                L = 80 - yard * 6 + 20
                d.line([(mx - L, yy), (mx + L, yy)], fill=(30, 255), width=4)
                # furled sail
                d.line([(mx - L + 6, yy + 6), (mx + L - 6, yy + 6)], fill=(170, 255), width=6)
            d.line([(mx, top), (sx - 190, hull_y)], fill=(40, 200), width=2)
            d.line([(mx, top), (sx + 210, hull_y)], fill=(40, 200), width=2)
        d.line([(sx + 190, hull_y), (sx + 320, hull_y - 140)], fill=(25, 255), width=6)
    return lay


def build():
    rng = np.random.default_rng(31)
    sky = _sky()
    skyline = _skyline(rng)
    wharf = _wharf(rng)
    # water: reflection of everything above, rippled
    above = Image.fromarray((sky * 255).astype(np.uint8)).convert("L")
    above.paste(skyline.convert("L"), (0, 0), skyline.split()[1])
    above.paste(wharf.convert("L"), (0, 0), wharf.split()[1])
    a = np.asarray(above, np.float32) / 255.0
    yy, xx = np.mgrid[HZ + 120:PH, 0:PW].astype(np.float32)
    srcy = np.clip(2 * (HZ + 120) - yy, 0, PH - 1)
    rip = (tex(NOISE2, xx * 0.6, yy * 3.0) - 0.5) * 40 * ((yy - HZ) / 900 + 0.3)
    srcx = np.clip(xx + rip, 0, PW - 1)
    refl = a[srcy.astype(int), srcx.astype(int)] * 0.7 + 0.08
    refl += (tex(NOISE_FINE, xx * 1, yy * 8) - 0.5) * 0.12
    water = Image.new("LA", (PW, PH), (0, 0))
    w = np.zeros((PH, PW, 2), np.uint8)
    w[HZ + 120:, :, 0] = np.clip(refl * 255, 0, 255).astype(np.uint8)
    w[HZ + 120:, :, 1] = 255
    water = Image.fromarray(w, "LA").filter(ImageFilter.GaussianBlur(1.5))
    sky_img = Image.fromarray((sky * 255).astype(np.uint8)).convert("LA")
    return [sky_img, skyline.filter(ImageFilter.GaussianBlur(1.6)), water, wharf.filter(ImageFilter.GaussianBlur(0.9))]


def mist(time):
    yy, xx = np.mgrid[0:PH // 4, 0:PW // 4].astype(np.float32)
    m = tex(NOISE, xx * 0.6 + time * 6, yy * 2.5)
    band = np.exp(-((yy * 4 - (HZ - 40)) / 160) ** 2)
    v = np.clip((m - 0.4) * 1.6, 0, 1) * band * 0.55
    return Image.fromarray((v * 255).astype(np.uint8)).resize((PW, PH), Image.BILINEAR)


class Photo:
    def __init__(self):
        self.layers = build()
        self.par = [0.0, 0.35, 0.8, 1.0]

    def render(self, p, time, W, H):
        """Compose layers with parallax zoom; p = progress 0..1. Returns grayscale float array."""
        from common import cam2d
        canvas = None
        fx, fy = 330, 980  # zoom toward the Old North Church / North End
        for lay, k in zip(self.layers, self.par):
            z = (W / PW) * 1.04 * (1 + (0.3 + 0.12 * k) * p)
            cx = PW / 2 + (fx - PW / 2) * p * (0.45 + 0.1 * k)
            cy = PH / 2 + (fy - PH / 2) * p * (0.45 + 0.1 * k)
            im = cam2d(lay.convert("RGBA"), z, cx, cy, 0, 0, 0, out=(W, H), fill=(0, 0, 0, 0))
            canvas = im if canvas is None else Image.alpha_composite(canvas, im)
        z = (W / PW) * 1.04 * (1 + 0.38 * p)
        cx = PW / 2 + (fx - PW / 2) * p * 0.5
        cy = PH / 2 + (fy - PH / 2) * p * 0.5
        mi = cam2d(mist(time), z, cx, cy, 0, 0, 0, out=(W, H), fill=0)
        g = np.asarray(canvas.convert("L"), np.float32) / 255.0
        mm = np.asarray(mi, np.float32) / 255.0
        g = g * (1 - mm) + 0.85 * mm
        return g
