"""Period newspaper front pages (fictional mastheads, factual headlines) on a dark desk."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from common import W, H, NOISE, NOISE2, NOISE_FINE, tex, font, fractal_noise

PW, PH = 1500, 2150
INK = (28, 24, 20)

BODY = ("Without warning, shortly after noon yesterday, a great steel tank holding more than two million "
        "gallons of molasses burst asunder on Commercial Street, and a wave of the dark syrup, said by "
        "witnesses to have stood fully twenty-five feet high, swept through the North End at a speed no man "
        "could outrun. Buildings were torn from their foundations, the elevated railway structure was "
        "wrecked, and wagons, horses and men were caught fast in the flood. A fire house was crushed and "
        "carried toward the harbor. Rescuers waded waist deep in the sticky mass, which hardened in the "
        "cold of the afternoon and held victims as in a trap. Doctors and nurses of the Red Cross worked "
        "through the night, and sailors from the training ship nearby were among the first to aid the "
        "injured. Residents describe a roar like thunder and a sound like the firing of machine guns as "
        "the rivets gave way. ")


def _justify(d, text, x, y, w, h, f, lh, fill=INK):
    words = text.split()
    i = 0
    yy = y
    while yy < y + h - lh:
        line = []
        while i < len(words) * 50:
            wd = words[i % len(words)]
            test = " ".join(line + [wd])
            if d.textlength(test, font=f) > w and line:
                break
            line.append(wd)
            i += 1
        tw = sum(d.textlength(wd, font=f) for wd in line)
        gap = (w - tw) / max(1, len(line) - 1)
        xx = x
        for wd in line:
            d.text((xx, yy), wd, font=f, fill=fill)
            xx += d.textlength(wd, font=f) + gap
        yy += lh


def _center(d, text, y, f, fill=INK, x0=0, x1=PW):
    w = d.textlength(text, font=f)
    d.text(((x0 + x1 - w) / 2, y), text, font=f, fill=fill)
    bb = d.textbbox((0, 0), text, font=f)
    return y + bb[3]


def _fit(d, text, name, maxw, start):
    s = start
    while s > 20:
        f = font(name, s)
        if d.textlength(text, font=f) <= maxw:
            return f
        s -= 4
    return font(name, 20)


def halftone(photo, w, h, cell=7):
    g = ImageOps.grayscale(photo).resize((w, h))
    g = ImageOps.autocontrast(g, cutoff=1)
    a = np.asarray(g, np.float32) / 255.0
    out = Image.new("L", (w, h), 235)
    d = ImageDraw.Draw(out)
    for yy in range(0, h, cell):
        for xx in range(0, w, cell):
            dark = 1 - a[min(h - 1, yy + cell // 2), min(w - 1, xx + cell // 2)]
            r = cell * 0.62 * np.sqrt(dark)
            if r > 0.4:
                cx, cy = xx + cell / 2, yy + cell / 2
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=30)
    return out.convert("RGB")


def age_paper(img, seed):
    a = np.asarray(img, np.float32) / 255.0
    hh, ww = a.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    n = tex(NOISE, xx * 0.35 + seed * 50, yy * 0.35)
    n2 = tex(NOISE2, xx * 0.8, yy * 0.8 + seed * 90)
    paper = np.array([0.86, 0.8, 0.66], np.float32)
    stain = np.clip((n - 0.62) * 3, 0, 1)[..., None] * np.array([0.18, 0.22, 0.3], np.float32)
    edge = np.minimum(np.minimum(xx, ww - xx), np.minimum(yy, hh - yy))
    edgek = np.clip(1 - edge / 120, 0, 1)[..., None] * np.array([0.25, 0.3, 0.38], np.float32)
    fold = np.exp(-((yy - hh * 0.5) / 5) ** 2)[..., None] * 0.18 + np.exp(-((xx - ww * 0.5) / 4) ** 2)[..., None] * 0.1
    ink_var = 0.85 + 0.3 * n2[..., None]
    out = paper * (a ** 1.0) * (1 - stain - edgek) - fold
    dark = 1 - a
    out = out * (1 - dark * (ink_var - 1) * 0.5)
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def build_page(kind, photo):
    img = Image.new("RGB", (PW, PH), (255, 255, 255))
    d = ImageDraw.Draw(img)
    m = 70
    if kind == 0:
        d.text((m, 30), "EXTRA", font=font("oswald", 46), fill=INK)
        wtxt = "PRICE TWO CENTS"
        d.text((PW - m - d.textlength(wtxt, font=font("oswald", 34)), 40), wtxt, font=font("oswald", 34), fill=INK)
        y = _center(d, "The Evening Courier", 100, font("black", 160)) + 30
        d.line([(m, y), (PW - m, y)], fill=INK, width=4)
        y = _center(d, "BOSTON, WEDNESDAY, JANUARY 15, 1919", y + 10, font("oldb", 36)) + 14
        d.line([(m, y), (PW - m, y)], fill=INK, width=4)
        d.line([(m, y + 8), (PW - m, y + 8)], fill=INK, width=2)
        y += 30
        y = _center(d, "MOLASSES TANK BURSTS;", y, _fit(d, "MOLASSES TANK BURSTS;", "play", PW - 2 * m, 170)) + 20
        y = _center(d, "WAVE SWEEPS NORTH END", y, _fit(d, "WAVE SWEEPS NORTH END", "play", PW - 2 * m, 170)) + 26
        d.line([(m, y), (PW - m, y)], fill=INK, width=3)
        y = _center(d, "2,300,000 Gallons Loosed on Commercial Street", y + 14, _fit(d, "2,300,000 Gallons Loosed on Commercial Street", "playi", PW - 2 * m, 70)) + 14
        y = _center(d, "ELEVATED WRECKED — HOUSES CRUSHED — MANY DEAD AND HURT", y, _fit(d, "ELEVATED WRECKED — HOUSES CRUSHED — MANY DEAD AND HURT", "oldb", PW - 2 * m, 50)) + 22
        d.line([(m, y), (PW - m, y)], fill=INK, width=3)
        y += 22
    else:
        y = _center(d, "The Morning Ledger", 20, font("black", 160)) + 30
        d.line([(m, y), (PW - m, y)], fill=INK, width=4)
        y = _center(d, "BOSTON, FRIDAY, JANUARY 17, 1919", y + 10, font("oldb", 36)) + 14
        d.line([(m, y), (PW - m, y)], fill=INK, width=4)
        y += 30
        y = _center(d, "DEATH LIST GROWS", y, _fit(d, "DEATH LIST GROWS", "play", PW - 2 * m, 220)) + 20
        y = _center(d, "IN MOLASSES FLOOD", y, _fit(d, "IN MOLASSES FLOOD", "play", PW - 2 * m, 220)) + 20
        d.line([(m, y), (PW - m, y)], fill=INK, width=3)
        y = _center(d, "Rescuers Still Digging in the North End Ruins", y + 14, _fit(d, "Rescuers Still Digging in the North End Ruins", "playi", PW - 2 * m, 72)) + 14
        y = _center(d, "SCORES IN HOSPITALS — HORSES SHOT WHERE THEY LAY MIRED", y, _fit(d, "SCORES IN HOSPITALS — HORSES SHOT WHERE THEY LAY MIRED", "oldb", PW - 2 * m, 50)) + 22
        d.line([(m, y), (PW - m, y)], fill=INK, width=3)
        y += 22
    # columns + photo
    ncol = 5
    gut = 22
    cw = (PW - 2 * m - gut * (ncol - 1)) / ncol
    pf = font("old", 25)
    px0 = m + cw + gut
    pw = cw * 3 + gut * 2
    ph = int(pw * 0.62)
    pimg = halftone(photo, int(pw), ph)
    img.paste(pimg, (int(px0), int(y)))
    d.rectangle([px0, y, px0 + pw, y + ph], outline=INK, width=3)
    cap = "SCENE OF THE DISASTER — WRECKAGE ON COMMERCIAL STREET" if kind == 0 else "WHERE THE GREAT TANK STOOD — VIEW FROM THE ELEVATED"
    _center(d, cap, y + ph + 8, _fit(d, cap, "oldb", pw, 26), x0=px0, x1=px0 + pw)
    for c in range(ncol):
        x = m + c * (cw + gut)
        top = y if c in (0, ncol - 1) else y + ph + 50
        _justify(d, BODY, x, top, cw, PH - top - 60, pf, 31)
        if c < ncol - 1:
            d.line([(x + cw + gut / 2, y), (x + cw + gut / 2, PH - 60)], fill=INK, width=1)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return age_paper(img, kind)


def desk():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grain = tex(NOISE2, xx * 0.08, yy * 1.6)
    v = 0.12 + 0.08 * grain + 0.04 * tex(NOISE_FINE, xx * 0.5, yy * 3)
    return (v[..., None] * np.array([1.0, 0.85, 0.7], np.float32)).astype(np.float32)


class Papers:
    def __init__(self, photo1, photo2):
        self.pages = []
        for k, ph in enumerate((photo1, photo2)):
            p = build_page(k, ph).convert("RGBA")
            blurred = p.filter(ImageFilter.GaussianBlur(6))
            self.pages.append((p, blurred))
        self.desk = desk()

    def place(self, base, k, zoom, cx, cy, rot, dx, dy, focus_y=None):
        from common import cam2d
        p, pb = self.pages[k]
        sharp = cam2d(p, zoom, cx, cy, rot, dx, dy, fill=(0, 0, 0, 0), resample=Image.BILINEAR)
        soft = cam2d(pb, zoom, cx, cy, rot, dx, dy, fill=(0, 0, 0, 0), resample=Image.BILINEAR)
        s = np.asarray(sharp, np.float32) / 255.0
        b = np.asarray(soft, np.float32) / 255.0
        fy = H * 0.42 if focus_y is None else focus_y
        rows = np.arange(H, dtype=np.float32)
        k_ = np.clip(np.abs(rows - fy) / (H * 0.42), 0, 1)[:, None, None] ** 1.5
        mix = s * (1 - k_) + b * k_
        a = mix[..., 3:4]
        # drop shadow
        sh = Image.fromarray((a[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(25))
        sha = np.asarray(sh, np.float32)[..., None] / 255.0
        sha = np.roll(np.roll(sha, 25, 0), 18, 1)
        base = base * (1 - sha * 0.7)
        rgb = mix[..., :3] / np.maximum(a, 1e-3)
        return base * (1 - a) + rgb * a
