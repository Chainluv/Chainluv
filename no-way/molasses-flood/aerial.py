"""Aerial view of the North End after the flood: rooftops, wharves, the ruptured tank
and the spreading molasses, with a tilted drone-style camera."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import W, H, NOISE, NOISE2, NOISE_FINE, tex, hash2

MS = 2600
TANK = (1150, 1250)


def build():
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[0:MS, 0:MS].astype(np.float32)
    ground = 0.5 + 0.12 * tex(NOISE, xx * 0.25, yy * 0.25)
    img = np.repeat(ground[..., None], 3, 2) * np.array([0.95, 0.92, 0.86], np.float32)
    # harbor (right side), wavy shoreline
    shore = 1750 + 120 * (tex(NOISE, yy * 0.08, 5.0) - 0.5) + (yy - 1300) * 0.12
    water = xx > shore
    wv = 0.22 + 0.1 * tex(NOISE2, xx * 1.2, yy * 0.4)
    img[water] = (wv[water][:, None] * np.array([0.9, 0.95, 1.0]))
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    # piers + ships
    for k in range(9):
        y0 = 150 + k * 280 + rng.integers(-40, 40)
        x0 = int(1750 + (y0 - 1300) * 0.12) - 40
        L = rng.integers(380, 620)
        d.rectangle([x0, y0, x0 + L, y0 + 90], fill=(120, 112, 100))
        for j in range(0, L, 36):
            d.rectangle([x0 + j, y0 + 6, x0 + j + 28, y0 + 84], fill=(90 + rng.integers(0, 40),) * 3)
        if rng.random() < 0.7:
            sy = y0 + 110 + rng.integers(0, 40)
            sl = rng.integers(220, 380)
            d.ellipse([x0 + 60, sy, x0 + 60 + sl, sy + 60], fill=(40, 36, 32))
            d.rectangle([x0 + 120, sy + 18, x0 + sl - 20, sy + 42], fill=(110, 100, 90))
    # crooked street grid + blocks of rooftops
    for bx in range(-1, 9):
        for by in range(-1, 10):
            x0 = bx * 230 + by * 18 + rng.integers(-10, 10)
            y0 = by * 270 - bx * 14 + rng.integers(-10, 10)
            if x0 > 1650:
                continue
            x1, y1 = x0 + 190, y0 + 225
            # lots
            lx = x0
            while lx < x1 - 20:
                lw = rng.integers(40, 90)
                ly = y0
                while ly < y1 - 20:
                    lh = rng.integers(50, 110)
                    tone = 70 + rng.integers(0, 90)
                    d.rectangle([lx + 8, ly + 8, min(lx + lw, x1) + 8, min(ly + lh, y1) + 8], fill=(35, 32, 30))
                    d.rectangle([lx, ly, min(lx + lw, x1), min(ly + lh, y1)], fill=(tone, tone - 6, tone - 14))
                    d.line([(lx, (ly + min(ly + lh, y1)) / 2), (min(lx + lw, x1), (ly + min(ly + lh, y1)) / 2)], fill=(tone - 25,) * 3, width=2)
                    for _ in range(rng.integers(0, 3)):
                        cx, cy = lx + rng.integers(5, max(6, lw - 5)), ly + rng.integers(5, max(6, lh - 5))
                        d.rectangle([cx, cy, cx + 9, cy + 9], fill=(40, 38, 36))
                    ly += lh
                lx += lw
    # Commercial Street + elevated line along the shore
    pts = [(1610 + (y - 1300) * 0.12, y) for y in range(-50, MS + 50, 50)]
    d.line(pts, fill=(150, 145, 135), width=80)
    d.line(pts, fill=(45, 42, 40), width=26)
    for i in range(0, len(pts) - 1):
        x, y = pts[i]
        for j in range(0, 50, 10):
            d.line([(x - 14, y + j), (x + 14, y + j)], fill=(95, 90, 85), width=3)
    pil = pil.filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(pil, np.float32) / 255.0
    # molasses spill: noisy blob from the tank flowing toward the street and harbor
    tx, ty = TANK
    ang = np.arctan2(yy - ty, xx - tx)
    r = np.sqrt((xx - tx) ** 2 + (yy - ty) ** 2)
    reach = 260 + 260 * tex(NOISE, ang * 30 + 100, 3.0) + 650 * np.clip(np.cos(ang - 0.1), 0, 1) ** 3
    spill = np.clip((reach - r) / 30, 0, 1)
    lobes = np.clip((tex(NOISE, xx * 0.12 + 40, yy * 0.12) - 0.52) * 8, 0, 1) * (r < 750)
    spill = np.maximum(spill, lobes * np.clip((800 - r) / 200, 0, 1))
    fl = tex(NOISE, xx * 0.35 + 7, yy * 0.2)
    fl2 = tex(NOISE_FINE, xx * 0.9, yy * 0.9)
    edge = np.clip(1 - np.abs(spill - 0.5) * 2, 0, 1)
    mol = np.array([0.07, 0.035, 0.012], np.float32) * (0.7 + 0.6 * fl2[..., None]) + \
        np.array([0.9, 0.55, 0.22], np.float32) * (np.clip((fl - 0.6) * 3, 0, 1) ** 2 * 0.45 + edge * 0.25)[..., None]
    # buildings partly poke out (rooftops still visible at edges)
    k = spill[..., None] * 0.93
    a = a * (1 - k) + mol * k
    pil = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    # ruptured tank: broken ring + scattered plates
    R = 150
    for s in range(0, 360, 4):
        if 60 < s < 150:
            continue
        a0 = np.deg2rad(s)
        x, y = tx + R * np.cos(a0), ty + R * np.sin(a0)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(150, 140, 125))
    d.ellipse([tx - R + 20, ty - R + 20, tx + R - 20, ty + R - 20], outline=(60, 55, 50), width=6)
    for _ in range(26):
        a0 = rng.uniform(np.deg2rad(40), np.deg2rad(170))
        rr = rng.uniform(R + 30, R + 420)
        x, y = tx + rr * np.cos(a0), ty + rr * np.sin(a0)
        L, th = rng.uniform(20, 60), rng.uniform(0, 3.14)
        d.line([(x, y), (x + L * np.cos(th), y + L * np.sin(th))], fill=(150, 140, 125), width=7)
    return pil


def homography(src, dst):
    A = []
    B = []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(B, float))


class Aerial:
    def __init__(self):
        self.map = build()
        self.clouds = Image.fromarray((NOISE2 * 255).astype(np.uint8)).resize((1600, 1600), Image.BICUBIC)

    def render(self, p, time):
        """p in [0,1] progress through the shot."""
        cx, cy = TANK[0] + 120 - 160 * p, TANK[1] + 80 - 60 * p
        zoom = 0.85 + 0.35 * p
        rot = np.deg2rad(-12 + 10 * p)
        tilt = 0.32  # top of frame further away
        # screen corners -> map points (ground-plane perspective)
        corners = [(0, 0), (W, 0), (W, H), (0, H)]
        src = []
        for (x, y) in corners:
            ny = (y - H / 2) / H
            depth = 1 + tilt * (-ny * 2)
            mx = (x - W / 2) / zoom * depth
            my = (y - H / 2) / zoom * (1 + tilt * 0.5) * (depth ** 0.5)
            rx = mx * np.cos(rot) - my * np.sin(rot) + cx
            ry = mx * np.sin(rot) + my * np.cos(rot) + cy
            src.append((rx, ry))
        coeffs = homography(src, corners)
        im = self.map.transform((W, H), Image.PERSPECTIVE, tuple(coeffs), resample=Image.BILINEAR, fillcolor=(60, 58, 55))
        a = np.asarray(im, np.float32) / 255.0
        # atmospheric haze toward the top + drifting cloud shadows (parallax layer)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        haze = np.clip(0.55 - yy * 0.9, 0, 1) * 0.8
        a = a * (1 - haze) + np.array([0.78, 0.76, 0.72]) * haze
        cl = np.asarray(self.clouds.crop((int(200 + time * 60), int(300 + time * 25), int(200 + time * 60) + W // 2, int(300 + time * 25) + H // 2)).resize((W, H)), np.float32) / 255.0
        cs = np.clip((cl - 0.55) * 2.2, 0, 1)[..., None]
        a = a * (1 - cs * 0.35)
        wisp = np.clip((cl - 0.7) * 3, 0, 1)[..., None] * (1 - yy) * 0.5
        a = a * (1 - wisp) + 0.85 * wisp
        return a
