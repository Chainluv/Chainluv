"""The riveted molasses storage tank: creaks, cracks, bursts."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import W, H, NOISE, NOISE2, NOISE_FINE, tex, hash2, smoothstep, ease_out

XG, YG = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)

TCX, TR = 540.0, 520.0
TTOP, TBOT = 260.0, 1640.0
GROUND = 1640.0
NROWS = 7
MOL = np.array([0.075, 0.035, 0.012], np.float32)
AMBER = np.array([1.0, 0.62, 0.24], np.float32)


def rim_curve(u, ybase):
    b = (GROUND + 150 - ybase) * 0.085
    return ybase + b * np.sqrt(np.clip(1 - u * u, 0, 1))


def _crack_paths(seed=4):
    rng = np.random.default_rng(seed)
    paths = []

    def walk(u, y, ang, n, depth):
        pts = [(u, y)]
        base = ang
        for i in range(n):
            ang += rng.normal(0, 0.28)
            ang = ang * 0.85 + base * 0.15
            step = 10 + rng.random() * 16
            u += np.cos(ang) * step / TR
            y += np.sin(ang) * step
            u = float(np.clip(u, -0.92, 0.92))
            pts.append((u, y))
            if depth < 1 and rng.random() < 0.06:
                walk(u, y, ang + rng.choice([-1, 1]) * (0.6 + rng.random()), n // 2, depth + 1)
        paths.append((pts, depth))
    # main seam split rising up the tank
    walk(-0.05, 1560, -np.pi / 2 + 0.1, 60, 0)
    walk(-0.05, 1520, -0.15, 22, 1)
    walk(-0.08, 1400, np.pi + 0.2, 20, 1)
    walk(0.0, 1180, -np.pi / 2 - 0.5, 20, 1)
    walk(-0.1, 1000, np.pi + 0.5, 14, 1)
    return paths


CRACKS = _crack_paths()

_rng = np.random.default_rng(9)
FRAGS = [dict(u=-0.05 + _rng.normal(0, 0.12), y=1350 + _rng.normal(0, 160),
              vx=_rng.normal(0, 900), vy=-500 - _rng.random() * 900, w=25 + _rng.random() * 70,
              h=4 + _rng.random() * 9, rot=_rng.random() * 6, vr=_rng.normal(0, 9)) for _ in range(14)]
RIVETS = [dict(u=_rng.uniform(-0.8, 0.8), row=_rng.integers(1, NROWS), t0=_rng.uniform(3.9, 5.05),
               vx=_rng.normal(0, 500), vy=-200 - _rng.random() * 500) for _ in range(40)]


def sky(wx, wy, time):
    g = np.clip((GROUND - wy) / 1800, 0, 1)
    c = tex(NOISE2, wx * 0.25 + time * 8, wy * 0.35)
    v = (0.55 + 0.3 * g) * (0.82 + 0.3 * c)
    return v


def render(time, zoom=1.0, cxw=540, cyw=960, burst_t=5.1):
    wx = (XG - W / 2) / zoom + cxw
    wy = (YG - H / 2) / zoom + cyw
    img = np.empty((H, W, 3), np.float32)
    v = sky(wx, wy, time)
    img[:] = (v[..., None] * np.array([0.95, 0.93, 0.9], np.float32))

    # distant warehouses / harbor cranes silhouettes
    bh = 1460 + 90 * tex(NOISE, wx * 0.02, 1.0) - (np.floor(wx / 140) % 3 == 0) * 120
    bm = wy > bh
    img[bm] = (0.32 + 0.05 * tex(NOISE_FINE, wx[bm] * 0.5, wy[bm] * 0.5))[:, None] * np.array([0.9, 0.87, 0.83])

    # tank body
    u = (wx - TCX) / TR
    inside_u = np.abs(u) < 1
    ytop = rim_curve(u, TTOP)
    tm = inside_u & (wy > ytop) & (wy < GROUND + 10)
    uu = u[tm]
    yy = wy[tm]
    th = np.arcsin(np.clip(uu, -1, 1))
    lam = 0.12 + 0.7 * np.clip(np.cos(th + 0.6), 0, 1) ** 1.6 + 0.25 * np.exp(-((th + 0.45) / 0.18) ** 2)
    lam *= 0.65 + 0.35 * np.sqrt(np.clip(1 - uu * uu, 0, 1))
    # plate rows
    rowy = np.linspace(TTOP, GROUND, NROWS + 1)
    row = np.zeros_like(yy, dtype=np.int64)
    seam_d = np.full_like(yy, 1e9)
    for k in range(NROWS + 1):
        yk = rim_curve(uu, rowy[k])
        row += (yy > yk).astype(np.int64)
        seam_d = np.minimum(seam_d, np.abs(yy - yk))
    nvert = 9
    colf = (th / np.pi + 0.5) * nvert + (row % 2) * 0.5
    col = np.floor(colf)
    vert_d = np.abs(colf - col - 0.5)  # 0.5 at seam
    vert_px = (0.5 - vert_d) * (TR * np.pi / nvert) * np.cos(th)
    plate = 0.92 + 0.12 * hash2(col, row, 2)
    rust = tex(NOISE, uu * 300, yy * 0.15)
    streak = tex(NOISE2, uu * 700, yy * 0.012 + 20)
    base = lam * plate * (0.82 + 0.25 * rust)
    # leak stains running down from seams (the tank was painted brown to hide them)
    leak = np.clip((streak - 0.6) * 3.0, 0, 1) * (row > 1)
    shade = base * (1 - 0.55 * leak)
    # seams + rivets
    seam = (seam_d < 2.5) | (vert_px < 2.0)
    shade = np.where(seam, shade * 0.72, shade)
    rv_along = (th * 90) % 1.0
    rivet = (np.abs(seam_d - 6) < 3) & (np.abs(rv_along - 0.5) < 0.2)
    shade = np.where(rivet, shade * 1.25 + 0.03, shade)
    shade *= 0.9 + 0.2 * tex(NOISE_FINE, uu * 600, yy * 0.6)
    tcol = shade[:, None] * np.array([0.8, 0.68, 0.55], np.float32)[None, :]
    tcol = tcol * (1 - leak[:, None] * 0.2) + MOL[None, :] * leak[:, None] * 0.5
    img[tm] = tcol
    # rim railing
    rail = inside_u & (wy > ytop - 34) & (wy <= ytop) & ((np.abs(wy - (ytop - 30)) < 2.5) | (((np.arcsin(np.clip(u, -1, 1)) * 30) % 1.0) < 0.12))
    img[rail] = 0.18

    # ground
    gm = wy >= GROUND + 10
    g = 0.28 + 0.08 * tex(NOISE, wx[gm] * 0.1, wy[gm] * 0.3)
    img[gm] = g[:, None] * np.array([0.95, 0.9, 0.85])

    # elevated railway trestle crossing the foreground
    ely0, ely1 = 1500.0, 1570.0
    em = (wy > ely0) & (wy < ely1)
    lat = (wy < ely0 + 8) | (wy > ely1 - 8) | ((wx + (wy - ely0)) % 60 < 6) | ((wx - (wy - ely0)) % 60 < 6)
    img[em & lat] = 0.1
    cm = (wy >= ely1) & ((wx % 260) < 18)
    img[cm] = 0.11

    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")

    def S(uw, yw):  # world (u, y) on tank -> screen
        xw = TCX + uw * TR
        return ((xw - cxw) * zoom + W / 2, (yw - cyw) * zoom + H / 2)

    def Sw(xw, yw):
        return ((xw - cxw) * zoom + W / 2, (yw - cyw) * zoom + H / 2)

    # tiny workers at the base for scale
    for i, px in enumerate([150, 190, 860, 900, 935]):
        x, y = Sw(px, GROUND + 6)
        s = 26 * zoom
        d.rectangle([x - s * 0.18, y - s, x + s * 0.18, y], fill=(25, 22, 20))
        d.ellipse([x - s * 0.12, y - s * 1.25, x + s * 0.12, y - s * 0.98], fill=(25, 22, 20))

    # cracks
    crack_p = np.clip((time - 3.7) / (burst_t - 3.7), 0, 1)
    if crack_p > 0:
        for pts, depth in CRACKS:
            n = int(len(pts) * min(1, crack_p * (1.4 - depth * 0.25)))
            if n < 2:
                continue
            sp = [S(*p) for p in pts[:n]]
            wdt = max(1, int((5 - depth * 1.5) * zoom * (0.6 + crack_p)))
            d.line(sp, fill=(8, 4, 2, 255), width=wdt + 3)
            d.line(sp, fill=(70, 35, 12, 255), width=max(1, wdt - 1))
            # oozing drips from the crack
            for j in range(2, n, 7 if depth == 0 else 99):
                ph = hash2(j, depth, 5)
                age = max(0.0, time - (3.7 + (j / max(1, len(pts))) * (burst_t - 3.7) / max(crack_p, 1e-3) * crack_p))
                L = min(260, age * (40 + 90 * ph)) * zoom
                x, y = sp[j]
                if L > 2:
                    ww = max(2, int(5 * zoom * (0.5 + ph)))
                    d.line([(x, y), (x, y + L)], fill=(14, 7, 2, 240), width=ww)
                    d.ellipse([x - ww * 0.8, y + L - ww * 0.6, x + ww * 0.8, y + L + ww], fill=(14, 7, 2, 240))
                    d.line([(x - ww * 0.25, y), (x - ww * 0.25, y + L * 0.9)], fill=(160, 95, 35, 120), width=1)
    # popping rivets
    for r in RIVETS:
        a = time - r["t0"]
        if 0 < a < 0.7 and r["t0"] < burst_t:
            yb = rowy[r["row"]]
            u0 = r["u"]
            y0 = rim_curve(u0, yb)
            x, y = S(u0, y0)
            x += r["vx"] * a * zoom
            y += (r["vy"] * a + 900 * a * a) * zoom
            d.line([(x, y), (x - r["vx"] * 0.02 * zoom, y - (r["vy"] + 1800 * a) * 0.02 * zoom)], fill=(240, 210, 150, 230), width=max(2, int(4 * zoom)))
            if a < 0.06:
                d.ellipse([x - 14 * zoom, y - 14 * zoom, x + 14 * zoom, y + 14 * zoom], fill=(255, 230, 180, 160))

    # burst: molasses torrent + flying steel plates
    if time >= burst_t:
        a = time - burst_t
        cx_, cy_ = S(-0.05, 1450)
        torrent = Image.new("L", (W, H), 0)
        td = ImageDraw.Draw(torrent)
        Rx = (180 + a * 2200) * zoom
        Ry = (140 + a * 1500) * zoom
        poly = []
        for k in range(72):
            ang = k / 72 * 2 * np.pi
            nk = 0.75 + 0.5 * tex(NOISE, np.array([k * 7.0 + a * 60]), np.array([a * 30]))[0]
            down = 1.0 + 0.9 * max(0, np.sin(ang))  # spreads more downward
            poly.append((cx_ + np.cos(ang) * Rx * nk, cy_ + np.sin(ang) * Ry * nk * down))
        td.polygon(poly, fill=255)
        tmask = np.asarray(torrent.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255.0
        arr = np.asarray(pil, np.float32) / 255.0
        fl = tex(NOISE2, (XG - cx_) * 0.12, (YG - cy_) * 0.05 - a * 300)
        fl2 = tex(NOISE, (XG - cx_) * 0.3, (YG - cy_) * 0.3 - a * 500)
        mcol = MOL[None, None, :] * (0.6 + 0.9 * fl2[..., None]) + AMBER[None, None, :] * (np.clip((fl - 0.6) * 3, 0, 1) ** 2)[..., None] * 0.8
        arr = arr * (1 - tmask[..., None]) + mcol * tmask[..., None]
        pil = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil, "RGBA")
        for fr in FRAGS:
            x, y = S(fr["u"], fr["y"])
            x += fr["vx"] * a * zoom
            y += (fr["vy"] * a + 700 * a * a) * zoom
            ang = fr["rot"] + fr["vr"] * a
            c, s = np.cos(ang), np.sin(ang)
            w2, h2 = fr["w"] * zoom, fr["h"] * zoom
            pts = [(x + c * px - s * py, y + s * px + c * py) for px, py in [(-w2, -h2), (w2, -h2 * 0.7), (w2 * 0.8, h2), (-w2, h2 * 0.8)]]
            d.polygon(pts, fill=(40, 34, 30, 255), outline=(120, 100, 85, 255))
    return np.asarray(pil, np.float32) / 255.0
