"""Procedural ray-cast 1919 Boston street: brick warehouses, cobblestones, trolley rails,
an elevated railway overhead, and a molasses wave / flood surface."""
import numpy as np
from PIL import Image, ImageDraw
from common import (W, H, NOISE, NOISE2, NOISE_FINE, tex, hash2, smoothstep, font)

XG, YG = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)

WF = 26.0       # facade plane |X|
SW = 17.0       # curb |X|
EL_W = 7.0      # elevated structure half width
EL_Y = 17.0     # el underside height
SEGL = 34.0
RAILS = [-7.4, -2.6, 2.6, 7.4]

MOL = np.array([0.075, 0.035, 0.012], np.float32)
AMBER = np.array([1.0, 0.62, 0.24], np.float32)
FOG = np.array([0.74, 0.73, 0.70], np.float32)

SIGNS = ["SHIP CHANDLERY", "FREIGHT DEPOT", "HARDWARE & PAINTS", "N. E. COAL CO.",
         "WHOLESALE GROCERS", "ROPE & TACKLE", "PAVING DEPT.", "LIVERY STABLE",
         "COOPERAGE", "IRON WORKS", "TEA & COFFEE", "SALT FISH"]


def _make_signs():
    out = []
    for i, s in enumerate(SIGNS):
        img = Image.new("L", (640, 80), 40 + (i * 37) % 50)
        d = ImageDraw.Draw(img)
        f = font("oswaldm", 54)
        w = d.textlength(s, font=f)
        d.text(((640 - w) / 2, 6), s, font=f, fill=205)
        d.rectangle([3, 3, 636, 76], outline=150, width=3)
        out.append(np.asarray(img, np.float32) / 255.0)
    return np.stack(out)


SIGN_TEX = _make_signs()

# per-building random parameters (index by segment id + side)
NSEG = 400
_r = np.random.default_rng(5)
SEG_H = (32 + _r.random((2, NSEG)) * 34).astype(np.float32)
SEG_TONE = (0.36 + _r.random((2, NSEG)) * 0.26).astype(np.float32)
SEG_SPC = (6.0 + _r.integers(0, 3, (2, NSEG)) * 1.0).astype(np.float32)
SEG_SIGN = _r.integers(-4, len(SIGNS), (2, NSEG))


class Street:
    def __init__(self, **kw):
        self.f = 1000.0
        self.yh = 1150.0
        self.ch = 5.5
        self.camX = 0.0
        self.camZ = 0.0
        self.time = 0.0
        self.wave_z = None      # world Z of molasses wave front
        self.wave_h = 30.0
        self.flood_h = None     # flood surface height behind the front (or everywhere if front None)
        self.flood_front = None
        self.flood_speed = 30.0
        self.el_gap = None      # (z0, z1) collapsed section
        self.collapsed = []     # list of (side, seg) collapsed buildings
        self.fog = 380.0
        self.el_c = 15.0
        self.sky_bright = 1.0
        self.__dict__.update(kw)

    # ---------------------------------------------------------- helpers
    def project(self, X, Y, Z):
        t = Z - self.camZ
        x = W / 2 + self.f * (X - self.camX) / t
        y = self.yh - self.f * (Y - self.ch) / t
        return x, y, t

    def wave_height(self, X):
        tm = self.time
        k = min(1.0, self.wave_h / 32)
        h = self.wave_h + (3.5 * np.sin(X * 0.21 + tm * 2.3) + 2.5 * np.sin(X * 0.57 - tm * 3.1)) * k
        h += (tex(NOISE, X * 6 + tm * 40, tm * 25) - 0.5) * 9 * k
        h += smoothstep(WF - 8, WF, np.abs(X)) * 7 * (self.wave_h / 32)  # climbs the walls
        return h

    def sky(self, dx, dy):
        g = np.clip(dy * 1.2, 0, 1)[..., None]
        base = FOG * (1 - g) + np.array([0.86, 0.85, 0.83], np.float32) * g
        u = dx / (dy + 0.25) * 140 + self.time * 6
        v = 1 / (dy + 0.25) * 90
        c = tex(NOISE2, u, v)[..., None]
        sky = base * (0.78 + c * 0.32) * self.sky_bright
        return sky.astype(np.float32)

    # ---------------------------------------------------------- render
    def render(self):
        f, ch = self.f, self.ch
        dx = (XG - W / 2) / f
        dy = (self.yh - YG) / f
        INF = np.float32(1e9)
        cands = []

        # ground
        tg = np.where(dy < -1e-4, ch / np.maximum(-dy, 1e-4), INF)
        cands.append(tg)

        # flood surface
        if self.flood_h is not None:
            tf = np.where(dy < -1e-4, (ch - self.flood_h) / np.maximum(-dy, 1e-4), INF)
            if self.flood_front is not None:
                zf = self.camZ + tf
                tf = np.where(zf > self.flood_front, tf, INF)
            tf = np.where(tf > 0, tf, INF)
        else:
            tf = np.full_like(dx, INF)
        cands.append(tf)

        # facades (left / right)
        tfac = np.where(np.abs(dx) > 1e-4, (np.sign(dx) * WF - self.camX) / np.where(np.abs(dx) > 1e-4, dx, 1), INF)
        tfac = np.where(tfac > 0, tfac, INF)
        Yf = ch + dy * tfac
        Zf = self.camZ + tfac
        side = (dx > 0).astype(np.int64)
        seg = np.clip(np.floor(Zf / SEGL + side * 0.37).astype(np.int64), 0, NSEG - 1)
        hb = SEG_H[side, seg]
        for (sd, sg) in self.collapsed:
            m = (side == sd) & (seg == sg)
            jag = 6 + 9 * tex(NOISE, Zf * 9, 3.0)
            hb = np.where(m, jag, hb)
        tfac = np.where(Yf < hb, tfac, INF)
        cands.append(tfac)

        # el deck underside (el runs along the street, offset to one side)
        ec, ew = self.el_c, EL_W
        td = np.where(dy > 1e-4, (EL_Y - ch) / np.maximum(dy, 1e-4), INF)
        Xd = self.camX + dx * td
        Zd = self.camZ + td
        dm = np.abs(Xd - ec) < ew
        if self.el_gap:
            dm &= ~((Zd > self.el_gap[0]) & (Zd < self.el_gap[1]))
        td = np.where(dm, td, INF)
        cands.append(td)

        # el side trusses + columns (planes X = ec +- ew)
        te = np.full_like(dx, INF)
        Ye = np.zeros_like(dx)
        Ze = np.zeros_like(dx)
        safe = np.where(np.abs(dx) > 1e-4, dx, 1e-4)
        for px in (ec - ew, ec + ew):
            tt = (px - self.camX) / safe
            tt = np.where((tt > 0) & (np.abs(dx) > 1e-4), tt, INF)
            Yt = ch + dy * tt
            Zt = self.camZ + tt
            ly = Yt - EL_Y
            trussm = (ly >= 0) & (ly <= 5.5) & (
                (ly < 0.7) | (ly > 4.8) | (np.mod(Zt + ly, 5.5) < 0.55) | (np.mod(Zt - ly, 5.5) < 0.55) | (np.mod(Zt, 11) < 0.5))
            zc = np.mod(Zt + 3, 40)
            cw = 1.3 + np.maximum(0, Yt - (EL_Y - 3.5)) * 0.9
            colm = (Yt < EL_Y) & (zc < cw)
            em = trussm | colm
            if self.el_gap:
                em &= ~((Zt > self.el_gap[0]) & (Zt < self.el_gap[1]))
            tt = np.where(em, tt, INF)
            better = tt < te
            te = np.where(better, tt, te)
            Ye = np.where(better, Yt, Ye)
            Ze = np.where(better, Zt, Ze)
        cands.append(te)

        # wave wall
        if self.wave_z is not None and self.wave_z - self.camZ > 0.3:
            tw0 = self.wave_z - self.camZ
            Xw = self.camX + dx * tw0
            Yw = ch + dy * tw0
            hw = self.wave_height(Xw)
            tw = np.where((Yw <= hw) & (Yw > -1), np.float32(tw0), INF)
        else:
            tw = np.full_like(dx, INF)
            Xw = Yw = hw = None
        cands.append(tw)

        T = np.stack(cands)
        idx = np.argmin(T, axis=0)
        tmin = np.min(T, axis=0)
        img = np.zeros((H, W, 3), np.float32)

        # sky
        m = tmin >= INF
        img[m] = self.sky(dx[m], dy[m])

        # ground
        m = (idx == 0) & ~(tmin >= INF)
        if m.any():
            img[m] = self.shade_ground(dx[m], dy[m], tg[m])
        m = idx == 1
        if m.any():
            img[m] = self.shade_flood(dx[m], dy[m], tf[m])
        m = idx == 2
        if m.any():
            img[m] = self.shade_facade(side[m], seg[m], Zf[m], Yf[m], tfac[m])
        m = idx == 3
        if m.any():
            img[m] = self.shade_deck(Xd[m], Zd[m], dx[m], dy[m])
        m = idx == 4
        if m.any():
            v = 0.13 + 0.07 * tex(NOISE_FINE, Ze[m] * 20, Ye[m] * 20)
            v = v + 0.06 * (np.sign(dx[m]) < 0)
            img[m] = (v[:, None] * np.array([1, 0.98, 0.95], np.float32))
        m = idx == 5
        if m.any():
            img[m] = self.shade_wave(Xw[m], Yw[m], hw[m])

        # fog
        fogk = (1 - np.exp(-np.minimum(tmin, 5000) / self.fog))[..., None]
        fogk = np.where((tmin >= INF)[..., None], 0, fogk)
        img = img * (1 - fogk) + FOG * self.sky_bright * fogk
        self.depth = tmin
        return img

    # ---------------------------------------------------------- shaders
    def shade_ground(self, dx, dy, t):
        X = self.camX + dx * t
        Z = self.camZ + t
        fp = t * t / (self.f * self.ch)  # ground footprint along z per px
        fade = np.clip(1 - fp / 0.35, 0, 1)
        n = tex(NOISE, X * 6, Z * 6)
        street = np.abs(X) < SW
        # cobbles
        row = np.floor(Z / 0.42)
        off = np.mod(row, 2) * 0.45
        cx = (X + off * 0.7) / 0.62
        fx = cx - np.floor(cx)
        fz = Z / 0.42 - row
        edge = np.minimum(np.minimum(fx, 1 - fx) * 0.62, np.minimum(fz, 1 - fz) * 0.42)
        stone = 0.3 + 0.18 * hash2(np.floor(cx), row)
        cob = stone * (0.45 + 0.55 * smoothstep(0.0, 0.11, edge)) * (0.85 + 0.3 * smoothstep(0.05, 0.2, edge))
        cob = cob * fade + 0.3 * (1 - fade)
        # sidewalk slabs
        sz = np.mod(Z, 5.0)
        slab = 0.5 + 0.06 * hash2(np.floor(Z / 5), np.floor(X / 6), 3)
        slab = np.where(sz < 0.12 * (1 + fp * 4), slab * (1 - 0.35 * fade), slab)
        curb = np.abs(np.abs(X) - SW) < 0.5
        v = np.where(street, cob, slab)
        v = np.where(curb, 0.62, v)
        # rails
        pxw = t / self.f
        for r in RAILS:
            cov = np.clip(0.22 / np.maximum(pxw, 1e-3), 0, 1)
            d = np.abs(X - r)
            v = np.where(d < np.maximum(0.11, pxw * 0.7), v * (1 - cov) + 0.78 * cov, v)
            v = np.where((d > 0.11) & (d < 0.35), v * 0.8, v)
        v = v * (0.8 + 0.4 * n)
        # under the el is darker
        v = v * np.where(np.abs(X - self.el_c) < EL_W, 0.78, 1.0)
        # wet sheen near horizon
        sheen = np.clip(1 - (-dy) * 7, 0, 1) * 0.22
        col = v[:, None] * np.array([1.0, 0.97, 0.92], np.float32) + sheen[:, None]
        # molasses slick right in front of the wave
        if self.wave_z is not None:
            dzw = self.wave_z - Z
            slick = (dzw > 0) & (dzw < 9 + 0.3 * self.wave_h)
            k = smoothstep(9 + 0.3 * self.wave_h, 0, dzw) * slick
            fl = tex(NOISE2, X * 3, (Z - self.time * 60) * 2)
            mcol = MOL[None, :] * (0.8 + 0.6 * fl[:, None]) + AMBER[None, :] * np.clip(fl - 0.62, 0, 1)[:, None] * 1.4
            col = col * (1 - k[:, None]) + mcol * k[:, None]
        return col

    def shade_flood(self, dx, dy, t):
        X = self.camX + dx * t
        Z = self.camZ + t
        tm = self.time
        n = tex(NOISE2, X * 2.5, (Z + tm * self.flood_speed) * 0.9)
        n2 = tex(NOISE, X * 6 + 50, (Z + tm * self.flood_speed * 1.3) * 2.5)
        cos = -dy / np.sqrt(dx * dx + dy * dy + 1)
        fres = 0.05 + 0.95 * (1 - np.clip(cos * 2.2 + (n - 0.5) * 0.35, 0, 1)) ** 4
        refl = (FOG * 1.05 * self.sky_bright)[None, :] * np.array([1.0, 0.8, 0.6], np.float32)[None, :]
        body = MOL[None, :] * (0.6 + 0.8 * n2[:, None])
        col = body * (1 - fres[:, None]) + refl * fres[:, None]
        spec = np.clip((n2 - 0.6) * 4, 0, 1) ** 2 * np.clip(1 - fres, 0.2, 1)
        col = col + AMBER[None, :] * spec[:, None] * 0.55
        # shadow strip under the el
        col = col * np.where(np.abs(X - self.el_c) < EL_W, 0.85, 1.0)[:, None]
        return col

    def shade_facade(self, side, seg, Z, Y, t):
        u = Z - (seg - side * 0.37) * SEGL
        tone = SEG_TONE[side, seg]
        spc = SEG_SPC[side, seg]
        hb = SEG_H[side, seg]
        pxw = t / self.f
        detail = np.clip(1 - (pxw - 0.06) / 0.25, 0, 1)
        n = tex(NOISE, u * 5 + seg * 37, Y * 5)
        nf = tex(NOISE_FINE, u * 30, Y * 30)
        brick = tone * (0.78 + 0.32 * n) * (0.9 + 0.2 * nf)
        v = brick
        # gaps between buildings
        v = np.where(u < 0.5, v * 0.45, v)
        # storefront level
        sf_win = (Y > 1.8) & (Y < 9.2) & (u > 2.2) & (u < SEGL - 2.2)
        mull = np.mod(u - 2.2, 4.0) < 0.35
        glass = 0.08 + 0.12 * (Y / 9.2) + 0.08 * tex(NOISE2, u * 3, Y * 3)
        v = np.where(sf_win & ~mull, glass, v)
        v = np.where(sf_win & mull, 0.5, v)
        # sign band
        sid = SEG_SIGN[side, seg]
        has_sign = (sid >= 0) & (Y > 9.8) & (Y < 12.2) & (u > 3) & (u < SEGL - 3)
        if has_sign.any():
            su = np.clip(((u - 3) / (SEGL - 6)) * 639, 0, 639).astype(np.int64)
            sv = np.clip(((12.2 - Y) / 2.4) * 79, 0, 79).astype(np.int64)
            sv_ = SIGN_TEX[np.clip(sid, 0, None), sv, su] * 0.9
            v = np.where(has_sign, sv_ * detail + 0.25 * (1 - detail), v)
        # upper windows
        up = Y > 13
        fy = np.mod(Y - 13, 10.5)
        flr = np.floor((Y - 13) / 10.5)
        cu = (u - 3.0) / spc
        fu = (cu - np.floor(cu)) * spc
        win = up & (fy > 2.6) & (fy < 8.2) & (fu < 3.2) & (u > 3) & (u < SEGL - 3) & (Y < hb - 4)
        frame = win & ((fy < 2.95) | (fy > 7.85) | (fu < 0.3) | (fu > 2.9) | (np.abs(fy - 5.4) < 0.15))
        wid = hash2(np.floor(cu) + seg * 13, flr + side * 7, 9)
        pane = 0.06 + 0.18 * (fy - 2.6) / 5.6 * (wid > 0.3) + (wid > 0.82) * 0.32
        pane = pane + 0.06 * tex(NOISE2, u * 4, Y * 4)
        lint = up & (fy > 8.2) & (fy < 8.9) & (fu < 3.6) & (u > 2.6) & (u < SEGL - 2.6) & (Y < hb - 4)
        v = np.where(win, np.where(frame, 0.66, pane) * detail + v * (1 - detail) * 0.6, v)
        v = np.where(lint, 0.62 * detail + v * (1 - detail), v)
        # string course + cornice
        v = np.where(np.abs(Y - 12.6) < 0.35, 0.6, v)
        top = hb - Y
        v = np.where(top < 2.6, np.where(top < 0.6, 0.62, np.where(top < 1.4, 0.18, 0.48)), v)
        # collapsed buildings look splintered
        for (sd, sg) in self.collapsed:
            m = (side == sd) & (seg == sg)
            v = np.where(m, 0.22 + 0.25 * tex(NOISE_FINE, u * 12, Y * 3), v)
        # ambient occlusion near street + light side
        v = v * (0.62 + 0.38 * np.clip(Y / 8, 0, 1))
        v = v * np.where(side == 0, 1.0, 0.82)
        return v[:, None] * np.array([1.0, 0.94, 0.88], np.float32)[None, :]

    def shade_deck(self, X, Z, dx, dy):
        Xr = X - self.el_c
        ties = np.mod(Z, 1.15) < 0.8
        gird = (np.abs(np.abs(Xr) - EL_W * 0.45) < 0.6) | (np.abs(Xr) > EL_W - 0.8)
        solid = ties | gird
        skyc = self.sky(dx, dy) * 0.75
        iron = (0.1 + 0.05 * tex(NOISE_FINE, X * 20, Z * 20))[:, None] * np.ones(3, np.float32)
        return np.where(solid[:, None], iron, skyc)

    def shade_wave(self, X, Y, hw):
        tm = self.time
        d = hw - Y  # depth below crest
        s1 = tex(NOISE2, X * 7, Y * 1.1 - tm * 70)
        s2 = tex(NOISE, X * 14 + 99, Y * 2.5 - tm * 110)
        s3 = tex(NOISE_FINE, X * 25, Y * 6 - tm * 160)
        # pseudo normal: face curves from vertical (bottom) to overhanging lip (top)
        curve = np.clip(1 - d / 10, 0, 1)
        body = MOL[None, :] * (0.5 + 0.9 * s2[:, None]) * (0.7 + 0.6 * curve[:, None])
        spec = np.clip((s1 - 0.5) * 3.2, 0, 1) ** 2 * (0.35 + curve)
        streak = np.clip((s3 - 0.55) * 4, 0, 1) * 0.5
        band = np.exp(-((d - 4.5) / 2.8) ** 2)
        rim = np.exp(-d / 0.7) * 0.7 * min(1.0, self.wave_h / 25)
        lip = np.exp(-((d - 2.2) / 0.9) ** 2) * 0.75
        col = body * (1 - lip[:, None])
        col = col + AMBER[None, :] * (spec * 0.6 + rim * 0.9 + band * s1 * 0.45 + streak * 0.25)[:, None]
        col = col + np.array([0.95, 0.88, 0.75], np.float32)[None, :] * (spec * band * 0.5 + rim * 0.15)[:, None]
        col = col * (0.45 + 0.55 * np.clip(Y / 5, 0, 1))[:, None]
        return col


# ------------------------------------------------------------ sprites
def person_poly(kind, run=0.0):
    """Silhouette polygons in normalized coords (x in [-.5,.5], y up 0..1)."""
    if kind == 0:  # man in long coat and bowler
        body = [(-0.2, 0.0), (-0.12, 0.0), (-0.09, 0.42), (0.09, 0.42), (0.12, 0.0), (0.2, 0.0),
                (0.23, 0.5), (0.24, 0.78), (0.12, 0.84), (-0.12, 0.84), (-0.24, 0.78), (-0.23, 0.5)]
        head = (0.0, 0.9, 0.075)
        hat = [(-0.13, 0.93), (0.13, 0.93), (0.08, 0.95), (0.08, 1.0), (-0.08, 1.0), (-0.08, 0.95)]
    elif kind == 1:  # woman, long skirt, wide hat
        body = [(-0.27, 0.0), (0.27, 0.0), (0.12, 0.55), (0.18, 0.8), (0.1, 0.84), (-0.1, 0.84), (-0.18, 0.8), (-0.12, 0.55)]
        head = (0.0, 0.9, 0.07)
        hat = [(-0.2, 0.94), (0.2, 0.94), (0.1, 0.97), (0.06, 1.0), (-0.06, 1.0), (-0.1, 0.97)]
    else:  # worker with cap
        body = [(-0.18, 0.0), (-0.1, 0.0), (-0.05, 0.45), (0.05, 0.45), (0.1, 0.0), (0.18, 0.0),
                (0.2, 0.5), (0.25, 0.8), (0.12, 0.85), (-0.12, 0.85), (-0.25, 0.8), (-0.2, 0.5)]
        head = (0.0, 0.91, 0.075)
        hat = [(-0.1, 0.95), (0.14, 0.95), (0.1, 0.99), (-0.08, 0.99)]
    return body, head, hat


def draw_sprites(img, st, sprites):
    """sprites: list of dict(kind, X, Z, h, shade). Depth tested against st.depth."""
    for sp in sprites:
        t = sp["Z"] - st.camZ
        if t < 1.5:
            continue
        x0, y0, _ = st.project(sp["X"], sp.get("Y", 0.0), sp["Z"])
        scale = st.f * sp["h"] / t
        if scale < 3 or scale > 4000:
            continue
        pad = 4
        bw = int(scale * 0.7) + pad * 2
        bh = int(scale * 1.05) + pad * 2
        lx, ly = int(x0 - bw / 2), int(y0 - scale - pad)
        if lx > W or ly > H or lx + bw < 0 or ly + bh < 0:
            continue
        mask = Image.new("L", (bw, bh), 0)
        d = ImageDraw.Draw(mask)

        def P(p):
            return (bw / 2 + p[0] * scale, bh - pad - p[1] * scale)
        if sp["kind"] == "cart":
            # horse + wagon side-ish silhouette
            pts = [(-0.9, 0.35), (0.2, 0.35), (0.2, 0.7), (-0.9, 0.7)]
            d.polygon([P(p) for p in pts], fill=255)
            for wx in (-0.7, 0.0):
                cx, cy = P((wx, 0.18))
                r = 0.18 * scale
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=max(1, int(scale * 0.03)))
            horse = [(0.25, 0.45), (0.85, 0.45), (0.95, 0.75), (1.1, 0.8), (1.05, 0.62), (0.9, 0.5),
                     (0.85, 0.0), (0.8, 0.0), (0.78, 0.4), (0.4, 0.4), (0.35, 0.0), (0.3, 0.0)]
            d.polygon([P(p) for p in horse], fill=255)
            d.line([P((0.2, 0.55)), P((0.5, 0.55))], fill=255, width=max(1, int(scale * 0.03)))
        elif sp["kind"] == "plank":
            ang = sp.get("ang", 0.3)
            c, s = np.cos(ang), np.sin(ang)
            pts = [(-0.5, -0.04), (0.5, -0.04), (0.5, 0.04), (-0.5, 0.04)]
            d.polygon([P((x * c - y * s, 0.3 + x * s + y * c)) for x, y in pts], fill=255)
        else:
            body, head, hat = person_poly(sp["kind"])
            d.polygon([P(p) for p in body], fill=255)
            cx, cy = P(head[:2])
            r = head[2] * scale
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            d.polygon([P(p) for p in hat], fill=255)
        m = np.asarray(mask, np.float32) / 255.0
        # clip to screen
        sx0, sy0 = max(0, lx), max(0, ly)
        sx1, sy1 = min(W, lx + bw), min(H, ly + bh)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        mm = m[sy0 - ly:sy1 - ly, sx0 - lx:sx1 - lx]
        vis = (st.depth[sy0:sy1, sx0:sx1] > t).astype(np.float32)
        a = (mm * vis)[..., None]
        fogk = 1 - np.exp(-t / st.fog)
        col = np.array(sp.get("col", (0.11, 0.1, 0.09)), np.float32) * (1 - fogk) + np.array([0.74, 0.73, 0.7]) * fogk
        img[sy0:sy1, sx0:sx1] = img[sy0:sy1, sx0:sx1] * (1 - a) + col * a
    return img


def draw_beam(img, st, p0, p1, width, col=(0.1, 0.09, 0.08), steps=24):
    """Thick 3D beam (e.g. a fallen girder) projected and depth-tested coarsely."""
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    pts = []
    for i in range(steps + 1):
        a = i / steps
        X = p0[0] + (p1[0] - p0[0]) * a
        Y = p0[1] + (p1[1] - p0[1]) * a - np.sin(a * np.pi) * p0[3] if len(p0) > 3 else p0[1] + (p1[1] - p0[1]) * a
        Z = p0[2] + (p1[2] - p0[2]) * a
        x, y, t = st.project(X, Y, Z)
        if t > 1:
            pts.append((x, y, st.f * width / t))
    for i in range(len(pts) - 1):
        d.line([pts[i][:2], pts[i + 1][:2]], fill=255, width=max(1, int(pts[i][2])))
        # lattice ticks
        if i % 2 == 0:
            d.line([pts[i][:2], (pts[i + 1][0], pts[i + 1][1] - pts[i][2] * 0.8)], fill=200, width=max(1, int(pts[i][2] * 0.2)))
    m = (np.asarray(layer, np.float32) / 255.0)[..., None]
    img[:] = img * (1 - m) + np.array(col, np.float32) * m
    return img


def draw_spray(img, st, n=140, seed=3, color_k=1.0):
    """Molasses droplets / blobs flung from the wave crest."""
    if st.wave_z is None or st.wave_z - st.camZ < 2:
        return img
    rng = np.random.default_rng(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tm = st.time
    for i in range(n):
        X = (rng.random() * 2 - 1) * WF
        life = 0.55 + rng.random() * 0.5
        ph = rng.random()
        age = ((tm / life + ph) % 1.0) * life
        vy = 8 + rng.random() * 14
        vz = -(4 + rng.random() * 10)
        hw = float(st.wave_height(np.array([X]))[0])
        Y = hw - 1 + vy * age - 16 * age * age
        Z = st.wave_z - 1 + vz * age
        if Y < 0:
            continue
        t = Z - st.camZ
        if t < 1.5:
            continue
        x, y, _ = st.project(X, Y, Z)
        r = min(12.0, st.f * (0.07 + rng.random() ** 2 * 0.3) / t)
        if not (-50 < x < W + 50 and -50 < y < H + 50):
            continue
        if st.depth[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] < t:
            continue
        hi = rng.random() < 0.2
        c = (int(255 * 0.55 * color_k), int(255 * 0.3 * color_k), int(255 * 0.1 * color_k), 220) if hi else (20, 10, 4, 235)
        stretch = 1 + min(3.0, abs(vy - 32 * age) * 0.12)
        d.ellipse([x - r, y - r * stretch, x + r, y + r], fill=c)
    arr = np.asarray(layer, np.float32) / 255.0
    a = arr[..., 3:4]
    img[:] = img * (1 - a) + arr[..., :3] * a
    return img
