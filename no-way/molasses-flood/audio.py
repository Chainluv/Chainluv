"""Synthesized score + sound design + narration mix (48 kHz stereo WAV)."""
import sys
import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000
DUR = 20.0
N = int(SR * DUR)
rng = np.random.default_rng(42)
NARR = {"l1": 0.15, "l2": 3.55, "l3": 9.35, "l4": 16.35, "l5": 18.7}  # l5 = "No way." brand tag


def T(n):
    return np.arange(n) / SR


def filt(x, kind, f, order=4):
    sos = signal.butter(order, f, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def noise(n):
    return rng.standard_normal(n)


def brown(n):
    b = np.cumsum(noise(n))
    b = filt(b, "highpass", 20, 2)
    return b / (np.abs(b).max() + 1e-9)


def env_adsr(n, a, r, curve=1.0):
    t = T(n)
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / max(r, 1e-4)) ** curve
    return e


def place(buf, x, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if x.ndim == 1:
        l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        x = np.stack([x * l, x * r], 1) * 1.414
    if i < 0:
        x = x[-i:]
        i = 0
    j = min(N, i + len(x))
    if j > i:
        buf[i:j] += x[:j - i] * gain


def reverb(x, secs=2.5, wet=0.3, damp=3000):
    n = int(secs * SR)
    ir = np.stack([noise(n), noise(n)], 1) * np.exp(-T(n) * 6.9 / secs)[:, None]
    ir = filt(ir, "lowpass", damp, 2)
    ir[: int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))[:, None]
    ir /= np.sqrt((ir ** 2).sum(0))
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    w = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    return x * (1 - wet) + w * wet


def saw(f, n, detune=0.0):
    t = T(n)
    ph = np.cumsum(np.full(n, f) * (1 + detune)) / SR if np.isscalar(f) else np.cumsum(f) / SR
    return 2 * (ph % 1.0) - 1


def sine_sweep(f0, f1, n, curve="exp"):
    if curve == "exp":
        f = f0 * (f1 / f0) ** (np.arange(n) / n)
    else:
        f = np.linspace(f0, f1, n)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


# ------------------------------------------------------------- music
def drone(t0, t1, notes, cutoff=700, gain=1.0, swell=1.0):
    n = int((t1 - t0) * SR)
    x = np.zeros(n)
    for f in notes:
        for d in (-0.004, 0.0, 0.0045):
            x += saw(f, n, d)
    x = filt(x, "lowpass", cutoff, 2)
    t = T(n)
    e = np.minimum(1, t / swell) * np.minimum(1, (t1 - t0 - t) / 0.6)
    x *= e * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * t))
    return x / (np.abs(x).max() + 1e-9) * gain


def braam(dur=3.0, root=36.7):
    n = int(dur * SR)
    t = T(n)
    x = np.zeros(n)
    for m in (1, 2, 3, 2 * 1.189, 4):  # root, octave, fifth, minor third, 2 oct
        for d in (-0.006, 0.006):
            x += saw(root * m, n, d) / m ** 0.3
    cut = 180 + 1800 * np.exp(-t / 0.5)
    # time-varying lowpass via blocks
    out = np.zeros(n)
    blk = 2048
    zi = None
    for i in range(0, n, blk):
        sos = signal.butter(2, cut[i], "lowpass", fs=SR, output="sos")
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        out[i:i + blk], zi = signal.sosfilt(sos, x[i:i + blk], zi=zi)
    out = np.tanh(out * 1.8)
    out *= env_adsr(n, 0.02, dur * 0.4)
    return out / np.abs(out).max()


def boom(dur=2.5, f0=90, f1=28):
    n = int(dur * SR)
    s = sine_sweep(f0, f1, n) * env_adsr(n, 0.005, 0.7)
    nz = filt(noise(n), "lowpass", 400) * env_adsr(n, 0.002, 0.25) * 0.8
    click = filt(noise(n), "highpass", 2000) * env_adsr(n, 0.0005, 0.015) * 0.6
    x = np.tanh((s + nz + click) * 1.5)
    return x / np.abs(x).max()


def piano(f, dur=3.5):
    n = int(dur * SR)
    t = T(n)
    x = np.zeros(n)
    for k, a in enumerate([1, 0.5, 0.3, 0.18, 0.1, 0.06]):
        fk = f * (k + 1) * (1 + 0.0004 * k * k)
        x += a * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.2 + k * 0.9))
    x += filt(noise(n), "bandpass", [800, 4000]) * env_adsr(n, 0.001, 0.01) * 0.2
    return x * np.minimum(1, t / 0.003)


def bell(f0=196, dur=5.0):
    n = int(dur * SR)
    t = T(n)
    x = np.zeros(n)
    for r, a, d in [(0.5, 0.6, 3.5), (1, 1, 2.6), (1.19, 0.7, 2.0), (1.5, 0.5, 1.6), (2.0, 0.45, 1.3), (2.51, 0.3, 0.9), (3.01, 0.2, 0.7)]:
        x += a * np.sin(2 * np.pi * f0 * r * t + rng.random() * 6) * np.exp(-t / d)
    x += filt(noise(n), "bandpass", [500, 3000]) * env_adsr(n, 0.001, 0.02) * 0.5
    return x / np.abs(x).max()


def heartbeat(t0, t1):
    out = []
    t = t0
    while t < t1:
        p = (t - t0) / (t1 - t0)
        out.append(t)
        out.append(t + 0.16)
        t += 0.62 - 0.26 * p
    return out


def thump():
    n = int(0.35 * SR)
    x = sine_sweep(70, 38, n) * env_adsr(n, 0.004, 0.09)
    return np.tanh(x * 2)


def tremolo_strings(t0, t1, notes):
    n = int((t1 - t0) * SR)
    t = T(n)
    x = np.zeros(n)
    for f in notes:
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
        x += saw(f * vib, n)
    x = filt(x, "bandpass", [300, 3500], 2)
    trem = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 13 * t)) * 0.5
    e = (t / (t1 - t0)) ** 1.6
    return x * trem * e / (np.abs(x).max() + 1e-9)


# ------------------------------------------------------------- sfx
def roar(dur, lo=150, hi=2500, mod=0.6):
    n = int(dur * SR)
    x = filt(noise(n), "bandpass", [lo, hi], 2) * 0.6 + brown(n) * 1.2
    t = T(n)
    m = 1 + mod * (filt(noise(n), "lowpass", 3, 1) * 20).clip(-1, 1)
    return x * m / (np.abs(x).max() + 1e-9)


def gurgles(dur, rate=14):
    n = int(dur * SR)
    x = np.zeros(n)
    for _ in range(int(dur * rate)):
        L = int(rng.uniform(0.03, 0.09) * SR)
        f0 = rng.uniform(180, 520)
        g = sine_sweep(f0, f0 * rng.uniform(1.3, 2.2), L) * np.hanning(L)
        i = rng.integers(0, n - L)
        x[i:i + L] += g * rng.uniform(0.3, 1)
    return x / (np.abs(x).max() + 1e-9)


def creak(dur, f0=420, rate0=25, rate1=60):
    n = int(dur * SR)
    t = T(n)
    rate = rate0 + (rate1 - rate0) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + rng.random() * 6))
    ph = np.cumsum(rate * (1 + 0.15 * noise(n).clip(-2, 2))) / SR
    imp = (np.diff(np.floor(ph), prepend=0) > 0).astype(float)
    x = np.zeros(n)
    for fr in (f0, f0 * 1.63, f0 * 2.4):
        b, a = signal.iirpeak(fr, 25, fs=SR)
        x += signal.lfilter(b, a, imp)
    e = np.sin(np.pi * t / dur) ** 0.7
    return x * e / (np.abs(x).max() + 1e-9)


def groan(dur):
    n = int(dur * SR)
    t = T(n)
    x = np.zeros(n)
    for f, a in [(55, 1), (89, 0.7), (143, 0.5), (231, 0.35)]:
        glide = f * (1 + 0.08 * np.sin(2 * np.pi * 0.3 * t + rng.random() * 6))
        x += a * saw(glide, n)
    x = filt(x, "lowpass", 500, 2)
    e = np.sin(np.pi * t / dur) ** 1.5
    return x * e / (np.abs(x).max() + 1e-9)


def pop():
    n = int(0.25 * SR)
    t = T(n)
    x = filt(noise(n), "highpass", 1500) * env_adsr(n, 0.0003, 0.008) * 1.5
    x += np.sin(2 * np.pi * rng.uniform(2500, 4200) * t) * np.exp(-t / 0.05) * 0.5
    x += sine_sweep(180, 60, n) * env_adsr(n, 0.001, 0.03) * 0.6
    return x


def clang(dur=3.0):
    n = int(dur * SR)
    t = T(n)
    x = np.zeros(n)
    for _ in range(14):
        f = rng.uniform(180, 2600)
        x += np.sin(2 * np.pi * f * t) * np.exp(-t / rng.uniform(0.3, 1.6)) * rng.uniform(0.2, 1)
    return x / np.abs(x).max()


def snap():
    n = int(0.4 * SR)
    x = np.zeros(n)
    for k in range(rng.integers(3, 7)):
        i = int(rng.uniform(0, 0.12) * SR)
        L = int(0.03 * SR)
        x[i:i + L] += filt(noise(L), "bandpass", [900, 5000]) * np.exp(-T(L) / 0.006) * rng.uniform(0.4, 1)
    x += filt(noise(n), "bandpass", [200, 1200]) * env_adsr(n, 0.001, 0.08) * 0.4
    return x / np.abs(x).max()


def whoosh(dur=0.6, f0=300, f1=4000, rev=False):
    n = int(dur * SR)
    t = T(n)
    x = noise(n)
    out = np.zeros(n)
    blk = 1024
    zi = None
    for i in range(0, n, blk):
        p = i / n
        fc = f0 * (f1 / f0) ** p if not rev else f1 * (f0 / f1) ** p
        sos = signal.butter(2, [fc * 0.7, min(fc * 1.4, 20000)], "bandpass", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[i:i + blk], zi = signal.sosfilt(sos, x[i:i + blk], zi=zi)
    e = np.sin(np.pi * t / dur) ** 2
    return out * e / (np.abs(out).max() + 1e-9)


def paper_slap():
    n = int(0.35 * SR)
    x = filt(noise(n), "bandpass", [200, 6000]) * env_adsr(n, 0.001, 0.04)
    x += filt(noise(n), "bandpass", [2000, 8000]) * env_adsr(n, 0.01, 0.12) * 0.4  # rustle
    return x / np.abs(x).max()


def glitch(dur=0.25):
    n = int(dur * SR)
    x = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.uniform(0.008, 0.04) * SR)
        kind = rng.integers(0, 3)
        if kind == 0:
            seg = np.sign(np.sin(2 * np.pi * rng.uniform(200, 2500) * T(L)))
        elif kind == 1:
            seg = np.round(noise(L) * 2) / 2
        else:
            seg = np.zeros(L)
        x[i:i + L] = seg[: n - i] * rng.uniform(0.3, 1)
        i += L
    return filt(x, "highpass", 150)


def projector(dur):
    n = int(dur * SR)
    t = T(n)
    clicks = np.zeros(n)
    for k in np.arange(0, dur, 1 / 18):
        i = int(k * SR)
        L = int(0.006 * SR)
        if i + L < n:
            clicks[i:i + L] += filt(noise(L), "bandpass", [1500, 6000]) * np.exp(-T(L) / 0.0015)
    hum = np.sin(2 * np.pi * 60 * t) * 0.15 + np.sin(2 * np.pi * 120 * t) * 0.08
    return clicks * 0.6 + hum


def wind(dur):
    n = int(dur * SR)
    x = filt(noise(n), "bandpass", [250, 1400], 2)
    m = 0.5 + 0.5 * np.clip(filt(noise(n), "lowpass", 0.6, 1) * 60, -1, 1)
    return x * m / (np.abs(x).max() + 1e-9)


# ------------------------------------------------------------- build
def build(voice_dir, out):
    music = np.zeros((N, 2))
    sfx = np.zeros((N, 2))
    D2, A2, D3, F3, A3 = 73.42, 110.0, 146.83, 174.61, 220.0

    # --- score
    place(music, drone(0.0, 3.0, [D2, A2, D3], 500, 0.5, 0.2), 0.0)
    place(music, drone(3.0, 9.1, [D2, A2, D3, F3], 650, 0.55, 0.4), 3.0)
    place(music, tremolo_strings(3.0, 9.0, [587.3, 622.3, 880.0]) * 0.28, 3.0, pan=0.2)
    for tb in heartbeat(3.1, 8.9):
        place(music, thump() * 0.8, tb)
    place(music, braam(3.2), 8.98, 0.9)
    place(music, drone(9.0, 16.3, [36.7, D2, F3 / 2, A2], 450, 0.5, 1.0), 9.0)
    place(music, tremolo_strings(11.7, 16.0, [293.7, 349.2, 440.0]) * 0.18, 11.7, pan=-0.2)
    for tn, f in [(16.15, 293.66), (16.9, 349.23), (17.55, 440.0), (18.1, 329.63)]:
        place(music, piano(f) * 0.35, tn, pan=rng.uniform(-0.3, 0.3))
    place(music, drone(16.0, 18.5, [D2, A2], 400, 0.3, 0.5), 16.0)
    place(music, whoosh(0.9, 200, 9000) * 0.35, 17.6)
    place(music, boom(2.0) * 0.9, 18.45)
    place(music, drone(18.45, 20.0, [36.7, D2, A2, F3], 380, 0.4, 0.05), 18.45)
    for tb in (0.12, 11.75, 13.62):
        place(music, boom(1.8, 80, 30) * (0.9 if tb < 1 else 0.5), tb)
    music = reverb(music, 2.8, 0.28)

    # --- sound design
    place(sfx, wind(20.0) * 0.12, 0.0)
    r = roar(2.95, 120, 2000) * np.linspace(0.15, 1.0, int(2.95 * SR)) ** 1.4
    place(sfx, r * 0.75, 0.0)
    place(sfx, gurgles(2.9, 10) * np.linspace(0.1, 1, int(2.9 * SR)) * 0.3, 0.05)
    u = filt(roar(0.45, 80, 600), "lowpass", 350)  # engulfed: muffled
    place(sfx, u * np.linspace(1, 0, len(u)) * 0.6, 2.92)
    place(sfx, whoosh(0.35, 3000, 200) * 0.35, 2.82)
    # tank stress
    place(sfx, groan(2.2) * 0.55, 3.05)
    place(sfx, creak(1.4, 380) * 0.35, 3.4, pan=-0.4)
    place(sfx, creak(1.2, 520, 30, 80) * 0.3, 4.2, pan=0.4)
    place(sfx, groan(1.3) * 0.6, 3.9)
    import tank as _tank  # rivet pop timing matches the visuals
    for rv in sorted(_tank.RIVETS, key=lambda r: r["t0"]):
        if rv["t0"] < 5.1:
            place(sfx, pop() * 0.45, rv["t0"], pan=float(np.clip(rv["u"], -0.8, 0.8)))
    # burst
    place(sfx, boom(3.0, 110, 25) * 1.0, 5.1)
    place(sfx, clang(3.0) * 0.4, 5.1)
    place(sfx, roar(1.0, 80, 4000) * env_adsr(int(1.0 * SR), 0.005, 0.4) * 0.8, 5.1)
    # flood
    fr = roar(3.3, 150, 3000)
    fe = np.minimum(1, T(len(fr)) / 0.15) * np.linspace(0.75, 1.0, len(fr))
    place(sfx, fr * fe * 0.8, 5.65)
    place(sfx, gurgles(3.2, 20) * 0.35, 5.7)
    for ts in (6.35, 7.25, 8.05, 8.5):
        place(sfx, snap() * 0.45, ts, pan=rng.uniform(-0.6, 0.6))
    place(sfx, whoosh(0.4, 4000, 300) * 0.4, 8.6)
    # newspapers
    place(sfx, whoosh(0.45, 600, 7000) * 0.4, 8.92)
    place(sfx, paper_slap() * 0.55, 9.32)
    place(sfx, whoosh(0.4, 700, 7000) * 0.35, 10.5)
    place(sfx, paper_slap() * 0.5, 10.88)
    place(sfx, whoosh(0.5, 300, 3000) * 0.25, 11.6)
    place(sfx, wind(4.5) * 0.18, 11.75)
    place(sfx, whoosh(0.5, 300, 3000) * 0.25, 13.45)
    b = reverb(bell(196, 5.0), 3.5, 0.45)
    place(sfx, b * 0.28, 13.7, pan=-0.2)
    place(sfx, reverb(bell(185, 4.0), 3.5, 0.5) * 0.18, 16.2, pan=0.2)
    place(sfx, projector(2.4) * 0.12, 16.0)
    place(sfx, glitch(0.22) * 0.35, 18.24)
    place(sfx, glitch(0.12) * 0.25, 19.4)
    place(sfx, glitch(0.08) * 0.2, 19.85)
    sfx = reverb(sfx, 1.2, 0.12)

    # --- narration
    voice = np.zeros((N, 2))
    for k, t0 in NARR.items():
        v, sr = sf.read(f"{voice_dir}/{k}.wav")
        v = signal.resample_poly(v, SR, sr)
        v = filt(v, "highpass", 70, 2)
        v = v + filt(v, "bandpass", [2500, 5000], 2) * 0.35  # presence
        v = np.tanh(v * 2.2) / np.tanh(2.2)
        place(voice, v, t0)
    voice = reverb(voice, 0.9, 0.08, 5000)
    # ducking
    ve = np.abs(voice).max(1)
    ve = filt(ve, "lowpass", 4, 1)
    ve = np.clip(ve / (ve.max() + 1e-9) * 3, 0, 1)
    duck = 1 - 0.5 * ve
    bed = music / (np.abs(music).max() + 1e-9) * 0.42 + sfx / (np.abs(sfx).max() + 1e-9) * 0.55
    mixd = bed * duck[:, None] + voice / (np.abs(voice).max() + 1e-9) * 0.95
    # fade-out tail
    tail = int(0.35 * SR)
    mixd[-tail:] *= np.linspace(1, 0, tail)[:, None]
    mixd = np.tanh(mixd * 1.1) * 0.9
    sf.write(out, mixd.astype(np.float32), SR, subtype="FLOAT")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
