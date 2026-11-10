"""Synthesize the reel soundtrack: funky 120 BPM groove + SFX. Writes audio.wav."""
import numpy as np, wave, sys

SR = 48000
DUR = 30.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
BEAT = 0.5
T0 = 1.0          # groove starts here; bars land on odd seconds
DROP = 11.0       # transformation drop
END = 29.0        # final hit

music = np.zeros((N, 2))
sfx = np.zeros((N, 2))


def t_(n): return np.arange(n) / SR


def add(buf, x, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= N: return
    if x.ndim == 1:
        l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        x = np.stack([x * l, x * r], 1)
    x = x[: N - i]
    buf[i:i + len(x)] += x * gain


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if lo: m *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)
    if hi: m *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * m, len(x))


def env(n, a=0.002, d=0.2):
    t = t_(n)
    return np.minimum(t / a, 1) * np.exp(-t / d)


# ---------- drums ----------
def kick(n=int(0.45 * SR)):
    t = t_(n)
    f = 45 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t / 0.28)
    x += 0.4 * fft_filter(rng.standard_normal(n), lo=2000) * np.exp(-t / 0.004)
    return np.tanh(1.6 * x)


def snare(n=int(0.3 * SR)):
    t = t_(n)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    nz = fft_filter(rng.standard_normal(n), lo=1500, hi=9000) * np.exp(-t / 0.09)
    return 0.5 * tone + 0.8 * nz


def clap(n=int(0.3 * SR)):
    t = t_(n)
    nz = fft_filter(rng.standard_normal(n), lo=900, hi=5000)
    e = np.zeros(n)
    for k, o in enumerate([0, 0.011, 0.022]):
        e += (t >= o) * np.exp(-np.maximum(t - o, 0) / (0.008 if k < 2 else 0.12))
    return nz * e * 0.9


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = t_(n)
    return fft_filter(rng.standard_normal(n), lo=7000) * np.exp(-t / (0.08 if open_ else 0.015))


K, SN, CL, HC, HO = kick(), snare(), clap(), hat(), hat(True)

# ---------- bass (funky octave-jumping, E minor) ----------
def note_hz(m): return 440 * 2 ** ((m - 69) / 12)


def bass_note(midi, dur, slap=False):
    n = int(dur * SR)
    t = t_(n)
    f = note_hz(midi)
    ph = 2 * np.pi * f * t
    x = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph + 0.3) + 0.12 * np.sin(4 * ph)
    e = np.minimum(t / 0.004, 1) * np.exp(-t / (0.10 if slap else 0.22))
    e *= np.clip((dur - t) / 0.015, 0, 1)
    x = np.tanh(1.8 * x * e)
    if slap:
        x += 0.35 * fft_filter(rng.standard_normal(n), lo=1500, hi=6000) * np.exp(-t / 0.01)
    return x


E1, G1, A1, B1, D2, E2 = 28, 31, 33, 35, 38, 40
E1 += 12; G1 += 12; A1 += 12; B1 += 12; D2 += 12; E2 += 12  # bring into E2 range
# one-bar pattern in 16ths: (step, midi, len16, slap)
BASS_A = [(0, E1, 2, False), (3, E1, 1, False), (4, E2, 1, True), (6, E1, 1, False),
          (8, G1, 2, False), (10, A1, 1, False), (11, E2, 1, True), (14, D2, 1, False), (15, B1, 1, False)]
BASS_B = [(0, A1, 2, False), (3, A1, 1, False), (4, A1 + 12, 1, True), (6, G1, 1, False),
          (8, E1, 2, False), (10, G1, 1, False), (12, E2, 1, True), (13, D2, 1, False), (14, B1, 2, False)]

# ---------- clav / stabs ----------
def clav(midis, dur=0.12, bright=1.0):
    n = int(dur * SR)
    t = t_(n)
    x = np.zeros(n)
    for m in midis:
        f = note_hz(m)
        ph = 2 * np.pi * f * t
        x += np.sign(np.sin(ph)) * 0.5 + np.sin(ph)
    x = fft_filter(x, lo=200, hi=2500 * bright)
    return x * env(n, 0.002, 0.06) / len(midis)


EM9 = [64, 67, 71, 74, 78]   # E G B D F#
AM9 = [69, 72, 76, 79, 83]   # A C E G B
CMAJ7 = [60, 64, 67, 71]

# ---------- pad (after section) ----------
def pad(midis, dur):
    n = int(dur * SR)
    t = t_(n)
    x = np.zeros((n, 2))
    for m in midis:
        for det, ch in [(-0.08, 0), (0.08, 1)]:
            f = note_hz(m + det)
            ph = 2 * np.pi * f * t
            x[:, ch] += np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    e = np.minimum(t / 0.4, 1) * np.clip((dur - t) / 0.4, 0, 1)
    return x * e[:, None] / len(midis)


# ---------- arrangement ----------
s16 = BEAT / 4
bar = 4 * BEAT
nbars = int((END - T0) / bar)          # 14 bars: 1..29
for b in range(nbars):
    t_bar = T0 + b * bar
    sec = 'hook' if t_bar < 3 else ('before' if t_bar < DROP else ('after' if t_bar < 25 else 'finale'))
    build = (t_bar >= 9 and t_bar < DROP)
    # drums
    for beat in range(4):
        tb = t_bar + beat * BEAT
        if sec == 'hook':
            if beat in (0, 2): add(music, K, tb, 0.9)
            if beat == 3: add(music, CL, tb, 0.5, 0.1)
            continue
        if build and tb >= 10.0:
            continue
        add(music, K, tb, 0.95 if beat in (0, 2) else 0.7)
        if beat in (1, 3):
            add(music, SN, tb, 0.55, 0.05); add(music, CL, tb, 0.45, -0.1)
    if sec != 'hook':
        for st in range(16):
            tb = t_bar + st * s16
            if build and tb >= 10.0: break
            if st % 2 == 0: add(music, HC, tb, 0.22 if st % 4 else 0.3, 0.3)
            elif sec in ('after', 'finale'): add(music, HC, tb, 0.12, -0.3)
            if st in (6, 14) and sec in ('after', 'finale'): add(music, HO, tb, 0.15, 0.4)
        if sec == 'before' and b % 2 == 1 and not build:
            add(music, K, t_bar + 3.5 * BEAT, 0.6)   # syncopated ghost kick
    # bass
    pat = BASS_A if b % 2 == 0 else BASS_B
    for st, m, ln, slap in pat:
        tb = t_bar + st * s16
        if build and tb >= 10.0: break
        add(music, bass_note(m, ln * s16 * 0.95, slap), tb, 0.55)
    # clav stabs on offbeats
    ch = EM9 if b % 2 == 0 else AM9
    for st in (2, 7, 10, 15) if sec != 'hook' else (2, 10):
        tb = t_bar + st * s16
        if build and tb >= 10.0: break
        add(music, clav(ch, bright=0.7 if sec in ('hook', 'before') else 1.3), tb, 0.35, -0.35 + 0.7 * (st % 2))
    # pad in after/finale
    if sec in ('after', 'finale'):
        add(music, pad(ch[:4], bar), t_bar, 0.11)

# hook lowpass ("muffled until the beat hits harder at 3s") is done by mixing
# build: snare roll 10..11 accelerating
tb = 10.0
for k in range(16):
    add(music, SN, 10.0 + k * (1.0 / 16), 0.15 + 0.03 * k, 0.0)
# final hit at END
add(music, K, END, 1.0); add(music, CL, END, 0.6)
add(music, bass_note(E1, 1.2), END, 0.7)
add(music, pad(EM9, 1.8), END, 0.25)
add(music, clav(EM9, 0.4, 1.4), END, 0.5)

# ---------- SFX ----------
def bass_hit(dur=1.4):
    n = int(dur * SR); t = t_(n)
    f = 38 + 70 * np.exp(-t / 0.06)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5)
    x += 0.3 * fft_filter(rng.standard_normal(n), lo=1000) * np.exp(-t / 0.01)
    return np.tanh(2.2 * x)


def pop(f0=900, f1=250, dur=0.06):
    n = int(dur * SR); t = t_(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.012)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.02)


def camera_click():
    out = np.zeros(int(0.18 * SR))
    for o, g, d in [(0.0, 1.0, 0.004), (0.07, 0.8, 0.006)]:
        n = int(0.05 * SR); t = t_(n)
        c = fft_filter(rng.standard_normal(n), lo=2500) * np.exp(-t / d)
        c += 0.5 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.006)
        i = int(o * SR); out[i:i + n] += c * g
    n = int(0.12 * SR); t = t_(n)
    out[int(0.02 * SR):int(0.02 * SR) + n] += 0.3 * fft_filter(rng.standard_normal(n), lo=300, hi=1500) * np.exp(-t / 0.03)
    return out


def whoosh(dur=0.45, f0=300, f1=4000, peak=0.6):
    n = int(dur * SR)
    noise = rng.standard_normal(n + 2048)
    out = np.zeros(n + 2048)
    hop, win = 512, 2048
    w = np.hanning(win)
    for i in range(0, n, hop):
        p = i / n
        fc = f0 * (f1 / f0) ** p
        seg = noise[i:i + win] * w
        X = np.fft.rfft(seg); f = np.fft.rfftfreq(win, 1 / SR)
        m = np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) / 0.7) ** 2)
        out[i:i + win] += np.fft.irfft(X * m, win)
    out = out[:n]
    t = t_(n) / dur
    e = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 1.5)
    return out * e / (np.abs(out).max() + 1e-9)


def riser(dur=1.0):
    n = int(dur * SR); t = t_(n)
    w = whoosh(dur, 200, 8000, 0.97)
    f = 200 * (8) ** (t / dur)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / dur) ** 2 * 0.3
    return w + s


def scratch(dur=0.45):
    # pitch-wobbled clav chord played back and forth, like a record scratch
    src = np.concatenate([clav(EM9, 0.5, 1.5) * 3 + 0.3 * rng.standard_normal(int(0.5 * SR)) * np.exp(-t_(int(0.5 * SR)) / 0.2)] * 2)
    n = int(dur * SR); t = t_(n)
    rate = 2.2 * np.sin(2 * np.pi * t / dur * 2.0) * np.sign(np.sin(2 * np.pi * t / dur * 2.0 + 0.0001))
    rate = 2.5 * np.sin(2 * np.pi * (t / dur) * 2.0)
    pos = 0.15 * SR + np.cumsum(rate)
    pos = np.clip(pos, 0, len(src) - 2)
    i = pos.astype(int); fr = pos - i
    y = src[i] * (1 - fr) + src[i + 1] * fr
    y = fft_filter(y, lo=300, hi=6000)
    y *= np.minimum(np.abs(rate) / 0.6, 1)
    return y / (np.abs(y).max() + 1e-9)


def chime(notes=(96, 100, 103, 108), gap=0.05):
    out = np.zeros(int(1.4 * SR))
    for k, m in enumerate(notes):
        n = int(1.0 * SR); t = t_(n); f = note_hz(m)
        x = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.1)) * env(n, 0.001, 0.35)
        i = int(k * gap * SR); out[i:i + n] += x * (0.8 ** k)
    return out


add(sfx, bass_hit(), 0.0, 0.9)
add(sfx, pop(), 0.06, 0.45)
add(sfx, scratch(0.42), 0.56, 0.55)
add(sfx, bass_hit(0.9), 1.0, 0.55)
add(sfx, pop(1200, 400), 1.02, 0.35, 0.2)
add(sfx, pop(1400, 500), 1.55, 0.3, -0.2)
add(sfx, camera_click(), 2.5, 0.6)
add(sfx, camera_click(), 4.5, 0.55)
for k in range(4):
    add(sfx, pop(700 + 200 * k, 250 + 60 * k), 6.0 + 0.25 * k, 0.3, (-1) ** k * 0.4)
add(sfx, whoosh(0.35), 7.4, 0.35, -0.5)
add(sfx, whoosh(0.35), 7.55, 0.3, 0.5)
add(sfx, riser(1.0), 10.0, 0.75)
add(sfx, whoosh(0.3, 500, 6000, 0.8), 10.78, 0.6)
add(sfx, bass_hit(1.6), 11.0, 0.85)
add(sfx, chime(), 11.05, 0.18)
add(sfx, whoosh(0.7, 300, 3000, 0.5), 12.05, 0.3, 0.6)
add(sfx, chime((103, 108, 112)), 19.5, 0.15)
add(sfx, whoosh(0.4, 600, 5000, 0.85), 22.15, 0.35)
add(sfx, whoosh(0.8, 200, 2500, 0.4), 25.0, 0.22)
add(sfx, pop(1000, 350), 27.0, 0.35)
add(sfx, chime((100, 103, 108, 112), 0.06), 27.05, 0.15)

# ---------- mix ----------
# simple reverb for music
ir_n = int(1.2 * SR)
ir = rng.standard_normal((ir_n, 2)) * np.exp(-t_(ir_n) / 0.35)[:, None] * 0.02
ir[0] = 0
def conv(x, h):
    L = len(x) + len(h)
    nf = 1 << (L - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(h, nf), nf)[:len(x)]
wet = np.stack([conv(music[:, c], ir[:, c]) for c in range(2)], 1)
music = music + 0.6 * wet

# hook (1..3): slightly lowpassed; before (3..11): mild "flat" lowpass; after: full
def seg_lp(buf, a, b, hi):
    i, j = int(a * SR), int(b * SR)
    for c in range(2):
        buf[i:j, c] = fft_filter(buf[i:j, c], hi=hi)
seg_lp(music, 1.0, 3.0, 1800)
seg_lp(music, 3.0, 10.0, 7000)

# sidechain-ish duck on kicks after the drop
mix = music * 0.8 + sfx
# fade out tail
t = t_(N)
mix *= np.clip((DUR - t) / 0.9, 0, 1)[:, None]
mix = np.tanh(mix * 1.1) * 0.9
mix /= np.abs(mix).max() / 0.89
pcm = (mix * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok', pcm.shape)
