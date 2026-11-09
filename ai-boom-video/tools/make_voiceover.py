"""Make a rough-draft AI voiceover from narration.txt with Piper (offline TTS),
plus captions timed exactly to that draft audio.

Draft only: the free Piper voices are licensed for non-commercial use.
Replace with your own voice (or a commercially licensed AI voice) before publishing.

Usage: python3 tools/make_voiceover.py path/to/voice.onnx
Outputs (in draft/): voiceover-draft.mp3, captions-draft.en.srt/.vtt, chapters-draft.txt, timeline.json
"""
import json
import re
import subprocess
import sys
import textwrap
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

sys.path.insert(0, str(Path(__file__).parent))
from build_captions import chunk_sentence, fmt, MAX_LINE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "draft"
SENTENCE_PAUSE, PARAGRAPH_PAUSE, CHAPTER_PAUSE = 0.30, 0.55, 1.4


def speakable(text):
    """Rewrite text so the TTS reads it naturally. Captions keep the original."""
    text = re.sub(r"\b20(\d\d)\b", lambda m: "twenty " + ("oh " if m.group(1)[0] == "0" else "") + str(int(m.group(1))), text)
    text = re.sub(r"\bAI\b", "A.I.", text)
    text = text.replace("P E", "P.E.")
    return text


def main(model):
    voice = PiperVoice.load(model)
    cfg = SynthesisConfig(length_scale=1.0)
    rate = voice.config.sample_rate
    OUT.mkdir(exist_ok=True)
    pcm, t = bytearray(), 0.0
    cues, chapters = [], []

    def silence(sec):
        nonlocal t
        pcm.extend(b"\x00\x00" * int(rate * sec))
        t += sec

    for line in (ROOT / "narration.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            silence(PARAGRAPH_PAUSE)
            continue
        if line.startswith("## "):
            if cues:
                silence(CHAPTER_PAUSE)
            chapters.append((t, line[3:]))
            continue
        for sentence in re.split(r"(?<=[.?!])\s+", line):
            for chunk in chunk_sentence(sentence):
                audio = b"".join(c.audio_int16_bytes for c in voice.synthesize(speakable(chunk), cfg))
                dur = len(audio) / 2 / rate
                cues.append((t, t + dur, "\n".join(textwrap.wrap(chunk, MAX_LINE))))
                pcm.extend(audio)
                t += dur
            silence(SENTENCE_PAUSE)

    wav = OUT / "voiceover-draft.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(bytes(pcm))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "64k", str(OUT / "voiceover-draft.mp3")], check=True)
    wav.unlink()

    (OUT / "captions-draft.en.srt").write_text("\n".join(
        f"{i}\n{fmt(a, ',')} --> {fmt(b, ',')}\n{x}\n" for i, (a, b, x) in enumerate(cues, 1)))
    (OUT / "captions-draft.en.vtt").write_text("\n".join(
        ["WEBVTT\n"] + [f"{fmt(a, '.')} --> {fmt(b, '.')}\n{x}\n" for a, b, x in cues]))
    ts = lambda s: f"{int(s) // 60}:{int(s) % 60:02}"
    (OUT / "chapters-draft.txt").write_text("\n".join(f"{ts(a)} {n}" for a, n in chapters) + "\n")
    (OUT / "timeline.json").write_text(json.dumps({"duration": t, "chapters": chapters}, indent=1))
    print(f"{len(cues)} cues, runtime {ts(t)}")


if __name__ == "__main__":
    main(sys.argv[1])
