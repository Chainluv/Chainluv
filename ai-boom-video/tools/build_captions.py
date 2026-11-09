"""Build sidecar caption files (SRT + VTT), a plain transcript, and YouTube
chapter timestamps from narration.txt.

Timings are estimated from speaking pace. After you record the voiceover,
either upload transcript.txt to YouTube and let it auto-sync the timings,
or re-time the .srt in your editor.

Usage: python3 tools/build_captions.py [words_per_minute]
"""
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WPM = 150  # overridden by the command-line argument in main()
MAX_LINE = 42          # chars per caption line (Netflix/YouTube-friendly)
MAX_LINES = 2
SENTENCE_PAUSE = 0.25  # seconds
PARAGRAPH_PAUSE = 0.6
CHAPTER_PAUSE = 1.5


def chunk_sentence(sentence):
    """Split a sentence into caption-sized pieces (<= 2 lines of 42 chars)."""
    limit = MAX_LINE * MAX_LINES
    if len(sentence) <= limit:
        return [sentence]
    # Prefer breaking at commas / colons, then fall back to words.
    parts = re.split(r"(?<=[,:;])\s+", sentence)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= limit:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                chunks.append(cur)
            while len(p) > limit:
                cut = p.rfind(" ", 0, limit)
                chunks.append(p[:cut])
                p = p[cut + 1:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def fmt(t, sep):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}{sep}{int(round((s % 1) * 1000)) % 1000:03}"


def main():
    global WPM
    if len(sys.argv) > 1:
        WPM = float(sys.argv[1])
    lines = (ROOT / "narration.txt").read_text().splitlines()
    t, cues, chapters, transcript = 0.0, [], [], []
    for line in lines:
        line = line.strip()
        if not line:
            t += PARAGRAPH_PAUSE
            continue
        if line.startswith("## "):
            if cues:
                t += CHAPTER_PAUSE
            chapters.append((t, line[3:]))
            transcript.append("")
            continue
        transcript.append(line)
        for sentence in re.split(r"(?<=[.?!])\s+", line):
            for chunk in chunk_sentence(sentence):
                words = len(chunk.split())
                dur = max(1.2, words / WPM * 60)
                text = "\n".join(textwrap.wrap(chunk, MAX_LINE))
                cues.append((t, t + dur, text))
                t += dur
            t += SENTENCE_PAUSE

    srt = [f"{i}\n{fmt(a, ',')} --> {fmt(b, ',')}\n{txt}\n" for i, (a, b, txt) in enumerate(cues, 1)]
    vtt = ["WEBVTT\n"] + [f"{fmt(a, '.')} --> {fmt(b, '.')}\n{txt}\n" for a, b, txt in cues]
    out = ROOT / "captions"
    (out / "captions.en.srt").write_text("\n".join(srt))
    (out / "captions.en.vtt").write_text("\n".join(vtt))
    (out / "transcript.txt").write_text("\n".join(transcript).strip() + "\n")

    def ts(sec):
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02}"
    chap_txt = "\n".join(f"{ts(a)} {name}" for a, name in chapters)
    (out / "youtube-chapters.txt").write_text(chap_txt + "\n")
    print(f"{len(cues)} cues, est. runtime {ts(t)} at {WPM:.0f} wpm")
    print(chap_txt)


if __name__ == "__main__":
    main()
