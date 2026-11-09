"""Build a rough-cut video (title cards + charts) timed to the draft voiceover.
Captions go in as a separate, switchable subtitle track. They are not burned in.

Run tools/make_voiceover.py first. Usage: python3 tools/make_roughcut.py
Output: draft/rough-cut.mp4
"""
import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_charts import SURFACE, INK, INK2, MUTED, BLUE, ORANGE

ROOT = Path(__file__).resolve().parent.parent
DRAFT, CHARTS = ROOT / "draft", ROOT / "charts"
CARDS = DRAFT / "cards"

LESSONS = {
    1: "Find the bottlenecks AI creates before everyone else prices them in.",
    2: "Research suppliers can win even when individual drugs fail. They can still be overpriced.",
    3: "Follow who supplies, deploys, and profits from automation, not just who builds robots.",
    4: "Predicting the future isn't enough. The price you pay matters enormously.",
    5: "Think beyond the breakthrough. What happens when it becomes normal?",
    6: "Look for companies that keep pricing power when production gets cheap.",
    7: "Know where profits pile up, and how much is already priced in.",
}


def card(name, kicker, title, body="", accent=BLUE):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.94, 0.94, "ROUGH CUT  ·  DRAFT VOICE", ha="right", fontsize=16, color=MUTED)
    if kicker:
        fig.text(0.08, 0.62, kicker.upper(), fontsize=26, color=accent, weight="bold")
    fig.text(0.08, 0.55, "\n".join(textwrap.wrap(title, 30)), fontsize=64, weight="bold", color=INK, va="top", linespacing=1.15)
    if body:
        fig.text(0.08, 0.22, "\n".join(textwrap.wrap(body, 60)), fontsize=30, color=INK2, va="top", linespacing=1.4)
    fig.text(0.08, 0.06, "Hypothetical scenario for education. Not a forecast or investment advice.", fontsize=17, color=MUTED)
    path = CARDS / f"{name}.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    CARDS.mkdir(parents=True, exist_ok=True)
    info = json.loads((DRAFT / "timeline.json").read_text())
    chapters, total = info["chapters"], info["duration"]
    shots = []  # (start_time, image)

    for i, (start, name) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else total
        length = end - start
        if name == "Intro":
            shots.append((start, card("intro", "A stock market story", "The Great AI Boom, 2026–2045",
                                      "Why investors who predicted the future still lost money.")))
            shots.append((start + length * 0.55, CHARTS / "00-timeline.png"))
        elif name.startswith("Chapter"):
            num = int(name.split()[1].rstrip(":"))
            label, _, years = name.split(": ", 1)[1].rpartition(" (")
            shots.append((start, card(f"ch{num}", f"Chapter {num}  ·  {years.rstrip(')')}", label,
                                      accent=ORANGE if num == 4 else BLUE)))
            if num == 4:
                shots.append((start + 6, CHARTS / "01-index-boom-and-crash.png"))
            shots.append((start + length * 0.8, card(f"ch{num}-lesson", "The lesson", LESSONS[num],
                                                     accent=ORANGE if num == 4 else BLUE)))
        elif name == "The Investing Playbook":
            shots.append((start, card("playbook", "Part two", "The Investing Playbook", "Six ideas.")))
        elif name.startswith("1."):
            shots.append((start, CHARTS / "04-chain-of-consequences.png"))
        elif name.startswith("2."):
            shots.append((start, card("pb2", "Idea 2", "A good company is not the same as a good stock")))
            shots.append((start + 5, CHARTS / "02-good-company-vs-good-stock.png"))
        elif name.startswith("3."):
            shots.append((start, card("pb3", "Idea 3", "Build an early-signal watchlist",
                                      "ETN · VRT · GLW · RXRX · TMO · DHR · TER · ABB · ROK · ILMN · ISRG · GEV · CEG · V · BLK. Companies to research, not automatic buys.")))
        elif name.startswith("4."):
            shots.append((start, CHARTS / "03-expectations-cycle.png"))
        elif name.startswith("5."):
            shots.append((start, card("pb5", "Idea 5", "If it becomes 100× more useful, what will we need 10× more of?",
                                      "Then ask: how much of that is already in the stock price?")))
        elif name.startswith("6."):
            shots.append((start, card("pb6", "Idea 6", "Look for three signals",
                                      "Bottlenecks  ·  Adoption inflection points  ·  Valuation extremes")))
        elif name == "Final Thought":
            shots.append((start, card("final", "Final thought", "The goal isn't just to find the future.",
                                      "It's to understand who captures its value, without paying a price that assumes everything goes perfectly.")))
            shots.append((max(start, total - 14), card("disclaimer", "Disclaimer", "Illustrative scenario only",
                                                       "All dates, events, and market returns are hypothetical. Not a prediction or personal investment advice.")))

    lst = DRAFT / "shots.txt"
    lines = []
    for j, (t, img) in enumerate(shots):
        nxt = shots[j + 1][0] if j + 1 < len(shots) else total
        lines += [f"file '{img}'", f"duration {nxt - t:.3f}"]
    lines.append(f"file '{shots[-1][1]}'")
    lst.write_text("\n".join(lines) + "\n")

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-i", str(DRAFT / "voiceover-draft.mp3"),
        "-i", str(DRAFT / "captions-draft.en.srt"),
        "-map", "0:v", "-map", "1:a", "-map", "2:s",
        "-vf", "fps=5,format=yuv420p", "-c:v", "libx264", "-tune", "stillimage", "-crf", "28", "-preset", "medium",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1",
        "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-disposition:s:0", "0",
        "-movflags", "+faststart", "-shortest", str(DRAFT / "rough-cut.mp4"),
    ], check=True)
    lst.unlink()
    size = (DRAFT / "rough-cut.mp4").stat().st_size / 1e6
    print(f"{len(shots)} shots, rough-cut.mp4 {size:.1f} MB")


if __name__ == "__main__":
    main()
