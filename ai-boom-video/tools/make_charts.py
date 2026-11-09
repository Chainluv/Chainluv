"""Render 1920x1080 chart frames for the video and blog (dark theme).
Usage: python3 tools/make_charts.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "charts"
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8a897f", "#33332f"
BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 22, "font.family": "DejaVu Sans",
})


def frame(title, subtitle):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.text(0.06, 0.92, title, fontsize=46, weight="bold", color=INK)
    fig.text(0.06, 0.865, subtitle, fontsize=24, color=INK2)
    fig.text(0.06, 0.04, "Hypothetical scenario for education. Not a forecast or investment advice.",
             fontsize=17, color=MUTED)
    return fig


def style_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(length=0, pad=10)
    ax.set_axisbelow(True)


def index_chart():
    years = list(range(2026, 2041))
    vals = [100, 135, 175, 230, 265, 310, 365, 420, 231, 245, 275, 305, 335, 370, 405]
    fig = frame("The crash nobody expected", "Hypothetical AI growth-stock index, 2026 = 100")
    ax = fig.add_axes([0.08, 0.14, 0.86, 0.66])
    style_axes(ax)
    ax.plot(years, vals, color=BLUE, linewidth=4, solid_capstyle="round")
    for (x, y, label, dy) in [(2026, 100, "100", -38), (2033, 420, "Peak: 420", 22), (2034, 231, "231  (−45%)", -40)]:
        ax.plot(x, y, "o", color=BLUE, markersize=14, markeredgecolor=SURFACE, markeredgewidth=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(-20 if label.startswith("Peak") else 0, dy), ha="right" if label.startswith("Peak") else "center",
                    fontsize=26, weight="bold", color=INK)
    ax.axvspan(2033, 2035, color=ORANGE, alpha=0.12, lw=0)
    ax.text(2034.1, 485, "Expectations fail", ha="center", fontsize=22, color=INK2)
    ax.set_ylim(0, 500)
    ax.set_xlim(2025.5, 2040.5)
    ax.set_xticks(range(2026, 2041, 2))
    fig.savefig(OUT / "01-index-boom-and-crash.png")
    plt.close(fig)


def pe_chart():
    g = 1.25
    years = list(range(6))
    def path(start_pe, end_pe=25):
        return [100 * (g ** t) * (start_pe + (end_pe - start_pe) * t / 5) / start_pe for t in years]
    a, b = path(80), path(20)
    fig = frame("Same company results. Opposite returns.",
                "Both grow earnings 25%/yr for 5 years and end at a P/E of 25. Stock price, start = 100")
    ax = fig.add_axes([0.08, 0.14, 0.64, 0.66])
    style_axes(ax)
    ax.plot(years, b, color=BLUE, linewidth=4)
    ax.plot(years, a, color=ORANGE, linewidth=4)
    ax.axhline(100, color=MUTED, linewidth=1.5, linestyle=(0, (4, 4)))
    for series, color, name, ret in [(b, BLUE, "Company B: P/E 20 → 25", b[-1] / 100 - 1),
                                     (a, ORANGE, "Company A: P/E 80 → 25", a[-1] / 100 - 1)]:
        ax.plot(5, series[-1], "o", color=color, markersize=14, markeredgecolor=SURFACE, markeredgewidth=3)
        ax.annotate(f"{name}\n{ret:+.0%}", (5, series[-1]), textcoords="offset points",
                    xytext=(22, -14), fontsize=24, weight="bold", color=INK, annotation_clip=False)
    ax.set_xticks(years, ["Start", "Yr 1", "Yr 2", "Yr 3", "Yr 4", "Yr 5"])
    ax.set_ylim(0, 420)
    fig.savefig(OUT / "02-good-company-vs-good-stock.png")
    plt.close(fig)


def cycle_chart():
    import numpy as np
    x = np.linspace(0, 10, 400)
    y = 1.8 * np.exp(-((x - 3.4) ** 2) / 1.6) + 0.12 * x + 0.3
    # (name, x position, label offset in points, horizontal alignment)
    stages = [("Discovery", 0.8, (-10, 24), "right"), ("Optimism", 2.2, (-24, 0), "right"),
              ("Euphoria", 3.4, (0, 26), "center"), ("Disappointment", 4.9, (24, 0), "left"),
              ("Despair", 6.2, (0, -50), "center"), ("Adoption", 9.0, (0, 26), "center")]
    fig = frame("The psychology of expectations", "A conceptual cycle, not a guaranteed sequence")
    ax = fig.add_axes([0.10, 0.12, 0.84, 0.68])
    ax.axis("off")
    ax.plot(x, y, color=VIOLET, linewidth=5)
    for name, sx, offset, ha in stages:
        sy = float(np.interp(sx, x, y))
        ax.plot(sx, sy, "o", color=VIOLET, markersize=18, markeredgecolor=SURFACE, markeredgewidth=4)
        ax.annotate(name, (sx, sy), textcoords="offset points", xytext=offset, ha=ha, va="center",
                    fontsize=26, weight="bold", color=INK)
    ax.text(0, -0.1, "Time →", fontsize=20, color=MUTED)
    ax.set_ylim(-0.2, 2.8)
    fig.savefig(OUT / "03-expectations-cycle.png")
    plt.close(fig)


def chain_chart():
    stages = [("Stage 1", "Compute", "Chips, electricity,\ndata centers"),
              ("Stage 2", "Scientific tools", "Sequencing,\nlab automation"),
              ("Stage 3", "Clinical", "Trials, drug developers,\ndiagnostics"),
              ("Stage 4", "Longevity", "Insurance, retirement,\nconsumer spending")]
    fig = frame("Follow the chain of consequences", "Example: AI helps cure cancer")
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.66])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    for i, (tag, head, body) in enumerate(stages):
        x0 = i + 0.08
        ax.add_patch(FancyBboxPatch((x0, 0.2), 0.78, 0.6, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="#262624", edgecolor=BLUE, linewidth=2.5))
        ax.text(x0 + 0.39, 0.7, tag, ha="center", fontsize=20, color=INK2)
        ax.text(x0 + 0.39, 0.58, head, ha="center", fontsize=28, weight="bold", color=INK)
        ax.text(x0 + 0.39, 0.38, body, ha="center", va="center", fontsize=20, color=INK2, linespacing=1.5)
        if i < 3:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(x0 + 0.8, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2.5, mutation_scale=28))
    ax.text(2, 0.06, "Each stage may get priced in before the next one becomes obvious.",
            ha="center", fontsize=22, color=INK2)
    fig.savefig(OUT / "04-chain-of-consequences.png")
    plt.close(fig)


def timeline_chart():
    chapters = [("2026–28", "Infrastructure\ngold rush"), ("2028–30", "First AI biotech\nbreakthrough"),
                ("2030–33", "Robots leave\nthe factory"), ("2033–35", "The great\nAI crash"),
                ("2035–38", "Cancer becomes\ntreatable"), ("2038–42", "Age of\nabundance"),
                ("2042–45", "The market\nin 2045")]
    fig = frame("The Great AI Boom, 2026–2045", "Seven chapters of a hypothetical future")
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.66])
    ax.axis("off")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 1)
    ax.plot([0, 6], [0.5, 0.5], color=GRID, linewidth=4, zorder=1)
    for i, (yrs, name) in enumerate(chapters):
        color = ORANGE if i == 3 else BLUE
        ax.plot(i, 0.5, "o", color=color, markersize=26, markeredgecolor=SURFACE, markeredgewidth=4, zorder=2)
        ax.text(i, 0.5, str(i + 1), ha="center", va="center", fontsize=16, weight="bold", color=SURFACE, zorder=3)
        ax.text(i, 0.64, yrs, ha="center", fontsize=22, weight="bold", color=INK)
        ax.text(i, 0.36, name, ha="center", va="top", fontsize=20, color=INK2, linespacing=1.4)
    fig.savefig(OUT / "00-timeline.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for fn in (timeline_chart, index_chart, pe_chart, cycle_chart, chain_chart):
        fn()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))
