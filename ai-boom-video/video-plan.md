# Video Production Plan

Target: ~14-minute long-form YouTube video, 1920×1080, dark "documentary explainer" look.
Narration script: `narration.txt`. Captions: `captions/` (separate files, **not burned in**).

## 1. Where to get assets

**Use these, in this order:**

| Need | Source | Cost | Notes |
|---|---|---|---|
| Charts & diagrams | `charts/` (made for you, re-render with `tools/make_charts.py`) | Free | Already 1920×1080 |
| Stock B-roll (data centers, labs, robots, cities) | Pexels Videos, Pixabay Videos, Mixkit | Free | Free for YouTube, no attribution required (credit anyway) |
| Higher-end B-roll | Storyblocks, Envato Elements, Artgrid | ~$15–30/mo | Unlimited downloads; best value for long-form |
| "Future" scenes (2040 cities, humanoid robots, AI labs) | AI video/image generators (Runway, Kling, Sora, Midjourney, Ideogram) | Paid tiers | Check each tool's commercial-use terms; label as AI-generated in YouTube Studio if realistic |
| Company logos / tickers | Don't use logos; show **ticker text** on a lower-third | Free | Avoids trademark issues and looks cleaner |
| Real stock charts (for intros) | TradingView screenshots | Free tier | TradingView allows screenshots with their branding visible |
| Music | YouTube Audio Library, Pixabay Music, Epidemic Sound / Artlist | Free / paid | Low, ambient, "tech documentary" tracks under voice |
| Sound effects | YouTube Audio Library, Pixabay SFX, freesound.org (check license per file) | Free | Whooshes for chapter cards, a low hit for the crash |

**Search terms that work on stock sites:** "server room", "data center aisle", "power lines aerial", "electrical substation", "fiber optic cable", "laboratory pipette", "DNA sequencing", "robot arm factory", "warehouse automation", "stock ticker screen", "trading floor", "city skyline night timelapse", "elderly couple hiking", "solar farm aerial".

**Rules of thumb**
- Keep a `credits.txt` as you download (file name → source → license). Paste it in the description.
- Avoid footage with visible brand logos or recognizable faces from unknown sources.
- Change the shot every 4–8 seconds. That's ~150–200 clips for 14 minutes, so lean on charts and text cards to cut the B-roll count.

## 2. Shot list by chapter

Times are estimates at 150 words/min; final times come from your recording.

| Time | Section | Visuals |
|---|---|---|
| 0:00 | Intro | Future city timelapse → hospital/lab → robot arms → solar farm → stock ticker. Text card: **"The market rewards *better than expected*."** End on disclaimer card. |
| 1:01 | Ch 1 Infrastructure | Server racks, cooling fans, substations, cable spools. Lower-third tickers: ETN, VRT, GLW, GEV. Text card: "Who supplies the AI economy?" |
| 2:25 | Ch 2 Biotech | Lab footage, a fake headline card "+70% in one day" (clearly labeled hypothetical), then red "−40% to −60%" card. Tickers: TMO, DHR, ILMN, RXRX |
| 3:37 | Ch 3 Robots | Robot arms, warehouse bots, humanoid robot (AI-generated), conveyor belts. Tickers: ROK, ABB, TER, CGNX. Brief shot of workers for the social-impact line |
| 4:56 | Ch 4 Crash | **`charts/01-index-boom-and-crash.png`** (animate the line drawing on, then the drop). Red-tinted trading screens, quiet music drop |
| 6:17 | Ch 5 Cancer | Doctor/patient, imaging scans, older adults traveling/exercising. On-screen question list |
| 7:23 | Ch 6 Abundance | Construction robots, vertical farms, solar/wind. Text card: "−60% cost → who keeps the savings?" Scarce-assets list as bullet card |
| 8:36 | Ch 7 2045 | Calm future skyline. Text card: "The biggest winners barely called themselves AI companies." |
| 9:27 | Playbook | Title card "The Investing Playbook", six numbered cards |
| 9:34 | 1. Chain | **`charts/04-chain-of-consequences.png`** (reveal box by box) |
| 10:08 | 2. Good company vs stock | **`charts/02-good-company-vs-good-stock.png`** (draw both lines together) |
| 10:54 | 3. Watchlist | Table card per theme with tickers (one card per line of narration) |
| 11:52 | 4. Psychology | **`charts/03-expectations-cycle.png`** (dot moves along curve) |
| 12:20 | 5. Second-order | Big quote card: "100× more useful → what do we need 10× more of?" |
| 12:48 | 6. Signals | Three icon cards: bottleneck / inflection / valuation |
| 13:12 | Final thought | Return to opening city shot; end card with subscribe + disclaimer |

Animating the charts: drop the PNGs into your editor and use a left-to-right wipe/mask to "draw" the lines. In CapCut or DaVinci Resolve that's a crop/mask keyframe.

## 3. Voiceover

- **Your own voice (best for trust and monetization):** a USB mic (e.g., Samson Q2U, Shure MV7) in a closet or under a blanket. Read `narration.txt` one paragraph at a time; re-read flubs instead of stopping.
- **AI voice:** ElevenLabs or similar. Check the plan allows commercial YouTube use. YouTube may require you to disclose realistic synthetic voice in Studio.
- Target pace ~150 words/min. Leave a short breath between chapters.

## 4. Captions (separate, not burned in)

The files in `captions/` are sidecar files. Nothing is burned into the video, so viewers can turn them on/off and YouTube indexes the text for search.

**Easiest, and gives the most accurate timing:**
1. Upload the video to YouTube.
2. YouTube Studio → **Subtitles** → your video → **Add language: English** → **Add** → **Upload file** → **Without timing** → choose `captions/transcript.txt`.
3. YouTube auto-syncs the text to your voice. Review and publish.

**If you want to use the pre-timed file:** Upload `captions/captions.en.srt` → **With timing**. Its timings are estimated from the script at 150 wpm, so they will drift from your real recording. Fix it in your editor (DaVinci Resolve, Premiere, CapCut, and Subtitle Edit all import .srt), or use the auto-sync route above.

**For the blog / website video player:** use `captions/captions.en.vtt` with an HTML5 `<track>` tag.

Rebuild captions after editing the script:
```
python3 tools/build_captions.py        # default 150 wpm
python3 tools/build_captions.py 165    # if you talk faster
```

## 5. YouTube description template

```
What if AI cures cancer, robots do most physical work, and energy gets cheap... and investors who saw it coming still lose money? A hypothetical stock market story from 2026 to 2045, and a 6-step playbook for thinking about tech booms.

Read the blog version: [link]

CHAPTERS
(paste captions/youtube-chapters.txt here, then fix times to match your final edit)

This is a fictional, educational scenario. Not a forecast and not investment advice. Tickers mentioned are for research only. Do your own research.

Footage & music credits:
(paste credits.txt)
```

YouTube chapters need: first timestamp at 0:00, at least 3 chapters, each at least 10 seconds long.

## 6. Thumbnail ideas

- Split image: glowing future city (left) / red crashing chart (right). Text: **"RIGHT ABOUT AI. STILL LOST MONEY."**
- Robot hand holding a falling stock chart. Text: **"The AI Crash of 2033"** (small "hypothetical" tag).
