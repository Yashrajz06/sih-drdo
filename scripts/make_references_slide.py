#!/usr/bin/env python3
"""Generate docs/references_slide.svg -- the research/references slide content.

Laid out as three tall cards with header pills, matching the SIH deck's own
slide-6 layout, and sized 2100x820 (2.56:1) to sit inside that box region.

Grouped by what each reference does in the project. The surveyed-only papers sit
in their own block inside card 1 rather than mixed into the main list: a flat run
of nineteen citations where five are load-bearing reads as padding, and invites a
question about a paper nobody on the team has read.

    python3 scripts/make_references_slide.py
"""
from pathlib import Path

W = 2100  # H is computed from the tallest card
INK, MUTED, FAINT = "#0F172A", "#475569", "#8A97A8"

THEMES = {
    "blue":  ("#1E3A8A", "#BFDBFE", "#7EA6E8", "#EFF5FF", "#DCE9FC"),
    "amber": ("#7C4A03", "#FBD87F", "#E0A93A", "#FFFBEF", "#FDF0CF"),
    "green": ("#14532D", "#C7EFA6", "#8FCB6B", "#F4FCEE", "#E2F6D5"),
}

o = []
def a(s): o.append(s)
def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(text, width_px, ppc):
    limit = max(8, int(width_px / ppc))
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if len(t) <= limit:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


a('__OPEN__')
a('<defs>')
for key, (_, _, _, top, bot) in THEMES.items():
    a(f'<linearGradient id="g{key}" x1="0" y1="0" x2="0" y2="1">'
      f'<stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bot}"/></linearGradient>')
a('</defs>')
a('__BG__')

CARDS = [
    ("blue", "Model and Method", [
        ("e", "GTCRN", "Rong et al., ICASSP 2024",
         "github.com/Xiaobin-Rong/gtcrn  ·  MIT  ·  the model we fine-tune"),
        ("e", "Complex Ratio Masking", "Williamson, Wang & Wang, IEEE/ACM TASLP 24(3), 2016",
         "the complex mask our model outputs"),
        ("e", "ERB auditory filters", "Glasberg & Moore, Hearing Research 47, 1990",
         "the perceptual sub-band front end"),
        ("h", "LOSS FUNCTIONS IMPLEMENTED FROM", "", ""),
        ("e", "Braun & Tashev, TSP 2021", "compressed magnitude, asymmetric anti-over-suppression", ""),
        ("e", "Yamamoto, Song & Kim, ICASSP 2020", "multi-resolution STFT loss", ""),
        ("h", "SURVEYED FOR CONTEXT -- NOT IMPLEMENTED", "", ""),
        ("t", "Wang & Chen, IEEE/ACM TASLP 2018 (overview)  ·  DCCRN, Interspeech 2020  ·  "
              "FullSubNet, ICASSP 2021  ·  DPRNN, ICASSP 2020  ·  DNS Challenge", "", ""),
    ]),
    ("amber", "Datasets", [
        ("e", "Military Audio Dataset", "Kim, Yoon & Jung, Scientific Data 11:668, 2024",
         "CC BY 4.0  ·  primary defence noise  ·  5,655 train / 830 test"),
        ("e", "LibriSpeech", "Panayotov et al., ICASSP 2015",
         "openslr.org/12  ·  clean speech for training"),
        ("e", "VCTK-DEMAND", "Valentini-Botinhao et al., Interspeech 2016",
         "held-out evaluation only  ·  never trained on"),
        ("e", "DEMAND noise database", "Thiemann, Ito & Vincent, Proc. Meetings on Acoustics, 2013",
         "the noise half of VCTK-DEMAND"),
        ("e", "ESC-50", "Piczak, ACM Multimedia 2015",
         "CC BY-NC  ·  siren and wind  ·  evaluation only"),
        ("h", "LICENSING", "", ""),
        ("t", "Every dataset licence verified. ESC-50 is CC BY-NC, so it is used for evaluation "
              "only and excluded from every deployment-facing claim.", "", ""),
    ]),
    ("green", "Metrics, Standards and Baseline", [
        ("e", "PESQ", "ITU-T P.862 / P.862.2  ·  Rix et al., ICASSP 2001",
         "perceptual quality  ·  target > 2.5"),
        ("e", "STOI", "Taal, Hendriks, Heusdens & Jensen, IEEE TASLP 19(7), 2011",
         "intelligibility  ·  target > 0.85"),
        ("e", "SI-SNR", "Le Roux, Wisdom, Erdogan & Hershey, ICASSP 2019",
         "arXiv:1811.02508  ·  target > 15 dB"),
        ("e", "ITU-T G.114", "One-way transmission time",
         "the 150 ms budget our 83.6 ms is measured against"),
        ("e", "Spectral subtraction", "Boll, IEEE TASSP 27(2), 1979",
         "our classical baseline  ·  +0.57 on engines, +0.07 on gunfire"),
        ("h", "TOOLCHAIN", "", ""),
        ("t", "PyTorch 2.13  ·  ONNX 1.22 / ONNX Runtime 1.29 (opset 11)  ·  onnx-simplifier 0.7.3  ·  "
              "silero-vad 6.2.1  ·  pesq 0.0.4 / pystoi 0.4.1  ·  Raspberry Pi 5", "", ""),
    ]),
]

M, GAP = 16, 34
CW = (W - 2 * M - 2 * GAP) / 3
card_slots, card_bottoms = [], []

for i, (theme, head, rows) in enumerate(CARDS):
    dark, pill, pill_ed, _, _ = THEMES[theme]
    x = M + i * (CW + GAP)
    card_slots.append((len(o), x, theme, pill_ed))
    a('__CARD__')
    a(f'<rect x="{x+22:.0f}" y="10" width="{CW-44:.0f}" height="46" rx="14" '
      f'fill="{pill}" stroke="{pill_ed}" stroke-width="1.8"/>')
    a(f'<text x="{x+CW/2:.0f}" y="41" text-anchor="middle" font-size="25" font-weight="700" '
      f'fill="{dark}">{esc(head)}</text>')

    tx, tw = x + 32, CW - 64
    y = 100
    for kind, name, cite, note in rows:
        if kind == "h":
            y += 10
            a(f'<line x1="{tx}" y1="{y-16}" x2="{x+CW-32:.0f}" y2="{y-16}" '
              f'stroke="{pill_ed}" stroke-width="1.5"/>')
            a(f'<text x="{tx}" y="{y+10}" font-size="18" font-weight="700" fill="{dark}" '
              f'letter-spacing="0.6">{esc(name)}</text>')
            y += 38
        elif kind == "t":
            for ln in wrap(name, tw, 9.3):
                a(f'<text x="{tx}" y="{y}" font-size="18" fill="{MUTED}">{esc(ln)}</text>')
                y += 25
            y += 8
        else:
            a(f'<circle cx="{tx+6}" cy="{y-8}" r="5" fill="{dark}"/>')
            a(f'<text x="{tx+22}" y="{y}" font-size="24" font-weight="700" fill="{INK}">{esc(name)}</text>')
            y += 26
            if cite:
                for ln in wrap(cite, tw - 22, 9.8):
                    a(f'<text x="{tx+22}" y="{y}" font-size="19" fill="{MUTED}">{esc(ln)}</text>')
                    y += 23
            if note:
                for ln in wrap(note, tw - 22, 9.3):
                    a(f'<text x="{tx+22}" y="{y}" font-size="18" fill="{FAINT}">{esc(ln)}</text>')
                    y += 22
            y += 16
    card_bottoms.append(y)

H = int(max(card_bottoms)) + 22
for idx, x, theme, pill_ed in card_slots:
    o[idx] = (f'<rect x="{x:.0f}" y="0" width="{CW:.0f}" height="{H}" rx="18" '
              f'fill="url(#g{theme})" stroke="{pill_ed}" stroke-width="2"/>')
a('</svg>')
o[o.index('__OPEN__')] = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'font-family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif">')
o[o.index('__BG__')] = f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>'
out = Path("docs/references_slide.svg")
out.write_text("\n".join(o), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes), {W}x{H}")
