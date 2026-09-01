#!/usr/bin/env python3
"""Generate docs/pipeline_architecture.svg -- the three-band system diagram.

Bands: (1) offline model engineering, (2) online real-time runtime, (3) edge
deployment. Every figure here is checked against the code in scripts/ and the
artefacts in models/; anything not yet built is drawn dashed grey and labelled
as planned, never as done.

All icons are hand-drawn inline SVG on a 24x24 grid -- no external assets, so
the file renders identically in a browser, in cairosvg, and once pasted into
PowerPoint. Text is ASCII apart from the middot: cairosvg draws arrows, minus
signs and radicals as empty boxes.

Card and band heights are computed from content rather than hardcoded.

    python3 scripts/make_pipeline_diagram.py
"""
from pathlib import Path

W = 1740
BX, BW = 56, 1640
IL, IR = BX + 22, BX + BW - 22

INK, MUTED, LINE = "#0F172A", "#64748B", "#475569"
ORANGE, ORANGE_BG, ORANGE_ED = "#EA8C2B", "#FFF7ED", "#FCD9A8"
TEAL, TEAL_BG, TEAL_ED = "#0D9488", "#F0FDFA", "#99F6E4"
BLUE, BLUE_BG, BLUE_ED = "#2563EB", "#EFF6FF", "#BFDBFE"
GREEN, GREEN_BG, GREEN_ED = "#059669", "#ECFDF5", "#A7F3D0"
SLATE_BG, SLATE_ED = "#F8FAFC", "#CBD5E1"
PLAN = "#94A3B8"

# ---------------------------------------------------------------- icon set
# Each entry is inner markup on a 24x24 grid. Stroke and fill are inherited
# from the wrapping <g>, except where an element sets fill explicitly.
ICON = {
"mic":      '<path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><path d="M8.5 21h7"/>',
"headset":  '<path d="M4 14v-2a8 8 0 0 1 16 0v2"/><path d="M4 13h3v7H6a2 2 0 0 1-2-2z"/><path d="M20 13h-3v7h1a2 2 0 0 0 2-2z"/>',
"wave":     '<path d="M2 12h2.5l2-7 3 14 3-18 3 21 2.5-10 1.5 3H22"/>',
"truck":    '<path d="M1 6h12v11H1z"/><path d="M13 10h4l3.5 3.5V17H13z"/><circle cx="6" cy="18.5" r="2.2"/><circle cx="17" cy="18.5" r="2.2"/>',
"heli":     '<path d="M3 4h18"/><path d="M12 4v3"/><rect x="7" y="7" width="9" height="7" rx="3"/><path d="M16 10.5h6"/><path d="M21 8v5"/><path d="M9 14v3"/><path d="M5.5 17h8"/>',
"jet":      '<path d="M2 12.5l8.5-1.2L13.5 3h2l-1.2 8.3 7.7 1.2-7.7 1.2L15.5 22h-2l-3-8.3z"/>',
"burst":    '<path d="M12 2l2.2 6.3 6.3-2.2-3.9 5.6 3.9 5.6-6.3-2.2L12 21.6l-2.2-6.5-6.3 2.2 3.9-5.6-3.9-5.6 6.3 2.2z"/>',
"wind":     '<path d="M2 8h11a3 3 0 1 0-3-3"/><path d="M2 12.5h15a3 3 0 1 1-3 3"/><path d="M2 17h8.5a2.4 2.4 0 1 1-2.4 2.4"/>',
"siren":    '<path d="M6 20h12v-5a6 6 0 0 0-12 0z"/><path d="M4 21h16"/><path d="M12 3v3"/><path d="M4 8l2.2 1.6"/><path d="M20 8l-2.2 1.6"/>',
"chat":     '<rect x="2" y="4" width="14" height="10" rx="2"/><path d="M7 18h11a2 2 0 0 0 2-2V9"/>',
"filter":   '<path d="M3 5h18l-7 8v6.5l-4-2.2V13z"/>',
"shuffle":  '<path d="M3 6h4l10 12h4"/><path d="M17 3l4 3-4 3"/><path d="M3 18h4l2.6-3.1"/><path d="M14.4 9.1L17 6"/><path d="M17 15l4 3-4 3"/>',
"sliders":  '<path d="M3 6h18"/><path d="M3 12h18"/><path d="M3 18h18"/><circle cx="9" cy="6" r="2.2" fill="#FFFFFF"/><circle cx="15.5" cy="12" r="2.2" fill="#FFFFFF"/><circle cx="7" cy="18" r="2.2" fill="#FFFFFF"/>',
"db":       '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13a8 3 0 0 0 16 0v-13"/><path d="M4 12a8 3 0 0 0 16 0"/>',
"net":      '<circle cx="5" cy="5.5" r="2"/><circle cx="5" cy="12" r="2"/><circle cx="5" cy="18.5" r="2"/><circle cx="12.5" cy="8.5" r="2"/><circle cx="12.5" cy="15.5" r="2"/><circle cx="20" cy="12" r="2"/><path d="M7 5.5l3.6 2.4M7 12l3.6-2.9M7 12l3.6 2.9M7 18.5l3.6-2.4M14.5 8.9l3.7 2.3M14.5 15.1l3.7-2.3"/>',
"target":   '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/>',
"chip":     '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9 6V3M15 6V3M9 21v-3M15 21v-3M6 9H3M6 15H3M21 9h-3M21 15h-3"/>',
"package":  '<path d="M12 2.5l9 4.8v9.4l-9 4.8-9-4.8V7.3z"/><path d="M3 7.3l9 4.8 9-4.8"/><path d="M12 12.1v9.4"/>',
"gear":     '<circle cx="12" cy="12" r="3.4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2.1 2.1M16.9 16.9L19 19M19 5l-2.1 2.1M7.1 16.9L5 19"/>',
"clock":    '<circle cx="12" cy="12" r="9"/><path d="M12 6.5V12l3.8 2.2"/>',
"shield":   '<path d="M12 2.5l8 3v6.2c0 5-3.4 8.9-8 10.8-4.6-1.9-8-5.8-8-10.8V5.5z"/><path d="M8.5 12l2.5 2.5 4.5-4.7"/>',
"cloudoff": '<path d="M7 18h9.5a4 4 0 0 0 .8-7.9A6 6 0 0 0 7.6 7.6"/><path d="M6.5 10A4 4 0 0 0 7 18"/><path d="M3 3l18 18"/>',
"pi":       '<circle cx="8.6" cy="14.4" r="3.3"/><circle cx="15.4" cy="14.4" r="3.3"/><circle cx="12" cy="18.6" r="3.3"/><path d="M12 10.4C10.6 7.2 8 6.4 6.2 7.6"/><path d="M12 10.4c1.4-3.2 4-4 5.8-2.8"/>',
"bars":     '<rect x="2.5" y="13" width="3" height="8" rx="1.2"/><rect x="7.5" y="8" width="3" height="13" rx="1.2"/><rect x="12.5" y="3" width="3" height="18" rx="1.2"/><rect x="17.5" y="10" width="3" height="11" rx="1.2"/>',
"bell":     '<path d="M2 18C6.5 18 7 5 12 5s5.5 13 10 13"/><path d="M2 21h20"/>',
"grid":     '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/><rect x="9" y="9" width="6" height="6" fill="currentColor" stroke="none" opacity="0.35"/>',
"layers":   '<path d="M12 2.8l9 4.7-9 4.7-9-4.7z"/><path d="M3 12.5l9 4.7 9-4.7"/><path d="M3 17.2l9 4.7 9-4.7"/>',
"loop":     '<path d="M4 10.5a8 8 0 0 1 13.6-3.6"/><path d="M17.6 3.2v3.9h-3.9"/><path d="M20 13.5a8 8 0 0 1-13.6 3.6"/><path d="M6.4 20.8v-3.9h3.9"/>',
"ring":     '<circle cx="12" cy="12" r="8.5" stroke-dasharray="3.4 2.6"/><path d="M12 3.5v4"/><path d="M15.4 5.1L12 3.5 15.4 1.9"/>',
"terminal": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6.5 9.5l3.2 2.9-3.2 2.9"/><path d="M12.5 15.3H18"/>',
"check":    '<circle cx="12" cy="12" r="9"/><path d="M8 12.2l2.9 2.9L16 9"/>',
"usb":      '<path d="M12 22V3"/><path d="M12 3l-2.4 3.6h4.8z" fill="currentColor"/><path d="M12 13.5l4-3.6"/><circle cx="16.6" cy="9.2" r="1.7" fill="currentColor"/><path d="M12 17.2l-4-3.8"/><rect x="6.2" y="11.4" width="3.6" height="3.6" rx="0.8" fill="currentColor"/>',
"bolt":     '<path d="M13.5 2L4 14h6l-1.2 8L20 10h-6.4z"/>',
"scissors": '<circle cx="6" cy="6" r="2.6"/><circle cx="6" cy="18" r="2.6"/><path d="M8.2 7.6L20 18"/><path d="M8.2 16.4L20 6"/>',
"speech":   '<path d="M12 3a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M6 11a6 6 0 0 0 12 0"/><path d="M2 14.5h2.5M19.5 14.5H22"/><path d="M12 17v4"/>',
"ruler":    '<rect x="2" y="8" width="20" height="8" rx="1.6"/><path d="M6.5 8v3.4M11 8v4.6M15.5 8v3.4"/>',
"split":    '<path d="M4 12h5"/><path d="M9 12l5-6h6"/><path d="M9 12l5 6h6"/><path d="M17 3l3 3-3 3"/><path d="M17 15l3 3-3 3"/>',
"box":      '<rect x="3" y="3" width="18" height="18" rx="2.5"/><path d="M3 9h18"/><circle cx="6.6" cy="6" r="0.9" fill="currentColor" stroke="none"/><circle cx="9.6" cy="6" r="0.9" fill="currentColor" stroke="none"/>',
"copy":     '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
}

o = []
def a(s): o.append(s)
def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def icon(name, x, y, size=22, col=INK, sw=1.75):
    s = size / 24.0
    body = ICON.get(name, ICON["box"]).replace("currentColor", col)
    a(f'<g transform="translate({x:.1f},{y:.1f}) scale({s:.4f})" fill="none" stroke="{col}" '
      f'stroke-width="{sw/s:.2f}" stroke-linecap="round" stroke-linejoin="round">{body}</g>')

def wrap(text, width_px, ppc):
    limit = max(6, int(width_px / ppc))
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if len(t) <= limit:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def card(x, y, w, title, title_icon, rows, accent=BLUE, bg=BLUE_BG, ed=BLUE_ED, sub=None):
    """rows: (icon, head, sub) tuples; a '~' prefix on the icon marks it planned."""
    tx = x + 14
    head_y = y + 25
    cur = head_y + (28 if sub else 10)
    drawn = []
    for ic, head, s2 in rows:
        planned = ic.startswith("~")
        ic = ic.lstrip("~")
        hl = wrap(head, w - 62, 5.7)
        sl = wrap(s2, w - 58, 5.1) if s2 else []
        block = 12.5 * len(hl) + 11.3 * len(sl)
        drawn.append((ic, planned, cur, hl, sl))
        cur += max(26.0, block + 6.0)
    h = cur - y + 6

    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h:.0f}" rx="9" fill="{bg}" stroke="{ed}" stroke-width="1.5"/>')
    a(f'<rect x="{x}" y="{y}" width="{w}" height="4" rx="2" fill="{accent}"/>')
    icon(title_icon, tx, y + 12, 19, accent, 1.9)
    a(f'<text x="{tx+27}" y="{head_y}" font-size="12.4" font-weight="700" fill="{INK}">{esc(title)}</text>')
    if sub:
        a(f'<text x="{tx}" y="{head_y+15}" font-size="9.3" fill="{MUTED}">{esc(sub)}</text>')

    for ic, planned, ry, hl, sl in drawn:
        col = PLAN if planned else accent
        tcol = PLAN if planned else INK
        scol = PLAN if planned else MUTED
        icon(ic, tx, ry - 2, 19, col, 1.7)
        yy = ry + 9
        for ln in hl:
            a(f'<text x="{tx+28}" y="{yy:.1f}" font-size="10.3" font-weight="600" fill="{tcol}">{esc(ln)}</text>')
            yy += 12.5
        for ln in sl:
            a(f'<text x="{tx+28}" y="{yy:.1f}" font-size="9.5" fill="{scol}">{esc(ln)}</text>')
            yy += 11.3
    return y + h

def band(y, h, num, title, note=None):
    a(f'<rect x="{BX}" y="{y}" width="{BW}" height="{h:.0f}" rx="12" fill="none" stroke="{LINE}" stroke-width="1.5" stroke-dasharray="2 4"/>')
    a(f'<rect x="{BX+16}" y="{y-13}" width="{7.05*len(title)+54:.0f}" height="26" rx="13" fill="#FFFFFF"/>')
    a(f'<circle cx="{BX+31}" cy="{y}" r="9.5" fill="{LINE}"/>')
    a(f'<text x="{BX+31}" y="{y+3.7}" text-anchor="middle" font-size="11" font-weight="700" fill="#FFFFFF">{num}</text>')
    a(f'<text x="{BX+47}" y="{y+4}" font-size="12.5" font-weight="700" fill="{LINE}" letter-spacing="0.5">{esc(title)}</text>')
    if note:
        a(f'<text x="{BX+BW-18}" y="{y+h-14:.0f}" text-anchor="end" font-size="10.2" fill="{MUTED}">{esc(note)}</text>')

def arrow(x1, y1, x2, y2, col=LINE, dash=None, mk="a"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{col}" stroke-width="1.6"{d} marker-end="url(#{mk})"/>')

def path(d, col=LINE, dash=None, mk="a"):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    a(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6"{ds} marker-end="url(#{mk})"/>')

# ------------------------------------------------------------------ header
a('__OPEN__'); a('<defs>')
for mid, col in (("a", LINE), ("t", TEAL), ("g", GREEN), ("p", PLAN)):
    a(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
      f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{col}"/></marker>')
a('</defs>'); a('__BG__')
a(f'<text x="{BX}" y="42" font-size="24" font-weight="700" fill="{INK}">AI-Based Noise Suppression for Defence Communications</text>')
a(f'<text x="{BX}" y="64" font-size="13" fill="{MUTED}">GTCRN speech enhancement  ·  streaming ONNX runtime  ·  Raspberry Pi 5 edge target</text>')
lx, lw = BX + BW - 476, 476
a(f'<rect x="{lx}" y="23" width="{lw}" height="47" rx="8" fill="{SLATE_BG}" stroke="{SLATE_ED}" stroke-width="1"/>')
icon("check", lx + 13, 30, 15, TEAL, 1.9)
a(f'<text x="{lx+34}" y="{42}" font-size="10.4" fill="{INK}">Solid colour = built and measured in this system</text>')
icon("gear", lx + 13, 49, 15, PLAN, 1.9)
a(f'<text x="{lx+34}" y="{61}" font-size="10.4" fill="{PLAN}">Grey = declared future work, not claimed as done</text>')

# ==================================================================== BAND 1
B1Y = 112
cy = B1Y + 32
xA, wA = IL, 200
xB, wB = 296, 216
xC, wC = 530, 660
xD, wD = 1208, 210
xE, wE = 1432, 202

bA = card(xA, cy, wA, "Audio Sources", "db", [
    ("speech", "Clean speech", "LibriSpeech dev-clean, 2,703 files -- training pool"),
    ("check", "Held-out evaluation", "VCTK-DEMAND, 824 pairs -- never trained on"),
    ("truck", "Stationary noise", "vehicles, engines"),
    ("heli", "Non-stationary noise", "helicopters, fighter jets"),
    ("burst", "Impulsive noise", "gunshots, shelling, footsteps"),
    ("db", "Noise corpus", "Military Audio Dataset, 7,466 clips, CC BY 4.0"),
    ("~layers", "Room impulse responses", "in code, disabled by default"),
], ORANGE, ORANGE_BG, ORANGE_ED)

bB = card(xB, cy, wB, "Data Preparation", "filter", [
    ("filter", "Speech screening", "silero-VAD removes noise clips containing speech: 221 of 5,655 dropped"),
    ("shuffle", "Dynamic on-the-fly mixing", "randomised SNR from -5 to +20 dB, drawn per sample"),
    ("ruler", "Active-speech level", "SNR referenced to voiced frames, not whole-clip RMS"),
    ("burst", "Impulsive oversampling", "about 50% of batches; 0-3 transients placed over voiced speech"),
    ("sliders", "Augmentation", "gain, clipping, band-limiting (radio channel)"),
    ("copy", "Pair generation", "16 kHz mono, 4 s segments, float32 noisy/clean tensors"),
], ORANGE, ORANGE_BG, ORANGE_ED)

a('__CFRAME__')
icon("net", xC + 14, cy + 11, 20, TEAL, 1.9)
a(f'<text x="{xC+42}" y="{cy+26}" font-size="13.2" font-weight="700" fill="#134E4A">Model Training and Fine-Tuning -- GTCRN</text>')
a(f'<text x="{xC+14}" y="{cy+42}" font-size="9.6" fill="{MUTED}">Grouped Temporal Convolutional Recurrent Network (ICASSP 2024), fine-tuned on real defence noise</text>')
sY = cy + 54
b1 = card(xC + 14, sY, 310, "Architecture", "chip", [
    ("net", "48,245 parameters", "23.7K learned weights plus a fixed, non-learned ERB matrix"),
    ("bolt", "33 MMACs/s", "compute cost per second of audio"),
    ("bars", "ERB sub-band front end", "257 FFT bins folded to 64 perceptual bands above bin 65"),
    ("grid", "SFE", "sub-band feature extraction, kernel-3 neighbour unfold"),
    ("layers", "Grouped conv encoder / decoder", "5 blocks, dilations 1 / 2 / 5, causal left-only padding"),
    ("loop", "2x DPGRNN", "bidirectional across frequency, unidirectional across time"),
    ("target", "TRA", "temporal recurrent attention"),
], TEAL, "#FFFFFF", TEAL_ED, sub="causal, zero look-ahead")
b2 = card(xC + 336, sY, 310, "Training objective and recipe", "target", [
    ("wave", "Hybrid loss", "SI-SNR + compressed magnitude (power 0.3) + complex real/imag"),
    ("bars", "Multi-resolution STFT loss", "FFT 512 / 1024 / 2048; spectral convergence and log-magnitude"),
    ("shield", "Asymmetric anti-over-suppression", "weight 0.1; penalises deleting speech harder than residual noise"),
    ("sliders", "Optimiser", "AdamW, lr 5e-4, cosine anneal, grad clip 5.0, batch 16"),
    ("clock", "Checkpointing", "every epoch, resumable -- Colab sessions disconnect"),
    ("check", "Selection", "best checkpoint by validation metric"),
], TEAL, "#FFFFFF", TEAL_ED, sub="what the model is optimised for")

sy = max(b1, b2) + 16
a(f'<rect x="{xC+14}" y="{sy:.0f}" width="{wC-28}" height="64" rx="8" fill="#FFFFFF" stroke="{TEAL_ED}" stroke-width="1.3"/>')
icon("split", xC + 26, sy + 12, 19, TEAL, 1.8)
a(f'<text x="{xC+53}" y="{sy+25:.0f}" font-size="10.8" font-weight="700" fill="{INK}">Data split</text>')
a(f'<text x="{xC+53}" y="{sy+40:.0f}" font-size="9.5" fill="{MUTED}">MAD ships its own training/test CSVs: 5,655 vs 830 usable clips. No test clip is seen in training.</text>')
a(f'<text x="{xC+53}" y="{sy+54:.0f}" font-size="9.5" fill="{MUTED}">Clean speech 95/5 by sorted path, chapter-contiguous. VCTK-DEMAND held out for the benchmark.</text>')
bx0, bw0 = xC + wC - 236, 210
a(f'<rect x="{bx0}" y="{sy+22:.0f}" width="{bw0*0.872:.0f}" height="15" rx="3.5" fill="{TEAL}"/>')
a(f'<rect x="{bx0+bw0*0.872:.0f}" y="{sy+22:.0f}" width="{bw0*0.128:.0f}" height="15" rx="3.5" fill="{PLAN}"/>')
a(f'<text x="{bx0+bw0*0.436:.0f}" y="{sy+33:.0f}" text-anchor="middle" font-size="8.8" font-weight="700" fill="#FFFFFF">5,655 TRAIN</text>')
a(f'<text x="{bx0+bw0*0.936:.0f}" y="{sy+33:.0f}" text-anchor="middle" font-size="7.6" font-weight="700" fill="#FFFFFF">830</text>')
bC = sy + 64 + 14
o[o.index('__CFRAME__')] = (
    f'<rect x="{xC}" y="{cy}" width="{wC}" height="{bC-cy:.0f}" rx="9" fill="{TEAL_BG}" stroke="{TEAL}" stroke-width="1.9"/>'
    f'<rect x="{xC}" y="{cy}" width="{wC}" height="4" rx="2" fill="{TEAL}"/>')

bD = card(xD, cy, wD, "Model Optimisation", "gear", [
    ("box", "ONNX export, opset 11", "a hard performance requirement: opset 18 adds about 30 nodes and runs 3.5x slower, 5.41 vs 1.53 ms per hop"),
    ("loop", "Streaming conversion", "Conv2d becomes StreamConv2d with explicit caches, so the graph runs one 16 ms frame at a time"),
    ("scissors", "Graph optimisation (onnxsim)", "constant folding, BatchNorm fusion"),
    ("~sliders", "INT8 dynamic quantisation", "planned; the shipped model is FP32"),
    ("~scissors", "Structured pruning", "planned; not attempted"),
], GREEN, GREEN_BG, GREEN_ED)

bE = card(xE, cy, wE, "Deployable Package", "package", [
    ("package", "gtcrn_defence.onnx", "562 KB on disk, FP32, opset 11"),
    ("layers", "Cache tensor spec", "conv, TRA and inter-frame state carried between hops"),
    ("check", "Measured on defence noise", "PESQ 2.49 · STOI 0.920 · SI-SNR 20.2 dB"),
    ("target", "Problem-statement targets", "PESQ 2.5 · STOI 0.85 · SI-SNR 15 dB"),
], GREEN, GREEN_BG, GREEN_ED)

for x1, x2 in ((xA+wA, xB), (xB+wB, xC), (xC+wC, xD), (xD+wD, xE)):
    arrow(x1+4, cy+118, x2-5, cy+118)

B1H = max(bA, bB, bC, bD, bE) + 40 - B1Y
band(B1Y, B1H, "1", "DEFENCE-AWARE MODEL ENGINEERING AND DATASET  (OFFLINE, ONE-TIME)",
     "Runs once on a Colab T4. Produces a single 562 KB ONNX file -- nothing else in this band ships to the device.")

# ==================================================================== BAND 2
B2Y = B1Y + B1H + 48
ry = B2Y + 32
bots = []
for x, w, t, ti, rows, ac, bg, ed, sb in [
    (78, 178, "Noise Environment", "wave", [
        ("truck", "Engine and vehicle noise", ""), ("heli", "Rotor and jet noise", ""),
        ("wind", "Wind, HVAC, broadband hiss", ""), ("siren", "Sirens and alarms", ""),
        ("burst", "Gunshots, shelling, artillery", ""), ("chat", "Background chatter", ""),
    ], ORANGE, ORANGE_BG, ORANGE_ED, "what the mic actually hears"),
    (267, 188, "Audio Acquisition", "mic", [
        ("mic", "Primary microphone", "single channel; no reference mic required"),
        ("wave", "Format", "16 kHz, mono, float32 via sounddevice"),
        ("terminal", "Driver", "sounddevice / PortAudio, ALSA on the Pi"),
        ("box", "Block size", "256 samples (16 ms) per callback"),
    ], BLUE, BLUE_BG, BLUE_ED, None),
    (466, 212, "Signal Processing", "bars", [
        ("ring", "Ring buffer", "512 samples; each new 256-sample hop shifts the window"),
        ("bars", "STFT", "512-point FFT"),
        ("bell", "Window", "periodic sqrt-Hann; the square root is what lets analysis and synthesis reconstruct exactly"),
        ("ruler", "Framing", "32 ms window, 16 ms hop = 50% overlap"),
        ("grid", "Output", "257 complex bins, 31.25 Hz apart, 62.5 frames per second"),
        ("clock", "Causal", "zero look-ahead; no future sample is ever read"),
    ], BLUE, BLUE_BG, BLUE_ED, "STFT analysis front-end"),
    (1130, 186, "Complex-Domain Mask", "grid", [
        ("grid", "Complex ratio mask (CRM)", "one complex gain per bin, applied to real and imaginary parts together"),
        ("wave", "Magnitude and phase move together", "so there is no separate phase estimation step"),
        ("shield", "It can only attenuate", "the mask cannot invent speech that is not there"),
    ], TEAL, TEAL_BG, TEAL_ED, None),
    (1328, 158, "Reconstruction", "layers", [
        ("bars", "Inverse FFT", "back to a 512-sample time frame"),
        ("bell", "Synthesis window", "sqrt-Hann, 50% overlap-add"),
        ("clock", "Algorithmic lag", "a fixed 256 samples, 16 ms"),
    ], BLUE, BLUE_BG, BLUE_ED, None),
    (1498, 152, "Output", "headset", [
        ("headset", "Headset or radio", ""),
        ("speech", "Enhanced speech, 16 kHz", ""),
        ("bolt", "83.6 ms end-to-end", "measured, not estimated"),
        ("check", "ITU-T G.114 budget", "150 ms"),
    ], GREEN, GREEN_BG, GREEN_ED, None),
]:
    bots.append(card(x, ry, w, t, ti, rows, accent=ac, bg=bg, ed=ed, sub=sb))

exx, exw = 692, 424
a('__EFRAME__')
icon("chip", exx + 14, ry + 11, 20, TEAL, 1.9)
a(f'<text x="{exx+42}" y="{ry+26}" font-size="13.2" font-weight="700" fill="#134E4A">Intelligent Enhancement Engine</text>')
a(f'<text x="{exx+14}" y="{ry+42}" font-size="9.6" fill="{MUTED}">ONNX Runtime 1.29 · CPUExecutionProvider · opset 11 · FP32 · single thread</text>')
e1 = card(exx + 12, ry + 54, 194, "Causal GTCRN", "net", [
    ("bars", "ERB sub-band + full-band 0-8 kHz", ""),
    ("grid", "SFE neighbour unfold", ""),
    ("layers", "Grouped conv encoder / decoder", ""),
    ("loop", "2x DPGRNN + TRA attention", ""),
    ("check", "The same graph that was trained", "nothing is retuned at runtime"),
], TEAL, "#FFFFFF", TEAL_ED)
e2 = card(exx + 218, ry + 54, 194, "Streaming state", "layers", [
    ("box", "conv_cache", "2 x 1 x 16 x 16 x 33"),
    ("box", "tra_cache", "2 x 3 x 1 x 1 x 16"),
    ("box", "inter_cache", "2 x 1 x 33 x 16"),
    ("loop", "Carried frame to frame", "this replaces look-ahead and gives the model a memory of past speech"),
], TEAL, "#FFFFFF", TEAL_ED)
eb = max(e1, e2) + 36
a(f'<text x="{exx+14}" y="{eb-14:.0f}" font-size="9.6" fill="{MUTED}">Runs on every frame regardless of the bypass toggle, so the recurrent state never goes stale.</text>')
o[o.index('__EFRAME__')] = (
    f'<rect x="{exx}" y="{ry}" width="{exw}" height="{eb-ry:.0f}" rx="9" fill="{TEAL_BG}" stroke="{TEAL}" stroke-width="1.9"/>'
    f'<rect x="{exx}" y="{ry}" width="{exw}" height="4" rx="2" fill="{TEAL}"/>')
bots.append(eb)

for x1, x2 in ((256, 267), (455, 466), (678, exx), (exx+exw, 1130), (1316, 1328), (1486, 1498)):
    arrow(x1+3, ry+96, x2-5, ry+96, col=TEAL, mk="t")

ny = max(bots) + 26
a(f'<rect x="{IL}" y="{ny:.0f}" width="{IR-IL}" height="80" rx="9" fill="#FFFFFF" stroke="{PLAN}" stroke-width="1.5" stroke-dasharray="6 4"/>')
icon("mic", IL + 16, ny + 10, 18, PLAN, 1.8)
a(f'<text x="{IL+42}" y="{ny+24:.0f}" font-size="11.5" font-weight="700" fill="{PLAN}">OPTIONAL ADAPTIVE NOISE CANCELLATION BRANCH -- NOT IMPLEMENTED</text>')
a(f'<text x="{IL+42}" y="{ny+39:.0f}" font-size="9.6" fill="{PLAN}">Needs a second, reference microphone. Scoped as future work; the shipped system is single-microphone.</text>')
sx = IL + 34
for i, (ic, s) in enumerate([("mic", "Reference mic"), ("filter", "VAD speech gate"),
                             ("sliders", "NLMS adaptive filter"), ("wave", "Correlated noise estimate")]):
    a(f'<rect x="{sx}" y="{ny+48:.0f}" width="186" height="24" rx="6" fill="{SLATE_BG}" stroke="{PLAN}" stroke-width="1" stroke-dasharray="4 3"/>')
    icon(ic, sx + 9, ny + 52, 15, PLAN, 1.7)
    a(f'<text x="{sx+30}" y="{ny+64:.0f}" font-size="9.8" fill="{PLAN}">{esc(s)}</text>')
    if i < 3:
        arrow(sx + 190, ny + 60, sx + 224, ny + 60, col=PLAN, dash="4 3", mk="p")
    sx += 228
path(f"M 880 {ny+44:.0f} V {eb+7:.0f}", col=PLAN, dash="6 4", mk="p")

B2H = ny + 80 + 40 - B2Y
band(B2Y, B2H, "2", "REAL-TIME SPEECH ENHANCEMENT  (ONLINE RUNTIME, ONE 16 ms FRAME AT A TIME)",
     "Pure streaming: no look-ahead and no buffering beyond a single frame. Measured 83.6 ms end-to-end, mic to ear.")

# ==================================================================== BAND 3
B3Y = B2Y + B2H + 48
dy = B3Y + 32
d1 = card(IL, dy, 262, "Deployment Flow", "copy", [
    ("package", "1. Optimised ONNX model", "built once in Band 1"),
    ("copy", "2. Copy to device", "one 562 KB file; PyTorch is never installed on the board"),
    ("terminal", "3. scripts/setup_pi.sh", "venv, ALSA device check, dependencies, throttle check"),
    ("ruler", "4. Benchmark", "per-hop latency and real-time factor, measured on the board"),
    ("headset", "5. scripts/live_demo.py", "mic to headset, with a live enhancement on/off toggle"),
], GREEN, GREEN_BG, GREEN_ED)
d2 = card(358, dy, 660, "Why the processing runs on the edge", "bolt", [
    ("bolt", "Latency", "83.6 ms measured end-to-end with a chirp and cross-correlation, not estimated. ITU-T G.114 treats anything under 150 ms as transparent to a talker."),
    ("cloudoff", "Connectivity independence", "a field radio cannot assume a network link, and a cloud round trip would exhaust the latency budget on its own."),
    ("shield", "Privacy and security", "operational audio never leaves the device."),
    ("clock", "Real-time factor 0.095", "the model uses under a tenth of the time budget it is given, which leaves headroom for a slower board."),
    ("package", "Footprint", "48,245 parameters, 562 KB. Small enough that ONNX Runtime threading buys nothing: 0.72 / 0.75 / 0.78 ms for 1 / 2 / 4 threads."),
], BLUE, BLUE_BG, BLUE_ED)
d3 = card(1036, dy, 250, "Edge Hardware Target", "pi", [
    ("pi", "Raspberry Pi 5, 8 GB", ""),
    ("chip", "Broadcom BCM2712", "quad-core ARM Cortex-A76 at 2.4 GHz, CPU inference only"),
    ("box", "ONNX Runtime", "CPUExecutionProvider, no accelerator required"),
    ("usb", "USB audio codec", "the Pi 5 has no 3.5 mm jack"),
], GREEN, GREEN_BG, GREEN_ED)
d4 = card(1306, dy, 328, "Verification on Target", "check", [
    ("ruler", "setup_pi.sh benchmark", "reports median and p95 ms per hop, and RTF, across 1 / 2 / 4 threads"),
    ("check", "Pass criterion", "RTF below 0.5"),
    ("shield", "Leaves system config alone", "the script deliberately does not touch /boot/firmware/config.txt or audio config: those need a reboot and can leave a board unbootable"),
], BLUE, BLUE_BG, BLUE_ED)

B3H = max(d1, d2, d3, d4) + 40 - B3Y
band(B3Y, B3H, "3", "EDGE DEPLOYMENT PLATFORM",
     "The Band 2 runtime executes entirely on this device. No cloud and no network dependency at inference time.")

# ---- cross-band routing, kept in the outer margins so nothing crosses a card
path(f"M {xE+wE/2:.0f} {bE+4:.0f} V {B1Y+B1H+16:.0f} H {W-26} V {B3Y-6:.0f}", col=GREEN, mk="g")
a(f'<text x="{W-32}" y="{B1Y+B1H+36:.0f}" text-anchor="end" font-size="10.2" font-weight="600" fill="{GREEN}">deploy model to device</text>')
path(f"M {IL+44} {dy-14:.0f} V {B3Y-18:.0f} H 28 V {ry+150:.0f} H {IL-7}", col=LINE, dash="5 4")
a(f'<text x="38" y="{B3Y-28:.0f}" font-size="10.2" font-weight="600" fill="{LINE}">hosts the Band 2 runtime</text>')

H = B3Y + B3H + 52
a(f'<text x="{BX}" y="{H-20:.0f}" font-size="9.8" fill="{MUTED}">Every figure quoted here is measured on this build. Grey dashed elements are declared future work and are not part of the current system.</text>')
a('</svg>')
o[o.index('__OPEN__')] = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" width="{W}" '
                         f'height="{H:.0f}" font-family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif">')
o[o.index('__BG__')] = f'<rect width="{W}" height="{H:.0f}" fill="#FFFFFF"/>'

out = Path("docs/pipeline_architecture.svg")
out.write_text("\n".join(o), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes), canvas {W} x {int(H)}, {len(ICON)} icons")
