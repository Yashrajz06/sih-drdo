#!/usr/bin/env python3
"""Render the illustration sections for the 3-minute project film.

Writes one MP4 per section into demo_package/film/ rather than a single locked
cut, because the narration is recorded separately: separate clips let the editor
stretch or trim a section to match the voice without re-rendering everything.

    1_cold_open      8s   noise, then the title over it        (audio)
    2_problem       26s   speech progressively buried          (audio)
    3_research      20s   two candidates eliminated, one left  (silent)
    4_architecture  50s   the pipeline, built stage by stage   (silent)
    5_demo          --    built by make_demo_video.py          (audio)
    6_conclusion    25s   measured results, then deployment    (silent)

Sections 3, 4 and 6 are silent by design -- they exist to be talked over.

Every frame is composited in numpy on top of a small number of pre-rendered
matplotlib panels, and piped straight into ffmpeg. Rendering a fresh figure per
frame would be roughly 20x slower for output that looks identical.

    python3 scripts/make_film.py                 # all sections, 1920x1080
    python3 scripts/make_film.py --fast          # 1280x720 preview
    python3 scripts/make_film.py --only 4        # re-render one section
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyBboxPatch
from scipy import signal as sps

from make_demo_video import (ACCENT, BG, FPS, INK, MUTED, PEAK, SAMPLE_RATE, WARN,
                             fig_to_rgb, fit_text, load_mono, new_fig, peak_norm,
                             spectrogram_db)
from mad_noise import load_clip, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
BLUE = "#2563EB"
GREY = "#94A3B8"
RED = "#DC2626"


# --------------------------------------------------------------- ffmpeg sink
def encode(frames, audio: np.ndarray | None, out: Path, size, fps=FPS) -> None:
    W, H = size
    with tempfile.TemporaryDirectory() as td:
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
        if audio is not None:
            ap = Path(td) / "a.wav"
            sf.write(ap, audio, SAMPLE_RATE)
            cmd += ["-i", str(ap), "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-shortest", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = 0
        for f in frames:
            p.stdin.write(f.tobytes())
            n += 1
            if n % 100 == 0:
                print(f"    {n} frames", end="\r", flush=True)
        p.stdin.close()
        if p.wait() != 0:
            sys.exit(f"ffmpeg failed on {out.name}")
    print(f"  wrote {out.name}  ({n} frames, {n/fps:.1f}s, {out.stat().st_size/1e6:.1f} MB)")


def vrect(img, x0, y0, x1, y1, colour, width=4):
    """Outline rectangle, drawn in place."""
    c = np.array(colour, dtype=np.uint8)
    x0, x1 = max(0, x0), min(img.shape[1], x1)
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    img[y0:y0 + width, x0:x1] = c
    img[y1 - width:y1, x0:x1] = c
    img[y0:y1, x0:x0 + width] = c
    img[y0:y1, x1 - width:x1] = c


def vdot(img, cx, cy, r, colour):
    h, w = img.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[y0:y1, x0:x1][m] = np.array(colour, dtype=np.uint8)


def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------ shared drawing
def spec_panel(size, x, title, sub, colour, note=None, bare=False, meter=False):
    fig = new_fig(size)
    ax = fig.add_axes([0, 0, 1, 1] if bare else [0.07, 0.16, 0.86, 0.56])
    f, t, db = spectrogram_db(x)
    ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma")
    if bare:
        ax.set_axis_off()
        return fig_to_rgb(fig)
    ax.set_ylabel("kHz", fontsize=13, color=MUTED)
    ax.set_xlabel("seconds", fontsize=13, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in ax.spines.values():
        s.set_color("#CBD5E1")
    fit_text(fig, 0.07, 0.86, title, 40, color=colour, weight="bold", family="sans-serif")
    fig.text(0.07, 0.80, sub, fontsize=18, color=MUTED, family="sans-serif")
    if note:
        fig.text(0.07, 0.06, note, fontsize=15, color=MUTED, family="sans-serif")
    if meter:
        fig.text(0.62, 0.895, "INTELLIGIBILITY", fontsize=14, color=MUTED,
                 weight="bold", family="sans-serif")
    return fig_to_rgb(fig)


def box(fig, x, y, w, h, title, lines, colour, face="#FFFFFF"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=face,
                                      edgecolor=colour, linewidth=2.4, zorder=2))
    fig.patches.append(plt.Rectangle((x, y + h - 0.008), w, 0.008, transform=fig.transFigure,
                                     facecolor=colour, edgecolor="none", zorder=3))
    fig.text(x + w / 2, y + h - 0.062, title, fontsize=19, color=INK, weight="bold",
             ha="center", family="sans-serif", zorder=4)
    yy = y + h - 0.115
    for ln in lines:
        fig.text(x + w / 2, yy, ln, fontsize=14, color=MUTED, ha="center",
                 family="sans-serif", zorder=4)
        yy -= 0.042


# ================================================================= section 1
def sec_cold_open(size, outdir, noisy, name, expansion, tagline):
    print("1_cold_open")
    dur = 8.0
    clip = noisy[: int(dur * SAMPLE_RATE)]
    base = spec_panel(size, clip, "", "", WARN, bare=True)

    # The title is rendered black-on-white purely to get a clean pixel mask.
    # Rendering it white-on-transparent does not survive dropping the alpha
    # channel -- the text vanishes into the background it is composited onto.
    # Two masks, because the accent line is painted in a different colour.
    fig = new_fig(size)
    fit_text(fig, 0.5, 0.545, name, 132, max_frac=0.80, color="#000000",
             weight="bold", ha="center", family="sans-serif")
    fit_text(fig, 0.5, 0.385, tagline, 34, max_frac=0.84, color="#000000",
             ha="center", family="sans-serif")
    mask_white = (fig_to_rgb(fig) < 120).any(axis=2)

    fig = new_fig(size)
    fit_text(fig, 0.5, 0.475, expansion, 25, max_frac=0.70, color="#000000",
             weight="bold", ha="center", family="sans-serif")
    mask_accent = (fig_to_rgb(fig) < 120).any(axis=2)

    def frames():
        for i in range(int(dur * FPS)):
            t = i / FPS
            if t < 3.4:
                yield base
                continue
            k = float(np.clip((t - 3.4) / 0.5, 0, 1))
            img = (base * (1 - 0.74 * k)).astype(np.uint8)
            if k > 0.4:
                img[mask_white] = (255, 255, 255)
                img[mask_accent] = hexc("#5EEAD4")
            yield img
    encode(frames(), peak_norm(clip), outdir / "1_cold_open.mp4", size)


# ================================================================= section 2
def sec_problem(size, outdir, clean):
    print("2_problem")
    dur = 26.0
    n = int(dur * SAMPLE_RATE)
    clean = peak_norm(np.tile(clean, int(np.ceil(n / len(clean))))[:n])

    noise = load_split("test")
    rng = np.random.default_rng(3)

    def bed(cat, gain, start_s):
        b = np.zeros(n, np.float32)
        pos = int(start_s * SAMPLE_RATE)
        while pos < n:
            c = load_clip(rng.choice(noise[cat]))
            take = min(len(c), n - pos)
            b[pos:pos + take] += c[:take] * gain
            pos += take
        return b

    layers = [("stationary", 0.5, 6.0), ("non_stationary", 0.7, 12.0), ("impulsive", 1.5, 18.0)]
    stages, cum = [clean.copy()], clean.copy()
    for cat, g, st in layers:
        cum = cum + bed(cat, g, st)
        stages.append(cum.copy())
    audio = peak_norm(cum)

    labels = [("Clean speech", "what the listener needs to hear"),
              ("+ Vehicle and engine noise", "broadband, steady -- classical filters handle this"),
              ("+ Rotor and jet noise", "non-stationary: the spectrum moves"),
              ("+ Gunfire and shelling", "impulsive: no warning, and classical filters fail")]
    cols = [ACCENT, BLUE, WARN, RED]
    panels = [spec_panel(size, s[: int(8 * SAMPLE_RATE)], lab, sub, c, meter=True,
                         note="Horizontal bands are speech harmonics. Vertical streaks are impulsive noise.")
              for s, (lab, sub), c in zip(stages, labels, cols)]

    W, H = size
    mx0, mx1 = int(0.62 * W), int(0.93 * W)
    my, mh = int(0.125 * H), int(0.026 * H)

    def frames():
        total = int(dur * FPS)
        for i in range(total):
            t = i / FPS
            idx = sum(1 for _, _, st in layers if t >= st)
            prev = max(0, idx - 1)
            st = layers[idx - 1][2] if idx else 0.0
            k = 1.0 if idx == 0 else min(1.0, (t - st) / 1.1)
            img = (panels[prev] * (1 - k) + panels[idx] * k).astype(np.uint8) if k < 1 else panels[idx].copy()
            score = [1.0, 0.72, 0.5, 0.22][idx]
            if idx:
                score = [1.0, 0.72, 0.5, 0.22][prev] * (1 - k) + score * k
            img[my:my + mh, mx0:mx1] = hexc("#E2E8F0")
            img[my:my + mh, mx0:mx0 + int((mx1 - mx0) * score)] = hexc(
                ACCENT if score > 0.6 else (WARN if score > 0.35 else RED))
            yield img
    encode(frames(), audio, outdir / "2_problem.mp4", size)


# ================================================================= section 3
def sec_research(size, outdir):
    print("3_research")
    dur = 20.0
    fig = new_fig(size)
    fit_text(fig, 0.07, 0.87, "We eliminated, we did not guess", 44,
             color=INK, weight="bold", family="sans-serif")
    fig.text(0.07, 0.815, "Every rejection below is a number we measured ourselves.",
             fontsize=19, color=MUTED, family="sans-serif")
    rows = [("Classical DSP", "spectral subtraction, Wiener filtering",
             "+0.07 PESQ on gunfire", "effectively nothing", RED),
            ("Large neural models", "high quality, high compute",
             "cannot hold 16 ms", "no real-time on edge hardware", RED),
            ("GTCRN", "causal, complex-domain, 48,245 parameters",
             "+0.71 PESQ on gunfire", "runs on a Raspberry Pi CPU", ACCENT)]
    ys = [0.60, 0.41, 0.22]
    for (name, sub, metric, verdict, col), y in zip(rows, ys):
        fig.patches.append(FancyBboxPatch((0.07, y - 0.02), 0.86, 0.145,
                                          boxstyle="round,pad=0.004,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor="#F8FAFC",
                                          edgecolor="#E2E8F0", linewidth=1.6, zorder=1))
        fig.patches.append(plt.Rectangle((0.07, y - 0.02), 0.008, 0.145, transform=fig.transFigure,
                                         facecolor=col, edgecolor="none", zorder=2))
        fig.text(0.10, y + 0.078, name, fontsize=27, color=INK, weight="bold",
                 family="sans-serif", zorder=3)
        fig.text(0.10, y + 0.032, sub, fontsize=16, color=MUTED, family="sans-serif", zorder=3)
        fig.text(0.50, y + 0.078, metric, fontsize=25, color=col, weight="bold",
                 family="sans-serif", zorder=3)
        fig.text(0.50, y + 0.032, verdict, fontsize=16, color=MUTED, family="sans-serif", zorder=3)
    panel = fig_to_rgb(fig)

    W, H = size
    strikes = [(int(0.092 * W), int(0.44 * W), int((1 - (ys[0] + 0.085)) * H), 5.5),
               (int(0.092 * W), int(0.44 * W), int((1 - (ys[1] + 0.085)) * H), 11.0)]

    def frames():
        for i in range(int(dur * FPS)):
            t = i / FPS
            img = panel.copy()
            for x0, x1, y, at in strikes:
                if t > at:
                    k = min(1.0, (t - at) / 0.7)
                    img[y - 3:y + 3, x0:x0 + int((x1 - x0) * k)] = hexc(RED)
            if t > 15.0:
                k = min(1.0, (t - 15.0) / 0.6)
                y0 = int((1 - (ys[2] + 0.125)) * H)
                y1 = int((1 - (ys[2] - 0.02)) * H)
                x1 = int(0.07 * W + 0.86 * W * k)
                vrect(img, int(0.07 * W), y0, x1, y1, hexc(ACCENT), 5)
            yield img
    encode(frames(), None, outdir / "3_research.mp4", size)


# ================================================================= section 4
# Five stages, not seven. An earlier cut split framing from the transform and
# the inverse transform from the output; at seven boxes across a 16:9 frame the
# captions no longer fit inside their own boxes.
STAGES = [
    ("Microphone", ["16 kHz, mono", "256-sample blocks"], 3.0, "#2563EB"),
    ("STFT", ["32 ms window, 16 ms hop", "257 complex bins"], 8.0, "#2563EB"),
    ("GTCRN", ["48,245 parameters", "full-band + ERB sub-band"], 15.0, "#0D9488"),
    ("Complex mask", ["one gain per bin", "attenuates, never invents"], 35.0, "#0D9488"),
    ("Output", ["iSTFT overlap-add", "83.6 ms end to end"], 43.0, "#059669"),
]
CACHE_AT = 27.0


def fit_in_box(fig, x, y, text, size, box_w, **kw):
    """Centre text in a box, shrinking until it fits inside it."""
    t = fig.text(x, y, text, fontsize=size, ha="center", **kw)
    r = fig.canvas.get_renderer()
    while size > 9 and t.get_window_extent(r).width / fig.bbox.width > box_w * 0.92:
        size -= 1
        t.set_fontsize(size)
    return t


def sec_architecture(size, outdir):
    print("4_architecture")
    dur = 50.0
    W, H = size
    n = len(STAGES)
    bw, gap = 0.165, 0.022
    x0 = (1.0 - (n * bw + (n - 1) * gap)) / 2
    by, bh = 0.42, 0.24
    xs = [x0 + i * (bw + gap) for i in range(n)]

    def panel_for(k, cache):
        fig = new_fig(size)
        fit_text(fig, 0.5, 0.86, "What happens to 16 milliseconds of audio", 44,
                 color=INK, weight="bold", ha="center", family="sans-serif")
        fig.text(0.5, 0.805, "One frame in, one frame out. No look-ahead, no buffering.",
                 fontsize=20, color=MUTED, ha="center", family="sans-serif")
        for i in range(k):
            name, lines, _, col = STAGES[i]
            fig.patches.append(FancyBboxPatch((xs[i], by), bw, bh,
                                              boxstyle="round,pad=0.004,rounding_size=0.014",
                                              transform=fig.transFigure, facecolor="#FFFFFF",
                                              edgecolor=col, linewidth=2.6, zorder=2))
            fig.patches.append(plt.Rectangle((xs[i], by + bh - 0.009), bw, 0.009,
                                             transform=fig.transFigure, facecolor=col,
                                             edgecolor="none", zorder=3))
            fit_in_box(fig, xs[i] + bw / 2, by + bh - 0.075, name, 23, bw,
                       color=INK, weight="bold", family="sans-serif", zorder=4)
            yy = by + bh - 0.128
            for ln in lines:
                fit_in_box(fig, xs[i] + bw / 2, yy, ln, 15, bw,
                           color=MUTED, family="sans-serif", zorder=4)
                yy -= 0.045
            if i:
                fig.patches.append(plt.Rectangle((xs[i] - gap - 0.002, by + bh / 2 - 0.004),
                                                 gap + 0.004, 0.008, transform=fig.transFigure,
                                                 facecolor="#94A3B8", edgecolor="none", zorder=1))
        if cache:
            cy = by - 0.155
            fig.patches.append(FancyBboxPatch((xs[2] - 0.03, cy), bw + 0.06, 0.105,
                                              boxstyle="round,pad=0.004,rounding_size=0.014",
                                              transform=fig.transFigure, facecolor="#F0FDFA",
                                              edgecolor=ACCENT, linewidth=2.4, zorder=2))
            fig.text(xs[2] + bw / 2, cy + 0.065, "Carried state", fontsize=19, color=INK,
                     weight="bold", ha="center", family="sans-serif", zorder=3)
            fit_in_box(fig, xs[2] + bw / 2, cy + 0.024, "conv · TRA · inter-frame caches",
                       14, bw + 0.06, color=MUTED, family="sans-serif", zorder=3)
            fig.patches.append(plt.Rectangle((xs[2] + bw / 2 - 0.002, cy + 0.105),
                                             0.004, by - cy - 0.105, transform=fig.transFigure,
                                             facecolor=ACCENT, edgecolor="none", zorder=1))
            fig.text(0.5, 0.135, "The caches replace look-ahead: the model remembers past speech "
                                 "instead of waiting for future audio.",
                     fontsize=18, color=ACCENT, ha="center", family="sans-serif", zorder=3)
        return fig_to_rgb(fig)

    panels = {(k, False): panel_for(k, False) for k in range(n + 1)}
    panels.update({(k, True): panel_for(k, True) for k in range(3, n + 1)})

    px = [(int(x * W), int((x + bw) * W)) for x in xs]
    pyt, pyb = int((1 - by - bh) * H), int((1 - by) * H)
    ymid = (pyt + pyb) // 2

    def frames():
        for i in range(int(dur * FPS)):
            t = i / FPS
            k = sum(1 for _, _, at, _ in STAGES if t >= at)
            cache = t >= CACHE_AT and k >= 3
            img = panels[(k, cache)].copy()
            if k:
                at, col = STAGES[k - 1][2], STAGES[k - 1][3]
                if t - at < 1.5:
                    vrect(img, px[k - 1][0] - 9, pyt - 9, px[k - 1][1] + 9, pyb + 9, hexc(col), 5)
            if t >= 45.5 and k == n:
                cyc = (t - 45.5) % 2.2 / 2.2
                cx = int(px[0][0] + (px[-1][1] - px[0][0]) * cyc)
                vdot(img, cx, ymid, 14, hexc(ACCENT))
                vdot(img, cx, ymid, 6, (255, 255, 255))
            yield img
    encode(frames(), None, outdir / "4_architecture.mp4", size)


# ================================================================= section 6
def sec_conclusion(size, outdir):
    print("6_conclusion")
    dur = 25.0
    W, H = size
    rows = [("STOI", 0.920, 0.85, "0.920", "target 0.85", 1.0, ACCENT),
            ("SI-SNR", 20.2, 15.0, "20.2 dB", "target 15 dB", 25.0, ACCENT),
            ("PESQ", 2.49, 2.5, "2.49", "target 2.5, at target within error", 3.0, WARN),
            ("Latency", 83.6, 150.0, "83.6 ms", "budget 150 ms (ITU-T G.114)", 160.0, ACCENT)]
    fig = new_fig(size)
    fit_text(fig, 0.07, 0.88, "Measured on real defence noise", 44, color=INK,
             weight="bold", family="sans-serif")
    fig.text(0.07, 0.828, "Gunshots and shelling at +15 dB, scored against clean reference speech.",
             fontsize=19, color=MUTED, family="sans-serif")
    ys = [0.66, 0.53, 0.40, 0.27]
    for (name, _, _, val, note, _, col), y in zip(rows, ys):
        fig.text(0.07, y, name, fontsize=24, color=INK, weight="bold", family="sans-serif")
        fig.text(0.235, y, val, fontsize=26, color=col, weight="bold", family="sans-serif")
        fig.text(0.70, y, note, fontsize=16, color=MUTED, family="sans-serif")
    fig.text(0.07, 0.15, "Raspberry Pi 5  ·  CPU only, no GPU  ·  no cloud, no network",
             fontsize=21, color=INK, weight="bold", family="sans-serif")
    fig.text(0.07, 0.10, "Real-Time  ·  Causal  ·  Complex-Domain  ·  Defence-Ready",
             fontsize=17, color=ACCENT, family="sans-serif")
    panel = fig_to_rgb(fig)

    bx0, bx1 = int(0.40 * W), int(0.675 * W)

    def frames():
        for i in range(int(dur * FPS)):
            t = i / FPS
            img = panel.copy()
            for j, ((_, v, tgt, _, _, full, col), y) in enumerate(zip(rows, ys)):
                start = 2.0 + j * 2.6
                k = float(np.clip((t - start) / 1.5, 0, 1))
                yy = int((1 - y) * H) - 22
                img[yy:yy + 22, bx0:bx1] = hexc("#E2E8F0")
                frac = min(1.0, v / full) * k
                img[yy:yy + 22, bx0:bx0 + int((bx1 - bx0) * frac)] = hexc(col)
                tx = bx0 + int((bx1 - bx0) * min(1.0, tgt / full))
                if k > 0.05:
                    img[yy - 6:yy + 28, tx:tx + 3] = hexc(INK)
            yield img
    encode(frames(), None, outdir / "6_conclusion.mp4", size)


# ===================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "demo_package" / "film")
    ap.add_argument("--fast", action="store_true", help="720p preview render")
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4, 6], help="render one section")
    ap.add_argument("--noisy", type=Path, default=REPO_ROOT / "demo_package" / "1_BEFORE_noisy.wav")
    ap.add_argument("--clean", type=Path,
                    default=REPO_ROOT / "data" / "vctk_demand_testset" / "clean" / "p232_021.wav")
    ap.add_argument("--name", default="DHVANI", help="project name on the cold-open card")
    ap.add_argument("--expansion", default="Defence-Hardened Voice And Noise Intelligence")
    ap.add_argument("--tagline",
                    default="AI-Enabled Adaptive Noise Cancellation for Defence Communications")
    a = ap.parse_args()

    size = (1280, 720) if a.fast else (1920, 1080)
    a.outdir.mkdir(parents=True, exist_ok=True)
    print(f"rendering at {size[0]}x{size[1]} into {a.outdir}\n")

    if a.only in (None, 1):
        sec_cold_open(size, a.outdir, load_mono(a.noisy), a.name, a.expansion, a.tagline)
    if a.only in (None, 2):
        sec_problem(size, a.outdir, load_mono(a.clean))
    if a.only in (None, 3):
        sec_research(size, a.outdir)
    if a.only in (None, 4):
        sec_architecture(size, a.outdir)
    if a.only in (None, 6):
        sec_conclusion(size, a.outdir)
    print("\ndone. Section 5 is demo_package/demo_video.mp4 (make_demo_video.py).")


if __name__ == "__main__":
    main()
