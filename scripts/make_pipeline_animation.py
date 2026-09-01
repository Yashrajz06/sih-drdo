#!/usr/bin/env python3
"""Render demo_package/pipeline_animation.mp4 -- the live pipeline, end to end.

Nothing here is illustrated or faked. The real ONNX model is run hop by hop over
real audio, and every panel plots what that run actually produced: the analysis
window, the 257-bin input spectrum, the mask the model chose for this frame, the
enhanced spectrum, and the overlap-added output.

The mask is recovered as |enhanced| / |noisy| per bin. GTCRN emits an enhanced
complex spectrum rather than a mask tensor, so this is the gain it effectively
applied -- measured from the output, not read from a config.

Two acts:
  ACT 1  one frame traced through every stage at 1/8 speed, with stage captions
  ACT 2  the whole dashboard at true speed, 62.5 frames per second

Frames are composited by updating a single matplotlib figure in place and piping
raw RGB to ffmpeg. Rebuilding the figure each frame is ~20x slower for output
that looks identical.

    .venv/bin/python scripts/make_pipeline_animation.py
    .venv/bin/python scripts/make_pipeline_animation.py --fast     # 720p preview
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
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import signal as sps

from streaming_engine import HOP, N_FFT, StreamingEnhancer

SR = 16000
FPS = 25
INK, MUTED, EDGE = "#0F172A", "#64748B", "#CBD5E1"
TEAL, AMBER, BLUE, GREY = "#0D9488", "#EA8C2B", "#2563EB", "#94A3B8"
FAINT_TEAL, FAINT_AMBER = "#CCFBF1", "#FEF3C7"

STAGES = [
    (0.0,  2.4,  "1", "MICROPHONE", "256 new samples arrive every 16 ms", "in"),
    (2.4,  4.8,  "2", "ANALYSIS WINDOW", "32 ms of audio, sqrt-Hann weighted, 50% overlap", "in"),
    (4.8,  7.4,  "3", "STFT", "a 512-point FFT turns it into 257 complex frequency bins", "specin"),
    (7.4,  10.8, "4", "GTCRN DECIDES", "one gain per bin  ·  1.0 keeps the bin, 0.0 removes it", "mask"),
    (10.8, 13.0, "5", "MASK APPLIED", "noise bins attenuated, speech harmonics preserved", "specout"),
    (13.0, 15.0, "6", "OVERLAP-ADD", "back to a waveform, a fixed 16 ms later", "out"),
]
ACT1 = 15.0


def load(path: Path) -> np.ndarray:
    w, fs = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    if fs != SR:
        w = sps.resample(w, int(len(w) * SR / fs)).astype(np.float32)
    m = float(np.abs(w).max())
    return (w / m * 0.89).astype(np.float32) if m > 1e-9 else w


def run_pipeline(noisy: np.ndarray, onnx: Path | None):
    """Run the real model hop by hop, capturing every intermediate."""
    eng = StreamingEnhancer(onnx) if onnx else StreamingEnhancer()
    n = (len(noisy) // HOP) * HOP
    noisy = noisy[:n]
    win, spec_in, spec_out, out = [], [], [], []
    for i in range(0, n, HOP):
        out.append(eng.process_hop(noisy[i:i + HOP]))
        L = eng.last
        win.append(L["windowed"].copy())
        spec_in.append(L["spec"].copy())
        spec_out.append(L["spec_out"].copy())
    return (noisy, np.concatenate(out), np.array(win),
            np.array(spec_in), np.array(spec_out))


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noisy", type=Path,
                    default=Path("demo_package/1_BEFORE_noisy.wav"))
    ap.add_argument("--onnx", type=Path, default=Path("models/gtcrn_defence.onnx"))
    ap.add_argument("--out", type=Path, default=Path("demo_package/pipeline_animation.mp4"))
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    W, H = (1280, 720) if a.fast else (1920, 1080)
    onnx = a.onnx if a.onnx.exists() else None
    print(f"model: {onnx or 'upstream pretrained'}")

    noisy, enh, win, SI, SO = run_pipeline(load(a.noisy), onnx)
    nhop = len(SI)
    dur2 = nhop * HOP / SR
    print(f"ran {nhop} hops ({dur2:.1f}s of audio) through the real model")

    mag_in, mag_out = db(SI), db(SO)
    floor = max(mag_in.max() - 78, -80)
    mask = np.clip(np.abs(SO) / (np.abs(SI) + 1e-7), 0, 1.25)

    freqs = np.arange(257) * SR / N_FFT / 1000
    t_hop = np.arange(nhop) * HOP / SR

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#FFFFFF")

    def panel(rect, title, sub=""):
        ax = fig.add_axes(rect)
        ax.set_facecolor("#FCFDFE")
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.tick_params(colors=MUTED, labelsize=9)
        fig.text(rect[0], rect[1] + rect[3] + 0.028, title, fontsize=15,
                 fontweight="bold", color=INK, family="sans-serif")
        if sub:
            fig.text(rect[0], rect[1] + rect[3] + 0.008, sub, fontsize=11,
                     color=MUTED, family="sans-serif")
        return ax

    fig.text(0.05, 0.944, "Inside the pipeline", fontsize=27, fontweight="bold",
             color=INK, family="sans-serif")
    cap_stage = fig.text(0.05, 0.906, "", fontsize=17, fontweight="bold",
                         color=TEAL, family="sans-serif")
    cap_note = fig.text(0.05, 0.881, "", fontsize=13, color=MUTED, family="sans-serif")
    cap_clock = fig.text(0.955, 0.944, "", fontsize=15, color=MUTED,
                         ha="right", family="sans-serif")
    cap_hear = fig.text(0.955, 0.906, "", fontsize=13, fontweight="bold",
                        color=AMBER, ha="right", family="sans-serif")

    VIEW = int(1.6 * SR)
    ax_in = panel([0.05, 0.655, 0.42, 0.160], "Microphone input", "what the mic hears")
    ax_out = panel([0.535, 0.655, 0.42, 0.160], "Enhanced output", "what gets transmitted")
    for ax in (ax_in, ax_out):
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, VIEW)
        ax.set_yticks([])
        ax.set_xticks([])
    ln_in, = ax_in.plot([], [], lw=1.0, color=AMBER)
    ln_out, = ax_out.plot([], [], lw=1.0, color=TEAL)
    winbox = Rectangle((0, -1), N_FFT, 2, facecolor=TEAL, alpha=0.18,
                       edgecolor=TEAL, lw=1.4)
    ax_in.add_patch(winbox)

    ax_si = panel([0.05, 0.385, 0.26, 0.150], "Input spectrum", "257 bins, this frame")
    ax_mk = panel([0.37, 0.385, 0.26, 0.150], "The mask GTCRN chose", "1.0 keep  ·  0.0 remove")
    ax_so = panel([0.69, 0.385, 0.26, 0.150], "Enhanced spectrum", "after the mask")
    for ax in (ax_si, ax_so):
        ax.set_xlim(0, 8)
        ax.set_ylim(floor, mag_in.max() + 4)
        ax.set_xlabel("kHz", fontsize=10, color=MUTED)
        ax.set_yticks([])
    ax_mk.set_xlim(0, 8)
    ax_mk.set_ylim(0, 1.28)
    ax_mk.set_xlabel("kHz", fontsize=10, color=MUTED)
    ax_mk.axhline(1.0, color=GREY, lw=1.0, ls="--")
    ln_si, = ax_si.plot(freqs, np.full(257, floor), lw=1.3, color=AMBER)
    ln_so, = ax_so.plot(freqs, np.full(257, floor), lw=1.3, color=TEAL)
    ln_mk, = ax_mk.plot(freqs, np.zeros(257), lw=1.8, color=TEAL)
    fill = [ax_mk.fill_between(freqs, 0, np.zeros(257), color=TEAL, alpha=0.20)]

    ax_gi = panel([0.05, 0.205, 0.90, 0.082], "Input spectrogram")
    ax_go = panel([0.05, 0.068, 0.90, 0.082], "Output spectrogram")
    for ax, M in ((ax_gi, mag_in), (ax_go, mag_out)):
        ax.imshow(M.T, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, t_hop[-1], 0, 8], vmin=floor, vmax=mag_in.max())
        ax.set_ylabel("kHz", fontsize=10, color=MUTED)
        ax.set_xticks([])
    ax_go.set_xlabel("seconds", fontsize=10, color=MUTED)
    ax_go.set_xticks(np.arange(0, t_hop[-1], 1.0))
    ph1 = ax_gi.axvline(0, color="#FFFFFF", lw=1.8)
    ph2 = ax_go.axvline(0, color="#FFFFFF", lw=1.8)

    RECTS = {"in": [0.034, 0.636, 0.452, 0.232], "out": [0.519, 0.636, 0.452, 0.232],
             "specin": [0.034, 0.352, 0.292, 0.232], "mask": [0.354, 0.352, 0.292, 0.232],
             "specout": [0.674, 0.352, 0.292, 0.232]}
    hl = FancyBboxPatch((0, 0), 0.01, 0.01, boxstyle="round,pad=0.004,rounding_size=0.01",
                        transform=fig.transFigure, facecolor="none", edgecolor=TEAL,
                        lw=3, zorder=10)
    fig.patches.append(hl)

    # ---------------------------------------------------------------- audio
    split = int(0.55 * len(enh))
    xf = int(0.05 * SR)
    a2 = np.concatenate([noisy[:split - xf],
                         noisy[split - xf:split] * np.linspace(1, 0, xf)
                         + enh[split - xf:split] * np.linspace(0, 1, xf),
                         enh[split:]]).astype(np.float32)
    track = np.concatenate([np.zeros(int(ACT1 * SR), np.float32), a2])

    n1 = int(ACT1 * FPS)
    n2 = int(dur2 * FPS)
    total = n1 + n2
    hop_a1 = min(nhop - 1, int(2.0 * SR / HOP))
    print(f"encoding {total} frames ({total/FPS:.1f}s) at {W}x{H}...")

    with tempfile.TemporaryDirectory() as td:
        ap_ = Path(td) / "a.wav"
        sf.write(ap_, track, SR)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", str(ap_), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(a.out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        for f in range(total):
            if f < n1:                                  # ---- act 1, slow motion
                t = f / FPS
                k = int(hop_a1 * f / max(1, n1 - 1))
                st = next(s for s in STAGES if s[0] <= t < s[1] or s is STAGES[-1])
                cap_stage.set_text(f"{st[2]}  ·  {st[3]}")
                cap_note.set_text(st[4])
                cap_clock.set_text(f"slow motion  ·  1/8 speed  ·  frame {k}")
                cap_hear.set_text("")
                r = RECTS[st[5]]
                hl.set_bounds(r[0], r[1], r[2], r[3])
                hl.set_visible(True)
            else:                                       # ---- act 2, real time
                j = f - n1
                k = min(nhop - 1, int(j * nhop / max(1, n2 - 1)))
                cap_stage.set_text("REAL TIME  ·  62.5 frames every second")
                cap_note.set_text("every panel below is live output from the model, not a mock-up")
                cap_clock.set_text(f"{k * HOP / SR:5.2f} s  ·  frame {k}")
                cap_hear.set_text("HEARING: MICROPHONE INPUT" if k * HOP < split
                                  else "HEARING: MODEL OUTPUT")
                hl.set_visible(False)

            s0 = k * HOP
            lo = max(0, s0 - VIEW + N_FFT)
            seg_i = noisy[lo:lo + VIEW]
            seg_o = enh[lo:lo + VIEW]
            ln_in.set_data(np.arange(len(seg_i)), seg_i)
            ln_out.set_data(np.arange(len(seg_o)), seg_o)
            winbox.set_x(s0 - lo)

            ln_si.set_ydata(np.maximum(mag_in[k], floor))
            ln_so.set_ydata(np.maximum(mag_out[k], floor))
            ln_mk.set_ydata(mask[k])
            fill[0].remove()
            fill[0] = ax_mk.fill_between(freqs, 0, mask[k], color=TEAL, alpha=0.20)
            ph1.set_xdata([t_hop[k], t_hop[k]])
            ph2.set_xdata([t_hop[k], t_hop[k]])

            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
            if (f + 1) % 50 == 0:
                print(f"  {f+1}/{total}", end="\r", flush=True)

        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg failed")

    print(f"\nwrote {a.out}  ({a.out.stat().st_size/1e6:.1f} MB, {total/FPS:.1f}s, {W}x{H})")


if __name__ == "__main__":
    main()
