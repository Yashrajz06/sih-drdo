#!/usr/bin/env python3
"""Assemble a finished demo MP4 from a before/after audio pair.

The video is built from the two wavs that live_demo.py --capture-test already
writes (the raw mic signal and the enhanced output). Nothing is screen-captured
and no system-audio loopback is involved: the audio track is taken straight from
the files, so it is digitally clean rather than a re-recording of the speakers.

Structure:  title card -> BEFORE -> AFTER -> side-by-side -> results card.

Two deliberate choices, both of which affect whether the result is honest:

  * Both tracks are peak-normalised to the same level. If "before" were simply
    louder, the demo would be showing a volume knob rather than a model.
  * The two sections play the identical utterance. Cutting a different take for
    "after" would make the comparison meaningless.

Rendering pipes raw RGB frames straight into ffmpeg's stdin -- writing several
hundred PNGs to disk first is slower and leaves temp files behind.

    python3 scripts/make_demo_video.py
    python3 scripts/make_demo_video.py --raw demo_live_raw.wav --enhanced demo_live.wav
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
FPS = 25
PEAK = 0.89

# Measured on this build -- see README.md. Kept in one place so the card cannot
# drift away from the numbers the repo actually reports.
RESULTS = [
    ("PESQ (wideband)", "2.49", "target > 2.5 -- at target within error"),
    ("STOI", "0.920", "target > 0.85"),
    ("SI-SNR", "20.2 dB", "target > 15 dB"),
    ("End-to-end latency", "83.6 ms", "ITU-T G.114 budget: 150 ms"),
]

INK = "#0F172A"
MUTED = "#64748B"
ACCENT = "#0D9488"
WARN = "#EA8C2B"
BG = "#FFFFFF"


# ----------------------------------------------------------------- audio prep
def load_mono(path: Path) -> np.ndarray:
    wav, fs = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if fs != SAMPLE_RATE:
        n = int(round(len(wav) * SAMPLE_RATE / fs))
        wav = sps.resample(wav, n).astype(np.float32)
    return wav


def peak_norm(x: np.ndarray) -> np.ndarray:
    m = float(np.abs(x).max())
    return x if m < 1e-9 else (x / m * PEAK).astype(np.float32)


def align_pair(raw: np.ndarray, enh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-add emits each hop one frame late; trim so both start together."""
    n = min(len(raw), len(enh))
    raw, enh = raw[:n], enh[:n]
    probe = min(n, SAMPLE_RATE * 5)
    xc = sps.correlate(enh[:probe], raw[:probe], mode="full")
    lag = int(np.argmax(np.abs(xc))) - (probe - 1)
    if 0 < lag < SAMPLE_RATE // 2:          # enhanced trails the input
        enh = enh[lag:]
        raw = raw[: len(enh)]
    elif -SAMPLE_RATE // 2 < lag < 0:
        raw = raw[-lag:]
        enh = enh[: len(raw)]
    return raw, enh


def spectrogram_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Z = sps.stft(x, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
    db = 20 * np.log10(np.abs(Z) + 1e-8)
    return f, t, np.clip(db, db.max() - 80, db.max())


# ----------------------------------------------------------------- panel art
def new_fig(size):
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=BG)
    return fig


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def fit_text(fig, x, y, text, size, max_frac=0.86, floor=24, **kw):
    """Draw text, shrinking until it actually fits the frame.

    Measured against the renderer rather than guessed from character counts --
    a long project title silently ran off the right edge otherwise, and the
    overflow is invisible until you extract a frame from the encoded video.
    """
    t = fig.text(x, y, text, fontsize=size, **kw)
    r = fig.canvas.get_renderer()
    while size > floor and t.get_window_extent(r).width / fig.bbox.width > max_frac:
        size -= 2
        t.set_fontsize(size)
    return t


def card(size, kicker, title, lines, accent=ACCENT) -> np.ndarray:
    fig = new_fig(size)
    fig.text(0.08, 0.80, kicker, fontsize=17, color=accent, weight="bold", family="sans-serif")
    fit_text(fig, 0.08, 0.70, title, 46, color=INK, weight="bold", family="sans-serif")
    y = 0.56
    for ln in lines:
        fig.text(0.08, y, ln, fontsize=21, color=MUTED, family="sans-serif")
        y -= 0.075
    fig.patches.append(plt.Rectangle((0.08, 0.755), 0.10, 0.006, transform=fig.transFigure,
                                     facecolor=accent, edgecolor="none"))
    return fig_to_rgb(fig)


def results_card(size) -> np.ndarray:
    fig = new_fig(size)
    fig.text(0.08, 0.86, "MEASURED ON REAL DEFENCE NOISE", fontsize=16, color=ACCENT,
             weight="bold", family="sans-serif")
    fig.text(0.08, 0.76, "Results", fontsize=46, color=INK, weight="bold", family="sans-serif")
    y = 0.60
    for name, value, note in RESULTS:
        fig.text(0.08, y, name, fontsize=22, color=INK, weight="bold", family="sans-serif")
        fig.text(0.46, y, value, fontsize=26, color=ACCENT, weight="bold", family="sans-serif")
        fig.text(0.60, y, note, fontsize=16, color=MUTED, family="sans-serif")
        y -= 0.11
    fig.text(0.08, 0.08, "GTCRN, 48,245 parameters  ·  streaming ONNX  ·  CPU only, no GPU",
             fontsize=15, color=MUTED, family="sans-serif")
    return fig_to_rgb(fig)


def spec_panel(size, x, label, sub, accent):
    """Full-bleed spectrogram panel. Returns (rgb, x0_px, x1_px, y0_px, y1_px)."""
    fig = new_fig(size)
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.62])
    f, t, db = spectrogram_db(x)
    ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma")
    ax.set_ylabel("kHz", fontsize=13, color=MUTED)
    ax.set_xlabel("seconds", fontsize=13, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in ax.spines.values():
        s.set_color("#CBD5E1")
    fit_text(fig, 0.06, 0.88, label, 42, color=accent, weight="bold", family="sans-serif")
    fig.text(0.06, 0.82, sub, fontsize=18, color=MUTED, family="sans-serif")
    fig.canvas.draw()
    bb = ax.get_window_extent()
    h = size[1]
    return (fig_to_rgb(fig), int(bb.x0), int(bb.x1), int(h - bb.y1), int(h - bb.y0))


def compare_panel(size, raw, enh) -> np.ndarray:
    fig = new_fig(size)
    fit_text(fig, 0.06, 0.91, "Same three seconds, before and after", 34,
             color=INK, weight="bold", family="sans-serif")
    seg = min(len(raw), 3 * SAMPLE_RATE)
    for i, (sig, ttl, col) in enumerate([(raw[:seg], "BEFORE", WARN), (enh[:seg], "AFTER", ACCENT)]):
        ax = fig.add_axes([0.06 + i * 0.47, 0.14, 0.41, 0.66])
        f, t, db = spectrogram_db(sig)
        ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma")
        ax.set_title(ttl, fontsize=22, color=col, weight="bold", pad=12)
        ax.set_xlabel("seconds", fontsize=12, color=MUTED)
        if i == 0:
            ax.set_ylabel("kHz", fontsize=12, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        for s in ax.spines.values():
            s.set_color("#CBD5E1")
    fig.text(0.06, 0.05, "The vertical streaks are gunshots. The horizontal bands are speech harmonics.",
             fontsize=15, color=MUTED, family="sans-serif")
    return fig_to_rgb(fig)


# ----------------------------------------------------------------- assembly
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=REPO_ROOT / "demo_package" / "1_BEFORE_noisy.wav")
    p.add_argument("--enhanced", type=Path, default=REPO_ROOT / "demo_package" / "2_AFTER_enhanced.wav")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "demo_package" / "demo_video.mp4")
    p.add_argument("--title", default="AI-Based Noise Suppression for Defence Communications")
    p.add_argument("--team", default="SIH 2026")
    p.add_argument("--seconds", type=float, default=15.0, help="max length of each half")
    p.add_argument("--size", default="1920x1080")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found -- install it with: sudo apt-get install -y ffmpeg")
    for f in (args.raw, args.enhanced):
        if not f.exists():
            sys.exit(f"missing {f}\nRecord a pair first:\n"
                     f"  python3 scripts/live_demo.py --capture-test 20 --out demo_live.wav")

    W, H = (int(v) for v in args.size.lower().split("x"))
    size = (W, H)

    raw, enh = align_pair(load_mono(args.raw), load_mono(args.enhanced))
    keep = min(len(raw), int(args.seconds * SAMPLE_RATE))
    raw, enh = peak_norm(raw[:keep]), peak_norm(enh[:keep])
    dur = len(raw) / SAMPLE_RATE
    print(f"aligned pair: {dur:.1f}s each, peak-normalised to {PEAK}")

    print("rendering panels...")
    title = card(size, args.team, args.title,
                 ["Real-time speech enhancement for high-noise environments",
                  "Recorded in a room with defence noise played back over a speaker"])
    before, bx0, bx1, by0, by1 = spec_panel(size, raw, "BEFORE", "what the microphone hears", WARN)
    after, ax0, ax1, ay0, ay1 = spec_panel(size, enh, "AFTER", "what the model transmits", ACCENT)
    compare = compare_panel(size, raw, enh)
    results = results_card(size)

    gap = np.zeros(int(0.6 * SAMPLE_RATE), np.float32)
    sil = lambda s: np.zeros(int(s * SAMPLE_RATE), np.float32)
    track = np.concatenate([sil(4.0), raw, gap, enh, sil(5.0), sil(6.0)])

    # (frames, static panel, optional playhead geometry)
    timeline = [
        (int(4.0 * FPS), title, None),
        (int(dur * FPS), before, (bx0, bx1, by0, by1)),
        (int(0.6 * FPS), after, None),
        (int(dur * FPS), after, (ax0, ax1, ay0, ay1)),
        (int(5.0 * FPS), compare, None),
        (int(6.0 * FPS), results, None),
    ]
    total = sum(n for n, _, _ in timeline)
    print(f"encoding {total} frames at {FPS} fps ({total/FPS:.1f}s)...")

    with tempfile.TemporaryDirectory() as td:
        apath = Path(td) / "track.wav"
        sf.write(apath, track, SAMPLE_RATE)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", str(apath),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        done = 0
        for nframes, panel, head in timeline:
            for i in range(nframes):
                if head is None:
                    proc.stdin.write(panel.tobytes())
                else:
                    x0, x1, y0, y1 = head
                    frame = panel.copy()
                    px = x0 + int((i / max(1, nframes - 1)) * (x1 - x0))
                    frame[y0:y1, max(x0, px - 1):min(x1, px + 2)] = (255, 255, 255)
                    proc.stdin.write(frame.tobytes())
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total}", end="\r", flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg failed")

    mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({mb:.1f} MB, {total/FPS:.1f}s, {W}x{H})")


if __name__ == "__main__":
    main()
