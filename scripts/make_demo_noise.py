"""
Build a continuous defence-noise soundtrack for live demos.

MAD clips are short (3-10 s) and single-event. A demo needs one continuous file you
can loop for minutes: a steady bed (helicopter rotor, vehicle engine) with gunfire and
shelling landing on top at irregular intervals -- i.e. what a radio operator in the
field would actually be sitting in.

Two ways to use the output:

  ACOUSTIC (most convincing to judges)
    Play this file out loud on a phone or speaker, then speak into the mic. The
    microphone genuinely picks up your voice and the noise together, exactly as it
    would in the field. Nothing is added in software.

  DIGITAL INJECTION (most reproducible)
    python scripts/live_demo.py --inject-noise demo_noise.wav --noise-gain 0.3
    The noise is mixed into the captured signal before enhancement. Identical every
    run, so it is the right choice for a recorded video or a side-by-side comparison.

Noise comes from the MAD **test** split -- clips the model never saw in training, so
the demo isn't showing off on material it memorised.

Usage:
    python scripts/make_demo_noise.py
    python scripts/make_demo_noise.py --seconds 120 --bed helicopter --events-per-min 12
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from mad_noise import LABEL_NAMES, MAD_DIR, _read_split, load_clip  # noqa: E402

SAMPLE_RATE = 16000
EPS = 1e-8


def clips_for_class(split: str, class_name: str) -> list[Path]:
    return [r["path"] for r in _read_split(f"{split}.csv") if r["class_name"] == class_name]


def loudest_window(wav: np.ndarray, seconds: float) -> np.ndarray:
    """The most energetic slice -- for a transient, this is the event itself rather
    than the quiet lead-in of the recording."""
    win = int(seconds * SAMPLE_RATE)
    if len(wav) <= win:
        return wav
    energy = np.convolve(wav**2, np.ones(win), mode="valid")
    start = int(np.argmax(energy))
    return wav[start : start + win]


def build_bed(clips: list[Path], total: int, rng: random.Random) -> np.ndarray:
    """Continuous background, cross-faded between clips so there are no clicks."""
    bed = np.zeros(total, dtype=np.float32)
    if not clips:
        return bed
    fade = int(0.25 * SAMPLE_RATE)
    pos = 0
    while pos < total:
        clip = load_clip(rng.choice(clips))
        if len(clip) < fade * 2:
            continue
        peak = np.abs(clip).max()
        if peak < 1e-5:
            continue
        clip = clip / peak
        n = min(len(clip), total - pos)
        seg = clip[:n].copy()
        # Fade the joins; an abrupt splice is audible as a click and would be unfair
        # to the model (a click is itself an impulsive event).
        if n > fade * 2:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        bed[pos : pos + n] += seg
        pos += max(n - fade, fade)
    return bed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "demo_noise.wav")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--split", default="test", choices=["test", "training"])
    p.add_argument("--bed", default="helicopter", choices=sorted(set(LABEL_NAMES.values()) - {"communication"}))
    p.add_argument("--bed-level", type=float, default=0.25, help="steady-noise amplitude (0-1)")
    p.add_argument("--events-per-min", type=float, default=10.0, help="gunfire/shelling events per minute")
    p.add_argument("--event-level", type=float, default=0.9, help="transient peak amplitude (0-1)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not MAD_DIR.exists():
        sys.exit(f"MAD dataset not found at {MAD_DIR}. See scripts/mad_noise.py")

    rng = random.Random(args.seed)
    total = int(args.seconds * SAMPLE_RATE)

    bed_clips = clips_for_class(args.split, args.bed)
    event_clips = clips_for_class(args.split, "shooting") + clips_for_class(args.split, "shelling")
    print(f"bed: {args.bed} ({len(bed_clips)} clips)   events: shooting+shelling ({len(event_clips)} clips)")
    if not bed_clips and not event_clips:
        sys.exit("no usable clips found")

    print(f"building {args.seconds:.0f}s soundtrack...")
    track = build_bed(bed_clips, total, rng) * args.bed_level

    n_events = int(args.events_per_min * args.seconds / 60)
    placed = 0
    for _ in range(n_events):
        if not event_clips:
            break
        event = loudest_window(load_clip(rng.choice(event_clips)), 0.8)
        peak = np.abs(event).max()
        if peak < 1e-5:
            continue
        event = event / peak * args.event_level * rng.uniform(0.6, 1.0)
        pos = rng.randint(0, max(0, total - len(event)))
        end = min(total, pos + len(event))
        track[pos:end] += event[: end - pos]
        placed += 1

    # Leave headroom so the mixture with speech doesn't clip downstream.
    peak = np.abs(track).max() + EPS
    if peak > 0.95:
        track = track * (0.95 / peak)

    sf.write(args.out, track.astype(np.float32), SAMPLE_RATE)
    print(f"placed {placed} transient events")
    print(f"wrote {args.out}  ({args.seconds:.0f}s, {args.out.stat().st_size/1e6:.1f} MB)\n")
    print("ACOUSTIC demo  -- play this out loud, speak into the mic:")
    print(f"    aplay {args.out.name}          # in one terminal")
    print("    python scripts/live_demo.py --onnx models/gtcrn_defence.onnx   # in another")
    print("\nDIGITAL demo   -- noise mixed in software, identical every run:")
    print(f"    python scripts/live_demo.py --inject-noise {args.out.name} \\")
    print("        --noise-gain 0.3 --onnx models/gtcrn_defence.onnx")


if __name__ == "__main__":
    main()
