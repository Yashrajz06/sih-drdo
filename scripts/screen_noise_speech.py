"""
Gate 0.3 -- screen MAD noise clips for incidental speech.

Why this matters more than it sounds: MAD clips are cut from YouTube military
videos, so gunshot/shelling clips plausibly carry narration, radio chatter, or
shouting underneath. We already drop the explicit `communication` class, but speech
*inside* the other classes is invisible to that filter. Training a denoiser on
noise-that-contains-speech teaches it to remove speech -- the exact opposite of the
goal -- and it would surface only as mysteriously worse post-fine-tune numbers.

This ranks every noise clip by how speech-like a VAD thinks it is, then exports the
top-ranked ones for a human to listen to. The listening step is not optional: it
validates that the detector is right before we let it delete training data. A VAD
firing on gunshot transients would otherwise quietly throw away exactly the clips
the project cares most about.

Usage:
    python scripts/screen_noise_speech.py --split training
    python scripts/screen_noise_speech.py --split training --export 15
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from mad_noise import CATEGORIES, load_clip, load_split  # noqa: E402

SAMPLE_RATE = 16000


def speech_fraction(model, wav: np.ndarray) -> float:
    """Fraction of the clip's duration the VAD marks as speech."""
    from silero_vad import get_speech_timestamps

    if len(wav) < SAMPLE_RATE // 2:  # too short to judge
        return 0.0
    peak = np.abs(wav).max()
    if peak < 1e-6:
        return 0.0
    # Normalize: MAD clip levels vary wildly and the VAD is level-sensitive.
    wav = wav / peak

    stamps = get_speech_timestamps(torch.from_numpy(wav), model, sampling_rate=SAMPLE_RATE)
    speech_samples = sum(s["end"] - s["start"] for s in stamps)
    return float(speech_samples / len(wav))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="training", choices=["training", "test"])
    parser.add_argument("--limit", type=int, default=None, help="only screen the first N clips (smoke test)")
    parser.add_argument(
        "--export", type=int, default=15, help="copy this many top-ranked clips out for listening"
    )
    parser.add_argument("--export-dir", type=Path, default=REPO_ROOT / "listening_check")
    parser.add_argument(
        "--out", type=Path, default=None, help="manifest path (default: noise_speech_scores_<split>.json)"
    )
    args = parser.parse_args()
    out_path = args.out or REPO_ROOT / f"noise_speech_scores_{args.split}.json"

    by_category = load_split(args.split)
    clips = [(cat, p) for cat, paths in by_category.items() for p in paths]
    if not clips:
        sys.exit("No MAD clips found. Run scripts/mad_noise.py to check the dataset is in place.")
    if args.limit:
        clips = clips[: args.limit]

    from silero_vad import load_silero_vad

    model = load_silero_vad()

    print(f"screening {len(clips)} {args.split}-split clips for incidental speech...")
    scored = []
    for i, (category, path) in enumerate(clips, 1):
        try:
            frac = speech_fraction(model, load_clip(path))
        except Exception as exc:  # a corrupt clip shouldn't abort a 5k-clip run
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        scored.append({"path": str(path.relative_to(REPO_ROOT)), "category": category, "speech_fraction": frac})
        if i % 500 == 0:
            print(f"  {i}/{len(clips)}")

    scored.sort(key=lambda r: r["speech_fraction"], reverse=True)

    print(f"\n=== speech-likelihood by category ({args.split}) ===")
    for category in CATEGORIES:
        rows = [r for r in scored if r["category"] == category]
        if not rows:
            continue
        fracs = np.array([r["speech_fraction"] for r in rows])
        print(
            f"{category:15s} n={len(rows):5d}  "
            f"mean={fracs.mean():.3f}  median={np.median(fracs):.3f}  "
            f">30% speech: {int((fracs > 0.3).sum()):4d}  >50%: {int((fracs > 0.5).sum()):4d}"
        )

    out_path.write_text(json.dumps(scored, indent=2))
    print(f"\nwrote {out_path}")

    if args.export:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for f in args.export_dir.glob("*.wav"):
            f.unlink()
        print(f"\nexporting {args.export} most speech-like clips to {args.export_dir}/")
        for rank, row in enumerate(scored[: args.export], 1):
            src = REPO_ROOT / row["path"]
            dst = args.export_dir / f"{rank:02d}_{row['category']}_{row['speech_fraction']:.2f}_{src.parent.name}-{src.name}"
            shutil.copy(src, dst)
            print(f"  {rank:2d}. {row['category']:15s} speech_fraction={row['speech_fraction']:.2f}  {row['path']}")

    print(
        "\nGATE 0.3 (human): listen to the exported clips.\n"
        "  - If they DO contain speech -> the detector works; pick an exclusion threshold.\n"
        "  - If they DON'T -> the VAD is misfiring on impulsive noise; do NOT filter on it.\n"
        "This gate cannot be closed without a human listening -- it is not auto-passable."
    )


if __name__ == "__main__":
    main()
