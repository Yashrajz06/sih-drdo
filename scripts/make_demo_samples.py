"""
Generate before/after audio samples for listening tests and the demo.

Objective metrics tell you the model improved; they don't tell you what it *sounds*
like, and a judge will ask to hear it. This produces matched triples -- the same
mixture processed by nothing, by the stock pretrained model, and by our defence
fine-tuned model -- so the comparison is like-for-like and the difference is audible
rather than asserted.

Each sample set is written as:
    <name>_0_noisy.wav       the mixture, unprocessed
    <name>_1_pretrained.wav  stock GTCRN (what you'd get off the shelf)
    <name>_2_finetuned.wav   our defence-tuned model
    <name>_3_clean.wav       the ground-truth clean speech, for reference

Usage:
    python scripts/make_demo_samples.py
    python scripts/make_demo_samples.py --snrs 0,10 --per-category 2
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from eval_impulsive_noise import mix_at_snr  # noqa: E402
from mad_noise import load_clip, load_split  # noqa: E402
from run_baseline_eval import enhance, load_model, load_resampled, score_pair  # noqa: E402

SAMPLE_RATE = 16000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-dir", type=Path, default=REPO_ROOT / "data" / "vctk_demand_testset" / "clean")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "demo_samples")
    p.add_argument("--pretrained", default="model_trained_on_dns3.tar")
    p.add_argument("--finetuned", type=Path, default=REPO_ROOT / "checkpoints" / "finetuned_run2.pt")
    p.add_argument("--snrs", default="5,15", help="comma-separated SNRs in dB")
    p.add_argument("--per-category", type=int, default=1, help="samples per (category, SNR)")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    clean_files = sorted(args.clean_dir.glob("*.wav"))
    if not clean_files:
        sys.exit(f"no clean speech in {args.clean_dir}")
    if not args.finetuned.exists():
        sys.exit(f"no fine-tuned checkpoint at {args.finetuned}")

    noise_by_cat = load_split("test")  # held-out noise the model never trained on
    rng = random.Random(args.seed)

    print("loading models...")
    pre = load_model(args.pretrained)
    fine = load_model(str(args.finetuned))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.wav"):
        old.unlink()

    snrs = [float(s) for s in args.snrs.split(",")]
    print(f"\n{'sample':34s} {'noisy':>7s} {'stock':>7s} {'ours':>7s} {'gain':>7s}")
    rows = []
    for category, clips in noise_by_cat.items():
        if not clips:
            continue
        for snr in snrs:
            for k in range(args.per_category):
                clean = load_resampled(rng.choice(clean_files))
                noise = load_clip(rng.choice(clips))
                mix, target = mix_at_snr(clean, noise, snr, rng)

                enh_pre = enhance(pre, mix)
                enh_fine = enhance(fine, mix)

                name = f"{category}_{snr:+.0f}dB" + (f"_{k+1}" if args.per_category > 1 else "")
                sf.write(args.out_dir / f"{name}_0_noisy.wav", mix, SAMPLE_RATE)
                sf.write(args.out_dir / f"{name}_1_pretrained.wav", enh_pre, SAMPLE_RATE)
                sf.write(args.out_dir / f"{name}_2_finetuned.wav", enh_fine, SAMPLE_RATE)
                sf.write(args.out_dir / f"{name}_3_clean.wav", target, SAMPLE_RATE)

                s_noisy = score_pair(target, mix)["pesq_wb"]
                s_pre = score_pair(target, enh_pre)["pesq_wb"]
                s_fine = score_pair(target, enh_fine)["pesq_wb"]
                rows.append((name, s_noisy, s_pre, s_fine))
                print(f"{name:34s} {s_noisy:7.2f} {s_pre:7.2f} {s_fine:7.2f} {s_fine - s_pre:+7.2f}")

    if rows:
        n = len(rows)
        print(f"\n{'MEAN':34s} {sum(r[1] for r in rows)/n:7.2f} "
              f"{sum(r[2] for r in rows)/n:7.2f} {sum(r[3] for r in rows)/n:7.2f} "
              f"{sum(r[3]-r[2] for r in rows)/n:+7.2f}")
    print(f"\nwrote {len(rows) * 4} files to {args.out_dir}")
    print("\nListen in order: _0_noisy -> _1_pretrained -> _2_finetuned -> _3_clean")
    print("Note: PESQ on a single clip is noisy. These are for listening; the")
    print("statistical claims come from eval_impulsive_noise.py at 100 trials/cell.")


if __name__ == "__main__":
    main()
