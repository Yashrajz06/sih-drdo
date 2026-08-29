"""
Build a complete before/after demo package: two audio files, an annotated
spectrogram comparison, and a metrics table.

The point of the spectrogram is to make the problem *visible*. Judges hear the
difference for a few seconds and then have to take your word for it; a picture of
the noise -- gunfire as vertical bars, rotor as horizontal stripes, speech as the
harmonic stack in between -- lets them see exactly what was removed and, just as
importantly, that the speech survived.

Two modes:

  SYNTHETIC (default) -- mix known clean speech with the defence-noise soundtrack.
    Because the clean signal is known, full metrics (PESQ/STOI/SI-SNR) can be
    computed. Use this for the slide.

  REAL CAPTURE (--from-capture) -- process an actual microphone recording.
    More convincing, but there is no clean reference, so no PESQ. Honest framing:
    "recorded in this room" beats a better number nobody can verify.

Usage:
    python scripts/make_demo_package.py
    python scripts/make_demo_package.py --from-capture battlefield_demo_raw.wav
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_baseline_eval import load_resampled, score_pair  # noqa: E402
from streaming_engine import HOP, StreamingEnhancer  # noqa: E402

SAMPLE_RATE = 16000
DEFAULT_MODEL = REPO_ROOT / "models" / "gtcrn_defence.onnx"


def run_model(mix: np.ndarray, onnx_path: Path) -> np.ndarray:
    """Process through the real streaming engine -- the same code path the live
    demo uses, so the files you play are what the system actually produces."""
    enhancer = StreamingEnhancer(onnx_path)
    n = len(mix) // HOP
    return np.concatenate([enhancer.process_hop(mix[i * HOP : (i + 1) * HOP]) for i in range(n)])


def measure_lag(reference: np.ndarray, signal: np.ndarray, max_lag: int = 2048) -> int:
    """Samples by which `signal` trails `reference`, by cross-correlation."""
    from scipy.signal import correlate

    n = min(len(reference), len(signal))
    corr = correlate(signal[:n], reference[:n], mode="full")
    centre = n - 1
    window = corr[centre : centre + max_lag]
    return int(np.argmax(np.abs(window)))


def align(reference: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Compensate the streaming engine's constant output delay before scoring.

    Overlap-add reconstruction emits each hop one frame after the input that
    produced it -- a fixed 256-sample (16 ms) lag here. PESQ, STOI and SI-SNR all
    assume time-aligned signals; scoring without this correction reports the
    enhancement as catastrophic damage (measured once at STOI 0.649, SI-SNR
    -29 dB, when the aligned truth was an improvement). PESQ happens to survive
    because P.862 realigns internally, which is exactly what makes the bug easy
    to miss -- one metric looks fine while the others are nonsense.
    """
    lag = measure_lag(reference, signal)
    if lag > 0:
        signal = signal[lag:]
    n = min(len(reference), len(signal))
    return reference[:n], signal[:n], lag


def build_synthetic(args) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    clean_files = sorted((REPO_ROOT / "data" / "vctk_demand_testset" / "clean").glob("*.wav"))
    if not clean_files:
        sys.exit("no clean speech found -- download the VCTK-DEMAND test set first")
    noise_path = args.noise_file or (REPO_ROOT / "demo_noise.wav")
    if not noise_path.exists():
        sys.exit("demo_noise.wav missing -- run: python scripts/make_demo_noise.py")

    rng = random.Random(args.seed)
    # Concatenate a few utterances so there is enough speech to judge by ear.
    clean = np.concatenate([load_resampled(rng.choice(clean_files)) for _ in range(3)])
    clean = clean[: int(args.seconds * SAMPLE_RATE)]

    noise, _ = sf.read(noise_path, dtype="float32")
    idx = (np.arange(len(clean)) + rng.randint(0, len(noise))) % len(noise)
    noise = noise[idx]

    # Scale to a target SNR rather than a raw gain. A gain figure means nothing
    # across different noise files; an SNR is comparable and is what the metrics
    # tables are indexed by.
    def rms(x):
        return float(np.sqrt(np.mean(x**2) + 1e-12))

    noise = noise * (rms(clean) / (10 ** (args.snr / 20)) / rms(noise))
    mix = clean + noise

    peak = np.abs(mix).max()
    if peak > 0.99:
        scale = 0.99 / peak
        mix, clean = mix * scale, clean * scale
    return mix.astype(np.float32), clean.astype(np.float32), clean.astype(np.float32)


def spectrogram_figure(noisy, enhanced, out_path: Path, clean=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    panels = [("Before — microphone signal", noisy), ("After — enhanced output", enhanced)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 6.2), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (title, sig) in zip(axes, panels):
        f, t, Sxx = spectrogram(sig, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
        db = 10 * np.log10(Sxx + 1e-10)
        ax.pcolormesh(t, f / 1000, db, shading="gouraud", cmap="magma", vmin=-100, vmax=-25)
        ax.set_ylabel("kHz", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=6)
        ax.set_ylim(0, 8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("seconds", fontsize=10)

    fig.suptitle(
        "What the noise looks like, and what the model removes",
        fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=0.985,
    )
    fig.text(
        0.012, 0.925,
        "Vertical streaks = gunfire · horizontal bands = rotor/engine · "
        "stacked curves in the middle = the voice we must keep",
        fontsize=9.5, color="#475569",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-capture", type=Path, help="process a real mic recording instead of synthesising")
    p.add_argument("--onnx", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "demo_package")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--snr", type=float, default=3.0,
                   help="target signal-to-noise ratio in dB. Lower = harder to understand. "
                        "Below ~0 dB the model cannot fully recover the speech either.")
    p.add_argument("--noise-file", type=Path, default=None, help="override demo_noise.wav")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if not args.onnx.exists():
        sys.exit(f"model not found: {args.onnx}\nRun: python scripts/export_onnx.py --checkpoint checkpoints/finetuned_run2.pt --out {args.onnx}")

    if args.from_capture:
        if not args.from_capture.exists():
            sys.exit(f"no such file: {args.from_capture}")
        noisy = load_resampled(args.from_capture)
        reference = None
        print(f"source: real capture — {args.from_capture}")
    else:
        noisy, _, reference = build_synthetic(args)
        print("source: synthetic mix (clean speech + demo_noise.wav)")

    print("processing through the streaming engine...")
    enhanced = run_model(noisy, args.onnx)
    noisy = noisy[: len(enhanced)]
    if reference is not None:
        reference = reference[: len(enhanced)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    before = args.out_dir / "1_BEFORE_noisy.wav"
    after = args.out_dir / "2_AFTER_enhanced.wav"
    sf.write(before, noisy, SAMPLE_RATE)
    sf.write(after, enhanced, SAMPLE_RATE)

    fig_path = args.out_dir / "spectrogram_comparison.png"
    print("rendering spectrogram comparison...")
    spectrogram_figure(noisy, enhanced, fig_path, clean=reference)

    if reference is not None:
        ref_a, enh_a, lag = align(reference, enhanced)
        ref_b, noisy_a, _ = align(reference, noisy)
        n = min(len(ref_a), len(ref_b))
        b = score_pair(ref_b[:n], noisy_a[:n])
        a = score_pair(ref_a[:n], enh_a[:n])
        print(f"\nstreaming output delay: {lag} samples ({lag / SAMPLE_RATE * 1000:.0f} ms) "
              "— compensated before scoring")
        print(f"\n{'':22s} {'PESQ':>7s} {'STOI':>7s} {'SI-SNR':>9s}")
        print(f"{'BEFORE (noisy)':22s} {b['pesq_wb']:7.2f} {b['stoi']:7.3f} {b['si_snr']:8.1f} dB")
        print(f"{'AFTER  (enhanced)':22s} {a['pesq_wb']:7.2f} {a['stoi']:7.3f} {a['si_snr']:8.1f} dB")
        print(f"{'improvement':22s} {a['pesq_wb']-b['pesq_wb']:+7.2f} "
              f"{a['stoi']-b['stoi']:+7.3f} {a['si_snr']-b['si_snr']:+8.1f} dB")
    else:
        # No clean reference exists for a real recording, so intrusive metrics are
        # undefined. Saying so is better than quoting a number that cannot be computed.
        print("  (real capture — no clean reference, so PESQ/STOI/SI-SNR cannot be computed)")
        print(f"  noise floor before: {20*np.log10(np.sqrt(np.mean(noisy**2))+1e-9):6.1f} dBFS")
        print(f"  noise floor after:  {20*np.log10(np.sqrt(np.mean(enhanced**2))+1e-9):6.1f} dBFS")

    print(f"\nwrote to {args.out_dir}/")
    print(f"  {before.name}\n  {after.name}\n  {fig_path.name}")
    print(f"\nPlay them:\n  aplay {before}\n  aplay {after}")


if __name__ == "__main__":
    main()
