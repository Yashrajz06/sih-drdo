"""
Stage 0 baseline: run pretrained GTCRN on the VCTK-DEMAND test set and report
PESQ (wb) / STOI / SI-SNR, both for the enhanced output and for the raw noisy
input (so the improvement delta is visible). This is a sanity check that the
model + checkpoint + STFT pipeline are wired correctly -- not a from-scratch
reimplementation of anything in third_party/gtcrn.

Usage:
    python scripts/run_baseline_eval.py --limit 20
    python scripts/run_baseline_eval.py                 # full 824-pair test set
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi

REPO_ROOT = Path(__file__).resolve().parent.parent
GTCRN_DIR = REPO_ROOT / "third_party" / "gtcrn"
sys.path.insert(0, str(GTCRN_DIR))
from gtcrn import GTCRN  # noqa: E402

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 256
WINDOW = torch.hann_window(N_FFT).pow(0.5)


def load_model(checkpoint_name: str) -> torch.nn.Module:
    """Accepts either a bare filename in third_party/gtcrn/checkpoints/ or a path to
    one of our own fine-tuned checkpoints (which also carry optimizer state)."""
    candidate = Path(checkpoint_name)
    path = candidate if candidate.exists() else GTCRN_DIR / "checkpoints" / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    model = GTCRN().eval()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model


def load_resampled(path: Path) -> np.ndarray:
    wav, fs = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # downmix to mono
    if fs != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), fs, SAMPLE_RATE
        ).numpy()
    return wav


@torch.no_grad()
def enhance(model: torch.nn.Module, mix: np.ndarray) -> np.ndarray:
    # torch>=2.x requires stft/istft to use complex dtype; the model (written for an
    # older torch) expects/returns the (..., 2) real/imag layout, so we convert at the
    # boundary with view_as_real / view_as_complex instead of touching vendored code.
    spec_c = torch.stft(torch.from_numpy(mix), N_FFT, HOP, N_FFT, WINDOW, return_complex=True)
    spec = torch.view_as_real(spec_c)
    out_spec = model(spec[None])[0]
    out_c = torch.view_as_complex(out_spec.contiguous())
    enh = torch.istft(out_c, N_FFT, HOP, N_FFT, WINDOW, return_complex=False)
    return enh.numpy()


def si_snr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    proj = (np.dot(estimate, reference) / (np.dot(reference, reference) + eps)) * reference
    noise = estimate - proj
    return float(10 * np.log10((np.sum(proj**2) + eps) / (np.sum(noise**2) + eps)))


def score_pair(clean: np.ndarray, degraded: np.ndarray) -> dict:
    n = min(len(clean), len(degraded))
    clean, degraded = clean[:n], degraded[:n]
    return {
        "pesq_wb": float(pesq(SAMPLE_RATE, clean, degraded, "wb")),
        "stoi": float(stoi(clean, degraded, SAMPLE_RATE, extended=False)),
        "si_snr": si_snr(clean, degraded),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "vctk_demand_testset",
        help="directory containing clean/ and noisy/ subfolders",
    )
    parser.add_argument(
        "--checkpoint",
        default="model_trained_on_vctk.tar",
        help="checkpoint file in third_party/gtcrn/checkpoints/",
    )
    parser.add_argument("--limit", type=int, default=None, help="only score the first N pairs")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results.json", help="where to write per-file + aggregate results"
    )
    args = parser.parse_args()

    clean_dir = args.data_dir / "clean"
    noisy_dir = args.data_dir / "noisy"
    if not clean_dir.is_dir() or not noisy_dir.is_dir():
        sys.exit(
            f"Expected {clean_dir} and {noisy_dir} to exist. "
            "Download/unzip the VCTK-DEMAND test set first."
        )

    clean_files = sorted(clean_dir.glob("*.wav"))
    if args.limit:
        clean_files = clean_files[: args.limit]
    if not clean_files:
        sys.exit(f"No .wav files found in {clean_dir}")

    model = load_model(args.checkpoint)

    per_file = []
    for clean_path in clean_files:
        noisy_path = noisy_dir / clean_path.name
        if not noisy_path.exists():
            print(f"skip {clean_path.name}: no matching noisy file", file=sys.stderr)
            continue

        clean = load_resampled(clean_path)
        noisy = load_resampled(noisy_path)
        enh = enhance(model, noisy)

        entry = {
            "file": clean_path.name,
            "noisy": score_pair(clean, noisy),
            "enhanced": score_pair(clean, enh),
        }
        per_file.append(entry)
        print(
            f"{clean_path.name:20s} "
            f"PESQ {entry['noisy']['pesq_wb']:.2f} -> {entry['enhanced']['pesq_wb']:.2f}  "
            f"STOI {entry['noisy']['stoi']:.3f} -> {entry['enhanced']['stoi']:.3f}  "
            f"SI-SNR {entry['noisy']['si_snr']:.2f} -> {entry['enhanced']['si_snr']:.2f}"
        )

    def avg(key_outer, key_inner):
        vals = [e[key_outer][key_inner] for e in per_file]
        return sum(vals) / len(vals)

    summary = {
        "n_files": len(per_file),
        "checkpoint": args.checkpoint,
        "noisy": {
            "pesq_wb": avg("noisy", "pesq_wb"),
            "stoi": avg("noisy", "stoi"),
            "si_snr": avg("noisy", "si_snr"),
        },
        "enhanced": {
            "pesq_wb": avg("enhanced", "pesq_wb"),
            "stoi": avg("enhanced", "stoi"),
            "si_snr": avg("enhanced", "si_snr"),
        },
    }

    print("\n=== summary ({} files, checkpoint={}) ===".format(summary["n_files"], args.checkpoint))
    print(
        f"noisy:    PESQ {summary['noisy']['pesq_wb']:.3f}  STOI {summary['noisy']['stoi']:.3f}  SI-SNR {summary['noisy']['si_snr']:.2f} dB"
    )
    print(
        f"enhanced: PESQ {summary['enhanced']['pesq_wb']:.3f}  STOI {summary['enhanced']['stoi']:.3f}  SI-SNR {summary['enhanced']['si_snr']:.2f} dB"
    )

    args.out.write_text(json.dumps({"summary": summary, "per_file": per_file}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
