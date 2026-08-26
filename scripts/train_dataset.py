"""
Dataset pipeline -- problem-statement deliverable 1: "a scalable dataset pipeline
for generating realistic noisy-clean speech pairs."

Mixing happens on the fly, a fresh random combination every time an item is drawn,
rather than pre-generating a fixed corpus. The PS asks for coverage of "both
stationary and impulsive noise scenarios" at "varying SNR levels"; dynamic mixing
means the model sees far more unique speech/noise/SNR/position combinations than
disk would hold, which is the standard DNS-Challenge approach.

Design decisions that matter, and why:

- **Impulsive oversampling.** Gunfire and shelling are the cases the PS singles out
  and the ones our baseline showed classical DSP failing on, so they are drawn more
  often than their natural share of the corpus.

- **Transients are placed over *voiced* speech.** If a gunshot only ever lands in a
  silent gap, removing it is trivial and the model learns nothing useful. We locate
  high-energy (voiced) regions of the clean utterance and place events there.

- **Speech-contaminated noise clips are excluded** (see mad_noise.SPEECH_EXCLUSION_
  THRESHOLD). Some MAD clips carry soldiers shouting under the gunfire; training on
  those teaches the model that voices are noise to delete.

- **SNR is computed over active speech frames**, not the whole clip. LibriSpeech
  utterances have leading/trailing silence, and including it in the level estimate
  makes the effective in-speech SNR higher than the label claims.

Augmentation implements the PS's list -- "random noise mixing, reverberation,
clipping" -- plus band-limiting to emulate a radio channel.
"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from mad_noise import CATEGORIES, load_clip, load_split  # noqa: E402

SAMPLE_RATE = 16000
EPS = 1e-8


def active_speech_rms(wav: np.ndarray, frame: int = 400, threshold_db: float = -40.0) -> float:
    """RMS over frames that actually contain speech.

    A whole-clip RMS is dragged down by leading/trailing silence, which makes the
    real SNR during speech higher than the nominal label. This is a simple
    energy-gated approximation of ITU-T P.56 active speech level -- not the full
    standard, but it removes the dominant bias.
    """
    if len(wav) < frame:
        return float(np.sqrt(np.mean(wav**2) + EPS))
    n_frames = len(wav) // frame
    frames = wav[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt((frames**2).mean(axis=1) + EPS)
    peak = frame_rms.max()
    active = frame_rms[frame_rms > peak * (10 ** (threshold_db / 20))]
    return float(active.mean()) if active.size else float(frame_rms.mean())


def voiced_positions(wav: np.ndarray, frame: int = 400, top_fraction: float = 0.5) -> np.ndarray:
    """Sample indices inside the loudest (i.e. voiced) frames of the utterance.
    Used to place impulsive events where they actually overlap speech."""
    n_frames = len(wav) // frame
    if n_frames < 2:
        return np.array([0])
    frames = wav[: n_frames * frame].reshape(n_frames, frame)
    energy = (frames**2).mean(axis=1)
    n_keep = max(1, int(n_frames * top_fraction))
    loud = np.argsort(energy)[-n_keep:]
    return loud * frame


class NoisySpeechDataset(Dataset):
    """Yields (noisy, clean) float32 waveform pairs of fixed length."""

    def __init__(
        self,
        clean_files: list[Path],
        split: str = "training",
        segment_seconds: float = 4.0,
        snr_range: tuple[float, float] = (-5.0, 20.0),
        impulsive_weight: float = 0.5,
        epoch_size: int = 2000,
        rir_files: list[Path] | None = None,
        rir_prob: float = 0.0,
        clip_prob: float = 0.1,
        bandlimit_prob: float = 0.2,
        seed: int = 0,
    ):
        if not clean_files:
            raise ValueError("no clean speech files given")
        self.clean_files = clean_files
        self.segment = int(segment_seconds * SAMPLE_RATE)
        self.snr_range = snr_range
        self.epoch_size = epoch_size
        self.rir_files = rir_files or []
        self.rir_prob = rir_prob if self.rir_files else 0.0
        self.clip_prob = clip_prob
        self.bandlimit_prob = bandlimit_prob
        self.base_seed = seed

        # Training noise only, with speech-contaminated clips removed.
        by_cat = load_split(split, exclude_speech=True)
        self.noise_by_cat = {k: v for k, v in by_cat.items() if v}
        if not self.noise_by_cat:
            raise ValueError(f"no usable noise clips in MAD '{split}' split")

        # Oversample impulsive relative to its natural share -- the PS's hard case.
        other = [c for c in self.noise_by_cat if c != "impulsive"]
        if "impulsive" in self.noise_by_cat and other:
            rest = (1.0 - impulsive_weight) / len(other)
            self.cat_weights = {c: (impulsive_weight if c == "impulsive" else rest) for c in self.noise_by_cat}
        else:
            self.cat_weights = {c: 1.0 / len(self.noise_by_cat) for c in self.noise_by_cat}

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self, idx: int) -> random.Random:
        # Seeded per item so a given (seed, idx) is reproducible, but each epoch
        # still draws fresh combinations via set_epoch().
        return random.Random((self.base_seed, idx).__hash__())

    def set_epoch(self, epoch: int) -> None:
        self.base_seed = self.base_seed + 1000003 * epoch

    def _load_clean_segment(self, rng: random.Random) -> np.ndarray:
        for _ in range(10):
            path = rng.choice(self.clean_files)
            try:
                wav, fs = sf.read(path, dtype="float32")
            except Exception:
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if fs != SAMPLE_RATE:
                wav = torch.from_numpy(wav)
                import torchaudio

                wav = torchaudio.functional.resample(wav, fs, SAMPLE_RATE).numpy()
            if len(wav) < self.segment // 2:
                continue
            if len(wav) < self.segment:
                wav = np.pad(wav, (0, self.segment - len(wav)))
            else:
                start = rng.randint(0, len(wav) - self.segment)
                wav = wav[start : start + self.segment]
            if np.abs(wav).max() > 1e-4:
                return wav.astype(np.float32)
        return np.zeros(self.segment, dtype=np.float32)

    def _sample_category(self, rng: random.Random) -> str:
        cats = list(self.cat_weights)
        return rng.choices(cats, weights=[self.cat_weights[c] for c in cats], k=1)[0]

    def _noise_bed(self, rng: random.Random, category: str) -> np.ndarray:
        """A continuous noise bed the length of the segment (tiled/cropped)."""
        for _ in range(5):
            try:
                noise = load_clip(rng.choice(self.noise_by_cat[category]))
            except Exception:
                continue
            if len(noise) == 0:
                continue
            if len(noise) < self.segment:
                noise = np.tile(noise, int(np.ceil(self.segment / len(noise))))
            start = rng.randint(0, max(0, len(noise) - self.segment))
            return noise[start : start + self.segment].astype(np.float32)
        return np.zeros(self.segment, dtype=np.float32)

    def _add_impulsive_events(self, noise: np.ndarray, clean: np.ndarray, rng: random.Random) -> np.ndarray:
        """Drop 0-3 transients on top, aligned to voiced speech regions."""
        if "impulsive" not in self.noise_by_cat:
            return noise
        n_events = rng.choices([0, 1, 2, 3], weights=[0.25, 0.4, 0.25, 0.1], k=1)[0]
        if n_events == 0:
            return noise

        positions = voiced_positions(clean)
        out = noise.copy()
        for _ in range(n_events):
            try:
                event = load_clip(rng.choice(self.noise_by_cat["impulsive"]))
            except Exception:
                continue
            if len(event) == 0:
                continue
            # Take the loudest ~0.5 s window -- that's the transient itself, not the
            # quiet tail of the recording.
            win = min(len(event), int(0.5 * SAMPLE_RATE))
            if len(event) > win:
                energy = np.convolve(event**2, np.ones(win), mode="valid")
                event = event[int(np.argmax(energy)) : int(np.argmax(energy)) + win]
            pos = int(rng.choice(positions))
            end = min(len(out), pos + len(event))
            seg = event[: end - pos]
            if seg.size == 0:
                continue
            # Event-to-noise ratio varies so the model sees a range of severities.
            gain = 10 ** (rng.uniform(0.0, 12.0) / 20)
            out[pos:end] = out[pos:end] + seg * gain
        return out

    def _augment_clean(self, clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        """Returns (clean_for_mixing, target). With reverb the target stays the
        anechoic signal, so the model learns to dereverberate as well as denoise."""
        target = clean
        if self.rir_files and rng.random() < self.rir_prob:
            try:
                rir, fs = sf.read(rng.choice(self.rir_files), dtype="float32")
                if rir.ndim > 1:
                    rir = rir[:, 0]
                if fs == SAMPLE_RATE and len(rir) > 1:
                    rir = rir / (np.abs(rir).max() + EPS)
                    wet = np.convolve(clean, rir)[: len(clean)]
                    if np.abs(wet).max() > 1e-6:
                        return wet.astype(np.float32), target
            except Exception:
                pass
        return clean, target

    def __getitem__(self, idx: int):
        rng = self._rng(idx)

        clean_raw = self._load_clean_segment(rng)
        clean_mix, target = self._augment_clean(clean_raw, rng)

        category = self._sample_category(rng)
        noise = self._noise_bed(rng, category)
        noise = self._add_impulsive_events(noise, clean_mix, rng)

        # Scale noise to hit the requested SNR against ACTIVE speech level.
        snr_db = rng.uniform(*self.snr_range)
        speech_level = active_speech_rms(clean_mix)
        noise_level = np.sqrt(np.mean(noise**2) + EPS)
        target_noise_level = speech_level / (10 ** (snr_db / 20))
        noise = noise * (target_noise_level / (noise_level + EPS))

        noisy = clean_mix + noise

        # Band-limiting: crude emulation of a narrowband radio channel.
        if rng.random() < self.bandlimit_prob:
            from scipy.signal import butter, lfilter

            cutoff = rng.uniform(3000, 7000) / (SAMPLE_RATE / 2)
            b, a = butter(4, min(cutoff, 0.99), btype="low")
            noisy = lfilter(b, a, noisy).astype(np.float32)

        # Random overall gain, then clipping (both listed in the PS).
        gain = 10 ** (rng.uniform(-6, 3) / 20)
        noisy, target = noisy * gain, target * gain
        if rng.random() < self.clip_prob:
            ceiling = rng.uniform(0.3, 0.9) * (np.abs(noisy).max() + EPS)
            noisy = np.clip(noisy, -ceiling, ceiling)

        # Prevent digital clipping; scale mixture and target together so the
        # relationship the model must learn is unchanged.
        peak = max(np.abs(noisy).max(), np.abs(target).max())
        if peak > 0.99:
            scale = 0.99 / peak
            noisy, target = noisy * scale, target * scale

        return (
            torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
        )


def find_clean_files(root: Path, extensions=(".flac", ".wav")) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the dataset pipeline")
    parser.add_argument("--clean-dir", type=Path, default=REPO_ROOT / "data" / "librispeech")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    clean = find_clean_files(args.clean_dir)
    print(f"clean speech files found: {len(clean)}")
    if not clean:
        sys.exit(f"none under {args.clean_dir} -- download LibriSpeech first")

    ds = NoisySpeechDataset(clean_files=clean, epoch_size=args.n)
    print(f"noise categories: { {k: len(v) for k, v in ds.noise_by_cat.items()} }")
    print(f"category sampling weights: {ds.cat_weights}\n")

    for i in range(args.n):
        noisy, target = ds[i]
        # Verify the achieved SNR matches intent (measured on active speech).
        resid = (noisy - target).numpy()
        snr = 20 * np.log10(active_speech_rms(target.numpy()) / (np.sqrt(np.mean(resid**2)) + EPS) + EPS)
        print(
            f"  item {i}: noisy{tuple(noisy.shape)} target{tuple(target.shape)}  "
            f"peak {noisy.abs().max():.3f}  measured SNR {snr:+.1f} dB"
        )
