"""
Loader for the Military Audio Dataset (MAD) -- Kim, Yoon & Jung, "A Military
Audio Dataset for Situational Awareness and Surveillance," Scientific Data
11:668 (2024). CC BY 4.0. Source: https://github.com/kaen2891/military_audio_dataset
(audio via Kaggle: kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset,
not redistributed here -- data/ is gitignored).

Expects data/MAD_dataset/{training,test}.csv with columns
`path,label,...` (path relative to data/MAD_dataset/), and the audio at those
paths already 16kHz mono PCM (true for the Kaggle release -- verified, no
resampling done here).

Label -> class name (confirmed from the dataset repo's main.py):
    0 communication  1 shooting  2 footsteps  3 shelling
    4 vehicle        5 helicopter  6 fighter

`communication` is radio chatter / other people's speech, not background
noise -- excluded from every category below. Using it as "noise" would train
the model to suppress speech, which is the opposite of the goal.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
MAD_DIR = REPO_ROOT / "data" / "MAD_dataset"

# Gate 0.3: MAD clips are cut from YouTube combat footage, so some carry soldiers
# shouting or radio chatter *underneath* the gunfire. Training a denoiser on noise
# that contains speech teaches it to remove speech. scripts/screen_noise_speech.py
# scores every clip with a VAD; clips above this threshold are dropped.
#
# Threshold validated by human listening (2026-08-26): clips scoring 0.70-1.00 were
# confirmed to contain audible human voices mixed with gunfire, so the detector is
# finding real speech rather than mistaking transients for it. 0.1 is deliberately
# conservative -- the cost is asymmetric, since leaving speech in corrupts the
# training objective while dropping clips only costs a little data (3.9% overall,
# 7% of the impulsive class, leaving 2742 impulsive clips).
SPEECH_EXCLUSION_THRESHOLD = 0.1

LABEL_NAMES = {
    0: "communication",
    1: "shooting",
    2: "footsteps",
    3: "shelling",
    4: "vehicle",
    5: "helicopter",
    6: "fighter",
}

# Design-doc taxonomy (docs/solution-design.md SS4): stationary / non-stationary /
# impulsive. footsteps are brief transients like gunfire, not steady like engine
# drone, so grouped under impulsive -- a judgment call, not from the source paper.
CATEGORIES = {
    "stationary": {"vehicle"},
    "non_stationary": {"helicopter", "fighter"},
    "impulsive": {"shooting", "shelling", "footsteps"},
}


def _read_split(csv_name: str) -> list[dict]:
    csv_path = MAD_DIR / csv_name
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    out = []
    missing = 0
    for row in rows:
        label = int(row["label"])
        if label == 0:  # communication -- speech, not noise
            continue
        wav_path = MAD_DIR / row["path"]
        if not wav_path.exists():
            missing += 1
            continue
        out.append({"path": wav_path, "class_name": LABEL_NAMES[label]})

    if missing:
        print(f"note: {missing} clips listed in {csv_name} have no audio file on disk", file=sys.stderr)
    return out


def _speech_excluded(split: str, threshold: float) -> set[Path]:
    """Paths flagged as speech-contaminated by scripts/screen_noise_speech.py.
    Returns empty if the manifest hasn't been generated -- callers that require
    filtering should check `exclude_speech=True` actually removed something."""
    manifest = REPO_ROOT / f"noise_speech_scores_{split}.json"
    if not manifest.exists():
        return set()
    rows = json.loads(manifest.read_text())
    return {REPO_ROOT / r["path"] for r in rows if r["speech_fraction"] > threshold}


def load_split(
    split: str,
    exclude_speech: bool = False,
    threshold: float = SPEECH_EXCLUSION_THRESHOLD,
) -> dict[str, list[Path]]:
    """split: 'training' or 'test'. Returns {category: [wav paths]}.

    exclude_speech: drop clips a VAD flagged as containing speech. Use for *training*
    noise. Evaluation deliberately defaults to False so the held-out test set stays
    exactly as published -- filtering the eval set too would let us quietly grade
    ourselves on an easier problem than the one we describe.
    """
    if split not in ("training", "test"):
        raise ValueError("split must be 'training' or 'test'")
    rows = _read_split(f"{split}.csv")

    excluded = _speech_excluded(split, threshold) if exclude_speech else set()
    if exclude_speech and not excluded:
        print(
            f"warning: exclude_speech=True but no exclusions found for '{split}'. "
            f"Run scripts/screen_noise_speech.py --split {split} first.",
            file=sys.stderr,
        )

    by_category: dict[str, list[Path]] = {cat: [] for cat in CATEGORIES}
    for row in rows:
        if row["path"] in excluded:
            continue
        for cat, class_names in CATEGORIES.items():
            if row["class_name"] in class_names:
                by_category[cat].append(row["path"])
                break
    return by_category


def load_clip(path: Path, target_sr: int = 16000) -> np.ndarray:
    wav, fs = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if fs != target_sr:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(torch.from_numpy(wav), fs, target_sr).numpy()
    return wav


def _check() -> None:
    if not MAD_DIR.exists():
        print(f"MAD dataset not found at {MAD_DIR}.")
        print("Download from https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset")
        print(f"and extract so audio lands at {MAD_DIR}/{{training,test}}/<video_num>/<file_id>.wav")
        sys.exit(1)

    for split in ("training", "test"):
        by_cat = load_split(split)
        total = sum(len(v) for v in by_cat.values())
        print(f"\n{split}: {total} usable noise clips (communication excluded)")
        for cat, clips in by_cat.items():
            classes = ", ".join(sorted(CATEGORIES[cat]))
            print(f"  {cat:15s} ({classes}): {len(clips)} clips")

        if split != "training":
            # The eval split is intentionally left unfiltered -- see load_split().
            continue
        filtered = load_split(split, exclude_speech=True)
        n_filtered = sum(len(v) for v in filtered.values())
        if n_filtered != total:
            print(f"  -> with speech filter (>{SPEECH_EXCLUSION_THRESHOLD}): {n_filtered} clips "
                  f"({total - n_filtered} dropped)")
            for cat, clips in filtered.items():
                print(f"     {cat:15s}: {len(clips)}")


if __name__ == "__main__":
    _check()
