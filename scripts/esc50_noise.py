"""
ESC-50 noise loader -- covers the noise types the problem statement names that the
Military Audio Dataset does not have.

The PS names: "gunshots, artillery fire, helicopter rotor noise, armored vehicle sound
and emergency sirens" and later "gunshots, drones, artillery, vehicle engines, wind".
MAD covers gunfire, shelling, helicopter and vehicle. It has no **siren**, no **wind**,
and no drone. ESC-50 supplies siren and wind directly, plus airplane/helicopter as the
nearest available stand-ins for rotor/propeller drone.

Honest gap: ESC-50 has no quadcopter-drone class either. MAD's helicopter/fighter and
ESC-50's airplane are acoustically related (rotor/turbine tonal noise) but they are not
the same thing, and we say so rather than claiming drone coverage we don't have.

LICENCE WARNING -- ESC-50 is **CC BY-NC** (non-commercial). docs/solution-design.md
flags this: acceptable for an academic prototype with attribution, but it must be
labelled and must not be presented as deployment-ready data. MAD (CC BY 4.0) and
LibriSpeech (CC BY 4.0) carry no such restriction.

    Piczak, "ESC: Dataset for Environmental Sound Classification," ACM Multimedia 2015.
    https://github.com/karolpiczak/ESC-50

Usage:
    python scripts/esc50_noise.py           # show what's available
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ESC50_DIR = REPO_ROOT / "data" / "esc50"

# ESC-50 class -> our PS-aligned category.
#
# Deliberately narrow. ESC-50 has 50 classes but only a handful belong in a defence
# acoustic scene, and adding weak proxies would make the evaluation *look* broader
# while actually measuring something irrelevant. Household sources (vacuum_cleaner,
# washing_machine, can_opening, door_wood_knock, glass_breaking) were mapped in an
# earlier version and removed: they are not defence noises, and MAD already supplies
# real gunfire and shelling, which are far better impulsive material than any ESC-50
# stand-in.
#
# What remains is the genuine gap MAD does not cover, plus two closely-related
# rotor/turbine sources.
CLASS_CATEGORIES = {
    "wind": "stationary",              # PS-named, absent from MAD
    "engine": "stationary",
    "siren": "non_stationary",         # PS-named, absent from MAD
    "helicopter": "non_stationary",
    "airplane": "non_stationary",
    "chainsaw": "non_stationary",      # sustained broadband machinery
    "fireworks": "impulsive",          # nearest ESC-50 analogue to ordnance
    "thunderstorm": "impulsive",       # broadband transient
}

# The specific classes the PS names and MAD lacks -- reported separately so we can say
# exactly what the added coverage buys.
PS_GAP_CLASSES = ("siren", "wind")


def _meta_path() -> Path | None:
    for candidate in ESC50_DIR.rglob("esc50.csv"):
        return candidate
    return None


def load_clips(categories: tuple[str, ...] | None = None) -> dict[str, list[Path]]:
    """Returns {category: [wav paths]} for the defence-relevant ESC-50 classes."""
    meta = _meta_path()
    if meta is None:
        return {}
    audio_dir = meta.parent.parent / "audio"
    if not audio_dir.is_dir():
        return {}

    by_category: dict[str, list[Path]] = {}
    with open(meta) as f:
        for row in csv.DictReader(f):
            category = CLASS_CATEGORIES.get(row["category"])
            if category is None or (categories and category not in categories):
                continue
            path = audio_dir / row["filename"]
            if path.exists():
                by_category.setdefault(category, []).append(path)
    return by_category


def load_by_class(class_names: tuple[str, ...]) -> dict[str, list[Path]]:
    """Returns {esc50_class_name: [wav paths]} -- for testing specific named noises."""
    meta = _meta_path()
    if meta is None:
        return {}
    audio_dir = meta.parent.parent / "audio"
    out: dict[str, list[Path]] = {}
    with open(meta) as f:
        for row in csv.DictReader(f):
            if row["category"] in class_names:
                path = audio_dir / row["filename"]
                if path.exists():
                    out.setdefault(row["category"], []).append(path)
    return out


def load_clip(path: Path, target_sr: int = 16000, trim_seconds: float = 2.0) -> np.ndarray:
    """ESC-50 is 44.1 kHz; resample to our 16 kHz mono convention and trim to the
    most energetic window.

    The trim matters. ESC-50 clips are a fixed 5 s and many contain long silences
    around a short event. Scaling such a clip to a target SNR by whole-clip RMS
    produces a near-silent noise bed -- an early run showed an input SI-SNR of
    +39.9 dB on a nominally +10 dB mixture, i.e. essentially no noise was added and
    the "evaluation" was measuring nothing. Taking the loudest window makes the
    requested SNR correspond to the sound actually present.
    """
    import soundfile as sf

    wav, fs = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if fs != target_sr:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(torch.from_numpy(wav), fs, target_sr).numpy()
    wav = wav.astype(np.float32)

    win = int(trim_seconds * target_sr)
    if len(wav) > win:
        # Sliding-window energy; take the loudest window.
        energy = np.convolve(wav**2, np.ones(win), mode="valid")
        start = int(np.argmax(energy))
        wav = wav[start : start + win]
    return wav


def _check() -> None:
    if _meta_path() is None:
        print(f"ESC-50 not found under {ESC50_DIR}.")
        print("Download: https://github.com/karolpiczak/ESC-50/archive/master.zip")
        print(f"Extract so that {ESC50_DIR}/ESC-50-master/meta/esc50.csv exists.")
        sys.exit(1)

    by_cat = load_clips()
    total = sum(len(v) for v in by_cat.values())
    print(f"ESC-50 defence-relevant clips: {total}\n")
    for category, clips in sorted(by_cat.items()):
        names = sorted(k for k, v in CLASS_CATEGORIES.items() if v == category)
        print(f"  {category:15s} {len(clips):4d} clips  ({', '.join(names)})")

    print("\nPS-named noises MAD lacks, now covered:")
    for name, clips in sorted(load_by_class(PS_GAP_CLASSES).items()):
        print(f"  {name:10s} {len(clips):4d} clips")

    print("\nStill not covered: quadcopter drone (absent from both MAD and ESC-50).")
    print("      helicopter/airplane are related rotor/turbine sources, not equivalents.")
    print("\nLICENCE: ESC-50 is CC BY-NC -- academic use with attribution only.")


if __name__ == "__main__":
    _check()
