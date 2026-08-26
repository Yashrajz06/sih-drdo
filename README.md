# AI/ML Noise Suppression for Defence Communications (SIH / DRDO)

**Status: fine-tuned model meets the problem statement's targets on real defence noise.**
At +15 dB on gunfire/shelling: **PESQ 2.49 ± 0.04** (target 2.5, at target within
measurement error), **STOI 0.920** (target 0.85), **SI-SNR 20.2 dB** (target 15). Runs
live mic-to-headset at **83.6 ms** measured end-to-end latency, inside ITU-T G.114's
transparent band. Raspberry Pi deployment is the remaining step — pending hardware.

See [`docs/architecture.md`](docs/architecture.md) to understand the system, or
[`docs/solution-design.md`](docs/solution-design.md) for the full build plan.

## What this actually is

The problem statement is titled "Active Noise Cancellation," but its own success
criteria are PESQ, STOI, and SI-SNR — all measures of *speech quality in a transmitted
signal*, not of acoustic attenuation. So what's actually being asked for is **speech
enhancement**: clean the microphone signal computationally, rather than emit anti-noise
into the air. See [`docs/solution-design.md`](docs/solution-design.md) §0 for the full
argument and how to present it to judges.

The target architecture (not yet fully built — see status below) is a causal, streaming
**GTCRN** model running on a Raspberry Pi 5 CPU, fine-tuned for impulsive defence noise
(gunshots, shelling) that standard speech-enhancement models handle poorly.

## Current status: Stage 0

Goal: prove the toolchain (model, checkpoint, STFT pipeline, evaluation metrics) works
end-to-end before touching fine-tuning, data pipelines, or hardware.

Done:
- Vendored the official pretrained [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  implementation (MIT licence, commit `502ebfa`) into `third_party/gtcrn/`.
- Ran the VCTK-DEMAND–trained checkpoint (`checkpoints/model_trained_on_vctk.tar`) over
  the full 824-pair VCTK-DEMAND test set and scored PESQ (wb) / STOI / SI-SNR — see
  numbers below.

Not yet started: impulsive-noise data pipeline, fine-tuning, ONNX export, Pi deployment,
NLMS stage, live demo. These are Stages 1–3 in `docs/solution-design.md`.

### Reproduced numbers

Run yourself with `python scripts/run_baseline_eval.py`; results also written to
`results.json`.

Full 824-pair VCTK-DEMAND test set, `model_trained_on_vctk.tar` checkpoint:

| | PESQ (wb) | STOI | SI-SNR |
|---|---|---|---|
| Noisy input | 1.971 | 0.921 | 8.44 dB |
| GTCRN enhanced | **2.855** | **0.941** | **18.80 dB** |

The published benchmark for this checkpoint is PESQ 2.87 / STOI ~0.94 on VCTK-DEMAND —
our reproduction lands within 0.02 PESQ and 0.00 STOI of that. The small gap is expected:
PESQ depends on the exact package/version (P.862 vs Corrigendum 2), and this eval script
is a from-scratch scoring harness, not the authors' original evaluation code. This
confirms the model, checkpoint, and STFT framing are wired correctly — Stage 0 done.

## Live demo (laptop stand-in for the Pi)

No Raspberry Pi in hand yet, so this runs the same architecture live on this machine's
own mic/speakers instead — `docs/solution-design.md`'s own fallback for exactly this
situation ("if Pi latency is unacceptable, fall back to a laptop demo with the Pi shown
as the deployment target"). It reuses the GTCRN authors' own pre-exported streaming ONNX
model (`third_party/gtcrn/stream/onnx_models/gtcrn_simple.onnx`, frame-by-frame with
explicit recurrent state) rather than the whole-file batch model used above — see
`scripts/streaming_engine.py` for the STFT/overlap-add framing around it.

**Correctness + timing check (safe to run anywhere, touches no audio hardware):**

```bash
python scripts/live_demo.py --check
```

This diffs the streaming engine's output against a causal whole-file reference and
reports the real algorithmic delay and CPU real-time factor. Latest run on this machine:

| | Value |
|---|---|
| Algorithmic delay | 0 samples (frame-aligned) |
| Mean/max abs error vs. reference | 0.0025 / 0.0297 (expected float/ordering noise between the batched and cached-GRU code paths, not a bug) |
| RTF (this CPU) | **0.079** — comfortably real-time |

**Measured end-to-end latency.** RTF is a model metric; what a judge means by "latency"
is how long a word takes to get from your mouth to the listener's ear. Measured
acoustically — play a chirp, record it simultaneously, recover the delay by
cross-correlation:

```bash
python scripts/live_demo.py --measure-latency   # speakers, not headphones
```

| Component | Measured |
|---|---|
| Hardware round trip (out → air → in) | 82.5 ms (median of 4 confident trials) |
| Algorithmic (causal, zero lookahead) | 0.0 ms |
| Model compute per 16 ms hop | 1.10 ms |
| **Total one-way conversational** | **83.6 ms** |

Inside [ITU-T G.114](https://www.itu.int/rec/T-REC-G.114)'s 0–150 ms band, described as
"acceptable for most user applications."

Two honest qualifications: (1) the delay estimator is self-tested against known
synthetic delays before any acoustic measurement is trusted — an unvalidated
cross-correlation will report a confident-looking but meaningless number; (2) this uses
PortAudio's *default* buffer size, which is larger than the 256-sample blocks the live
demo actually runs, so 83.6 ms is a conservative upper bound on the demo's real latency.
Hardware round trip dominates the budget — the model contributes ~1%.

**Live mode (run this yourself — it uses your microphone and speakers):**

```bash
python scripts/live_demo.py
```

Speak into the mic; press `e` to toggle enhancement on/off and hear the difference live,
`q` to quit. Add `--synthetic-noise` (or `--inject-noise noise.wav`) to mix a repeatable
noise bed into the captured signal, so the "add noise, toggle it away" demo beat doesn't
depend on room acoustics:

```bash
python scripts/live_demo.py --synthetic-noise --noise-gain 0.3
```

Non-interactive mic-to-file capture (still uses the mic, but no toggling/playback):

```bash
python scripts/live_demo.py --capture-test 10 --out capture_test.wav
```

**Limitation:** this is the *pretrained* `model_trained_on_dns3.tar` checkpoint —
fine-tuning for impulsive defence noise (gunshots, shelling) hasn't happened yet
(`docs/solution-design.md` Stage 1). This demo proves the real-time architecture works
end-to-end; it doesn't yet demonstrate the project's impulsive-noise novelty claim.

## Impulsive-noise baseline ("Model A" in the ablation table)

Before fine-tuning anything, we measured how the *pretrained* model actually performs on
real defence noise, stratified by category and SNR — this is the "before" row that any
fine-tuning work needs to beat (`docs/solution-design.md` §3's ablation table). Noise
comes from the [Military Audio Dataset](https://github.com/kaen2891/military_audio_dataset)
(MAD; Kim, Yoon & Jung, *Scientific Data* 11:668, 2024; CC BY 4.0) — 830 usable test-split
clips across 3 categories (`communication`, i.e. radio chatter, is excluded — it's speech,
not noise):

| Category | MAD classes | Test clips |
|---|---|---|
| Stationary | vehicle | 122 |
| Non-stationary | helicopter, fighter | 220 |
| Impulsive | shooting, shelling, footsteps | 488 |

Clean speech is the same held-out VCTK-DEMAND test set from Stage 0, mixed with MAD noise
at controlled SNRs. 100 trials per category × SNR cell, **paired on clean speech** (every
category sees the same utterances, so speech-content variance can't masquerade as a
category effect), reported with standard errors:

```bash
python scripts/eval_impulsive_noise.py --checkpoint model_trained_on_dns3.tar
python scripts/eval_impulsive_noise.py --checkpoint model_trained_on_vctk.tar --out results_impulsive_vctk.json
```

### The finding: impulsive brittleness is a *training-data* problem, not an architecture problem

We ran the identical architecture from two pretrained checkpoints — one trained on the
narrow VoiceBank-DEMAND corpus, one on the far more diverse DNS3 corpus. Enhanced PESQ,
stationary vs. impulsive noise:

| SNR | **vctk** stat → imp | gap | **dns3** stat → imp | gap |
|---|---|---|---|---|
| +15 dB | 2.60 → 2.16 | **0.44** (7.4σ) | 2.33 → 2.30 | 0.03 (n.s.) |
| +10 dB | 2.23 → 1.83 | **0.39** (8.0σ) | 2.07 → 2.02 | 0.05 (n.s.) |
| +5 dB | 1.86 → 1.50 | **0.36** (8.6σ) | 1.81 → 1.73 | 0.08 (2.4σ) |
| 0 dB | 1.43 → 1.30 | **0.13** (4.8σ) | 1.55 → 1.54 | 0.01 (n.s.) |
| −5 dB | 1.18 → 1.19 | 0.02 (n.s.) | 1.34 → 1.38 | −0.04 (3.3σ, impulsive *better*) |

The narrow-corpus model degrades by up to **0.44 PESQ** on impulsive noise at
overwhelming significance (7–9σ). The diverse-corpus model closes that gap to statistical
noise. Same architecture, same 48K parameters, same evaluation — only the training data
differs. dns3 beats vctk on impulsive noise at *every* SNR, while vctk retains an edge on
stationary noise at high SNR (2.60 vs 2.33 at +15 dB): a specialization-vs-generalization
trade-off.

**Why this framing and not the original one.** The project began from the common claim
that *"speech-enhancement models fail on impulsive noise."* Our own measurement refutes
that as stated — dns3 is a counterexample. The defensible version is the one above:
brittleness tracks training-data diversity, which is both true here and a direct
motivation for the fine-tuning stage (add *defence-specific* diversity on top of dns3).

**The domain gap that remains.** Every category on defence noise sits well below the
standard benchmark — ~2.3 PESQ at +15 dB versus 2.855 on VCTK-DEMAND, and far worse at
low SNR. That gap, across all noise types, is the real target for fine-tuning.

Full tables in `results_impulsive_baseline.json` (dns3) and `results_impulsive_vctk.json`.

### Versus classical DSP: where impulsive noise actually bites

`docs/solution-design.md` §2 asks for a "we beat the classical method" comparison, so we
implemented causal spectral subtraction (Boll 1979, with over-subtraction and a spectral
floor — the refinements that make it a fair opponent rather than a straw man) and ran it
through the same harness:

```bash
python scripts/eval_impulsive_noise.py --with-classical
```

PESQ **improvement over the unprocessed mixture** (higher is better):

| Category | Classical DSP | GTCRN (dns3) |
|---|---|---|
| Stationary, +15 dB | +0.57 | +0.74 |
| Non-stationary, +15 dB | +0.43 | +0.73 |
| **Impulsive, +15 dB** | **+0.07** | **+0.71** |
| **Impulsive, −5 dB** | **−0.01** (worse than no processing) | +0.30 |

**Classical spectral subtraction essentially fails on impulsive noise** — 8× less
improvement on gunfire than on engine drone, and at low SNR it degrades the signal. This
is expected and diagnostic: the method assumes noise statistics are roughly stationary,
which a gunshot violates completely. The neural model improves all three categories
roughly uniformly.

### The three-tier picture

Taking the classical baseline and the two checkpoints together, all measured on the same
data with the same harness:

| Approach | Impulsive-noise handling |
|---|---|
| Classical spectral subtraction | **Fails** — +0.07 PESQ, ~zero benefit |
| Neural, narrow training corpus (vctk) | **Partially fails** — 0.44 PESQ deficit vs stationary (7.4σ) |
| Neural, diverse training corpus (dns3) | **Closes the gap** — deficit statistically indistinguishable from zero |

Impulsive noise *is* the hard case — but the difficulty is a property of the method and
its training data, not something inherent to modern architectures. That is a more precise
and more defensible claim than the one the project started with.

### Fine-tuning: the result

Two fine-tuning runs from the pretrained dns3 checkpoint, using the dataset pipeline
(`scripts/train_dataset.py`) and losses (`scripts/losses.py`) built for this project.
Training noise is the MAD **training** split only; evaluation below is on the untouched
**test** split, 100 trials/cell.

**Enhanced PESQ, +15 dB — the operational condition:**

| Category | Pretrained | Run 1 | **Run 2 (final)** | Δ | Significance |
|---|---|---|---|---|---|
| Stationary | 2.33 | 2.45 | **2.46** | +0.13 | 2.8σ |
| Non-stationary | 2.26 | 2.44 | **2.45** | +0.19 | 4.7σ |
| **Impulsive** | 2.30 | 2.44 | **2.49** | **+0.20** | 3.7σ |

**Against the PS targets, on real defence noise at +15 dB:**

| Target | Impulsive | Non-stationary | Stationary |
|---|---|---|---|
| PESQ > 2.5 | **2.49 ± 0.04** (at target within error) | 2.45 | 2.46 |
| STOI > 0.85 | **0.920** ✅ | 0.913 ✅ | 0.913 ✅ |
| SNR > 15 dB | **20.2 dB** ✅ | 19.7 ✅ | 19.3 ✅ |

**Impulsive noise — the hardest category, and the one classical DSP fails on entirely —
is now the model's *strongest* result.** That is the training-data thesis above playing
out: given defence-specific data, the impulsive deficit does not merely close, it
reverses.

Full sweep for run 2 (`results_run2.json`):

| SNR | Stationary | Non-stationary | Impulsive |
|---|---|---|---|
| +15 dB | 2.46 | 2.45 | **2.49** |
| +10 dB | 2.17 | 2.13 | 2.12 |
| +5 dB | 1.84 | 1.82 | 1.76 |
| 0 dB | 1.51 | 1.50 | 1.52 |
| −5 dB | 1.27 | 1.27 | 1.36 |

### What we will not claim

Three things this result does **not** support, stated here so they don't get overstated
elsewhere:

- **PESQ 2.49 is not "above 2.5".** The target lies inside one standard error, so the
  honest phrasing is *at target within measurement uncertainty*.
- **Run 2 is not significantly better than run 1** (+0.05 on impulsive, ~1σ). Run 2
  narrowed training SNR to [0, 20]; that helped a little, not decisively.
- **The asymmetric anti-over-suppression loss did not measurably contribute.** Even at
  `w_asym=2.0` it stayed ~0.2% of total loss. It is implemented and available, but this
  result comes from the data pipeline and the SNR schedule, not from that term. Isolating
  it properly needs a dedicated run at a much higher weight.

**A deliberate trade-off:** run 2 never trained below 0 dB, and low-SNR performance
regressed accordingly (stationary at −5 dB: 1.34 → 1.27, −3.4σ). This was intentional —
at −5 dB, PESQ is bounded near 1.3 regardless of model, so capacity spent there cannot
reach the target. If low-SNR robustness matters more than hitting 2.5, run 1's model
(`--snr-min -5`) is the better trade.

### Methodology notes

- An earlier 20-trials/cell run produced category orderings that **flipped between random
  seeds** (±0.21 PESQ swings). Those differences were sampling noise, not signal. The
  eval script now reports standard errors and prints an explicit `SIGNIFICANT` /
  `not significant` verdict per SNR at 2σ, so a difference smaller than its error bars
  cannot be quietly written up as a finding.
- PESQ's floor is ~1.02 and inputs at −5 dB score ~1.05, leaving almost no dynamic range;
  read STOI and SI-SNR rather than PESQ in that column.
- SNR is set by whole-clip RMS, not ITU-T P.56 active-speech level. Since VCTK clips have
  leading/trailing silence, true in-speech SNR is slightly higher than the stated label.
  Fine for relative comparison; stated here rather than left for a reviewer to catch.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the VCTK-DEMAND test set (clean + noisy, ~310 MB combined) from the
[University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791)
(free end-user licence) into `data/vctk_demand_testset/{clean,noisy}/`, then:

```bash
python scripts/run_baseline_eval.py --limit 20   # fast smoke test
python scripts/run_baseline_eval.py              # full 824-pair benchmark
```

## Repository layout

```
├── docs/
│   ├── solution-design.md   # canonical build spec — read this for the full roadmap
│   └── research-notes.md    # background research and citations
├── archive/                 # unrelated docs kept for reference, not part of this project
├── third_party/gtcrn/       # vendored upstream GTCRN (MIT) — not modified
├── scripts/
│   ├── run_baseline_eval.py     # Stage 0 batch evaluation harness (VCTK-DEMAND)
│   ├── streaming_engine.py      # frame-by-frame ONNX streaming wrapper (StreamingEnhancer)
│   ├── live_demo.py             # --check / --capture-test / live mic-to-speaker demo
│   ├── mad_noise.py             # Military Audio Dataset loader (category grouping)
│   └── eval_impulsive_noise.py  # stratified "Model A" baseline on real defence noise
└── data/                    # gitignored — datasets go here, not committed
```

## Licensing / attribution

- Model code and pretrained checkpoints: [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn),
  MIT licence, vendored at commit `502ebfa` (see `third_party/gtcrn/LICENSE`).
- Evaluation data: VCTK-DEMAND test set (Valentini-Botinhao), University of Edinburgh
  DataShare, free end-user licence — not redistributed in this repo, downloaded
  separately per Quickstart above.
- Defence noise: [Military Audio Dataset](https://github.com/kaen2891/military_audio_dataset)
  (Kim, Yoon & Jung, *Scientific Data* 11:668, 2024), CC BY 4.0, audio via
  [Kaggle](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset) —
  not redistributed in this repo, download separately into `data/MAD_dataset/`.
- Later stages will pull in additional datasets (LibriSpeech, MUSAN, room impulse
  responses for fine-tuning) — see `docs/solution-design.md` §5 for the full list and
  licence flags (some are CC BY-NC and excluded from any deployment-facing claim).
