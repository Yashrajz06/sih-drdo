# AI/ML Noise Suppression for Defence Communications (SIH / DRDO)

**Status: real-time live demo working on a laptop (no Raspberry Pi hardware yet) —
mic-to-speaker GTCRN enhancement with a live on/off toggle, RTF 0.079.** See
[`docs/solution-design.md`](docs/solution-design.md) for the full 14-day build plan.

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
│   ├── run_baseline_eval.py # Stage 0 batch evaluation harness
│   ├── streaming_engine.py  # frame-by-frame ONNX streaming wrapper (StreamingEnhancer)
│   └── live_demo.py         # --check / --capture-test / live mic-to-speaker demo
└── data/                    # gitignored — datasets go here, not committed
```

## Licensing / attribution

- Model code and pretrained checkpoints: [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn),
  MIT licence, vendored at commit `502ebfa` (see `third_party/gtcrn/LICENSE`).
- Evaluation data: VCTK-DEMAND test set (Valentini-Botinhao), University of Edinburgh
  DataShare, free end-user licence — not redistributed in this repo, downloaded
  separately per Quickstart above.
- Later stages will pull in additional datasets (LibriSpeech, MUSAN, Military Audio
  Dataset, etc.) — see `docs/solution-design.md` §5 for the full list and licence flags
  (some are CC BY-NC and excluded from any deployment-facing claim).
