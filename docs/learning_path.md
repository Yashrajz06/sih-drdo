# Learning Path

Only what is needed to understand and defend **this** project. Everything not
directly load-bearing has been cut. Total: **~12 hours.**

Order matters — each block assumes the one above it.

---

## 0. Math you actually need (1.5 h)

- [ ] **Complex numbers** — a+bi, magnitude, phase. *Every spectrogram cell is one.*
      3Blue1Brown, "What is Euler's formula actually saying?"
- [ ] **Decibels** — log scale, why +6 dB is a doubling. Any short explainer.
- [ ] **Standard error** — why we report error bars. StatQuest, "Standard Error".

Skip: linear algebra courses, integration, statistics beyond the above.

---

## 1. Signal processing (3 h) — **the highest-value block**

This is the gap most likely to be probed by a judge.

- [ ] Xiph.org, **"Digital Show and Tell"** (24 min) — sampling, Nyquist, quantization.
      The best 24 minutes available on digital audio.
- [ ] 3Blue1Brown, **"But what is the Fourier Transform?"** (20 min).
- [ ] Valerio Velardo, **"Audio Signal Processing for ML"** — only these videos:
      STFT · windowing · spectrograms. (~2 h)

**You must be able to answer, cold:**
- What is a sample? Why 16 kHz? (Nyquist → 8 kHz ceiling covers speech)
- What is an STFT? What is a frequency bin? (257 bins, 31.25 Hz each)
- Why 32 ms window / 16 ms hop? Why overlap? (windowing tapers edges; 50%
  overlap + sqrt-Hann sums back to 1.0)
- What is magnitude vs phase, and why does phase matter? (ignoring it causes
  musical noise)

---

## 2. Machine learning basics (1.5 h)

- [ ] 3Blue1Brown, **"Neural Networks"** — 4 videos (1 h).
- [ ] Colah's blog, **"Understanding LSTM Networks"** (45 min) — this is the
      memory mechanism inside our DPGRNN.

**You must be able to answer:**
- What is a weight? (48,245 numbers found by search, not written by hand)
- What is a loss function? What is gradient descent?
- What is fine-tuning vs training from scratch? (we forked pretrained DNS3 weights)
- What is overfitting, and why is our test split held out?

Skip: full courses (Ng, fast.ai). Wrong shape for the timeline.

---

## 3. Our model (2 h)

- [ ] **GTCRN paper** — Rong et al., ICASSP 2024. Read it twice.
- [ ] Colah's LSTM post already covers the GRU. Nothing else needed.

**You must be able to answer:**
- What does the model output? (a **mask** — a per-cell multiplier, not audio.
  It cannot hallucinate speech, only attenuate what is there.)
- What is a **CRM**? (complex mask — fixes loudness *and* timing)
- What is **ERB**? (perceptual frequency compression, 257 -> 64 bands; lossy
  compression tuned to human hearing, like JPEG chroma subsampling)
- What is **causality** here? (no lookahead across time; the frequency axis is
  free because frequency is not time)
- Why is it only 48K params? (grouped convolutions + ERB + small hidden sizes)

---

## 4. Speech enhancement context (2 h)

- [ ] **Wang & Chen (2018)**, "Supervised Speech Separation Based on Deep
      Learning: An Overview" — **masking sections only.** The single most useful
      paper for this project.
- [ ] **Boll (1979)**, spectral subtraction — short. Or just read our
      `scripts/spectral_subtraction.py`, which is the same thing in 76 lines.

**You must be able to answer:**
- **PESQ / STOI / SI-SNR** — what each measures, and why we report all three.
  (Each alone is gameable: silence scores perfectly on noise removal and zero on
  intelligibility.)
- Why classical spectral subtraction fails on gunfire. (It measures a noise
  profile during a quiet moment and subtracts it — assumes noise is *steady*.
  A gunshot is over before you can measure it. Measured: **+0.07 PESQ on
  gunfire vs +0.57 on engine noise.** That gap is why this project exists.)

---

## 5. Deployment (1 h) — mostly your home turf

- [ ] ONNX docs, **Concepts** section — graph format, **opset versioning**.
- [ ] ONNX Runtime, **quantization** guide — skim.
- [ ] ITU-T **G.114** summary (one page) — the 150 ms latency band.

**You must be able to answer:**
- What is **RTF**? (compute time / audio duration; ours is 0.095 — under 10% of
  the available budget)
- The **opset trap**: newer PyTorch silently upgrades opset 11 -> 18. Still
  correct, passes every accuracy test, but **3.5x slower** (5.41 vs 1.53 ms/hop).
  Survivable on a laptop; pushes RTF above 1.0 on the Pi, i.e. breaks real time.
- What is **INT8 quantization**? (8-bit weights instead of 32-bit floats; ~4x
  smaller, faster on CPU, small quality cost)

---

## 6. Read our own code (2 h) — highest value per minute

In this order. Everything above exists to make these readable.

- [ ] `scripts/streaming_engine.py` (72 lines) — the entire real-time loop
- [ ] `scripts/spectral_subtraction.py` (76 lines) — the classical baseline
- [ ] `scripts/losses.py` (200 lines) — every loss term, with a runnable self-test
- [ ] `scripts/train_dataset.py` — data synthesis, SNR, augmentation
- [ ] `scripts/export_onnx.py` — deployment, and the opset trap in a comment

Then: `docs/architecture.md` for the decisions and their reasons.

---

## 7. Do these four things (2 h) — reading alone will not stick

- [ ] **Round-trip a file.** Load a WAV, STFT, iSTFT, assert it matches to 1e-6.
      Then break it deliberately — remove the overlap — and listen.
- [ ] **Plot spectrograms.** Your voice, a gunshot, and the two mixed. You will
      *see* why the frequency view is the right representation.
- [ ] **Hand-write a mask.** Zero everything above 4 kHz. Listen. That is a crude
      version of what the model does with 48,000 learned numbers.
- [ ] **Run the demo with enhancement toggling** and watch the spectrogram.

---

## The three facts to have memorised

1. **PESQ 2.49 / STOI 0.920 / SI-SNR 20.2 dB** on real defence noise at +15 dB.
   Targets were 2.5 / 0.85 / 15. (PESQ is *at* target within error, not above it
   — say that, don't round up.)
2. **83 ms** end-to-end latency, measured with a chirp and cross-correlation,
   not estimated. ITU-T G.114 says under 150 ms feels natural.
3. **Classical: +0.07 PESQ on gunfire. Ours: +0.71.** The 8x gap on impulsive
   noise is the entire justification for the approach.

## The one weakness to state before you are asked

Below 0 dB — noise as loud as the voice — we reach about PESQ 1.3. That is a
limit of the problem, not the method: the information needed to reconstruct
those words largely is not in the signal. We hit the targets from +10 dB up.

*Volunteering this earns more credibility than any other answer in the Q&A.*

---

## If you have only 3 hours

Blocks **1** and **6**, then the three facts above. DSP fundamentals plus your
own code covers more ground than anything else in the same time.
