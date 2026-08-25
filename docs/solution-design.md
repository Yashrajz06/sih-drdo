# AI/ML-Based Noise Suppression for Defence Communications
## Complete Solution Design — v2

**Smart India Hackathon | 6-person team | Raspberry Pi 5 target**

---

## 0. What this system actually is (read this first)

The problem statement is titled "Active Noise Cancellation," but read its own success criteria: **PESQ, STOI, SI-SNR**. Those are all measures of *speech quality in a transmitted signal*. You cannot compute PESQ on an anti-noise acoustic field.

So the brief is asking for **speech enhancement**, not classical ANC:

| | Classical ANC | Speech Enhancement (this project) |
|---|---|---|
| Protects | The listener's ears | The transmitted message |
| Method | Generates anti-noise via speaker | Cleans the mic signal computationally |
| Measured by | Attenuation in dB SPL | PESQ, STOI, SI-SNR |

**How to present it:** *"AI-driven hybrid adaptive noise cancellation and speech enhancement system. The neural enhancement stage handles complex, non-stationary and impulsive noise; an optional adaptive filter stage handles correlated stationary noise such as engine and rotor drone."*

If a judge challenges the terminology, the metrics argument above is your answer. Have it ready — it converts a potential weakness into evidence that you read the brief more carefully than the people who wrote it.

---

## 1. Core architecture (locked)

```
        PRIMARY MIC (USB sound card or INMP441)
                    |
            Audio preprocessing
        (S32 to float, 48k to 16k resample, DC removal)
                    |
            Streaming STFT  (32 ms window, 16 ms hop, causal)
                    |
        +-----------------------------+
        |      GTCRN  (23.7K params)  |
        |  defence-noise fine-tuned   |
        |  impulsive-noise robust     |
        |  recurrent state carried    |
        +-----------------------------+
                    |
        Complex mask / deep filter apply
                    |
            iSTFT + overlap-add
                    |
            Enhanced speech
                    |
        HEADSET / COMMS OUTPUT
```

**Optional hybrid branch — demo only, not critical path:**

```
    REFERENCE MIC --> VAD gate --> NLMS adaptive filter
                                        |
                            correlated-noise suppression
                            (engine, rotor drone)
```

**Rule: the system must fully work without the optional branch.** If the reference mic doesn't produce a clean, speech-free channel, you drop NLMS entirely and lose nothing — the brief itself calls it optional.

---

## 2. Model selection

### Recommendation: GTCRN

Rong, Sun, Zhang, Hu, Zhu & Lu, *"GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources,"* ICASSP 2024, pp. 971–975.

| Property | Value |
|---|---|
| Parameters | 23.7K (48.2K counting unlearnable ERB module) |
| Compute | 39.6 MMACs/s (33.0 with simplified ERB) |
| VCTK-DEMAND PESQ | 2.87 |
| VCTK-DEMAND STOI | ~0.94 |
| DNS3 blind DNSMOS | 3.44 |
| Causal | Yes, zero lookahead |
| Licence | MIT |
| Tooling | Official PyTorch + ONNX streaming sample + sherpa-onnx |

**Fallback: DeepFilterNet2** (2.31M params, RTF 0.42 measured on Raspberry Pi 4, MIT/Apache-2.0). Higher perceptual quality, but carries a 2-frame lookahead giving ~40 ms algorithmic latency. Switch to this only if GTCRN quality proves inadequate after fine-tuning.

**Baselines to beat (and report):** RNNoise (~60K params, PESQ ~2.29) and classical spectral subtraction. Beating the classical DSP baseline is explicitly what the brief's background section asks for.

**Not for the Pi:** FullSubNet+, FRCRN, MP-SENet, CMGAN, TF-GridNet, SE-Mamba. Excellent quality, but large and mostly non-causal. Use only as offline upper-bound references, or as distillation teachers in the finale.

### Claim discipline

Never say *"our system achieves PESQ 2.87 on defence noise."* That number is from VCTK-DEMAND — read speech, no strong reverb, no gunshots. Say instead:

> *"Baseline GTCRN achieves PESQ 2.87 on the standard VCTK-DEMAND benchmark. Our contribution is adapting and evaluating it specifically for impulsive defence noise, measured on our own stratified test set."*

Then show your numbers.

---

## 3. Novelty: impulsive-noise robustness

### Why standard models fail on transients

Classical and most DNN systems assume quasi-stationary noise statistics. Impulsive events are sudden broadband spikes — a gunshot muzzle blast has rise time under ~60 µs and peak SPL above 125 dB — that smear across the STFT. They are under-represented in DNS-style corpora, and a single high-energy event dominates the training loss gradient, so models either ignore the transient or over-suppress the speech co-occurring with it.

### Implementable techniques

Ranked by payoff-to-effort for a 2-week window:

**1. Impulsive-event oversampling + SNR curriculum.** Mix online per epoch. Oversample MAD gunshot/shelling relative to natural frequency. Vary event count (0–3 per clip), peak level, and position. Start at higher SNR, widen downward in ~5 dB steps.

**2. Asymmetric anti-over-suppression loss.** Penalise attenuating speech bins harder than leaving residual noise:

```
L_asym = Σ max{ |S|^c − |Ŝ|^c , 0 }²      with c ≈ 0.3
```

(Braun et al., arXiv 2205.06931)

**3. Multi-resolution STFT loss.** Sum spectral-convergence and log-magnitude terms across FFT ∈ {512, 1024, 2048}, hop ∈ {50, 120, 240}, window ∈ {240, 600, 1200}. Short windows catch transients; long windows preserve harmonics. (Défossez et al., arXiv 2006.12847)

**4. Power-law magnitude compression.** Exponent c ≈ 0.3–0.6 on input and loss, so one loud transient cannot dominate gradients. GTCRN's own loss already uses SI-SNR + compressed magnitude/complex terms (α=0.01, β=0.3).

**5. Deep filtering over point masks.** Predict a short complex FIR filter (e.g. 5 taps) across adjacent T-F bins to recover phase and energy smeared by a transient.

**Finale stretch:** detect-then-restore hybrid (Vaseghi & Rayner, IEE Proc-I 1990), impulsive–stationary separation front-end (IS³, arXiv 2509.02622), dynamically-weighted transient loss.

### The ablation — this is what proves your contribution

**Without this table, your novelty claim is unfalsifiable and a judge will say so.**

| | Model A | Model B (finale) | Model C |
|---|---|---|---|
| | Pretrained GTCRN | + defence fine-tune | + impulsive strategy |
| Stationary (engine, wind) | | | |
| Non-stationary (siren, drone) | | | |
| Impulsive (gunshot, shelling) | | | |

Report PESQ / STOI / SI-SNR in each cell.

**For the internal round, run A vs C only.** Model B is the scientifically proper control but costs another full training run. Add it for the finale. Be upfront that B is missing — "our controlled ablation isolating the loss functions is in progress" is a fine answer.

---

## 4. Evaluation: three categories, matching the brief

The problem statement names stationary, non-stationary, and impulsive noise. Mirror that vocabulary exactly.

| Category | Sources |
|---|---|
| **Stationary** | Engine idle, HVAC, rotor drone, wind |
| **Non-stationary** | Sirens, moving vehicle, drone passes, varying rotor |
| **Impulsive** | Gunshot, shelling, artillery, explosions |

Stratify by input SNR: −5, 0, +5, +10, +15 dB.

**Metrics and libraries:**
- `pesq` — wideband ('wb') at 16 kHz. State the version; P.862 vs Corrigendum 2 differ by ~0.8 MOS.
- `pystoi` — STOI and ESTOI (`extended=True`)
- `torchmetrics.audio` — SI-SDR / SI-SNR
- SNR improvement = output SNR − input SNR
- DNSMOS for real recordings where no clean reference exists

**Realistic expectations.** PESQ > 2.5 and STOI > 0.85 are achievable for a causal 16 kHz model at moderate SNR. At ≤0 dB on unseen impulsive noise, expect PESQ 1.8–2.5. Going causal costs roughly 0.05–0.3 PESQ versus the non-causal version of the same architecture. **Set this expectation with judges before they find it themselves** — hit targets on the stationary/moderate-SNR set, show honest improving numbers on the hard set.

---

## 5. Datasets

### Clean speech
| Dataset | Size | Licence |
|---|---|---|
| LibriSpeech (OpenSLR SLR12) | ~1000 h | CC BY 4.0 |
| VCTK | ~44 h, 110 speakers | ODC / CC BY |
| DNS Challenge clean | Hundreds of hours | Permissive |
| **SPRING-INX** (IIT Madras) | ~2000 h, 10 Indian languages | Open |
| **IndicTTS / Indic TIMIT** | Indian-accented English | Check per-corpus |
| Svarah | 9.6 h Indian-English eval | Open |

Indian speech fine-tuning matters — your demo speakers are Indian, and train/test accent mismatch is a real and avoidable failure.

### Noise
| Dataset | Relevant classes | Licence |
|---|---|---|
| **Military Audio Dataset (MAD)** | gunshot, shelling, helicopter, vehicle, fighter, footsteps | **CC BY 4.0** |
| MUSAN (SLR17) | broad ambient | CC BY 4.0 |
| FSD50K | gunshot, explosion, siren, engine | Mixed — **filter out NC clips** |
| ESC-50 | helicopter, engine, siren, chainsaw | **CC BY-NC — flag** |
| UrbanSound8K | gun_shot, siren, engine_idling, jackhammer | **Non-commercial — flag** |
| DEMAND | real environmental | CC BY-SA |

**MAD caveat:** it is only ~12 hours / 8,075 samples across 7 classes. Small enough that you can overfit to its specific recording conditions and learn "this microphone's gunshot" rather than "gunshots." Supplement with FSD50K and UrbanSound8K gunshot/explosion classes and augment aggressively — level, position, RIR, band-limiting.

### Room impulse responses
- OpenSLR SLR28 (`rirs_noises.zip`, 1.3 GB, Apache-2.0) — used by the DNS Challenge
- OpenSLR SLR26 — ~60,000 simulated RIRs, Apache-2.0

### Licensing for submission
Prefer CC0 / CC BY / Apache-2.0. Keep an attribution manifest listing every asset and its licence. Label CC BY-NC assets clearly; acceptable for an academic prototype with attribution, but do not present them as deployment-ready data.

---

## 6. Your own recorded test set

**Mandatory, but keep it small.** For the internal round: 3–4 speakers, ~30 utterances, one afternoon. This exists to catch sim-to-real bugs in your capture chain, not to be a research corpus.

Record Indian-accented English at multiple distances, with noise played through a speaker into the room.

**Describe it accurately.** A consumer loudspeaker cannot reproduce a gunshot — peak SPL above 125 dB with sub-100 µs rise time is beyond the driver. What you record is a compressed, band-limited approximation. Call it *"played-back defence noise in a real room,"* never *"real defence conditions."* Judges respect the distinction; overclaiming gets caught.

**Never train on this set.** Held-out only.

---

## 7. Data pipeline

Follow DNS Challenge conventions:

- **SNR sampling:** uniform [−5, +20] dB training; low-SNR stress subset [−15, 0] dB; fixed test points for stratification
- **Mixing:** set clean speech to target active level, scale noise to hit desired SNR computed over *active speech frames*, sum, rescale mixture and target together if peak exceeds 0 dBFS
- **Sample rate:** 16 kHz
- **Dynamic mixing:** mix per epoch, not pre-generated. DNS reports 5000+ effective hours after ten epochs. Pre-generate only a fixed validation/test set for reproducibility.
- **Impulsive placement:** overlap transients with *voiced* speech, not only silence — otherwise the task is trivially easy. Avoid clips where the transient saturates everything.
- **RIR augmentation:** convolve ~50–75% of clean clips with a random RIR before adding noise. Decide and state whether your target is anechoic clean (denoise only) or reverberant clean (denoise + dereverb).
- **Other:** random gain, clipping simulation, band-limiting to simulate the radio channel

---

## 8. Training recipe

**Loss:**
```
L = α·L_SISNR + L_mag(compressed) + β·(L_real + L_imag) + L_MRSTFT + γ·L_asym
```
GTCRN published weights: α = 0.01, β = 0.3. Tune γ empirically — start around 0.1.

**Hyperparameters:** Adam, LR 1e-3 (or 5e-4), reduce-on-plateau or cosine, batch 12–32, 4-second chunks, gradient clipping, early stopping. 60–100 epochs typical; a 24K-param model converges fast.

**Free GPU reality:** Kaggle gives 30 GPU-h/week (P100 or T4×2); Colab ~12 h sessions with quotas. Enough for GTCRN-scale fine-tuning in 1–3 days of accumulated time. **Checkpoint to Drive constantly** — sessions disconnect.

**Two-week strategy: fine-tune, don't train from scratch.** Start from pretrained GTCRN weights. Scratch training is feasible for this model size but de-risking the timeline matters more than architectural purity. Note scratch training and distillation as finale work.

---

## 9. Streaming inference

- **Framing:** one hop per step. 32 ms window, 16 ms hop, 50% overlap. Ring buffer for input.
- **Recurrent state:** export ONNX with state tensors as explicit inputs/outputs; feed hidden states back each step. **Do not re-run the whole buffer per frame** — this is the most common streaming implementation bug.
- **Reconstruction:** overlap-add with synthesis window weighting.
- **Implementation:** `sounddevice` (PortAudio) callback + `onnxruntime` CPUExecutionProvider. Keep the callback allocation-free; precompute FFT plans.

### Latency budget — measure all three separately

| Component | Expected |
|---|---|
| Algorithmic (window + lookahead) | 16–32 ms (GTCRN, zero lookahead) |
| Model compute | <1 ms/hop on Pi 5 |
| Audio I/O (ALSA period + USB + DAC) | 20–40 ms |
| **End-to-end mic→headphone** | **target <100 ms** |

**Report end-to-end measured latency, not RTF.** RTF is a model metric; judges are asking about conversation latency. Measure by recording a click and timing the gap.

**Standard to cite:** ITU-T G.114 (05/2003) — 0–150 ms one-way is "acceptable for most user applications," 150–400 ms acceptable with awareness, >400 ms unacceptable. You are comfortably inside the first band.

---

## 10. Raspberry Pi 5 deployment

- **Runtime:** ONNX Runtime ARM64 wheel, 64-bit Raspberry Pi OS required
- **Quantisation:** dynamic INT8 (`quantize_dynamic`) — no calibration needed, good for GRU/Linear-heavy models. Static INT8 for the finale.
- **Threads:** Pi 5 has 4× Cortex-A76 @ 2.4 GHz. For a tiny model, 1–2 intra-op threads often minimise jitter; benchmark against 4.
- **NEON/SIMD:** ONNX Runtime ARM64 kernels use it automatically.
- **Expected RTF:** ~0.02–0.05.

**Claim carefully.** The ~0.048 RTF figure circulating for GTCRN on Pi 5 comes from a *community port of a modified GTCRN-AEC variant*, not the original authors. Say:

> *"Community experiments indicate strong Raspberry Pi 5 real-time performance; we benchmarked our exact ONNX model and pipeline on our own hardware and measured [X]."*

### Audio I/O
- **USB sound card:** find via `arecord -l`; tune ALSA `period_size` / `buffer_size` and the PortAudio latency hint
- **INMP441 I2S:** 3.3 V, 24-bit. Set `dtparam=i2s=on` plus an I2S overlay in `/boot/firmware/config.txt`. Native capture is S32_LE @ 48 kHz — convert to 16-bit/16 kHz in software. For two mics, share BCLK/LRCLK and set one to left, one to right via the L/R select pin.

---

## 11. Optional NLMS stage

**Algorithm:**
```
y(n) = wᵀx(n)
e(n) = d(n) − y(n)
w ← w + µ·x(n)·e(n) / (‖x(n)‖² + δ)
```
where d = primary (speech+noise), x = reference (noise), e = output.

**Requirements:** a reference mic capturing noise correlated with the primary's noise but *not* the speech. VAD-gate adaptation so the filter only updates during speech-absent frames — otherwise it learns to cancel speech.

**What it's good for:** periodic/correlated noise (engine, rotor). Reported ~13 dB suppression with NLMS vs ~10 dB with plain LMS. It does almost nothing for uncorrelated impulsive transients.

**Verdict:** implement the neural path first. Add NLMS as a documented hybrid enhancement and a talking point (classical + AI). Demo live only if the reference channel is genuinely clean.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| **Causal/non-causal confusion** — model works offline, fails live | Train and evaluate in the exact streaming config. Never normalise using future or global statistics. **Highest-priority risk.** |
| Buffer underruns → clicks, dropouts | Adequate ALSA period/buffer, lean callback, pre-warm model |
| Train–test mismatch | Indian speech fine-tuning + your own recorded held-out set |
| MAD overfitting | Supplement with FSD50K/UrbanSound8K, augment hard |
| Over-suppression at low SNR | Asymmetric loss, attenuation limiting |
| Musical noise | Complex/deep filtering, avoid aggressive gain flooring |
| Capture chain bugs (AGC, DC offset, 32→16-bit) | Verify with a known tone and FFT check before wiring the model |
| Free-GPU disconnects | Checkpoint every epoch to Drive |

---

## 13. Fourteen-day plan

**Days 1–2 — Baseline running.** Pretrained GTCRN denoising a file on a laptop. Download LibriSpeech, MUSAN, FSD50K (CC-BY subset), OpenSLR28, MAD. Image the Pi SD card and confirm simultaneous mic capture + headphone playback with no dropouts. *This Pi audio check is day 2, not day 12 — if Linux audio fights you, you need to know now.*

**Days 3–6 — The live loop.** Streaming inference: mic → 16 ms hops → model with carried state → headphones, with a bypass toggle. **Milestone: by end of day 6, someone outside the team can wear the headphones and hear the toggle work.** If you're not there, cut scope — drop the Pi, drop fine-tuning, keep the loop.

**Days 7–9 — Make it yours.** Fine-tune on the impulsive-heavy mix. Export to ONNX (stateful streaming), INT8 quantise, port to Pi, measure end-to-end latency. Compute the A-vs-C ablation across three noise categories.

**Days 10–11 — Record your own test set** (one afternoon) and build the PPT.

**Days 12–14 — Rehearse.** Two full run-throughs with a hostile-judge stand-in. Record a backup demo video.

### Team split (6 people)
| Role | People |
|---|---|
| Audio pipeline + streaming loop | 2 |
| Data pipeline + training | 2 |
| Pi deployment + latency/metrics | 1 |
| PPT + demo script + Q&A prep | 1 |

The PPT role is not filler — that person answers judges' questions and must understand the system fully.

---

## 14. Demo script

1. Baseline: teammate speaks, clean, into the headset a judge is wearing
2. Add noise: helicopter loop + gunshot played into the room
3. **Toggle enhancement ON/OFF live** — this is the moment that wins
4. On-screen: live end-to-end latency readout + before/after spectrogram
5. Show the ablation table and the three-category stratified metrics
6. Show the latency budget against G.114

Keep a recorded fallback video. Live demos fail in unfamiliar rooms.

---

## 15. Judge Q&A

**"Why Raspberry Pi, not the Jetson in the brief?"**
Design decision, with numbers. Pi 5 ≈ ₹8,000 vs Jetson AGX Orin 64GB ≈ ₹2.5 lakh — roughly 30×. GTCRN needs ~40 MMACs/s; the Pi CPU runs it at RTF ~0.02–0.05. A Jetson GPU would sit idle. Unit cost is a real requirement for anything issued at squad scale.

**"Is this really ANC?"**
See §0. The brief's own metrics — PESQ, STOI, SI-SNR — are speech-quality measures, only meaningful for the enhancement interpretation. We implement enhancement as the core and adaptive filtering as the hybrid branch.

**"What's your latency?"**
End-to-end measured [X] ms, of which [Y] ms is algorithmic (causal, zero lookahead) and [Z] ms is audio I/O. Inside ITU-T G.114's 150 ms transparent band.

**"How is this different from Krisp / RNNoise / DeepFilterNet?"**
Those target stationary office noise. We target impulsive defence noise specifically, via impulsive-event oversampling plus asymmetric and multi-resolution STFT losses — and we prove it with a stratified ablation rather than asserting it.

**"What exactly did you contribute?"**
Point at the ablation table.

**"Does it distort speech?"**
Asymmetric anti-over-suppression loss plus attenuation limiting. We report PESQ and STOI, not just SNR, precisely because SNR alone rewards over-suppression.

**"Indian accents?"**
Fine-tuned on SPRING-INX / IndicTTS; tested on our own held-out Indian-speaker recordings.

**"How do you handle phase?"**
Complex-domain processing and deep filtering rather than magnitude-only masking.

---

## 16. Finale additions (3 months)

- Model B in the ablation (fine-tune without special losses — the proper control)
- Train from scratch and/or distill from a non-causal teacher (MP-SENet, FullSubNet)
- Static INT8 + structured pruning
- Detect-then-restore impulsive stage; IS³-style impulsive/stationary separation
- Multi-channel beamforming with the mic pair
- Larger Indian-speech fine-tuning corpus
- Formal listening tests (ITU-T P.808) alongside DNSMOS
- Ruggedised enclosure, comms-unit integration

---

## Caveats

- VCTK-DEMAND numbers are read speech without strong reverb — a mismatched proxy for defence conditions. Your own stratified test set is the real evidence.
- The Pi 5 RTF figure is from a community port of a modified variant, not the original authors. Re-benchmark your own build.
- PESQ depends on package version (P.862 vs Corrigendum 2) and WB vs NB. Fix and report both.
- ESC-50, UrbanSound8K, and FSD50K's NC subset are non-commercial. Keep an attribution manifest.
- MAD is ~12 hours. Guard against overfitting to its recording conditions.
