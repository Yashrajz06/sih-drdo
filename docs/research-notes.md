# AI/ML Active Noise Cancellation for Defence Communications — Complete Solution Design (Smart India Hackathon)

## TL;DR
- **Build a causal, streaming GTCRN model (23.7K params, 39.6 MMACs/s, PESQ 2.87 on VCTK-DEMAND, DNSMOS 3.44 on DNS3) exported to ONNX and run on the Raspberry Pi 5 CPU**, where a community port already demonstrates ~0.048 real-time factor single-threaded — this is the single best fit for your 2-week, ₹8,000-hardware, CS-oriented team, with DeepFilterNet2 (RTF 0.42 on a Raspberry Pi 4) as the higher-quality fallback.
- **Your genuine novelty is impulsive-noise robustness**: standard models are trained on quasi-stationary noise and fail on gunshots/artillery. Win the round with a concrete, implementable recipe — asymmetric anti-over-suppression loss + multi-resolution STFT loss + heavy oversampling of impulsive events + power-law compression + an optional detect-then-restore stage — and prove it with a stratified evaluation.
- **The stated targets are realistic but only if honestly scoped**: PESQ > 2.5 and STOI > 0.85 are achievable for a causal 16 kHz model at moderate SNR; be explicit that causal/low-latency processing costs roughly 0.05–0.3 PESQ versus non-causal, report algorithmic latency (~16–40 ms) against the ITU-T G.114 150 ms one-way budget, and stratify results by noise type and input SNR.

## Key Findings

**Model choice.** Among real-time causal models, GTCRN (Grouped Temporal Convolutional Recurrent Network — Rong, Sun, Zhang, Hu, Zhu & Lu, "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources," ICASSP 2024, pp. 971–975) is uniquely suited to a CPU edge target. Its official poster states: "Only 23.7 K parameters and 39.6 MMACs per second… Achieves a PESQ of 2.87 on the VCTK-DEMAND dataset and a DNSMOS of 3.44 on the DNS3 blind test set." (The official GitHub repo updates these to 48.2K parameters / 33.0 MMACs/s when the unlearnable ERB module is counted.) It ships an MIT-licensed PyTorch implementation, sherpa-onnx support, and a ready ONNX real-time streaming sample. A community Raspberry Pi 5 port of a GTCRN variant measured ~0.048 real-time factor single-threaded (~0.78 ms per 16 ms hop). DeepFilterNet2 is the proven fallback: Schröter et al. ("DeepFilterNet2," arXiv:2205.05474, 2022) state "it can be run on a Raspberry Pi 4 with a real-time factor of 0.42"; it is dual MIT/Apache-2.0 licensed with a mature codebase and pretrained weights.

**The impulsive-noise gap is real and exploitable.** Standard speech enhancement is designed for quasi-stationary noise; transients (gunshots, artillery, door slams) are brief, high-energy, broadband events under-represented in DNS-style training data. Concrete fixes exist and are implementable in two weeks.

**Everything you need is free and appropriately licensed.** Clean speech (LibriSpeech, VCTK, DNS clean, plus Indian corpora such as SPRING-INX and IndicTTS), noise (MUSAN, FSD50K, ESC-50, UrbanSound8K, DEMAND, plus the Military Audio Dataset for gunshot/shelling/helicopter/vehicle), and room impulse responses (OpenSLR26/28, Apache-2.0) are all downloadable.

## Details

### 1. Model architecture selection

All benchmark numbers below are on VoiceBank/VCTK-DEMAND unless stated; note this is a mismatched benchmark (read speech, no strong reverb), so treat it as relative.

**Tier A — ultra-light causal (recommended for the Pi):**
- **GTCRN** — 23.7K params (48.2K counting the unlearnable ERB module), 39.6 MMACs/s (33.0 with the simplified ERB concatenation). PESQ 2.87, STOI ≈ 0.94 (independent FastEnhancer benchmark, arXiv 2509.21867), DNSMOS 3.44 on DNS3. Causal, streaming-capable, MIT license, official PyTorch + ONNX + sherpa-onnx support. It is a grouped, simplified derivative of DPCRN with sub-band feature extraction and a temporal recurrent attention (TRA) module. **This is the recommendation.**
- **RNNoise** — ~60K params (0.06M), PESQ ~2.29–2.34; the classic ultra-low-complexity RNN baseline, runs far below RTF 1.0 on a single CPU core. Use as a sanity baseline and the "we beat the classic" narrative.
- Newer 2025 ultra-light options: LiSenNet (~37K, PESQ ~3.07), UL-UNAS (a NAS refinement of GTCRN), CoFi-Lite (83K params, 12.87 MMACs/s), QC-GAN Tiny (35K, PESQ 3.23). Research-grade with weaker tooling — note as finale-stretch options.

**Tier B — small full-band, real-time on modest CPU:**
- **DeepFilterNet2 / DeepFilterNet3** — 2.31M params (DFN2). Two-stage ERB-envelope + deep-filtering architecture at 48 kHz. RTF 0.42 on Raspberry Pi 4; 0.19 on a single laptop thread (DFN1); ~0.04 on a notebook Core-i5 (DFN2). ~40 ms algorithmic latency (20 ms window, 10 ms hop, two-frame lookahead — note this lookahead means the stock model is NOT strictly no-lookahead). Dual MIT/Apache-2.0. Mature CLI, PyPI package, LADSPA/PipeWire plugin, Android bindings. Best fallback if GTCRN quality is insufficient.
- **DPCRN / DCCRN** — DCCRN is 3.67M params, ~14.4 GMACs; a well-known complex-valued CRN. A causal DCCRN variant exists but DCCRN scores lowest among modern baselines in low-SNR cross-corpus tests and is heavier than needed.

**Tier C — high quality but non-causal / too heavy for the Pi (finale reference or teacher models):**
- **FullSubNet / FullSubNet+ / Fast FullSubNet** — strong DNS performers; FullSubNet+'s multi-branch complexity constrains embedded deployment.
- **FRCRN** — complex CRN; its strong variant uses squeeze-excitation over the whole time axis, making it non-causal unless cumulative pooling is substituted.
- **MP-SENet** (~2M, PESQ ~3.50), **CMGAN** (PESQ 3.41), **TF-GridNet**, **CleanUNet**, **SE-Mamba** (PESQ 3.55) — state-of-the-art quality, but large and mostly offline/non-causal. Use only as offline "upper bound" references or teachers for knowledge distillation in the finale.

**Recommendation:** Primary = GTCRN (causal, streaming, ONNX, MIT, fits the Pi with huge headroom). Fallback = DeepFilterNet2 (proven on Pi 4, mature tooling) if you need better perceptual quality and can accept its 2-frame lookahead / ~40 ms latency. Report baselines = RNNoise and classical spectral subtraction / NLMS.

### 2. The impulsive-noise problem (your novelty)

**Why standard models fail.** Classic and many DNN systems assume quasi-stationary noise statistics; impulsive events are sudden broadband spikes (a gunshot muzzle blast has a rise time under ~60 µs and SPL >125 dB) that smear across the STFT and are under-represented in DNS-style corpora whose SNR sampling and DNSMOS filtering bias toward stationary noise. The complex speech–noise interplay in the STFT domain makes a robust learned mapping hard precisely at transients, and a single high-energy event dominates the training-loss gradient, so models either ignore it or over-suppress the co-occurring speech.

**Concrete, implementable techniques (ranked by effort/payoff for 2 weeks):**

1. **Oversample impulsive events + curriculum/SNR scheduling.** Mix noise online per epoch so each gunshot/artillery clip is seen at many amplitudes and positions; oversample rare transient classes; start at higher SNR and expand the SNR range downward in ~5 dB steps. (Braun & Fingscheidt curriculum, arXiv 1606.06864; per-epoch mixing in Reddy/Braun, "Data augmentation and loss normalization for deep noise suppression," arXiv 2008.06412.)
2. **Asymmetric anti-over-suppression loss.** Add a one-sided penalty that punishes attenuating speech bins more than leaving residual noise, so the model stops "deleting" speech during a transient: `L_asym = Σ max{|S|^c − |Ŝ|^c, 0}²`, c ≈ 0.3. (Braun et al., arXiv 2205.06931.)
3. **Multi-resolution STFT loss.** Sum spectral-convergence + log-magnitude losses across FFT ∈ {512,1024,2048}, hop ∈ {50,120,240}, win ∈ {240,600,1200}; short windows capture transients, long windows capture harmonics. (Défossez et al., DEMUCS, "Real Time Speech Enhancement in the Waveform Domain," arXiv 2006.12847.)
4. **Power-law / loudness compression of input and loss.** Compress magnitude with exponent c ≈ 0.3–0.6 so one loud transient does not dominate gradients. (Braun et al. loss survey, arXiv 2009.12286; compression c=0.6 in IS³, arXiv 2509.02622.) GTCRN's own loss already uses SI-SNR + compressed magnitude/complex terms (α=0.01, β=0.3).
5. **Detect-then-restore hybrid.** Stage 1 detects impulsive instants; Stage 2 blanks/clips or interpolates them before/alongside the neural stage. (Classical: Vaseghi & Rayner, IEE Proc-I 1990; Sugiyama impact-noise suppression, WASPAA 2007.) A clipping/limiter front-end bounds transient energy but must be paired with learned restoration to avoid its own distortion.
6. **Impulsive–stationary separation front-end (deep filtering).** Separate impulsive from stationary components with a dual-decoder deep-filter net, then treat each path differently. (Berger et al., "IS³: Generic Impulsive–Stationary Sound Separation using Deep Filtering," arXiv 2509.02622, 2025.)
7. **Deep filtering (multi-frame complex filter) instead of point masks.** Predict a short complex FIR filter (e.g., 5 taps) over adjacent T-F bins to recover phase/energy smeared by a transient — this is exactly what DeepFilterNet does on low bins. (Schröter et al., arXiv 2110.05588.)
8. **Attention / dynamically-weighted loss for transients.** Up-weight high-error frames (transients produce them) so capacity focuses there. (Causal dynamically-weighted-loss attention encoder-decoder, PMC10174555; CCAUNet complex coordinate attention + MR-STFT, SSRN 4492197.)

Your defensible contribution: take GTCRN, retrain with an impulsive-event-heavy data pipeline plus asymmetric and multi-resolution STFT losses, and report a stratified evaluation showing you beat the pretrained/stationary-trained baseline specifically on impulsive noise. That is a clean, honest, novel result for a hackathon.

### 3. Datasets (all free)

**Clean speech:**
- **LibriSpeech** (OpenSLR SLR12) — ~1000 h English audiobook read speech, 16 kHz, CC BY 4.0. Primary training clean set.
- **VCTK** (Edinburgh) — 110 English speakers, multiple accents, ~44 h, Open Data Commons / CC BY. Basis for the VoiceBank-DEMAND benchmark.
- **DNS Challenge clean speech** (Microsoft, GitHub) — Librivox-derived, hundreds of hours plus synthesis scripts; permissive.
- **Mozilla Common Voice** — multilingual incl. Indian English & Indian languages, CC0.
- **Indian corpora (important — your demo is Indian speakers):** SPRING-INX (IIT Madras, ~2000 h, 10 languages, open, arXiv 2310.14654); IndicTTS (IIT Madras); IISc-MILE Kannada (~350 h) / Tamil (~150 h); Indic TIMIT (~240 h Indian-English, 80 speakers); Svarah (9.6 h Indian-accented English eval); AccentDB. Check per-corpus license before redistribution; use for fine-tuning and for a matched demo/test set.

**Noise (general):**
- **MUSAN** (OpenSLR SLR17) — ~109 h music/speech/noise, CC BY 4.0. Point-source noises are reused in the OpenSLR RIR set.
- **FSD50K** (Fonseca et al., arXiv 2010.00475 / Zenodo 4060432) — "51,197 audio clips totalling 108.3 hours," 200 AudioSet classes (144 leaf + 56 intermediate); licenses are "CC0, CC-BY, CC-BY-NC and CC Sampling+" — **filter out the NC clips for a competition submission.** Contains gunshot, explosion, siren, engine classes; from Zenodo.
- **ESC-50** — 2,000 clips, 50 classes incl. engine, siren, helicopter, chainsaw, fireworks. **CC BY-NC — flag the non-commercial restriction.**
- **UrbanSound8K** — 8,732 clips, 10 classes incl. gun_shot, siren, engine_idling, jackhammer, drilling. **Non-commercial research license — flag.**
- **DEMAND** — multichannel real environmental noise, CC BY-SA.
- **WHAM!** — real ambient bar/café/park noise.
- **AudioSet** — approximately 2 million clips across 527 classes; only features/labels are officially released and audio must be scraped from YouTube (links rot) — use only for class references, not bulk download.

**Defence-specific noise:**
- **Military Audio Dataset (MAD)** — Kim, Yoon & Jung, "A Military Audio Dataset for Situational Awareness and Surveillance," Scientific Data 11(1):668 (2024); GitHub kaen2891/military_audio_dataset. It "contains 8,075 sound samples from 7 classes corresponding to approximately 12 hours." Classes: communication, gunshot, footsteps, shelling, vehicle, helicopter, fighter. Released under CC BY 4.0. **This is your headline defence noise source.**
- **BGG / "Enemy Spotted"** (game-engine gunshots), **C3GD** (Certus Caliber Classification Gunshot Dataset, 8,000+ field recordings, arXiv 2606.18135) — supplementary gunshot variety.
- Royalty-free SFX (Pixabay, ZapSplat free tier) for artillery/helicopter/drone/siren top-ups — check each clip's license.

**RIRs (reverberation augmentation):**
- **OpenSLR SLR28** (rirs_noises.zip, 1.3 GB, Apache-2.0) — real + simulated RIRs + isotropic/point-source noise. Used by the DNS Challenge.
- **OpenSLR SLR26** — ~60,000 simulated RIRs, Apache-2.0.
- **BUT ReverbDB** — real RIRs from 8+ rooms, free non-restrictive license.

**Licensing flag for the submission:** prefer CC0 / CC BY / Apache-2.0 assets (LibriSpeech, MUSAN, FSD50K CC-BY subset, OpenSLR RIRs, MAD). Explicitly exclude or clearly label CC BY-NC assets (ESC-50, UrbanSound8K, FSD50K NC clips); for a purely academic prototype they are usually acceptable with attribution. Keep an attribution manifest.

### 4. Data pipeline design

Follow DNS Challenge conventions:
- **SNR sampling:** uniform in [-5, +20] dB for training (DNS/FullSubNet convention); include a low-SNR tail [-15, 0] dB for a stress subset. Test at fixed points (e.g., -5, 0, 5, 10, 15 dB) to stratify. VoiceBank-DEMAND test uses {2.5, 7.5, 12.5, 17.5} dB.
- **Mixing convention:** set clean speech to a target active level, scale noise to hit the desired SNR (computed over active speech frames), sum, then check/prevent clipping (rescale mixture and target together if peak > 0 dBFS). Sample rate 16 kHz.
- **Dynamic (on-the-fly) mixing:** mix per epoch/batch so the model sees millions of unique combinations (DNS reports "over 5000 hours seen after ten epochs"). Preferred over pre-generated for generalization and disk economy; pre-generate only a fixed validation/test set for reproducible metrics.
- **Impulsive event handling (your special case):** place transients at random positions overlapping voiced speech (not only in silence, or the task is trivial); vary count (0–3 events/clip), peak level, and event-to-speech ratio; keep some clips transient-free; avoid impossible cases where the transient saturates the whole clip. Oversample MAD gunshot/shelling relative to their natural frequency.
- **RIR augmentation:** convolve ~50–75% of clean clips with a random RIR (DNS convention) before adding noise; optionally reverberate the noise too. Decide and state whether your target is the anechoic clean (denoise only) or reverberant clean (denoise + dereverb).
- **Other augmentation:** random gain, clipping simulation, band-limiting/low-pass (simulate the radio channel), optional SpecAugment.

### 5. Training recipe

- **Features:** 16 kHz, STFT 512-point, 32 ms Hann window, 16 ms hop is a common causal config; GTCRN uses its own sub-band + ERB front end. Power-law compress magnitude (c ≈ 0.3–0.5).
- **Losses:** combine (a) **SI-SNR** (time-domain, scale-invariant — stabilizes training but can over-suppress low frequencies if used alone), (b) **complex/compressed-magnitude spectral MSE** (preserves phase; GTCRN uses magnitude + real + imag terms), (c) **multi-resolution STFT loss**, and (d) an **asymmetric** anti-over-suppression term. Robust recipe: `L = α·L_SISNR + L_mag(compressed) + β·(L_real+L_imag) + L_MRSTFT + γ·L_asym`. GTCRN's published weights: α=0.01, β=0.3. Perceptual options (PMSQE, PESQ-surrogate/MetricGAN) improve perceptual scores but add complexity — leave for the finale.
- **Optimizer/schedule:** Adam, initial LR 1e-3 (or 5e-4), reduce-on-plateau or cosine, batch 12–32, 4-second chunks, gradient clipping, early stopping. ~60–100 epochs typical; GTCRN-size models converge fast.
- **Free-GPU reality:** Colab (T4/L4, ~12 h sessions, quota limits) and Kaggle (P100 or T4×2, 30 GPU-h/week) are enough for a 24–50K-param model — a usable model in 1–3 days of accumulated training. Checkpoint to Google Drive frequently (sessions disconnect). For a 2M-param DeepFilterNet, fine-tuning the pretrained weights is far more realistic than from scratch.
- **2-week strategy:** START from pretrained GTCRN (or DeepFilterNet) weights and fine-tune on your impulsive-heavy data. Training GTCRN from scratch is feasible but fine-tuning de-risks the timeline. Reserve scratch training and distillation for the finale.

### 6. Real-time streaming inference

- **Framing:** process one hop per step (e.g., 16 ms hop, 32 ms window → 50% overlap). Maintain a ring buffer of input samples; compute the STFT on the current window; run the model; reconstruct with overlap-add (weighted by the synthesis window).
- **Recurrent state:** for GRU/RNN models (GTCRN, DeepFilterNet), carry hidden states across frames — export the ONNX model with state tensors as explicit inputs/outputs and feed them back each step (the GTCRN streaming ONNX sample and DeepFilterNet's stateful streaming implementation both do this). Do NOT re-run the whole buffer each frame.
- **Latency budget (report this explicitly):**
  - Algorithmic = window + lookahead. Causal GTCRN with 32 ms window / 16 ms hop ≈ one-frame (~16–32 ms) algorithmic latency and **zero lookahead**. DeepFilterNet adds a 2-frame lookahead → ~40 ms total added latency.
  - Compute latency = model time per hop (Pi 5: sub-millisecond for GTCRN; RTF ≪ 1).
  - Audio I/O: ALSA period/buffer (tune to ~10–20 ms period, 3–4 periods) + USB audio + DAC, roughly ~20–40 ms.
  - Total one-way should stay well under the **ITU-T G.114 (05/2003)** 0–150 ms band, described as "Acceptable for most user applications" — the standard notes that "if delays can be kept below this figure, most applications, both speech and non-speech, will experience essentially transparent interactivity"; 150–400 ms is acceptable with awareness of the impact, and >400 ms is unacceptable. Aim for <100 ms total; you can honestly claim algorithmic latency in the 16–40 ms range.
- **Implementation:** Python with `sounddevice` (PortAudio) callback or blocking stream + `onnxruntime` (CPUExecutionProvider). Keep the callback lean (no allocations); precompute STFT/iSTFT plans. Set `onnxruntime` intra-op threads deliberately (see §7).

### 7. Edge deployment on Raspberry Pi 5

- **Runtime:** ONNX Runtime on ARM64 (install the aarch64 wheel; requires 64-bit Raspberry Pi OS). PyTorch also has out-of-the-box Pi 4/5 support but ONNX Runtime is leaner for inference.
- **Quantization:** dynamic INT8 (`quantize_dynamic`) is the easy win for RNN/GRU/Linear-heavy models like GTCRN — no calibration data needed, typically ~2–4× smaller/faster with minor quality loss. Static INT8 needs a calibration set and gives more speedup but is more work; try it for the finale. Because GTCRN is already tiny, quantization is about latency/power headroom, not fitting in memory.
- **Threads:** set intra-op threads to the number of performance cores (Pi 5 has 4× Cortex-A76). The community Pi 5 GTCRN port scaled from RTF 0.048 (1 thread) to 0.020 (4 threads). For a tiny model, 1–2 threads often minimizes latency jitter; benchmark both.
- **NEON/SIMD:** ONNX Runtime's ARM64 kernels use NEON automatically; the Cortex-A76's wider pipeline is why the Pi 5 is markedly faster than the Pi 4.
- **Expected RTF:** GTCRN ~0.02–0.05 on Pi 5 (large real-time headroom); DeepFilterNet2 ~0.42 on Pi 4 (comfortably real-time, expect better on Pi 5).
- **Audio I/O setup:**
  - USB sound card: appears as an ALSA card; select via `arecord -l` / the device index in sounddevice; tune ALSA `period_size`/`buffer_size` (and the PortAudio latency hint) to minimize latency without underruns.
  - INMP441 I2S MEMS mics: 3.3 V, 24-bit, I2S; configure `dtparam=i2s=on` plus an I2S device-tree overlay (e.g., `googlevoicehat-soundcard`, or a `simple-audio-card`/`i2s-mmap` + Adafruit `i2smic.py` approach) in `/boot/firmware/config.txt`. Native capture is S32_LE at 48 kHz; record stereo with `arecord -D plughw:<card> -c2 -r 48000 -f S32_LE`, then convert to 16-bit / 16 kHz in software. For two mics (primary + reference), wire both on the shared BCLK/LRCLK with one set to the left channel and one to the right (via the L/R select pin), giving a stereo pair for the NLMS reference. Note: combining an I2S mic and an I2S amp needs a single combined overlay, since only the last overlay may bind.

### 8. The LMS/NLMS adaptive filter stage

- **What it needs:** a reference microphone that captures noise correlated with the primary-mic noise but NOT the speech. The adaptive filter models the acoustic path from reference to primary, predicts the noise component, and subtracts it. NLMS (normalized LMS) is standard because speech is highly non-stationary; adaptive-cancellation studies report ~13 dB suppression with NLMS versus ~10 dB with plain LMS.
- **Algorithm (NLMS):** `y(n)=wᵀx(n); e(n)=d(n)−y(n); w ← w + µ·x(n)·e(n)/(‖x(n)‖²+δ)`, where d = primary (speech+noise), x = reference (noise), e = output (cleaned). Adapt only when speech is absent (VAD-gated) to avoid cancelling speech.
- **Role in the hybrid pipeline:** place NLMS EITHER before the neural net (remove correlated low-frequency engine/rotor drone using the reference mic, then let GTCRN clean the residual) OR after it (mop up residual stationary noise). Its strength is periodic/correlated noise (engine, rotor); it does little for uncorrelated impulsive transients.
- **Worth it?** For a 2-week single-channel neural demo, NLMS is OPTIONAL and adds integration risk (needs a good, speech-free reference and careful VAD gating; poor mic separation lets speech leak into the reference and get cancelled). Recommendation: implement the neural path first; add NLMS as a documented "hybrid" enhancement and a talking point (classical + AI), and demo it live only if you get a clean reference channel working with the second INMP441.

### 9. Evaluation and metrics

- **Libraries:** `pesq` (ludlows/python-pesq; wideband `'wb'` at 16 kHz, narrowband `'nb'` at 8 kHz; use the `pesqc2` fork for P.862 Corrigendum 2), `pystoi` (STOI + ESTOI via `extended=True`), `torchmetrics.audio` (SI-SDR/SI-SNR, PESQ, STOI wrappers), `speechmetrics` (wrapper for SI-SDR/STOI/PESQ/SRMR/MOSNet). SNR improvement = output SNR − input SNR. DNSMOS (non-intrusive) is useful when you lack a clean reference (real recordings).
- **Report WB-PESQ at 16 kHz** (state which; WB and NB differ, and Corrigendum 2 raised scores by ~0.8 MOS on average — fix and cite the exact package/version for reproducibility).
- **Stratify results** by (a) noise type — separate stationary (engine, wind) from impulsive (gunshot, shelling) — and (b) input SNR bins. This stratification is where your impulsive-noise story is proven.
- **Realistic expectations vs targets:** PESQ > 2.5 and STOI > 0.85 are achievable for a causal 16 kHz model at moderate SNR (GTCRN posts PESQ 2.87 / STOI ~0.94 on VCTK-DEMAND). But at very low SNR (≤0 dB) and on unseen impulsive noise, expect lower PESQ (often 1.8–2.5). Going causal / low-lookahead typically costs ~0.05–0.3 PESQ and a few STOI points versus the non-causal version of the same model (documented for DCCRN, LaCo-SENet, BASENet). Set judge expectations: hit the targets on the stationary/moderate-SNR set, and show honest, improving numbers on the hard impulsive/low-SNR set.

### 10. Architecture diagram / system design

Signal flow:

`[Primary mic (INMP441 or USB)] → [ALSA capture, S32_LE 48 kHz] → [resample to 16 kHz, to float] → [ring buffer]`
`[Reference mic] → (optional) [VAD-gated NLMS adaptive filter] →`
`→ [STFT / sub-band front-end] → [GTCRN causal model + recurrent state] → [mask / complex-filter apply] → [iSTFT + overlap-add]`
`→ [optional residual NLMS / post-filter] → [resample to output rate] → [ALSA playback] → [headphones / comms unit]`

Side branches: a metrics/logging path (offline PESQ/STOI/SI-SNR on recorded pairs) and an optional impulsive-event detector feeding the detect-then-restore stage.

### 11. Risks and failure modes

- **Buffer underruns/overruns** → clicks/dropouts. Mitigate with adequate ALSA period/buffer and a lean callback; pre-warm the model.
- **Causal/non-causal confusion** → a model that "works" offline (with lookahead or whole-file normalization) fails live. Train and evaluate in the exact streaming configuration; never normalize using future/global statistics.
- **Train–test mismatch** → great on synthetic, poor on the live Indian-speaker demo. Fine-tune on Indian speech; record a small in-room test set with your actual mics; include your own recorded noises.
- **Over-suppression / speech distortion at low SNR** → use the asymmetric loss and attenuation limiting; don't chase maximum noise removal.
- **Musical noise** → the classic spectral-subtraction artifact; the neural model plus complex/deep-filtering reduces it; avoid aggressive gain flooring.
- **Live vs offline gap** → mic gain/AGC, DC offset, and 32→16-bit conversion bugs. Verify the capture chain with a known tone (FFT check) before wiring the model.
- **Free-GPU disconnects** → checkpoint constantly; keep a small reproducible subset for fast iteration.

### 12. Demo design and likely judge questions

**Live demo structure:** (1) show the clean baseline; (2) play/inject a defence noise (helicopter loop + a gunshot) into the room or the input while a teammate speaks Indian-accented English; (3) toggle enhancement ON/OFF live so judges hear the difference; (4) show a live latency readout and an on-screen before/after spectrogram; (5) show the stratified metrics table (stationary vs impulsive, by SNR) and the latency budget vs G.114. Keep a recorded fallback video for hardware failure.

**Likely questions + strong answers:**
- *"Why Raspberry Pi, not Jetson?"* — Cost-efficiency by design: Pi 5 ~₹8,000 vs Jetson AGX Orin 64GB ~₹2.5 lakh (~30×). GTCRN needs ~40 MMACs/s; the Pi CPU runs it at RTF ~0.02–0.05 — a Jetson GPU is idle overkill for this workload. Deployable at squad scale.
- *"Is it real-time / what's the latency?"* — Yes; algorithmic latency 16–40 ms (causal, no/low lookahead), total one-way <100 ms, inside ITU-T G.114's 150 ms band; RTF ≪ 1.
- *"How is this different from Krisp/RNNoise/DeepFilterNet?"* — Those target stationary/office noise; we specifically target impulsive defence noise via an impulsive-heavy data pipeline plus asymmetric/multi-resolution losses, and we prove it with stratified metrics.
- *"Why not just a bigger model?"* — Edge power/thermal budget; we show a 24K-param causal model hits the targets, which matters for field deployment.
- *"How do you handle phase?"* — Complex-domain processing / deep filtering rather than magnitude-only masking.
- *"Does it distort speech?"* — Asymmetric anti-over-suppression loss + attenuation limiting; we report PESQ/STOI (intelligibility), not just SNR.
- *"Indian accents?"* — Fine-tuned on SPRING-INX/IndicTTS/Indic-TIMIT and tested on our own Indian-speaker recordings.

## Recommendations

**Stage 0 (days 1–2):** Freeze scope. Set up Colab/Kaggle; clone GTCRN (MIT) and its ONNX streaming sample, plus DeepFilterNet as fallback. Download LibriSpeech + MUSAN + FSD50K (CC-BY subset) + OpenSLR28 RIRs + MAD. Get the pretrained GTCRN running offline and reproduce a PESQ number.

**Stage 1 (days 3–7):** Build the data pipeline (dynamic mixing, DNS SNR conventions, impulsive-event oversampling, RIR convolution, clipping). Fine-tune GTCRN with SI-SNR + compressed-magnitude + multi-resolution STFT + asymmetric loss. Track PESQ/STOI/SI-SNR on a fixed validation set, stratified by noise type and SNR.

**Stage 2 (days 8–11):** Export to ONNX (streaming, stateful), dynamic INT8 quantize, deploy to Pi 5 with sounddevice + onnxruntime. Get the USB sound card capture→enhance→playback loop working; measure RTF and latency. Add the INMP441 mics if time allows.

**Stage 3 (days 12–14):** Build the demo (ON/OFF toggle, live spectrogram, latency readout, metrics table), record the fallback video, rehearse judge Q&A. Add NLMS only if a clean reference channel works.

**Benchmarks that change the plan:** If GTCRN quality is visibly poor on impulsive noise even after fine-tuning → switch primary to DeepFilterNet2 (accept ~40 ms lookahead) or add the detect-then-restore stage. If Pi latency/underruns are unacceptable → reduce hop overlap, pin threads, or fall back to a laptop demo with the Pi shown as the deployment target. If free-GPU time is insufficient → rely fully on fine-tuning (no scratch training) and shrink the dataset.

**Finale (3-month) additions:** train from scratch and/or distill from a non-causal teacher (MP-SENet/FullSubNet); add static INT8 + pruning; multi-channel/beamforming with the mic pair; a proper impulsive-event detector; larger Indian-speech fine-tuning; formal listening tests (P.808) and DNSMOS; ruggedized hardware and comms-unit integration.

## Caveats
- The Raspberry Pi 5 GTCRN RTF (~0.048 single-thread) is from a community port of a modified ~49K GTCRN-AEC variant, not the original authors — treat it as indicative and re-benchmark your own build.
- VCTK/VoiceBank-DEMAND numbers are on read speech without strong reverb and are a mismatched proxy for defence conditions; your own stratified test set is the real evidence.
- Vendor-blog claims of DeepFilterNet3 "PESQ 3.5–4.0 / STOI >0.95" are non-authoritative and excluded here.
- Some noise datasets (ESC-50, UrbanSound8K, and FSD50K's CC-BY-NC clips) are non-commercial; keep an attribution manifest and prefer CC0/CC-BY/Apache assets for the submission.
- PESQ values depend on the exact package/version (P.862 vs Corrigendum 2) and on WB vs NB — fix and report them for reproducibility.