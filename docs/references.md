# References, Datasets and Tooling

Everything this project builds on, split by whether we **used** it or merely
**surveyed** it. The distinction matters: a references slide that lists papers you
only skimmed invites a question you cannot answer.

Verified against `requirements.txt`, `third_party/gtcrn/`, and the installed
environment on 2026-08-31.

---

## 1. Core model

| Work | Detail |
|---|---|
| **GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources** — Rong et al., **ICASSP 2024** | The architecture this project fine-tunes. 48,245 parameters, 33 MMACs/s, causal. https://ieeexplore.ieee.org/document/10448310 |
| **Official implementation** — Xiaobin-Rong/gtcrn | Vendored at commit `502ebfa`, **MIT licence**. Includes pretrained checkpoints and the streaming reformulation. https://github.com/Xiaobin-Rong/gtcrn |

---

## 2. Datasets

| Dataset | Role in this project | Licence |
|---|---|---|
| **Military Audio Dataset (MAD)** — Kim, Yoon & Jung, *Scientific Data* 11:668 (2024) | **Primary defence noise.** 7,466 clips; we use 5,655 train / 830 test after screening. https://github.com/kaen2891/military_audio_dataset | **CC BY 4.0** |
| **LibriSpeech** — Panayotov et al., ICASSP 2015 | **Clean speech for training.** dev-clean, 2,703 files. https://www.openslr.org/12 | CC BY 4.0 |
| **VCTK-DEMAND (VoiceBank+DEMAND)** — Valentini-Botinhao et al., Univ. of Edinburgh | **Held-out evaluation only — never trained on.** 824 test pairs; used to reproduce the published benchmark. https://datashare.ed.ac.uk/handle/10283/2791 | Free end-user licence |
| **DEMAND noise database** — Thiemann, Ito & Vincent (2013) | The noise half of VCTK-DEMAND. https://zenodo.org/record/1227121 | CC BY-SA |
| **ESC-50** — Piczak, ACM Multimedia 2015 | Siren and wind — noise types the problem statement names that MAD does not contain. **Evaluation only, never trained on**, because of the licence. https://github.com/karolpiczak/ESC-50 | **CC BY-NC** (non-commercial) |

**Stated gap:** no dataset used here contains quadcopter-drone noise. MAD's helicopter
and fighter classes are acoustically related but not equivalent.

---

## 3. Evaluation metrics and standards

| Standard / paper | Use |
|---|---|
| **ITU-T P.862 / P.862.2** — PESQ, wideband mode | Perceptual quality. Target > 2.5. https://www.itu.int/rec/T-REC-P.862 |
| **STOI** — Taal, Hendriks, Heusdens & Jensen, ICASSP 2010; IEEE TASLP 2011 | Intelligibility. Target > 0.85. |
| **SI-SNR / SDR** — Le Roux, Wisdom, Erdogan & Hershey, *"SDR — Half-baked or Well Done?"*, ICASSP 2019 | Scale-invariant separation quality. Target > 15 dB. https://arxiv.org/abs/1811.02508 |
| **ITU-T G.114** — One-way transmission time | The 150 ms latency budget our 83.6 ms is measured against. https://www.itu.int/rec/T-REC-G.114 |
| **ITU-T P.56** — Active speech level | **Cited, not implemented.** We use an RMS proxy over voiced frames and say so. |

---

## 4. Classical baselines (implemented for comparison)

| Work | Result we measured |
|---|---|
| **Boll (1979)**, *Suppression of Acoustic Noise in Speech Using Spectral Subtraction*, IEEE TASSP 27(2) | Our `scripts/spectral_subtraction.py`. **+0.57 PESQ on engine noise, +0.07 on gunfire** — the core evidence that classical DSP fails on impulsive noise. |

---

## 5. Software and toolchain

| Tool | Version | Role |
|---|---|---|
| **PyTorch** | 2.13.0 | Training and fine-tuning |
| **ONNX** / **ONNX Runtime** | 1.22.0 / 1.29.0 | Edge inference, CPUExecutionProvider, opset 11 |
| **onnx-simplifier** | 0.7.3 | Constant folding, BatchNorm fusion |
| **silero-vad** | 6.2.1 | Speech-contamination screening of noise clips (MIT) |
| **pesq** / **pystoi** | 0.0.4 / 0.4.1 | Metric implementations |
| **sounddevice** / PortAudio | 0.5.6 | Real-time audio I/O |
| **NumPy** / **SciPy** / **librosa** / **soundfile** | 2.5.2 / 1.18.1 / 1.0.0 / 0.14.0 | Signal processing |
| **Raspberry Pi 5** (BCM2712, Cortex-A76) | — | Edge deployment target |

---

## 6. Surveyed for context — not used in the implementation

Listed separately so nothing here can be mistaken for a component of the system.

- **Wang & Chen (2018)**, *Supervised Speech Separation Based on Deep Learning: An Overview*, IEEE/ACM TASLP — the masking-vs-mapping framing behind our choice of a complex ratio mask.
- **DCCRN**, **FullSubNet**, **DPRNN** — architectural ancestors of GTCRN's dual-path and sub-band ideas.
- **Wiener filtering**, **classical LMS/NLMS ANC** — named in the problem statement as prior art; the NLMS branch is scoped as future work and requires a reference microphone we do not have.
- **DNS Challenge** (Interspeech/ICASSP) — the corpus family the pretrained checkpoint was trained on.

---

## Compressing this to one slide

A slide cannot hold the table above. Keep five lines and put the rest in the appendix:

1. **Model** — GTCRN, ICASSP 2024 (MIT)
2. **Defence noise** — Military Audio Dataset, *Scientific Data* 2024 (CC BY 4.0)
3. **Speech** — LibriSpeech (train) · VCTK-DEMAND (held-out eval)
4. **Metrics** — ITU-T P.862 PESQ · STOI · SI-SNR · ITU-T G.114 latency
5. **Deployment** — PyTorch → ONNX Runtime → Raspberry Pi 5

Then add one line that most teams will not have, and that judges notice:

> *Every dataset licence checked. ESC-50 is CC BY-NC, so it is used for evaluation only and excluded from any deployment claim.*
