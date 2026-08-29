# SIH 2026 — Idea Presentation Content

Everything that goes into the six slides, plus layout guidance for each.

**How to use this:** the text in the boxed sections is what goes *on the slide* — keep it
short, it's for reading at a glance. The notes underneath each slide are for you: what to
say out loud, what to put where, and what to avoid.

**Three rules for the whole deck:**

1. **Nothing on a slide that you have to read aloud verbatim.** Slides carry the numbers
   and the picture; you carry the sentences.
2. **Every number on these slides is measured, not estimated.** If a judge asks "how do
   you know?", there's a real answer for every one. Never add a number that isn't.
3. **Plain words beat technical words.** The judge may not be an audio engineer. "The AI
   learns which sounds are the voice" lands; "complex-domain spectral masking" doesn't.

---

## SLIDE 1 — TITLE PAGE

> **Problem Statement ID:** *[fill from portal]*
>
> **Problem Statement Title:** AI/ML-enabled adaptive noise cancellation for defence
> communications — suppressing stationary, non-stationary and impulsive noise while
> maintaining speech intelligibility and real-time performance on embedded hardware
>
> **Theme:** *[fill from portal]*
>
> **PS Category:** Software
>
> **Team ID:** *[fill]*
> **Team Name:** *[fill]*

**Notes:** Template-controlled — just fill the blanks. Don't restyle it.

---

## SLIDE 2 — IDEA / PROPOSED SOLUTION

> ### Clear Speech Through Gunfire
> **AI noise removal for defence radio — real-time, offline, on low-cost hardware**
>
> **The problem**
> - A soldier's radio mic captures their voice **and** the gunfire, shelling, rotors and
>   engines around them
> - The listener misses words. In a fire-support request, a missed grid reference is a
>   serious problem
> - Existing noise removal is built for offices and phone calls — not battlefields
>
> **Our solution**
> - A small AI model cleans the microphone signal **before it is transmitted**
> - Trained on **real defence sounds** — gunfire, shelling, helicopters, sirens, wind
> - Runs **live, offline, on a low-cost board** — no internet, no cloud, no GPU
>
> **Already working — not a proposal**
> - Live microphone → AI → headset, with an on/off switch you can hear
> - **83 milliseconds** end-to-end — faster than a blink, so conversation feels natural
> - Measured on real defence noise: **PESQ 2.49 · STOI 0.92 · SNR 20 dB**
>   *(targets: 2.5 / 0.85 / 15)*
>
> **What makes it different**
> - Ordinary noise removal is trained on office noise. **We trained ours on gunfire and
>   shelling** — and it shows: on gunfire, the traditional method gives almost no
>   improvement, ours gives a large one

**Notes — this is your most important slide.**

- **Lead with "already working."** Most idea submissions are proposals. You have a system
  with measured numbers. Say so in the first ten seconds.
- **Visual:** the `pipeline_diagram.svg` on the right half, or the spectrogram
  before/after. If you can only fit one picture in the whole deck, put it here.
- The three metric numbers should be **large**. They're your credibility.
- **Don't explain how the AI works on this slide.** That's Slide 3.

---

## SLIDE 3 — TECHNICAL APPROACH

> ### How It Works
>
> **The five steps** *(all happening ~60 times per second)*
> 1. **Listen** — microphone captures voice + battlefield noise
> 2. **Slice** — audio cut into tiny 16-millisecond pieces
> 3. **Decide** — AI examines each slice and separates voice from noise
> 4. **Rebuild** — cleaned slices stitched back into smooth speech
> 5. **Deliver** — clear voice sent to the headset
>
> **What we used**
>
> | | |
> |---|---|
> | **The AI model** | GTCRN — only 48,000 settings *(large AI models have billions)* |
> | **Training** | PyTorch, on a free cloud GPU |
> | **Running it** | ONNX — one file that runs on laptop or embedded board |
> | **Hardware** | Low-cost single-board computer. **No GPU required.** |
>
> **How we trained it**
> - Took thousands of clear speech recordings
> - Mixed them with **real military audio** (a published research dataset: gunfire,
>   shelling, helicopters, vehicles)
> - Created millions of practice examples at many different noise levels
> - The AI practised recovering the clean speech until it got good at it
>
> **Why it's fast enough**
> - The AI never waits to "hear what comes next" — it works on the present moment only
> - Uses just **1 millisecond of every 16** — about 6% of the available time

**Notes:**

- **Visual:** the workflow diagram, or the five steps as an icon row. This slide needs a
  picture more than any other.
- **"48,000 settings vs billions"** is the line that lands with non-technical people —
  it makes "small and efficient" concrete.
- If asked why a small model: *"a big model would need an expensive computer and a power
  supply. This has to run on a battery in the field."*

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

> ### Feasibility — Already Proven
>
> | Question | Answer |
> |---|---|
> | Does it work in real time? | ✅ Uses 6% of available processing time |
> | Is the delay acceptable? | ✅ 83 ms — international standard allows 150 ms |
> | Does the AI actually work? | ✅ Matches published benchmark results |
> | Will it fit on a cheap board? | ✅ 1 MB file, no GPU needed |
> | Can we afford it at scale? | ✅ Under ₹8,000 per unit |
>
> **Challenges and how we handled them**
>
> | Challenge | What we did |
> |---|---|
> | Sudden loud sounds (gunfire) break traditional methods | Trained specifically on gunfire and shelling |
> | Risk of deleting the speech along with the noise | Measure intelligibility, not just noise removal |
> | Hidden speech inside our noise recordings | Detected and removed 221 contaminated clips before training |
> | Results could be luck, not real improvement | Every result repeated 100 times with statistical testing |
>
> **Honest scope**
> - Works well when the voice is at least as loud as the background
> - When noise is *louder* than the voice, quality drops — that is a limit of the
>   problem, not of our method. We report where it works and where it doesn't.

**Notes:**

- **The "hidden speech" row is worth pausing on.** Some gunfire recordings secretly
  contain soldiers shouting. Training on those would teach the AI that *voices are
  noise* — the opposite of the goal. Finding and removing them shows real care.
- **The honest-scope line is deliberate.** It pre-empts the hardest question. Judges
  reward teams who state limits before being asked; volunteering it reads as confidence,
  hiding it reads as either ignorance or evasion.
- If asked to expand: *"below the point where noise equals speech, the information needed
  to reconstruct the words largely isn't in the signal any more. No system solves that."*

---

## SLIDE 5 — IMPACT AND BENEFITS

> ### Why It Matters
>
> **Operational**
> - Fewer misheard orders and repeated transmissions in high-noise operations
> - Directly improves reliability of battlefield communication
> - Less listener fatigue on long missions
>
> **Economic**
> - **Under ₹8,000 per unit** vs ~₹2.5 lakh for the GPU hardware normally assumed
> - Affordable at **squad scale**, not just command posts
> - Low power — runs on a battery
>
> **Strategic**
> - **Fully offline** — no internet, no cloud, no data leaves the device
> - Built entirely on open, freely-licensed components — **no foreign vendor lock-in**
> - Indigenous capability that can be maintained and retrained in-country
>
> **Beyond defence**
> - Disaster response · mining · aviation ground crew · industrial radio
> - Any situation where people must talk over dangerous noise
>
> **Scalable**
> - Same system retrains for new noise environments without changing the hardware

**Notes:**

- **"No data leaves the device"** is a strong point for a defence audience — say it
  clearly.
- **Visual:** big number callouts — **₹8,000 · 83 ms · 48K · offline**.
- Keep "beyond defence" brief. It shows breadth, but the judges are here for the defence
  application.

---

## SLIDE 6 — RESEARCH AND REFERENCES

> ### Research and References
>
> **Core model**
> - Rong, Sun, Zhang, Hu, Zhu, Lu — *"GTCRN: A Speech Enhancement Model Requiring
>   Ultralow Computational Resources"*, **ICASSP 2024**, pp. 971–975
> - github.com/Xiaobin-Rong/gtcrn *(MIT licence)*
>
> **Datasets**
> - **Military Audio Dataset** — Kim, Yoon, Jung, *Scientific Data* **11:668 (2024)**,
>   CC BY 4.0 — 8,075 clips: gunfire, shelling, helicopter, vehicle
> - **VCTK-DEMAND** — University of Edinburgh DataShare
> - **LibriSpeech** — OpenSLR SLR12, CC BY 4.0
> - **ESC-50** — Piczak, ACM Multimedia 2015 *(siren, wind)*
>
> **Methods**
> - Boll — *"Suppression of acoustic noise in speech using spectral subtraction"*,
>   **IEEE TASSP 27(2), 1979** — the classical baseline we compare against
> - Braun et al., arXiv 2205.06931 — anti-over-suppression loss
> - Défossez et al., arXiv 2006.12847 — multi-resolution STFT loss
>
> **Standards**
> - **ITU-T G.114** — one-way latency; under 150 ms is "acceptable for most user
>   applications"
> - **ITU-T P.862 (PESQ)** · **STOI** (Taal et al., 2011) — speech quality and
>   intelligibility measures

**Notes:**

- All datasets are **openly licensed** — mention this if asked about data provenance.
- ESC-50 is CC BY-NC (non-commercial). Fine for an academic prototype with attribution;
  flag it honestly if a judge asks about deployment licensing.

---

## Design guidelines

**Layout**
- Follow the SIH template. Don't restyle it — it's a submission requirement.
- **Maximum 6 slides**, including the title. Delete the instructions slide before upload.
- **Export as PDF.** The portal accepts nothing else.

**Text**
- Points and short phrases. **No paragraphs.**
- Slide titles ~32–36 pt, body ~16–18 pt, never below 14 pt.
- Bold the numbers. They're what a judge remembers.

**Visuals — put at least one on every content slide**
- Slide 2: `docs/pipeline_diagram.svg` or the before/after spectrogram
- Slide 3: `docs/eraser_workflow.png` (the plain-language five steps)
- Slide 4: the feasibility ✅ table — it reads as a visual on its own
- Slide 5: large number callouts

Available in the repo: `docs/pipeline_diagram.svg`, `docs/architecture_diagram.svg`,
`docs/eraser_workflow.png`, `docs/eraser_architecture.png`,
`demo_package/spectrogram_comparison.png`.

**Language — swap these**

| Don't write | Write |
|---|---|
| complex spectral masking | the AI decides which sounds to keep |
| causal streaming inference | works live, without waiting |
| 48.2K parameters, 33 MMACs/s | 48,000 settings — large AI models have billions |
| PESQ 2.49 ± 0.04 | speech quality score 2.49 *(target 2.5)* |
| impulsive noise | sudden loud sounds like gunfire |
| fine-tuned on in-domain data | trained on real defence recordings |

**What to leave off the slides entirely**

- The asymmetric loss function — it didn't measurably contribute, so claiming it invites
  a question you can't win
- Anything about the Jetson unless asked — you chose differently for good reasons, but
  raising it unprompted invites a comparison you don't need
- Any number not in this document

---

## The three questions you will definitely be asked

**"Is this actually working, or just an idea?"**
> It's working. We can demonstrate it live — speak into a microphone with gunfire playing
> in the room and hear the clean output through headphones, with an on/off switch.

**"How is this different from noise cancellation I already have?"**
> Consumer noise removal is trained on offices and cafés. It assumes noise is steady. A
> gunshot isn't steady — it's louder than the voice across all frequencies for a fraction
> of a second. We measured the traditional method on gunfire: almost no improvement.
> Ours handles it because it was trained on the real thing.

**"What are the limitations?"**
> When background noise is louder than the voice, quality drops significantly. That's a
> limit of the problem — the information needed to reconstruct the words isn't fully in
> the signal. We meet the targets from the point where voice and noise are comparable
> upward, and we're explicit about where we don't.

*Answer that third one well and it will do more for you than any other thirty seconds of
the presentation.*
