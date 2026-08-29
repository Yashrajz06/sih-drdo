# Demo Script & Run-of-Show

A tested, repeatable ~4-minute demo. Structure: **make them feel the problem, show
them the problem, then remove it.**

The single most common mistake is opening with the solution. Nobody is impressed by
clean audio until they have suffered the noisy version first. Budget real time for
the "before" — silence while they struggle to follow the words does more persuading
than any slide.

---

## Part 1 — Build the assets (do this once, before demo day)

### 1.1 Generate the noise soundtrack

```bash
cd /path/to/sih-drdo          # your clone
source .venv/bin/activate
python scripts/make_demo_noise.py --seconds 90
```

90 seconds of helicopter rotor with 15 gunfire/shelling events. Clips come from the
**test** split — noise the model never trained on, so nobody can accuse you of
demoing on memorised material. Say that out loud if you're asked.

### 1.2 Generate the before/after package

```bash
python scripts/make_demo_package.py
```

Produces in `demo_package/`:

| File | Use |
|---|---|
| `1_BEFORE_noisy.wav` | Play first. The problem. |
| `2_AFTER_enhanced.wav` | Play second. The solution. |
| `spectrogram_comparison.png` | **Put this on a slide.** It makes the noise visible. |

Current measured result on this package:

| | PESQ | STOI | SI-SNR |
|---|---|---|---|
| Before | 1.23 | 0.879 | 5.1 dB |
| After | 1.71 | 0.889 | 12.7 dB |
| Change | **+0.48** | +0.010 | **+7.6 dB** |

### 1.3 Record your own voice version (stronger, do it if you can)

```bash
# Terminal 1 — or play demo_noise.wav from a phone speaker
aplay demo_noise.wav

# Terminal 2
python scripts/live_demo.py --capture-test 15 --out demo_package/mine.wav \
    --onnx models/gtcrn_defence.onnx
```

Gives `mine_raw.wav` (what the mic heard) and `mine.wav` (enhanced). A judge hearing
*your teammate's* voice pulled out of gunfire is far more convincing than a stranger's
recording from a dataset.

**Say "recorded in this room" — and never say "real battlefield conditions."** A
laptop speaker cannot reproduce a gunshot's 125 dB peak. Overclaiming here is the
easiest way to lose credibility with a technical judge.

---

## Part 2 — Run of show

### Beat 1 — The problem, in one sentence (15 s)

> "A soldier calls in a position over the radio. The microphone picks up his voice —
> and also the helicopter above him and the firefight around him. The person
> receiving that call has to understand every word. Today, they often can't."

Don't explain the technology yet.

### Beat 2 — Let them suffer the "before" (30 s)

Play `1_BEFORE_noisy.wav` **at a realistic volume.** Say nothing while it plays.

Then: *"Could you tell me what he said?"* — and wait for the answer. The silence
does the work.

### Beat 3 — Show them exactly what's wrong (45 s)

Put `spectrogram_comparison.png` on screen, top panel only if you can reveal it in
stages. Point at three things:

| Point at | Say |
|---|---|
| The purple wash covering everything | *"Every one of those specks is noise energy sitting on top of the speech."* |
| The bright vertical bar (~2.3 s) | *"That's a gunshot. It's louder than the voice across every single frequency at once."* |
| The stacked orange curves near the bottom | *"That's the actual voice — those stripes are what we have to protect."* |

**This is the part most teams skip, and it's what turns "sounds better" into
"I understand what they did."**

### Beat 4 — The solution (30 s)

Reveal the bottom panel.

> "Same audio, after our model. The background is now black — that's silence. But
> look at the bottom: the voice stripes are still there, intact. The hard part isn't
> removing noise. It's removing noise *without* taking the speech with it."

Then play `2_AFTER_enhanced.wav`. Ask again what he said.

### Beat 5 — Live proof it isn't a recording (60 s)

The most important beat. A judge will privately suspect you pre-processed a file.

```bash
python scripts/live_demo.py --inject-noise demo_noise.wav --noise-gain 0.3 \
    --onnx models/gtcrn_defence.onnx
```

Hand a judge the headphones. Teammate speaks. **Start with enhancement OFF.**
Then press `e`.

Toggle two or three times. Let them ask for it.

### Beat 6 — The numbers (45 s)

Only now, once they've heard it:

> "On real defence noise — gunfire and shelling from a published military audio
> dataset — we measure PESQ 2.49, STOI 0.92, SI-SNR 20 dB. The problem statement
> asks for 2.5, 0.85 and 15."
>
> "End-to-end latency is 83 milliseconds, measured with a chirp and a stopwatch, not
> estimated. The telecom standard for a conversation that feels natural is under 150."
>
> "The model is 48,000 parameters. It runs on an ₹8,000 Raspberry Pi, offline, no
> internet."

### Beat 7 — Close (15 s)

> "Traditional noise removal gives +0.57 PESQ on engine noise but only **+0.07** on
> gunfire — it assumes noise is steady, and a gunshot isn't. Ours handles both.
> That's the gap we set out to close."

---

## Part 3 — The shortcomings, precisely

If asked *"what exactly is wrong with the original audio?"* — four specific answers:

1. **Broadband masking.** The noise isn't in one frequency band you could filter out.
   It covers the entire spectrum, overlapping the speech everywhere at once.
2. **Impulsive events.** A gunshot is louder than the voice across all frequencies
   simultaneously, for ~200 ms. Anything relying on "noise is steady, speech isn't"
   fails completely here.
3. **Non-stationary sources.** Rotor speed changes, sirens sweep. A noise profile
   measured a second ago is already wrong.
4. **The consequence:** words drop out. Not degraded audio — *missing information*.
   In a fire-support request, that's a wrong grid reference.

And how each is addressed:

| Shortcoming | Our answer |
|---|---|
| Broadband masking | Per-frequency decision, ~257 bands, every 16 ms |
| Impulsive events | Trained on real gunfire/shelling, oversampled during training |
| Non-stationary | Model carries memory between frames; adapts continuously |
| Phase damage | Complex-domain mask corrects timing, not just volume — avoids the watery artefacts older methods produce |

---

## Part 4 — Pre-flight checklist

Run through this the morning of.

- [ ] `python scripts/live_demo.py --check` passes
- [ ] `demo_package/` files exist and play
- [ ] **Wired** headphones (Bluetooth adds 100–200 ms and wrecks the latency claim)
- [ ] Input device correct: `python -c "import sounddevice;print(sounddevice.query_devices())"`
- [ ] Mic level healthy — `--capture-test` warns below −50 dBFS
- [ ] Laptop volume tested in *that room*, not yours
- [ ] Backup video on the desktop and on a phone
- [ ] Venv activated (`(.venv)` visible in the prompt) — the single most common failure
- [ ] Rehearsed twice end-to-end with a stopwatch

---

## Part 5 — Failure recovery

| If | Do |
|---|---|
| Live demo won't start | Go straight to the recorded files. Don't debug on stage. |
| Audio device error | `--inject-noise` mode needs no external playback device |
| Feedback squeal | Output is going to speakers, not headphones. Kill it immediately. |
| Judge can't hear a difference | Fall back to the spectrogram — the visual doesn't depend on room acoustics |
| Everything fails | Play the backup video. Narrate over it. |

**Record the backup video the day before.** Live demos fail in unfamiliar rooms, and
a recording turns a disaster into a minor inconvenience.

---

## Part 6 — Hard questions

**"Did you just add noise digitally and remove it?"**
> Fair question. The noise clips come from the dataset's held-out test split — the
> model never saw them in training. And we can do it acoustically: play the noise
> from a phone into the room and the microphone picks up both together. Same result.

**"How do we know it isn't just muting quiet parts?"**
> Because we report STOI and PESQ, not just noise reduction. Muting would score well
> on noise removal and badly on intelligibility. We also train with a loss term that
> specifically penalises deleting speech.

**"Is 83 ms real?"**
> Measured, not estimated — we play a chirp, record it coming back, and cross-correlate.
> The estimator is tested against known delays first. Slides show the method.

**"Why not the Jetson the problem statement mentions?"**
> It says "or similar platforms." We sized the hardware to the model rather than the
> reverse: 33 MMACs/s, using 1.1 ms of every 16 ms frame. A Jetson GPU would idle at
> 30× the cost. That matters for equipment issued at squad scale.

**"What doesn't work?"**
> Below 0 dB — when noise is as loud as the voice — we reach about PESQ 1.3. That's a
> limit of the problem, not of our approach; the information needed to reconstruct the
> speech largely isn't in the signal. We hit the targets from +10 dB upward and we're
> explicit about where we don't.

*Answering that last one well is worth more than any other beat in the demo.*
