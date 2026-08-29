# Teaching Guide — Bringing Two Non-Technical Members Up to Speed

**Who this is for:** the two teammates who will build the PPT and present. By the end they
should be able to explain the whole project to a stranger and survive follow-up questions.

**The goal is understanding, not memorisation.** A memorised answer collapses the moment a
judge asks "but why?". Everything below builds from the ground up so that when they're
asked something we didn't rehearse, they can reason to an answer.

---

# PART 1 — How to run the session (for you)

**Format:** ~2 hours, in two sittings if possible. Sitting 1 = Lessons 1–3 (the concepts).
Sitting 2 = Lessons 4–5 (our work and results) + the drill.

**Rules that make it work:**

1. **Play the audio before explaining anything.** Start by playing
   `demo_package/1_BEFORE_noisy.wav`, then the after. Let them *experience* the problem
   before a single technical word. Curiosity first, explanation second.
2. **Show the spectrogram early and often.** `demo_package/spectrogram_comparison.png` is
   the single most useful teaching object we have. Most of the project becomes obvious
   once someone can read that picture.
3. **Make them explain it back.** After each lesson, ask them to explain it to you in
   their own words. If they use a word they can't define, stop there. That's exactly what
   will fail on stage.
4. **Let them ask "stupid" questions.** The questions they're embarrassed to ask are the
   ones judges will ask.
5. **Don't teach the maths.** They don't need Fourier transforms. They need the *picture*
   and the *why*.

---

# PART 2 — The lessons

## Lesson 1 — How a computer "sees" sound

*This is the foundation. Everything else depends on it. Spend the most time here.*

### Sound is pressure over time

A microphone measures air pressure, thousands of times a second. Our system measures
**16,000 times per second** — each measurement is a number. So one second of audio is a
list of 16,000 numbers.

That's the raw form. It's not very useful for separating voice from noise, because voice
and noise are just... mixed together in those numbers.

### The useful view: frequency

Any sound can be described as a combination of **pitches** (frequencies) — low rumbles,
mid tones, high hisses — all present at once, in different amounts.

> **Analogy that works:** a piano chord. You hear one sound, but it's actually several
> notes played together. If you had good enough ears, you could name each note and how
> loud it is. That's what the computer does — it breaks the sound into its component
> pitches and measures how much of each is present.

### The spectrogram — the picture that unlocks everything

Now do that repeatedly — break the sound into pitches, **60 times per second** — and draw
the result as a picture:

- **Horizontal axis** = time, moving left to right
- **Vertical axis** = pitch, low at the bottom, high at the top
- **Brightness** = how loud that pitch is at that moment

**→ Open `demo_package/spectrogram_comparison.png` now and look at the top panel.**

Point out three things:

| What you see | What it is |
|---|---|
| Stacked orange curves near the bottom, moving in waves | **The human voice.** Speech has a stack of related pitches that slide up and down as you talk. That ladder pattern is the signature of a voice. |
| A bright vertical stripe running top to bottom | **A gunshot.** It's loud at *every* pitch simultaneously, for a fraction of a second. |
| A purple haze covering everything | **The noise floor** — rotor, engine, wind. It's everywhere, all the time. |

**Check they've got it:** ask them to point at the voice, then at a gunshot. If they can
do that, they understand the core of this project.

### Why this view matters

In the raw list of numbers, voice and noise are hopelessly mixed. In this picture,
**they look different**. And "looks different" is exactly what makes separation possible.

---

## Lesson 2 — Why removing noise is hard

### The naive idea, and why it fails

Obvious approach: measure the noise during a silent moment, then subtract it.

This is a real technique — **spectral subtraction**, invented in 1979, still used in a lot
of equipment today. It works reasonably on steady noise like a fan or an engine.

**It fails on a battlefield.** Two reasons:

**1. The noise doesn't stay still.** You measured the helicopter a second ago. It's moved.
The engine changed speed. Your measurement is already out of date.

**2. Gunshots break the assumption completely.** Spectral subtraction assumes noise is
roughly constant. A gunshot is a 200-millisecond explosion of energy at every pitch at
once. There's nothing to subtract — by the time you notice it, it's over.

### We measured exactly how badly it fails

This is one of our strongest findings, and it's easy to explain:

| Noise type | Improvement from the 1979 method |
|---|---|
| Engine / vehicle | **+0.57** (works reasonably) |
| Gunfire / shelling | **+0.07** (essentially nothing) |

**Eight times worse on gunfire.** At high noise levels it actually makes things slightly
*worse* than doing nothing.

> **This is the justification for the entire project.** The problem statement says
> traditional methods are limited. We didn't just repeat that — we measured it.

### The trap: it's easy to remove noise badly

Here's the thing that makes this a genuinely hard problem rather than an easy one:

**You could remove 100% of the noise by outputting silence.** Perfect noise removal.
Completely useless.

The real task is removing noise **while keeping the speech intact**. That tension is the
whole game, and it's why we measure *intelligibility* and not just noise reduction. A
system optimised only for "less noise" learns to delete the voice too.

**→ Go back to the spectrogram, bottom panel.** The background is black — noise gone. But
the orange voice ladders are still there, intact. *That's* the achievement. Not the black
part; the fact that the orange survived.

---

## Lesson 3 — What our AI actually does

### The mixing-desk analogy

Picture a sound engineer's mixing desk with **257 volume sliders**, one for each pitch
band from low to high.

For every 16-millisecond slice of audio, our AI looks at the sound and sets all 257
sliders — turning down the ones that are mostly noise, leaving up the ones carrying voice.

Then it does it again for the next slice. **Sixty times a second.**

That set of slider positions is called a **mask**. That's the AI's entire job: look at a
slice, output a mask.

### What "trained" means

Nobody programmed rules like "if the pitch is above 4 kHz, turn it down." Instead:

1. We showed it **millions of examples**: a noisy recording, plus the clean version of the
   same speech
2. It guessed a mask, applied it, and compared the result to the clean version
3. It was wrong, so it adjusted itself slightly
4. Repeat — millions of times

Over time it learns, from the data alone, what voice looks like and what noise looks like.
Nobody told it. It worked it out.

> **If asked "how does it know?"** — it doesn't "know" in a human sense. It has seen
> millions of examples of voice-plus-noise alongside the clean answer, and has adjusted
> itself until its guesses match. It recognises the *pattern* of a voice the way you
> recognise a friend's face without being able to list the rules.

### Why ours is different from ordinary noise removal

Everything above describes any modern noise-removal AI. Here's our specific contribution,
and it's simple:

**What you train it on determines what it's good at.**

Ordinary noise-removal AI is trained on offices, cafés, traffic — the noise in the
datasets researchers usually have. It's very good at those and mediocre at gunfire,
because it's barely seen any.

**We trained ours on real military audio** — a published research dataset with 8,075
recordings of gunfire, shelling, helicopters, and armoured vehicles. That's the whole
innovation, and it's honest and easy to say.

### Why it has to work "live", and why that's hard

The system can't wait. If it buffered two seconds of audio to make a better decision, the
conversation would have a two-second delay and be unusable.

So the AI is **causal** — it only ever uses sound it has already heard, never sound from
the future. That's a real constraint that makes the job harder, and it's why we can claim
real-time operation.

It does keep a **memory** of recent moments, which is how it can tell "this rumble has
been going for ten seconds, that's a helicopter" from "this just started, that's a
gunshot."

---

## Lesson 4 — How we prove it works

*Presenters must understand the metrics, because "how do you measure that?" is a
guaranteed question.*

### The three measurements

| Name | What it measures | Scale | Our result |
|---|---|---|---|
| **PESQ** | How *good* the speech sounds | 1 to 4.5 | **2.49** (target 2.5) |
| **STOI** | How *understandable* the words are | 0 to 1 | **0.92** (target 0.85) |
| **SI-SNR** | How much louder the voice is than remaining noise | decibels | **20.2 dB** (target 15) |

**Why three and not one?** Because they can disagree, and the disagreement is informative.
A system that deletes speech would look fine on noise reduction and terrible on
intelligibility. Reporting all three makes it impossible to hide that.

### Where the numbers come from

Explain it as a fair test:

1. Take a **clean** speech recording — we know exactly what it should sound like
2. Add real defence noise at a **controlled loudness**
3. Run it through our system
4. Compare the output to the original clean version

Because we have the clean original, we can measure exactly how close we got.

### Why "100 trials" matters

We repeat every measurement **100 times** with different speech and noise combinations,
and report the uncertainty.

Explain why with the true story: early on we used 20 trials and saw a pattern that looked
real. We re-ran it with a different random selection and **got a different answer** — the
"pattern" was luck. So we increased to 100 trials and added a statistical test that says
outright whether a difference is real or noise.

> **This is worth saying to judges.** Most projects assert improvement. We can show that
> ours is statistically significant — and we say so when something *isn't*.

---

## Lesson 5 — What we actually built, in order

Give them the narrative. Presenters need the story, not just the facts.

**1. We worked out what was really being asked.**
The problem statement says "Active Noise Cancellation," which usually means
noise-cancelling headphones. But it measures success with PESQ, STOI and SI-SNR — which
are measures of *recorded speech quality*. You can't measure those on sound waves in a
room. So the actual task is cleaning the microphone signal in software. Getting this wrong
would have meant building the wrong system.

**2. We got a working AI and checked our measuring equipment.**
Before trusting anything, we tested against a standard 824-recording benchmark and matched
the published score to within 0.02. That proves our measurement setup is sound — otherwise
every number afterwards would be meaningless.

**3. We built the live system.**
Microphone → AI → headphones, working in real time with an on/off switch.

**4. We measured the delay properly.**
Not estimated — measured. We play a short chirp, record it coming back, and time the gap.
**83 milliseconds.** The international telecom standard says under 150 ms feels like a
natural conversation.

**5. We got real defence audio and found a hidden problem in it.**
The military recordings are cut from combat footage, so some "gunfire" clips **also
contain soldiers shouting**. Training on those would teach the AI that *human voices are
noise to delete* — the exact opposite of the goal. We ran a speech detector over all 5,655
clips, listened to the flagged ones, and removed 221.

> **Presenters should know this story.** It's the detail that shows genuine care rather
> than following a tutorial. It also has a satisfying shape: a subtle trap, found and
> fixed.

**6. We trained the AI on defence noise.**
Improved gunfire performance by **+0.20 PESQ**, statistically significant. Notably,
gunfire went from being the *worst* category to the *best* one.

**7. We proved the old method fails.**
The +0.07 vs +0.57 comparison from Lesson 2.

---

# PART 3 — Self-test

They should be able to answer these **without notes**. If they can't, go back to that
lesson.

1. What are the three things you can see in a spectrogram?
2. Why does the 1979 method fail on gunfire but work on engines?
3. Why can't we just remove all the noise?
4. What is a "mask", in the mixing-desk sense?
5. What makes our AI different from the noise removal in a phone?
6. Why does the AI have to work without hearing the future?
7. Why do we report three numbers instead of one?
8. Why 100 trials instead of 20?
9. What was wrong with some of the gunfire recordings, and why did it matter?
10. What is the system's main limitation?

**Answers to 3, 5, 9 and 10 are the ones that most impress judges.** Drill those hardest.

---

# PART 4 — Question drill

Run this as a hostile-judge role-play. Ask these in random order, out loud, and don't
accept a vague answer.

**"Is this working or is it an idea?"**
> Working. We can demonstrate live — speak into a mic with gunfire playing in the room and
> hear clean output through headphones, with an on/off switch.

**"How is it different from noise cancellation in my earphones?"**
> Yours is trained on offices and cafés, and assumes noise is steady. A gunshot isn't
> steady. We trained ours on real gunfire and shelling. We measured the difference: the
> traditional method gives almost no improvement on gunfire, ours gives a large one.

**"How do you know it isn't just deleting the quiet parts?"**
> Because we report intelligibility, not just noise reduction. Deleting speech would score
> well on one and badly on the other. We report both, precisely so that can't hide.

**"What's the delay?"**
> 83 milliseconds, measured — we play a chirp, record it coming back, and time the gap.
> The telecom standard for natural conversation is under 150.

**"Why not use the expensive hardware in the problem statement?"**
> It says "or similar platforms." We sized the hardware to the model rather than the other
> way round. The model needs about 6% of the processing available, so a GPU would sit
> idle at thirty times the cost. Unit cost matters for equipment issued at squad scale.

**"What are the limitations?"**
> When the background is louder than the voice, quality drops significantly. That's a
> limit of the problem, not of our method — the information needed to reconstruct the
> words isn't fully in the signal. We meet the targets from the point where voice and
> noise are comparable upward, and we say clearly where we don't.

**"What did you personally build?"**
> *(They must be able to answer this honestly.)* We started from a published open-source
> model. We built the training data pipeline, the training system, the evaluation
> framework, and the real-time streaming demonstration. The model was trained by us on
> defence audio.

---

# PART 5 — Rules for presenting

**Never say these:**

| Don't say | Why | Say instead |
|---|---|---|
| "It removes all noise" | Untrue, and easily disproved live | "It substantially reduces noise while preserving the speech" |
| "Tested in real battlefield conditions" | We played recordings through a speaker | "Tested with real defence audio recordings" |
| "It's 99% accurate" | Meaningless here — no such metric | Quote PESQ / STOI / SI-SNR |
| "It works in any condition" | It doesn't, and Lesson 5 explains why | "It meets the targets from this noise level upward" |

**When you don't know the answer:**

> "I don't know that offhand — [teammate] handled that part, or I can find out."

**This is a good answer.** Confident wrong answers are far more damaging than an honest
"I don't know." A judge who catches one invented answer will doubt everything else you
said.

**The single most important habit:** state limits before being asked. Volunteering "here's
where it doesn't work" reads as confidence and command of the material. Being caught
hiding it reads as either not knowing or not being straight.

---

# Appendix — Materials for the session

| File | Use |
|---|---|
| `demo_package/1_BEFORE_noisy.wav` | Play first — let them hear the problem |
| `demo_package/2_AFTER_enhanced.wav` | Play second |
| `demo_package/spectrogram_comparison.png` | **The key teaching object.** Lessons 1, 2, 3 |
| `docs/eraser_workflow.png` | The six steps in plain language |
| `docs/eraser_architecture.png` | Training vs field use |
| `docs/SIH2026_PPT_Content.docx` | What actually goes on the slides |
| `docs/demo_script.md` | Run-of-show for the live demo |

**Run the live demo during the session.** Let each of them wear the headphones and press
the toggle themselves. Someone who has personally experienced the on/off moment presents
it far more convincingly than someone who has only read about it.
