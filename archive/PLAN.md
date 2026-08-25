# PLAN.md — the week

Hackathon: **The Agent Harness Hackathon** (WeMakeDevs × TrueFoundry).
Runs Mon 24 Aug – Sun 30 Aug 2026. Submissions close **Sun 30 Aug, 8:00 PM London**
— that is **~00:30 IST on Mon 31 Aug**. Treat the real deadline as **Sun 18:00 IST**
and keep the rest as buffer.

Solo build, ~30+ hours. Read `ARCHITECTURE.md` first.

---

## Working rules (apply every day)

1. **Never commit to `main`.** Every change goes through a pull request, and Qodo
   reviews it before merge. The PR trail *is* the Best Code Quality submission and
   cannot be manufactured on Saturday.
2. **Small PRs.** One concern each. Two to four per day. A 40-file PR gets a
   useless review and tells a judge nothing.
3. **Address every Qodo finding.** Fix it, or reply in the thread saying why you
   disagree. An ignored finding is worse than no review.
4. **Verify the harness API against the live docs before implementing against it.**
   Do not invent SDK signatures. If the docs are ambiguous, write the smallest
   possible spike, run it, and record what actually worked in `docs/adr/`.
5. **End each day with something demoable.** If today's work does not visibly move
   the demo forward, you built the wrong thing.
6. **Record 30 seconds of screen capture whenever something works for the first
   time.** By Sunday you will want footage you did not think to take. This also
   feeds the social-post prize.
7. **Stop adding features at Saturday noon.** Nothing new after that. Ever. That
   rule is the difference between submitting and not submitting.

---

## Monday 24 — the harness runs, the repo is real

**Goal:** the harness is talking to your database, and the repo is set up so that
the code-quality track is winnable.

- [ ] Register for the hackathon (the form on the hackathon page). Join the
      WeMakeDevs Discord.
- [ ] Create the public repo `migration-gate`. MIT licence. Commit
      `ARCHITECTURE.md`, `PLAN.md`, and a stub README **before anything else** —
      first commit shows intent.
- [ ] Install **Qodo** on the repo. Today, not later. Protect `main` so PRs are
      required.
- [ ] `npx @truefoundry/trueforge`. Connect a model with your own API key.
      Confirm the chat UI responds.
- [ ] `docker compose up` a Postgres. Write `db/init/` creating `mg_reader` and
      `mg_writer` with the privilege split from ARCHITECTURE §4. Seed ~50k rows.
- [ ] Connect a **read-only** Postgres MCP server to the harness. Prove it: ask
      the agent to describe the `users` table and watch it answer from live
      introspection.
- [ ] Prove the negative too: ask it to `DROP TABLE users` and watch it fail on
      permissions. Screen-record that. It is a good social post and a good demo beat.

**PRs:** `chore: scaffold repo`, `feat: demo database and roles`,
`feat: connect read-only postgres mcp`

**Done when:** the agent answers a real question about your real schema, and
cannot write to it.

**If blocked:** the single highest-value thing today is the read-only MCP
connection. Drop the seeding polish before you drop that.

---

## Tuesday 25 — planning

**Goal:** English in, reviewed migration plan out.

- [ ] Scaffold `packages/migration-mcp`. Register it with the harness. Get a
      trivial tool (`ping`) called end-to-end before writing real logic.
- [ ] Implement `plan_migration` → the `MigrationPlan` type (ARCHITECTURE §5.3).
      Both `up` and `down`, always.
- [ ] Write `skills/postgres-migrations/` and load it. Verify it changes output:
      without the skill the agent will happily write `ADD COLUMN ... DEFAULT
      now()`; with it, it should split the operation.
- [ ] Add the **reviewer subagent** (§5.4). No write tools. Objections go into
      the plan notes.
- [ ] `docs/adr/001-*.md`: own MCP server vs off-the-shelf.

**PRs:** `feat: migration-mcp scaffold`, `feat: plan_migration`,
`feat: migration skill`, `feat: reviewer subagent`

**Done when:** "add a last_active_at column backfilled from sessions" produces a
sane two-phase plan with a rollback and at least one subagent objection.

**If blocked:** the subagent is the droppable piece. The plan is not.

---

## Wednesday 26 — the sandbox (the whole day, on purpose)

**Goal:** the migration is proven somewhere safe before anyone is asked to trust it.

This is the day that separates the project from a chat box. Guard it. Do not let
Tuesday's polish bleed into it.

- [ ] Provision a sandbox through the harness and run *anything* in it. Confirm
      that first.
- [ ] Stand up Postgres inside the sandbox. Load the schema snapshot from primary
      (schema only — invariant I2).
- [ ] Generate synthetic rows scaled to real row-count estimates.
- [ ] Implement `verify_in_sandbox`: apply `up`, capture timing / rows / locks,
      run assertions, apply `down`, compare schema hashes, return the report.
- [ ] Teardown, including on failure. A leaked sandbox on demo day is a bad look.
- [ ] `docs/adr/002-*.md`: how the scratch database gets its schema.

**PRs:** `feat: sandbox provisioning`, `feat: verify_in_sandbox`,
`feat: verification assertions`

**Done when:** a plan comes back with `reversible: true`, real timings, and a
verdict.

**If this fails by 20:00 Wednesday:** the escape hatch is a *simplified*
verification — run the migration against a second throwaway Postgres in the
sandbox with a hand-written schema fixture instead of a live snapshot. Weaker, but
it keeps the sandbox in the story. Take the escape hatch rather than losing
Thursday. Do not abandon the sandbox itself; it is a judging criterion.

---

## Thursday 27 — the gate

**Goal:** the agent stops, and a human decides.

- [ ] Register `apply_to_primary` as a tool requiring a human checkpoint. Confirm
      the harness genuinely blocks it — try to make it fire without approval and
      fail to.
- [ ] Implement `apply_to_primary`: preconditions (report exists, verdict not
      failed, schema hash unchanged), transaction, assertions, rollback on failure,
      commit on pass.
- [ ] Implement rejection-with-reason → agent revises the plan.
- [ ] Audit log (§8).
- [ ] **Test the reconnect:** trigger an approval, hard-refresh the browser, confirm
      it is still pending and still approvable. This is a scored capability and it
      takes ten minutes to verify.

**PRs:** `feat: approval checkpoint`, `feat: apply_to_primary`,
`feat: audit log`, `test: approval survives reconnect`

**Done when:** the full loop runs end to end for the happy path, and the rejection
path produces a revised plan.

---

## Friday 28 — the interface

**Goal:** a stranger could drive it. This is also the Best UI track, for free.

- [ ] Build the approval panel (§5.6): SQL diff, verification report, plain-English
      irreversibility banner, Approve / Reject.
- [ ] Make agent state legible while it works — introspecting / planning /
      reviewing / verifying / waiting on you.
- [ ] Handle the ugly states: verification failed, precondition failed, apply
      rolled back. Judges will hit these.
- [ ] Run the destructive path (`DROP COLUMN`) and make the refusal read well.
- [ ] `docs/adr/003-*.md`: UI surface choice.

**PRs:** `feat: approval panel`, `feat: agent status`, `feat: error states`

**Done when:** you can hand a laptop to someone who has not seen the project and
they understand what is being asked of them.

---

## Saturday 29 — freeze, then harden

**Feature freeze at 12:00.** After that, only fixes, docs, and rehearsal.

- [ ] `git clone` into a fresh directory on a clean path and follow your own README
      literally. Every gap you hit, fix in the README, not in your head.
- [ ] README: what it does, the three-things story, screenshot/GIF, quickstart,
      the two-role safety model, architecture summary, limitations.
- [ ] Confirm `.env` is gitignored and `.env.example` is committed. Grep history
      for anything that looks like a key.
- [ ] Clear every open Qodo finding.
- [ ] Rehearse the demo twice against a stopwatch. Fix whatever is slow or
      unexplainable.
- [ ] Write the blog post draft (open prize, roughly two hours: the job you gave
      the agent, how you wired it, what the harness handled, what broke).

**PRs:** `docs: readme`, `fix: *`, `chore: repo hygiene`

**Done when:** a stranger's clone runs, and you can narrate the demo without notes.

---

## Sunday 30 — record and submit

Target: **submitted by 18:00 IST**. Six hours of buffer, all of which you will use.

**The three-minute demo, shot list:**

| Time | Beat |
|---|---|
| 0:00–0:20 | The problem: schema changes on a live database are the scariest routine task in engineering. |
| 0:20–0:40 | Type the request. Agent introspects the real schema over MCP — show the tool call. |
| 0:40–1:10 | Plan appears with rollback. Subagent objection lands and is addressed. |
| 1:10–1:50 | **Sandbox.** Show where the code ran. Show the report: rows, timing, locks, reversible. |
| 1:50–2:20 | **The pause.** Let it sit on screen. Read the banner aloud. Then approve. |
| 2:20–2:40 | Applied. Assertions pass. Audit trail. |
| 2:40–3:00 | Destructive request refused, with the reason. Repo URL on screen. |

- [ ] Record. Multiple takes; the fifth is always the good one.
- [ ] Watch it back with the volume off, checking for keys, personal data, and
      stray browser tabs.
- [ ] Publish the blog post. Post the clip, tagging WeMakeDevs, TrueFoundry and Qodo.
- [ ] Submit: public repo URL, video, write-up covering what the agent does and how
      it uses the harness.
- [ ] **Submit at 18:00 even if something is imperfect.** An imperfect submission
      scores; a missed deadline does not.

---

## Triage, if the week goes badly

Cut in this order. Never cut upward past the line.

1. Multiple migration types → support `ADD COLUMN` + backfill only.
2. The reviewer subagent.
3. The audit log.
4. UI polish → the bundled chat UI plus a plain approval card.
5. Live schema snapshot → hand-written fixture in the sandbox.

— never cut below this line —

6. The sandbox verification step.
7. The human approval gate.
8. The read-only MCP connection to a real database.
9. The three-minute video.

Items 6–9 *are* the submission. Everything above them is decoration.
