# ARCHITECTURE.md — migration-gate

> **Read this file before writing any code.** It defines what we are building, what
> we are explicitly *not* building, and the invariants that must never be violated.
> If a task you are asked to do conflicts with the Safety Invariants section, stop
> and say so rather than working around it.

---

## 1. What this is

**migration-gate** is an agent that takes a schema change request in plain English,
plans the migration, proves it against a disposable copy of the database inside an
isolated sandbox, and then **stops and asks a human** before touching the real
database.

Example session:

```
User:   Add a last_active_at column to users, backfill it from sessions.
Agent:  [MCP] introspected schema: users (8 cols, 12,431 rows), sessions (5 cols, 98,220 rows)
Agent:  [plan] 001_add_last_active_at.up.sql + .down.sql generated
Agent:  [sandbox] restored schema snapshot + synthetic rows into scratch Postgres
Agent:  [sandbox] applied migration: 12,431 rows updated, 340ms, ACCESS EXCLUSIVE held 12ms
Agent:  [sandbox] verified rollback restores original schema hash
Agent:  ■ APPLYING TO PRIMARY IS IRREVERSIBLE — holding for approval
User:   [Approve]
Agent:  [MCP] applied to primary. Verification passed. Session logged.
```

This project is built for the WeMakeDevs / TrueFoundry **Agent Harness Hackathon**
(Aug 24–30, 2026). It is a solo build. Read `PLAN.md` for the schedule.

---

## 2. Why this shape (do not lose the thread)

The hackathon's disqualifying criterion is: *"If it would work just as well as a chat
box, change the project."* Three things must be visibly true in the demo:

| Requirement | How this project satisfies it |
|---|---|
| Reaches a **real tool** | Postgres over MCP — live introspection of a real database |
| Runs code in a **sandbox** | The migration is executed against a scratch DB in a Daytona sandbox before it is trusted |
| **Pauses** before the irreversible step | `apply_to_primary` is gated on a human checkpoint |
| Delegates to **subagents** | A separate reviewer subagent critiques the migration plan |
| Survives **reconnects** | A pending approval must still be pending after a page refresh |

Every architectural decision below exists to make one of those five lines true.
If a change makes one of them less visible in a three-minute demo, it is the wrong
change.

---

## 3. Non-goals

Do not build these. They cost days and win nothing:

- A general-purpose migration framework or a competitor to Flyway/Alembic/Prisma.
- Multi-database support. **Postgres only.**
- Authentication, multi-tenancy, user accounts, or RBAC.
- A hosted deployment. Local mode (SQLite-backed TrueForge) is the target.
- Rewriting anything the harness already does: the agent loop, tool dispatch,
  context compaction, session persistence, the approval mechanism itself.
  **We configure the harness. We do not reimplement it.**
- Handling every DDL statement in Postgres. See §7 for the supported subset.

---

## 4. System overview

```
┌──────────────────────────────────────────────────────────────┐
│  Browser: TrueForge chat UI (bundled) + our approval panel   │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼──────────────────────────────────┐
│  TrueForge harness  (npx @truefoundry/trueforge)             │
│  ─ agent loop, context mgmt, session persistence (SQLite)    │
│  ─ tool approval checkpoints                                 │
│  ─ subagent delegation                                       │
│  ─ sandbox provisioning (Daytona)                            │
└──┬─────────────────┬───────────────────┬─────────────────────┘
   │ MCP             │ MCP               │ sandbox tool
┌──▼──────────────┐ ┌▼────────────────┐ ┌▼────────────────────┐
│ postgres-mcp    │ │ migration-mcp   │ │ Daytona sandbox     │
│ (READ-ONLY)     │ │ (ours)          │ │  ─ scratch Postgres │
│                 │ │                 │ │  ─ psql, python     │
│ introspect      │ │ plan / verify / │ │  ─ NO network to    │
│ EXPLAIN         │ │ apply           │ │    primary DB       │
└──┬──────────────┘ └┬────────────────┘ └─────────────────────┘
   │ read-only role  │ writer role (gated)
┌──▼─────────────────▼─────────────────────────────────────────┐
│  Primary Postgres (docker compose, seeded demo data)         │
└──────────────────────────────────────────────────────────────┘
```

### Two database roles, always

- `mg_reader` — `CONNECT`, `SELECT`, `USAGE ON SCHEMA`. Nothing else. The
  read-only MCP server uses this and only this.
- `mg_writer` — DDL rights. Used by exactly one code path: `apply_to_primary`,
  which is unreachable without an approved checkpoint.

This split is not decoration. It means that even a fully compromised planning
loop cannot mutate the primary. Say this out loud in the demo.

---

## 5. Components

### 5.1 TrueForge harness — configured, not written

Local mode: `npx @truefoundry/trueforge`. SQLite-backed, single process.
Configuration lives in `config/` and is committed (secrets are not).

> **VERIFY BEFORE IMPLEMENTING.** The exact configuration format, SDK function
> names, and approval-hook API must be read from the live docs at
> <https://trueforge.dev> and the repo at
> <https://github.com/truefoundry/trueforge> — specifically the Quickstart,
> the MCP servers page, and the human-checkpoints / tool-approval section.
> Do not guess signatures from this document. Where this file names a function,
> treat the name as *intent*, not as API.

What we rely on the harness for:

| Capability | Our use |
|---|---|
| MCP connectors | Both MCP servers below |
| Sandbox as a tool | Scratch Postgres + migration execution |
| Human checkpoints (tool approval) | The gate on `apply_to_primary` |
| Subagents | The reviewer subagent (§5.4) |
| Session persistence | Pending approval survives refresh |
| Skills | The migration-authoring skill (§5.5) |
| Model providers | Bring-your-own key; configurable from the UI |

### 5.2 `postgres-mcp` — read-only introspection

Prefer an existing off-the-shelf Postgres MCP server if one can be pointed at a
read-only role cleanly. Only write our own if the available one insists on write
capability. Tools needed:

- `list_tables()`, `describe_table(name)` — columns, types, nullability, defaults
- `list_indexes(table)`, `list_constraints(table)`, `list_foreign_keys(table)`
- `row_count_estimate(table)` — from `pg_class.reltuples`, never `COUNT(*)` on
  large tables
- `explain(sql)` — `EXPLAIN` only, never `EXPLAIN ANALYZE` against primary
- `schema_snapshot()` — `pg_dump --schema-only` equivalent, returned as text

### 5.3 `migration-mcp` — ours, three tools

This is the project's own MCP server (`packages/migration-mcp`). Keep it small.

**`plan_migration(request: string) -> MigrationPlan`**
Produces, from the request plus a fresh schema snapshot:
```ts
type MigrationPlan = {
  id: string;                 // e.g. "001_add_last_active_at"
  intent: string;             // one-line restatement of the request
  up: string;                 // SQL
  down: string;               // SQL — REQUIRED, see invariant I3
  risk: {
    lockType: "ACCESS EXCLUSIVE" | "SHARE" | "ROW EXCLUSIVE" | "NONE";
    estimatedRowsTouched: number;
    rewritesTable: boolean;
    destructive: boolean;     // drops or type-narrows anything
  };
  notes: string[];            // e.g. "backfill is batched at 5k rows"
};
```

**`verify_in_sandbox(plan: MigrationPlan) -> VerificationReport`**
The heart of the project. Steps, in order:
1. Provision a sandbox (harness-managed).
2. Start Postgres inside it. Load the **schema snapshot** from primary.
3. Seed synthetic rows scaled to the real row-count estimate — **never copy real
   rows out of primary** (invariant I2).
4. Record `schemaHashBefore`.
5. Apply `plan.up`. Capture: wall time, rows affected, locks taken, errors.
6. Run the assertion set (§7.3).
7. Apply `plan.down`. Record `schemaHashAfter`.
8. Report `reversible = (schemaHashBefore === schemaHashAfter)`.
9. Tear the sandbox down.

```ts
type VerificationReport = {
  planId: string;
  applied: boolean;
  reversible: boolean;
  durationMs: number;
  rowsAffected: number;
  locksObserved: string[];
  assertions: { name: string; passed: boolean; detail: string }[];
  stdout: string;             // truncated; full log offloaded
  verdict: "safe" | "risky" | "failed";
};
```

**`apply_to_primary(planId: string) -> ApplyResult`**
The gated tool. Must be registered with the harness as requiring human approval.
Behaviour:
- Refuse if no `VerificationReport` exists for `planId`.
- Refuse if `verdict === "failed"`.
- Refuse if the schema hash of primary has changed since the plan was made
  (someone else migrated underneath us).
- Open a transaction. Apply `up`. Run the assertion set against primary.
  **Roll back and report if any assertion fails.** Commit only on a clean pass.
- Write an audit record (§8).

### 5.4 Reviewer subagent

Before verification, delegate the plan to a subagent with a single job: read
`plan.up` and `plan.down` against the schema snapshot and return objections.
It has **no tools that can write**. Its output is a list of concerns, each of
which the main agent either fixes or explicitly overrules in the plan notes.

This is cheap to build, uses a scored harness capability, and makes the demo
look considered rather than lucky. Prompts live in `prompts/reviewer.md`.

### 5.5 Migration skill

A TrueForge Skill (`skills/postgres-migrations/`) holding the house rules the
agent should apply when authoring SQL:

- `ADD COLUMN` with a volatile default rewrites the table — split into
  add-nullable, backfill in batches, then set default.
- Create indexes `CONCURRENTLY` outside a transaction.
- Never `DROP COLUMN` in the same migration that stops using it.
- Backfills over ~50k rows must be batched.
- Every `up` needs a `down`.

### 5.6 Approval UI

The bundled chat UI carries the conversation. We add one panel that renders a
pending approval as something a stranger can read in five seconds:

- Left: the diff (`up` SQL, syntax-highlighted).
- Right: the verification report — rows touched, duration, lock type, reversible
  yes/no, assertion pass/fail list.
- A red banner naming the irreversible act in plain words: *"This will run
  ACCESS EXCLUSIVE DDL on the primary database. 12,431 rows will be rewritten."*
- Two buttons: **Approve** / **Reject with reason**. A rejection reason goes back
  into the session as a user turn so the agent can revise.

> **VERIFY:** whether to build this as an embedded surface via
> `@truefoundry/trueforge-ui`, as Generative UI inside the chat, or as a small
> separate page reading the harness HTTP API. Read the docs, then choose the one
> that requires the least custom code. Record the decision in an ADR (§10).

---

## 6. Safety invariants

These are the project's spine. Every PR must preserve all of them.

- **I1 — The primary is read-only except behind an approval.** `mg_reader` for
  everything; `mg_writer` reachable only from `apply_to_primary`.
- **I2 — Real data never leaves the primary.** The sandbox gets *schema* plus
  synthetic rows. Never a data dump. This keeps customer data out of the sandbox,
  out of logs, and out of the demo video.
- **I3 — No plan without a rollback.** A plan whose `down` is empty or untested
  cannot reach the approval step.
- **I4 — Verification precedes approval.** The approval card cannot render
  without a `VerificationReport` attached.
- **I5 — Approval is per-plan, never blanket.** No "approve all", no remembered
  consent, no auto-approve flag. Not even behind an env var — a reviewer who
  finds `AUTO_APPROVE=true` in the repo has found the project's disqualification.
- **I6 — Secrets stay out of the repo and the video.** `.env` is gitignored;
  `.env.example` is committed with placeholder values. Before recording, check
  the terminal scrollback and browser tabs.
- **I7 — Apply is transactional with assertion rollback.** A failed
  post-condition rolls the transaction back rather than leaving a half-migrated
  primary.

---

## 7. Scope of supported migrations

### 7.1 Supported (build these)
- `ADD COLUMN` (nullable, with and without default)
- Backfill from another table via `UPDATE ... FROM`
- `CREATE INDEX CONCURRENTLY`
- `ADD CONSTRAINT ... NOT VALID` followed by `VALIDATE CONSTRAINT`
- `RENAME COLUMN`
- Widening type changes (e.g. `int4 -> int8`, `varchar(n) -> text`)

### 7.2 Detected and refused (but explained)
- `DROP COLUMN` / `DROP TABLE` — the agent should plan it, flag it as
  destructive, and recommend the two-phase alternative.
- Narrowing type changes that could truncate data.
- Anything the planner cannot produce a `down` for.

Refusing well is a feature here. A judge who sees the agent decline to drop a
column and explain why has seen the safety story land.

### 7.3 Assertion set
Run after `up` in the sandbox, and again inside the transaction on primary:
- Row counts on affected tables match expectation.
- No column that previously had zero nulls now has nulls (unless intended).
- Every declared foreign key still validates.
- A named smoke query supplied with the plan still returns rows.

---

## 8. Audit log

Every session writes an append-only JSONL record to `audit/`:

```jsonc
{ "ts": "...", "planId": "001_...", "event": "plan_created",  "actor": "agent" }
{ "ts": "...", "planId": "001_...", "event": "verified",      "verdict": "safe" }
{ "ts": "...", "planId": "001_...", "event": "approval_requested" }
{ "ts": "...", "planId": "001_...", "event": "approved",      "actor": "human" }
{ "ts": "...", "planId": "001_...", "event": "applied",       "durationMs": 340 }
```

Cheap to build, and it gives the demo a closing shot: the whole decision trail
in one file.

---

## 9. Repository layout

```
migration-gate/
├── README.md                 # what it does, how to run it, in that order
├── ARCHITECTURE.md           # this file
├── PLAN.md                   # the week
├── docker-compose.yml        # demo Postgres, seeded — judges run THIS
├── .env.example
├── config/
│   └── trueforge/            # harness config: MCP servers, model, approvals
├── packages/
│   ├── migration-mcp/        # our MCP server (§5.3)
│   └── ui/                   # approval panel (§5.6)
├── skills/
│   └── postgres-migrations/  # §5.5
├── prompts/
│   ├── system.md
│   └── reviewer.md
├── db/
│   ├── init/                 # roles: mg_reader, mg_writer
│   └── seed/                 # demo dataset (§11)
├── docs/adr/                 # one file per real decision
└── audit/                    # gitignored, created at runtime
```

TypeScript throughout — the harness, its SDK, and its UI are TypeScript, so
matching it avoids a second toolchain. Node 20+. pnpm.

---

## 10. Decision records

Any time you choose between two plausible approaches (UI surface, off-the-shelf
vs own Postgres MCP, sandbox DB strategy), write `docs/adr/NNN-title.md`:
context, options, decision, consequence. Four sentences each is enough. Judges
read repos; this is the cheapest possible signal that the repo is real software.

---

## 11. Demo database

Judges cannot connect to your database. `docker compose up` must give them a
Postgres with:

- `users`, `sessions`, `orders`, `order_items` — enough relational structure that
  a backfill is interesting.
- ~50k rows in `sessions` so timings are non-trivial but seeding is fast.
- Both roles created by `db/init/`.
- Entirely synthetic data. No real anything.

The README must state that the default target is this local demo DB, and that
pointing it at a real database is a deliberate act requiring a different
connection string.

---

## 12. Definition of done

- [ ] `git clone && cp .env.example .env && docker compose up` works on a clean machine.
- [ ] A judge can type a schema request and watch: MCP introspection → plan →
      subagent review → sandbox verification → approval card → apply.
- [ ] Refreshing the browser mid-approval leaves the approval still pending.
- [ ] Rejecting with a reason produces a revised plan.
- [ ] A destructive request is refused with an explanation.
- [ ] README explains the two-role model in the first screen.
- [ ] `docs/adr/` has at least three entries.
- [ ] Qodo has reviewed every merged PR.
- [ ] Demo video ≈3 minutes, approval moment on screen for ≥20 seconds.
