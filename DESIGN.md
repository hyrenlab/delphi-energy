<!-- These are design notes, not evidence of completed or measured capabilities. -->

# Delphi Energy — Design Notes

This document explains *why* each stage of the pipeline exists, what was
intentionally dropped from the original design, and how the adapter pattern
keeps the core platform-agnostic.

> Target reader: someone who wants to fork, modify, or integrate Delphi
> with their own agent framework. If you just want to use it, read the
> README.

---

## 1. Core philosophy

> **Do not serve the user's immediate preference. Serve the user's long-term judgment.**

Most AI assistants behave like agreeable summarizers. They validate the
user's framing, organize available information, and produce plausible
answers. This is useful for writing and research, but **dangerous for
high-stakes judgment**.

Delphi Energy challenges the user the way a trusted strategic advisor would.
It must respect intuition but not become a tool for decorating bias.

### Seven design intentions (prompt guidance, not guaranteed properties)

1. **Reframe before answering** — many bad judgments come from bad framings
2. **Generate structured opposition** — at least one role attacks the user's
   stated/implied preference
3. **Grade evidence** — A/B/C/D, hard data ≠ media buzz ≠ founder charisma
4. **Apply energy constraints** — drill only when it could change the verdict
5. **Convert judgment into action** — every run ends with a 24-hour First Domino
6. **Make uncertainty auditable** — confidence + what-would-change-my-mind +
   biggest-risk are mandatory fields
7. **Don't create noise while claiming to reduce it** — short brief first,
   deep audit second

---

## 2. Pipeline architecture

### 2.1 Event-driven, dynamic shape

The pipeline doesn't follow a fixed N-act script. Each stage emits a
`TranscriptEvent`. Each event self-renders to one section of the final
markdown. Simple questions produce 4-section transcripts. Complex
questions produce 12+ sections. The shape follows the actual debate.

This was an intentional architectural choice over "rigid template" because:
- Force-fitting content into template sections is itself a Delphi anti-pattern
- Different questions genuinely need different shapes
- The audit trail reads more naturally when sections appear *because*
  something happened, not because the template demanded one

### 2.2 Three paths

The pipeline auto-selects path based on `intake.cost_of_being_wrong` and
`intake.reversibility`:

**LITE path** (auto when `reversibility=high` AND `cost_of_being_wrong in
{low, medium}`):
- Skips Long-Termist, Game Theorist, Cross-Exam, Red Team
- 5 baseline successful LLM calls, excluding repairs/retries
- For "should I switch text editors" type questions

**STANDARD path** (default):
- All 7 opening roles + Cross-Exam + Red Team + Counterfactual + Evidence
  Auditor + Judge
- 13 baseline successful LLM calls, excluding optional review/repairs/retries
- For most strategic decisions

**HIGH-STAKES** (auto when `cost_of_being_wrong in {high, catastrophic}`):
- + Multi-Model Skeptic Dissent (run Skeptic on a 2nd LLM, surface
  disagreement to Judge)
- 15 baseline successful LLM calls, excluding optional review/repairs/retries
- For irreversible career / financial / identity decisions

### 2.3 Stage-by-stage rationale

Each stage mapped to the original design spec sections.

#### Intake (§ 7.1, 7.2, 7.3 + KAU 4-tier from § 7.11)

Reframes the user's raw question, classifies type/reversibility/cost-of-being-
wrong, extracts key variables, and (v0.9 critical addition) extracts a
`fact_boundary` 4-tier:
- **Known**: explicitly stated by user
- **Assumed**: reasonable but unverified
- **Unknown**: missing but important
- **Speculative**: weakly-supported inference

The KAU tiers are passed to Judge as a fact whitelist — this is the
anti-confabulation safeguard. Earlier versions invented specifics
(numbers, version names, competitor moves); v0.9 forced Judge to either
trace each fact to user/memory/role-output or use conditional language
("if X then Y").

#### Adaptive Bias check (§ none — added beyond spec)

If the user has accumulated historical preference data (e.g. via Honcho
or a custom adaptive layer), check whether their current question's topic
has a known bias. If yes, inject a warning section the Judge and Red Team
must address.

Default `NoOpBiasAdapter` returns None — this is a hook for custom
preference-tracking integrations.

#### Past Ledger lookup (§ 10.3 simplified)

File-glob the ledger directory, score each past entry by keyword overlap
with the current intake (with recency weighting), inject the top 3
briefs into the Judge prompt. Forces Judge to address consistency or
explicit divergence from past judgments.

Implementation is intentionally simple — no embeddings, no vector DB.
Keyword overlap is robust + transparent + zero-cost. Vector search is a
future upgrade once a vault crosses ~1000 entries.

#### Memory query (§ 7.7 "User Alignment Agent" — replaced)

The original design had a "User Alignment Agent" role that evaluated
alignment with the user's long-term direction. We replaced this with an
external memory adapter (e.g. Honcho dialectic) because:
- A role that knows the user's long-term context needs *real* user data;
  generating it from prompt-only tends toward generic platitudes
- Honcho-style external memory layers exist precisely to provide this
- Defaulting to a no-op makes Delphi work standalone, while the adapter
  hook lets richer integrations plug in

#### 7 opening roles (§ 7.7)

The original spec listed 11 roles. We landed on 7:

| Role | What it does | Source spec |
|---|---|---|
| Advocate | Strongest case for "do/believe/proceed" | § 7.7 Advocate |
| Skeptic | Attacks the hypothesis | § 7.7 Skeptic |
| Realist | Execution constraints | § 7.7 Realist |
| Long-Termist | 5-10 year compounding | § 7.7 Long-Termist |
| Game Theorist | Players + reaction functions + 2nd-order | § 7.16 + § 7.17 |
| Black Swan Scout | Low-prob/high-impact disruptions | § 7.7 Black Swan Scout |
| Analogist | Historical/cross-industry analogies + break points | § 7.7 Analogist |

Dropped from spec:
- **Evidence Auditor** as a role → made into a separate post-opening stage
  (it operates on what others said, so logically belongs after them)
- **User Alignment Agent** → replaced by Memory adapter
- **Sunk Cost Check + Identity Contamination Check** (§ 7.8) → folded into
  Red Team prompt
- **Dynamic Authority Weighting** (§ 7.6) → dropped. Judge naturally weights
  via prompt context; explicit weighting added complexity without proven
  benefit

#### Cross-Exam Round 1 (§ none — added)

After openings, Skeptic gets one explicit follow-up at Advocate. Forces
both to engage instead of monologuing past each other.

#### Red Team (§ 7.7 + § 7.8.1 + § 7.8.2)

Attacks *the user*, not the question. Sunk cost, identity defense,
stated-vs-revealed preference mismatch, motivated reasoning. Routed to a
"less agreeable" model (configurable via `ROLE_PROVIDER`).

#### Multi-Model Skeptic Dissent (§ none — added)

On high-stakes questions, run Skeptic on a second LLM (different provider
or model). Send both outputs to a small dissent-judge LLM that grades
agreement and surfaces structural divergence. If divergence is real,
flag it as its own transcript section the Judge must address.

This catches the "single LLM converges to one cognitive style" failure
mode. Adds two LLM calls; only fires when `cost_of_being_wrong` is high.

#### Counterfactual Baseline (§ 7.15)

Generates 2-3 real alternative paths to the proposed action. Forces the
verdict to be a *comparison*, not an isolated yes/no. Asks two specific
questions:
1. Is the current option truly better, or just more familiar?
2. Is there a smaller, more reversible test version?

#### Evidence Auditor (§ 7.9 + § 7.10)

Reads all role outputs, picks the 5-8 most consequential factual claims,
grades each A/B/C/D:
- **A**: hard evidence (audited financials, official data, signed contracts)
- **B**: strong signals (executive targets, hiring patterns, channel moves)
- **C**: weak signals (media reports, social buzz, rumors)
- **D**: narrative (founder charisma, brand vibe, ambition, analogy)

Then summarizes overall debate evidence quality so Judge can compare against
the threshold required by the question's risk level.

#### Judge (§ 7.20 + § 7.18 + § 7.4 + § 7.14 + § 7.10 + § 7.19 + § 7.21 + § 7.22 + § 7.24)

The single biggest stage. Synthesizes everything. Mandatory output fields:
- **Pre-Mortem** — assume failure, work backward
- **Time Horizon Split** — now / 3-12mo / 1-3yr / 3-10yr separate takes
- **Contradiction Map** — 2-4 core unresolved tensions
- **Counterfactual Check** — does verdict still hold vs strongest alternative?
- **Evidence Threshold** — current grade vs required grade for this risk level
- **Noise Warning** — what was actively filtered out
- **Judgment** — direct, no hedge
- **Confidence** — `low` | `medium` | `high`, with **frequency framing** ("in
  10 similar cases this judgment is right ~7 times"). Numeric percentages
  banned because they fake precision.
- **One-Line Bet** — verdict compressed into a future-checkable bet
- **Biggest Risk** — single sentence
- **24h First Domino** — `{action, cost, validates}` — must be doable in 24h,
  must validate one key assumption
- **What would change my mind** — 2-3 observable signals
- **Validation Signals** — 2-3 future indicators for the 6-week review

Plus a `narrative` field — 300-500 word readable verdict that integrates
all of the above.

JSON object requested with a schema in the system prompt; full schema validation is not implemented. Auto-repair retry on
parse fail (one extra LLM call asking the model to fix its own JSON).
Timeout bumped to 240s + retry-once because the Judge call is the most
expensive to lose.

#### Round-2 Cross-Exam (§ 7.18 design rule)

The original spec said "Pre-Mortem should happen before final convergence,
not after, because it must be allowed to change the conclusion."

We do this differently: Pre-Mortem is *inside* Judge, then we run a
**Round-2 Cross-Exam** that re-attacks Advocate using whatever the
Pre-Mortem revealed. This is supplementary post-verdict review: it does not
change the current brief or action. A subsequent run is not guaranteed to
retrieve or incorporate these questions; the user must carry them forward.

A cheap heuristic skips Round 2 if the Pre-Mortem theme already appears in
Round 1.

#### Action stage (§ 7.21 + § 7.22 + § 7.24)

Promotes the One-Line Bet, First Domino, and Validation Signals from the
Judge brief into a standalone transcript section. Visual separation in the
ledger — easy to find weeks later.

#### Optional Drilldowns (§ 7.12 simplified)

`--drill skeptic:"if runway were 1 month?"` runs an additional LLM call to
the named role *after* the main pipeline finishes (so the role can react to
the Judge's verdict, not just the openings). Repeatable.

Original spec had auto-recursive drilldown; we made it user-driven for two
reasons:
1. Auto-drilldown is an over-engineering trap (when to stop?)
2. The user knows which role's argument felt weakest — they should drive

---

## 3. Anti-confabulation (the v0.9 fix)

The v0.1 release exhibited a critical failure mode: when given fact-laden
questions with version numbers, Delphi confabulated plausible-sounding
follow-on details. Specifically: a question mentioning "v0.20" caused Judge
to invent a "v0.20 changeset" with imaginary impacts.

This is the most dangerous LLM judgment failure mode: not "wrong" but
"confidently fabricated specifics that look right." Users who don't know
the facts will act on fiction.

The fix has three parts:

1. **Intake extracts `fact_boundary.known`** — explicit whitelist of facts
   the user actually stated.
2. **Judge prompt enforces traceability**: every specific number / version /
   competitor / date must trace to (a) user's question, (b) memory output,
   (c) past ledger, or (d) a role's output. Anything else uses conditional
   language ("if X is true, then…") or marks `[推测:source]`.
3. **KAU 4-tier passed to Judge** — Judge sees Known, Assumed, Unknown, and
   Speculative tiers separately and must respect their boundaries.

Verification: in v1.0+ integration tests on questions mentioning "Hermes
0.12 version", Judge correctly cited only that fact and used conditional
language for everything downstream.

---

## 4. Adapter pattern

Four pluggable slots:

| Slot | Default | Purpose |
|---|---|---|
| `llm` | `OpenAILLMAdapter` | All LLM calls (required) |
| `memory` | `NoOpMemoryAdapter` | Long-term user context |
| `bias` | `NoOpBiasAdapter` | Historical bias warnings |
| `notifier` | `NoOpNotifierAdapter` (Discord if env set) | Push the brief externally |

Defaults are chosen to make the OS version "work standalone with just an
OpenAI key". Override at module load:

```python
from delphi import ADAPTERS, run_delphi
ADAPTERS["llm"] = MyAnthropicAdapter()
ADAPTERS["memory"] = MyHonchoAdapter()
transcript = run_delphi("...")
```

See `examples/` for full templates including Hermes integration, OpenClaw
integration, Honcho memory, and multi-provider routing.

---

## 5. What was intentionally dropped from the original spec

The original `delphi_energy_scheme_design.md` listed 24 stages. We landed
on a tighter pipeline. Dropped:

| Stage / feature | Why dropped |
|---|---|
| § 4.2 Auto-Trigger Detection | LLM-judged auto-trigger reliably over-triggers. User invokes explicitly. |
| § 7.4 Time Horizon Split as separate stage | Folded into Judge prompt as a mandatory output field |
| § 7.6 Dynamic Authority Weighting | Knob without proven benefit; Judge weights naturally via prompt context |
| § 10.3 reviewJudgment(id, newEvidence) | Use case: re-evaluate ledger 6 weeks later. Cleaner: re-run Delphi with new question that includes the past ledger reference |
| § 11/12 TypeScript interface | Python is the primary impl; TS interface adds maintenance burden without users |

Each drop is a deliberate trade-off, not an omission. If a use case calls
for one of these, the adapter pattern lets you re-add it without forking
core.

---

## 6. Cost & latency model

No reproducible live performance benchmark is included. Historical dollar and
latency claims have been removed because their source traces are unavailable.
Stages, including all seven openings, run sequentially.

`DELPHI_PRICES_JSON` configures exact provider/model token rates in USD per million
input/output tokens. Unconfigured models use explicitly labeled hypothetical
fallback assumptions, not verified vendor prices. The ledger records these rates.
See README for configuration and billing limitations.

`DELPHI_BUDGET_USD` is a positive finite **estimated soft budget**. The pipeline
warns at 80% and skips checked optional stages at 100%. Intake, Judge, and Judge
repair/retry can still run and exceed the limit. This is not a hard spend cap.

---

## 7. Provenance

Delphi Energy was originally designed and developed inside the Hermes
project (a personal AI architecture experiment). The design document was
written before any code, the implementation was iterated through 9
versions (v0.1 → v1.1) over a single overnight session, with each version
addressing concrete failure modes discovered by running Delphi on real
questions.

The OS release (this repo) is a clean extraction:
- All Hermes-specific imports replaced with adapter interfaces
- All personal data scrubbed (no user names, no Obsidian vault paths, no
  `agent.auxiliary_client` calls)
- Configuration moved to env vars
- Hermes integration moved to `examples/hermes_adapter.py` as a reference

The original Hermes-bound version remains in private use. This OS version
should be functionally equivalent for any user — the adapter pattern means
you can rebuild any Hermes-style integration on top of this core.
