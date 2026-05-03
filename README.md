# Delphi Energy

> A recursive adversarial judgment skill. Takes an important question, runs a
> structured multi-role debate, surfaces the user's blind spots, and converges
> into a fact-grounded verdict with a 24-hour first action.

**Status**: v1.1 — feature-complete against the design spec. MIT licensed.

## What it does

Most AI assistants behave like agreeable summarizers. They validate your
framing, organize information, and produce plausible answers. Useful for
writing. **Dangerous for high-stakes judgment.**

Delphi Energy challenges you the way a trusted strategic advisor would. It
runs your question through:

- **7 reasoning roles** in parallel — Advocate, Skeptic, Realist,
  Long-Termist, Game Theorist, Black Swan Scout, Analogist
- **Cross-Exam** — Skeptic interrogates Advocate
- **Red Team** — attacks *you*, not the question (sunk cost, identity
  defense, motivated reasoning)
- **Multi-Model Skeptic Dissent** — on high-stakes questions, runs Skeptic on
  two different LLMs and surfaces disagreement
- **Counterfactual Baseline** — generates 2-3 real alternatives so the verdict
  isn't a strawman comparison
- **Evidence Auditor** — grades each major claim A/B/C/D
- **Judge** — synthesizes everything with Pre-Mortem, Time Horizon split (now
  / 3-12mo / 1-3yr / 3-10yr), Contradiction Map, Evidence Threshold check,
  Noise Warning, Confidence (frequency framing — "in 10 similar cases this
  judgment is right ~7 times"), One-Line Bet, Biggest Risk, and a 24h First
  Domino
- **Round-2 Cross-Exam** — auto-fires if Pre-Mortem reveals an attack vector
  Round 1 missed
- **Optional drilldowns** — `--drill skeptic:"if runway were 1 month?"` to
  ask any role a follow-up

Output is two layers: a Decision Brief (Discord-friendly), and a full audit
trail markdown ledger you can re-read 6 weeks later.

## Quick start

```bash
git clone <your-fork> delphi-energy
cd delphi-energy
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
python delphi.py "should I take the job offer?"
```

You'll get the Decision Brief in your terminal, the full audit trail saved to
`./ledger/2026-MM-DD_<slug>.md`, and per-stage progress streamed to stderr.

## Configuration

All via environment variables (no config file needed):

| Variable                       | Default          | Purpose |
|---|---|---|
| `OPENAI_API_KEY`               | (required)       | LLM auth |
| `OPENAI_BASE_URL`              | OpenAI default   | Override base URL — works with DeepSeek, Together, Groq, OpenRouter, vLLM, etc. |
| `DELPHI_DEFAULT_MODEL`         | `gpt-4o-mini`    | Model for most roles |
| `DELPHI_JUDGE_MODEL`           | (same as default)| Model for Judge stage (use a stronger one if budget allows) |
| `DELPHI_INTAKE_MODEL`          | (same as default)| Model for intake (cheap utility) |
| `DELPHI_SKEPTIC_ALT_PROVIDER`  | (primary)        | Provider for the multi-model dissent Skeptic |
| `DELPHI_SKEPTIC_ALT_MODEL`     | `gpt-4o`         | Model for the multi-model dissent Skeptic |
| `DELPHI_LEDGER_DIR`            | `./ledger/`      | Where ledger md files go |
| `DELPHI_BUDGET_USD`            | `1.00`           | Per-run soft budget cap |
| `DELPHI_DISCORD_WEBHOOK_URL`   | (off)            | Enables DiscordWebhookNotifierAdapter |

## CLI

```bash
python delphi.py "<question>"             # standard pipeline
python delphi.py "<question>" --dry-run   # preview structure, no LLM calls
python delphi.py "<question>" --no-vault  # stdout only, don't write file
python delphi.py "<question>" --no-notify # skip Discord/etc, vault + stdout
python delphi.py "<question>" --no-stream # suppress per-stage progress
python delphi.py --describe               # print metadata JSON, exit
python delphi.py "<question>" --drill skeptic:"if runway were 1 month?"
```

## Adapter pattern (for plug-in integration)

The pipeline talks to 4 abstract adapters:

| Adapter slot | Default impl              | What it does |
|---|---|---|
| `llm`        | `OpenAILLMAdapter`        | LLM calls (required) |
| `memory`     | `NoOpMemoryAdapter`       | Long-term context (e.g. Honcho dialectic) |
| `bias`       | `NoOpBiasAdapter`         | Historical user bias warning (preference-layer) |
| `notifier`   | Discord if env, else no-op | Push the brief somewhere external |

Override any of them at module load:

```python
from delphi import ADAPTERS, run_delphi

# Drop-in your own LLM (e.g. Anthropic, local model, custom proxy)
ADAPTERS["llm"] = MyAnthropicAdapter()

# Wire long-term memory
ADAPTERS["memory"] = MyHonchoAdapter(...)

transcript = run_delphi("your question")
print(transcript.render())
```

See `examples/` for templates:
- `examples/basic.py` — minimal Python usage
- `examples/openclaw_adapter.py` — integrate with OpenClaw (or any agent
  framework — adapter pattern is the same)
- `examples/hermes_adapter.py` — integrate with Hermes (the framework Delphi
  was originally built for)
- `examples/honcho_adapter.py` — wire the Honcho memory adapter
- `examples/multi_provider.py` — route different roles to different providers

## Design philosophy

> **Do not serve the user's immediate preference. Serve the user's long-term
> judgment.**

7 principles, all enforced by the pipeline:

1. **Reframe before answering** — many bad judgments come from bad framings
2. **Generate structured opposition** — at least one role attacks the user's
   stated/implied preference
3. **Grade evidence** — A/B/C/D, hard data ≠ media buzz ≠ founder charisma
4. **Apply energy constraints** — drill only when it could change the verdict
5. **Convert judgment into action** — every run ends with a 24-hour First
   Domino
6. **Make uncertainty auditable** — confidence + what-would-change-my-mind +
   biggest-risk are mandatory fields
7. **Don't create noise while claiming to reduce it** — short brief first,
   deep audit second

Full design doc: see `DESIGN.md`.

## Cost & latency

Real measurements with `gpt-4o-mini` for most roles + `gpt-4o` for Judge:

| Path | LLM calls | Wall time | Cost |
|---|---|---|---|
| Lite (high-rev / low-cost question) | 6 | ~120s | ~$0.02 |
| Standard | 12 | ~240s | ~$0.05 |
| High-stakes (with multi-model dissent + round-2) | 16 | ~330s | ~$0.10 |

Wall time is dominated by the Judge call (60-100s). The cost cap is
`DELPHI_BUDGET_USD` (default $1.00) — Delphi prunes stages once it hits 80%
of cap.

## Anti-confabulation

Earlier versions confabulated specifics. v0.9 fixed this:
- Intake extracts a `user_asserted_facts` whitelist
- Judge prompt enforces: any specific number / version / competitor must
  trace back to user / memory / past ledger / role output, otherwise use
  conditional language ("if X then Y") or mark `[推测]`

## License

MIT — see `LICENSE`.

## Acknowledgments

Delphi Energy was originally designed and developed in the Hermes project as
a private skill. This repo is the open-sourced extraction; the original
design doc was the spec, and the pipeline was iterated through 9 versions
(v0.1 → v1.1) before being released. The original design rationale (why each
stage exists, which were dropped and why) lives in `DESIGN.md`.
