# Delphi Energy

> A sequential multi-role judgment prototype. It organizes competing arguments,
> highlights assumptions, and produces a model-generated Decision Brief and audit ledger.

**Status**: experimental v1.1 implementation; not feature-complete or independently
fact-verified. MIT licensed. Python **3.10–3.13** is the CI target range.

## What it does

Most AI assistants behave like agreeable summarizers. They validate your
framing, organize information, and produce plausible answers. Useful for
writing. **Dangerous for high-stakes judgment.**

Delphi Energy challenges you the way a trusted strategic advisor would. It
runs your question through:

- **7 reasoning roles** called sequentially — Advocate, Skeptic, Realist,
  Long-Termist, Game Theorist, Black Swan Scout, Analogist
- **Cross-Exam** — Skeptic interrogates Advocate
- **Red Team** — attacks *you*, not the question (sunk cost, identity
  defense, motivated reasoning)
- **Multi-Model Skeptic Dissent** — on high-stakes questions, runs Skeptic on
  two different LLMs and surfaces disagreement
- **Counterfactual Baseline** — generates 2-3 real alternatives so the verdict
  isn't a strawman comparison
- **Evidence Auditor** — asks a model to grade claims A/B/C/D from the debate; it does not retrieve or independently verify external sources
- **Judge** — synthesizes everything with Pre-Mortem, Time Horizon split (now
  / 3-12mo / 1-3yr / 3-10yr), Contradiction Map, Evidence Threshold check,
  Noise Warning, Confidence (frequency framing — "in 10 similar cases this
  judgment is right ~7 times"), One-Line Bet, Biggest Risk, and a 24h First
  Domino
- **Supplementary Round-2 Cross-Exam** — may run after Judge when a text-overlap heuristic flags a new Pre-Mortem topic. It is recorded in the ledger and does **not** revise the brief or action
- **Optional drilldowns** — `--drill skeptic:"if runway were 1 month?"` to
  ask any role a follow-up

Output is two layers: a Decision Brief (Discord-friendly), and a full audit
trail markdown ledger you can re-read 6 weeks later.

## Quick start

Start with the offline example (standard library only):

```bash
git clone https://github.com/hyrenlab/delphi-energy.git
cd delphi-energy
python3 examples/offline.py
python3 -m unittest discover -s tests -v
```

The example exercises the actual five-call lite pipeline using **scripted responses**.
It reads no personal history, sends no network requests, and saves no ledger.
[Sample output](examples/offline-transcript.md) is synthetic, not a model benchmark.

For live model calls, create an environment, install dependencies, and set your API
key locally. Calls can incur charges; configure the token price estimates below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Set OPENAI_API_KEY locally before running.
python delphi.py "should I try a different meeting format?" --no-notify
```

The live CLI prints a Decision Brief and writes an audit trail to `./ledger/`.
`--no-notify` suppresses external notifications even if a webhook is configured.
`--no-vault` skips ledger writes; it does not disable history reads or notifications.

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
| `DELPHI_BUDGET_USD`            | `1.00`           | Positive finite per-run **estimated** soft budget |
| `DELPHI_PRICES_JSON`           | `{}`             | Exact `provider/model` keys mapped to `[input, output]` USD per million tokens |
| `DELPHI_FALLBACK_INPUT_USD_PER_M` | `1.00`         | Hypothetical input rate for unconfigured models; not a vendor quote |
| `DELPHI_FALLBACK_OUTPUT_USD_PER_M` | `5.00`        | Hypothetical output rate for unconfigured models; not a vendor quote |
| `DELPHI_DISCORD_WEBHOOK_URL`   | (off)            | Enables DiscordWebhookNotifierAdapter |

## CLI

```bash
python delphi.py "<question>"             # standard pipeline
python delphi.py "<question>" --dry-run   # preview structure, no LLM calls
python delphi.py "<question>" --no-vault  # skip ledger write (notifications remain enabled)
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

7 design intentions reflected in the prompts and stages; these are not guaranteed output properties:

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

## Cost estimates and limits

No reproducible live cost or latency benchmark is included. Earlier numerical
performance claims were removed because their underlying traces are unavailable.
The baseline successful paths use 5 calls (lite), 13 (standard), or 15
(high-stakes), before optional round-2, drills, repairs, retries, or budget skips.
These are code-path counts, not performance measurements.

Prices are **user-configured estimates**, not fetched vendor prices. For example,
`DELPHI_PRICES_JSON='{"fixture/model": [2, 8]}'` demonstrates the format using
**fictional rates**. Configure every primary, alternate, and failover model using
its actual provider/model key and your applicable rates. Unconfigured models use
the clearly labeled hypothetical fallback rates. The ledger and CLI show the
pricing assumptions. Zero rates are allowed for local/free models; negative,
non-finite, and malformed prices are rejected. The budget must be finite and
strictly positive.

At 80% of estimated budget, the pipeline warns and continues. At 100%, checked
optional stages are skipped. Intake, Judge, and Judge repair/retry can still run;
one call or a multi-call stage can cross the limit. This is **not a hard spending
cap**. Failed requests, cache discounts, reasoning tokens, and provider billing
rules may make actual charges differ. Use provider-side controls for actual limits.

## Verification and limitations

Run the offline checks above; CI covers Python 3.10–3.13 without provider credentials.
See [verification and limitations](docs/verification.md) for tested behavior and gaps.

Intake's fact whitelist and Judge's source instructions are prompt safeguards.
User assertions, memory, and role outputs can all be wrong. There is no independent
source retrieval or factual verification in the default pipeline. A/B/C/D grades
and confidence frequencies are model opinions, not calibrated measurements.
JSON parsing and a single repair attempt do not validate the complete output schema;
a failed Judge produces raw text without a Decision Brief or action.

Round-2 is supplementary review after the verdict. Users must inspect it in the
ledger; neither the current brief nor the next run is guaranteed to incorporate it.

## License

MIT — see `LICENSE`.

## Acknowledgments

Delphi Energy was originally designed and developed in the Hermes project as
a private skill. This repo is the open-sourced extraction; the original
design doc was the spec, and the pipeline was iterated through 9 versions
(v0.1 → v1.1) before being released. The original design rationale (why each
stage exists, which were dropped and why) lives in `DESIGN.md`.
