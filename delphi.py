#!/usr/bin/env python3
"""
delphi.py — Delphi Energy: Sequential adversarial judgment prototype.

Open-source release (v1.1). License: MIT (see LICENSE).

What it does
============
Takes an important question, decomposes it, runs 7 specialized reasoning
roles (Advocate / Skeptic / Realist / Long-Termist / Game Theorist /
Black Swan Scout / Analogist), runs Cross-Exam + Red Team + Counterfactual
+ Evidence Auditor, then has a Judge produce a model-generated verdict
with Pre-Mortem, Time Horizon split, Contradiction Map, Evidence Threshold
check, Noise Warning, and a 24-hour first action.

Pipeline (event-driven dynamic shape — runs only what each Q needs):

  STANDARD path (13 calls before optional dissent, review, retries or drills):
    Intake (KAU 4-tier fact boundary + time horizon hints)
    → Adaptive Bias check (no-op unless adapter wired)
    → Past Ledger lookup (file-glob + keyword match)
    → Memory query (no-op unless adapter wired, e.g. Honcho)
    → Opening: Advocate / Skeptic / Realist / Long-Termist / Game Theorist
              / Black Swan Scout / Analogist
    → Cross-Exam Round 1 (Skeptic vs Advocate)
    → Red Team (attacks the user's stated/implied preference)
    → Multi-Model Skeptic Dissent (high-stakes only)
    → Counterfactual Baseline (2-3 alternative paths)
    → Evidence Auditor (A/B/C/D grading)
    → Judge (synthesis + Pre-Mortem + Time Horizon + Contradiction Map
             + Counterfactual Check + Evidence Threshold + Noise Warning
             + Confidence + One-Line Bet + 24h First Domino + Validation
             Signals; not externally verified; JSON-repair retry)
    → Supplementary Round-2 Cross-Exam (does not revise the verdict)
    → Action (one-line bet + first domino + validation signals)
    → Optional Drilldowns (--drill ROLE:Q for post-verdict follow-up)

  LITE path (auto when reversibility=high + cost_of_being_wrong in {low, medium}):
    Intake → Memory → Advocate + Skeptic + Realist → Judge → Action

Quick start
===========
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python delphi.py "your question"

ENV variables
=============
    OPENAI_API_KEY              required for default LLM adapter
    OPENAI_BASE_URL             override base URL (DeepSeek/Together/etc.)
    DELPHI_DEFAULT_MODEL        default model (default: "gpt-4o-mini")
    DELPHI_JUDGE_MODEL          model for Judge stage (default: same)
    DELPHI_LEDGER_DIR           where ledger files go (default: ./ledger/)
    DELPHI_BUDGET_USD           estimated soft budget in USD (default: 1.00)
    DELPHI_DISCORD_WEBHOOK_URL  enables Discord notifier
    DELPHI_HONCHO_BASE_URL      enables Honcho memory adapter
    DELPHI_HONCHO_PEER_ID       Honcho peer name (default: "user")

Plug-in adapters
================
The pipeline uses 4 abstract adapter slots: LLM, Memory, Bias, Notifier.
Defaults are OpenAI / no-op / no-op / no-op. Override at module load:

    from delphi import ADAPTERS, run_delphi
    ADAPTERS["memory"] = MyHonchoAdapter()
    transcript = run_delphi("your question")

See examples/ directory for adapter templates (OpenClaw, Hermes, etc.).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

WORKFLOW_NAME = "delphi"

# ── Configurable paths (env-driven, no personal data) ────────────────────
LEDGER_DIR = Path(os.environ.get(
    "DELPHI_LEDGER_DIR",
    str(Path.cwd() / "ledger"),
))

# ── Provider routing per role ────────────────────────────────────────────
# Per-role provider/model. Why per-role: LLM agreeableness varies. Adversarial
# roles (Skeptic, Red Team, Game Theorist, Black Swan) benefit from a model
# that doesn't soften "they will retaliate". Synthesis (Judge) benefits from
# a strong-reasoning model. You can override by setting env vars or by
# replacing this dict directly. Format: (provider_id, model_id).
#
# Default uses OpenAI for everything (max compatibility). To use multiple
# providers, see examples/multi_provider.py.
_DEFAULT_MODEL = os.environ.get("DELPHI_DEFAULT_MODEL", "gpt-4o-mini")
_JUDGE_MODEL   = os.environ.get("DELPHI_JUDGE_MODEL",   _DEFAULT_MODEL)
_INTAKE_MODEL  = os.environ.get("DELPHI_INTAKE_MODEL",  _DEFAULT_MODEL)

ROLE_PROVIDER = {
    "intake":         ("openai", _INTAKE_MODEL),
    "advocate":       ("openai", _DEFAULT_MODEL),
    "skeptic":        ("openai", _DEFAULT_MODEL),
    "realist":        ("openai", _DEFAULT_MODEL),
    "long_termist":   ("openai", _DEFAULT_MODEL),
    "game_theorist":  ("openai", _DEFAULT_MODEL),
    "black_swan":     ("openai", _DEFAULT_MODEL),
    "analogist":      ("openai", _DEFAULT_MODEL),
    "evidence_audit": ("openai", _DEFAULT_MODEL),
    "counterfactual": ("openai", _DEFAULT_MODEL),
    "red_team":       ("openai", _DEFAULT_MODEL),
    "cross_exam":     ("openai", _DEFAULT_MODEL),
    "judge":          ("openai", _JUDGE_MODEL),
}

def _valid_amount(value, name: str, *, positive: bool = False) -> float:
    """Reject invalid configuration before any provider call."""
    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if isinstance(value, bool) or not math.isfinite(amount) or amount < 0 or (positive and amount == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return amount


def _load_prices(raw: str) -> dict[str, tuple[float, float]]:
    """JSON provider/model -> [input, output] USD per million tokens."""
    try:
        values = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("DELPHI_PRICES_JSON must be a JSON object") from exc
    if not isinstance(values, dict):
        raise ValueError("DELPHI_PRICES_JSON must be a JSON object")
    prices = {}
    for key, pair in values.items():
        if not isinstance(key, str) or '/' not in key or not all(key.split('/', 1)):
            raise ValueError("Price keys must use provider/model")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Price for {key} must be [input, output]")
        prices[key] = tuple(_valid_amount(v, f"price for {key}") for v in pair)
    return prices


# Estimates only: fallback values are hypothetical planning assumptions,
# not vendor prices. Supply current rates for every routed model yourself.
BUDGET_USD_LIMIT = _valid_amount(os.environ.get("DELPHI_BUDGET_USD", "1.00"),
                               "DELPHI_BUDGET_USD", positive=True)
MODEL_PRICES = _load_prices(os.environ.get("DELPHI_PRICES_JSON", "{}"))
FALLBACK_PRICES = (
    _valid_amount(os.environ.get("DELPHI_FALLBACK_INPUT_USD_PER_M", "1.00"), "fallback input price"),
    _valid_amount(os.environ.get("DELPHI_FALLBACK_OUTPUT_USD_PER_M", "5.00"), "fallback output price"),
)


def _pricing_note() -> str:
    return ("Estimated cost only; configured USD/M input-output rates: "
            f"{json.dumps(MODEL_PRICES, sort_keys=True)}. Unconfigured models use "
            f"hypothetical fallback {FALLBACK_PRICES[0]:g}/{FALLBACK_PRICES[1]:g}, "
            "not verified vendor prices. Soft budget; actual charges may differ.")

# ── Failover routing per role ────────────────────────────────────────────
# If primary (provider, model) call fails (timeout/5xx/auth), try fallback.
# Empty by default (single-provider config). Populate when you have multiple
# providers configured. Format: (primary_provider, primary_model) → fallback.
ROLE_FAILOVER: dict[tuple[str, str], tuple[str, str]] = {}


# ─────────────────────────────────────────────────────────────────────────
# Adapter pattern — pluggable LLM, Memory, Bias, Notifier
# ─────────────────────────────────────────────────────────────────────────
# The pipeline never calls a vendor SDK directly. It calls these adapters.
# Defaults: OpenAI / no-op / no-op / no-op. Override at module load:
#   from delphi import ADAPTERS, run_delphi
#   ADAPTERS["llm"] = MyCustomLLMAdapter()
#   ADAPTERS["memory"] = MyHonchoAdapter()
#   transcript = run_delphi("...")

@dataclass
class LLMResponse:
    """Normalized LLM response. Adapters return this shape."""
    content: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMAdapter(Protocol):
    """Synchronous LLM call interface. Default impl uses OpenAI SDK."""
    def call(self, *, provider: str, model: str, messages: list[dict],
             max_tokens: int, temperature: float, timeout: float
             ) -> LLMResponse: ...


class MemoryAdapter(Protocol):
    """Long-term context provider (e.g. Honcho dialectic). Returns a string
    summary or None. Called once per pipeline with the reframed question."""
    def query(self, question: str) -> str | None: ...


class BiasAdapter(Protocol):
    """Returns a {topic, score, ...} dict if the user has historical bias on
    this topic class (e.g. via adaptive preference layer), or None."""
    def query(self, intake: dict) -> dict | None: ...


class NotifierAdapter(Protocol):
    """Pushes the decision brief somewhere external (Discord, Slack, etc.).
    Returns {ok: bool, error?: str}."""
    def push(self, brief_md: str, ledger_path: str | None) -> dict: ...


# ── Default adapter implementations ───────────────────────────────────────

class OpenAILLMAdapter:
    """OpenAI-compatible chat completions client. Works with OpenAI, DeepSeek,
    Together, Groq, OpenRouter, vLLM, etc. (anywhere with /v1/chat/completions).

    Reads OPENAI_API_KEY and optional OPENAI_BASE_URL from env.
    Lazy-imports `openai` so the module loads even if openai isn't installed
    (useful for --describe and dry-run).
    """
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install openai\n"
                "Or override ADAPTERS['llm'] with your own LLMAdapter."
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY env var not set. Set it or override "
                "ADAPTERS['llm']."
            )
        base_url = os.environ.get("OPENAI_BASE_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def call(self, *, provider: str, model: str, messages: list[dict],
             max_tokens: int, temperature: float,
             timeout: float) -> LLMResponse:
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        ti = getattr(usage, "prompt_tokens", 0) if usage else 0
        to = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMResponse(content=content, tokens_in=ti, tokens_out=to)


class NoOpMemoryAdapter:
    """Returns None — no long-term context provided."""
    def query(self, question: str) -> str | None:
        return None


class NoOpBiasAdapter:
    """Returns None — no historical bias data."""
    def query(self, intake: dict) -> dict | None:
        return None


class NoOpNotifierAdapter:
    """Silent no-op. Brief still goes to stdout + vault."""
    def push(self, brief_md: str, ledger_path: str | None) -> dict:
        return {"ok": True}


class DiscordWebhookNotifierAdapter:
    """Minimal Discord webhook poster. Set DELPHI_DISCORD_WEBHOOK_URL.
    Lazy-imports `requests` to avoid hard dep when not used.
    """
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.environ.get(
            "DELPHI_DISCORD_WEBHOOK_URL")

    def push(self, brief_md: str, ledger_path: str | None) -> dict:
        if not self.webhook_url:
            return {"ok": False,
                    "error": "DELPHI_DISCORD_WEBHOOK_URL not set"}
        try:
            import requests  # type: ignore
        except ImportError:
            return {"ok": False, "error": "requests not installed"}
        msg = brief_md
        if ledger_path:
            msg += f"\n\n📝 Full transcript: `{Path(ledger_path).name}`"
        # Discord 2000-char limit
        if len(msg) > 1900:
            msg = msg[:1900] + "\n\n_[truncated]_"
        try:
            r = requests.post(self.webhook_url, json={"content": msg},
                              timeout=15)
            r.raise_for_status()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# Active adapter registry — override these at module-load time.
ADAPTERS: dict[str, Any] = {
    "llm": OpenAILLMAdapter(),
    "memory": NoOpMemoryAdapter(),
    "bias": NoOpBiasAdapter(),
    "notifier": (
        DiscordWebhookNotifierAdapter()
        if os.environ.get("DELPHI_DISCORD_WEBHOOK_URL")
        else NoOpNotifierAdapter()
    ),
}

# ── Streaming progress to stderr ─────────────────────────────────────────
# v0.2 UX: 178s blackout in v0.1 was bad. Print one line per stage transition
# to stderr (stdout stays clean for brief). Set _STREAM_T0 at run start.
_STREAM_T0: float | None = None
_STREAM_ENABLED: bool = True  # set False by --no-stream

def _stream(stage: str, status: str = "...", **kw) -> None:
    if not _STREAM_ENABLED:
        return
    elapsed = (time.time() - _STREAM_T0) if _STREAM_T0 else 0
    parts = [f"[delphi {elapsed:5.1f}s] {stage:>14s} → {status}"]
    for k, v in kw.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}" if v < 1 else f"{k}={v:.1f}")
        else:
            parts.append(f"{k}={v}")
    print(" ".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────
# DelphiTranscript: event-log builder, dynamic structure, no fixed acts
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptEvent:
    """One event in the Delphi pipeline. Renders to one section of the md."""
    type: str           # e.g. "reframe", "role_opening", "cross_exam", "judge"
    payload: dict       # event-specific data
    timestamp: float = field(default_factory=time.time)
    # cost tracking — None if event isn't an LLM call
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    provider: str | None = None
    model: str | None = None


class DelphiTranscript:
    """Collects pipeline events; renders dynamic markdown transcript.

    Design: shape of transcript follows shape of actual debate. No fixed
    "Act 1, Act 2" template. Each event self-renders.
    """

    def __init__(self, raw_question: str):
        self.raw_question = raw_question
        self.start_ts = time.time()
        self.events: list[TranscriptEvent] = []
        self.brief: dict | None = None  # decision brief, set by judge stage

    def emit(self, event_type: str, *, tokens_in: int | None = None,
             tokens_out: int | None = None, cost_usd: float | None = None,
             provider: str | None = None, model: str | None = None,
             **payload) -> None:
        """Pipeline calls this to log a moment."""
        self.events.append(TranscriptEvent(
            type=event_type, payload=payload,
            tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd,
            provider=provider, model=model,
        ))

    def total_cost(self) -> float:
        return sum(e.cost_usd or 0 for e in self.events)

    def llm_call_count(self) -> int:
        """Count recorded calls, including zero-cost and grouped dissent calls."""
        return sum(e.payload.get("llm_calls", 1) for e in self.events
                   if e.cost_usd is not None)

    def total_tokens(self) -> tuple[int, int]:
        ti = sum(e.tokens_in or 0 for e in self.events)
        to = sum(e.tokens_out or 0 for e in self.events)
        return ti, to

    def latency_seconds(self) -> float:
        return time.time() - self.start_ts

    # ─── Rendering ───────────────────────────────────────────────────────

    def render(self) -> str:
        """Full markdown: brief on top, dynamic transcript, metadata appendix."""
        out = []
        out.append(self._render_header())
        out.append(self._render_brief())
        out.append("---\n")
        out.append("# 完整辩论 Trail\n")
        out.append("_下面是完整审计 trail。Brief 是 30 秒扫读版,这里是 deep audit。_\n")
        for ev in self.events:
            section = self._render_event(ev)
            if section:
                out.append(section)
        out.append(self._render_appendix())
        return "\n\n".join(out)

    def render_brief_only(self) -> str:
        """Discord-friendly: just decision brief + vault link placeholder."""
        return self._render_brief()

    def _render_header(self) -> str:
        ti, to = self.total_tokens()
        return (
            f"# Delphi Energy: {self._reframed_or_raw()}\n\n"
            f"> **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}** · "
            f"estimated cost ${self.total_cost():.3f} · "
            f"{self.llm_call_count()} LLM calls · "
            f"{ti:,} prompt tok / {to:,} completion tok · "
            f"{self.latency_seconds():.0f}s"
        )

    def _reframed_or_raw(self) -> str:
        for ev in self.events:
            if ev.type == "intake":
                return ev.payload.get("reframed", self.raw_question)
        return self.raw_question

    def _render_brief(self) -> str:
        if not self.brief:
            return "## 🎯 Decision Brief\n\n_(judgment 还没产生 — pipeline 中断或未完成)_"
        b = self.brief
        sigil = {"low": "🟡", "medium": "🟢", "high": "🟢"}.get(
            (b.get("confidence") or "").lower(), "⚪"
        )
        lines = ["## 🎯 Decision Brief"]
        if any(e.type == "cross_exam" and e.payload.get("round") == 2 for e in self.events):
            lines.append("\n_Post-verdict questions are in the full ledger; this brief and action have not been revised in response._")
        lines.append("")
        lines.append(f"**问题**:{self._reframed_or_raw()}")
        lines.append("")
        lines.append(f"**判断**:{b.get('judgment', '?')}")
        lines.append("")
        lines.append(f"**Confidence**:{sigil} {b.get('confidence', '?').upper()}"
                     f" — {b.get('confidence_framing', '')}")
        lines.append("")
        lines.append(f"**一行下注**:{b.get('one_line_bet', '?')}")
        lines.append("")
        lines.append(f"**Biggest Risk**:{b.get('biggest_risk', '?')}")
        lines.append("")
        fd = b.get("first_domino") or {}
        if isinstance(fd, dict):
            fd_str = (
                f"**做** {fd.get('action', '?')}"
                f" · 成本 {fd.get('cost', '?')}"
                f" · 验证 {fd.get('validates', '?')}"
            )
        else:
            fd_str = str(fd)
        lines.append(f"**24h First Domino**:{fd_str}")
        lines.append("")
        change = b.get("what_would_change_my_mind") or []
        if change:
            lines.append("**让我改变想法的条件**:")
            for c in change:
                lines.append(f"- {c}")
            lines.append("")

        # v1.1: 4 new sections (compact form for brief; full form in transcript)
        th = b.get("time_horizon") or {}
        if th:
            lines.append("**⏱ Time Horizon**:")
            for k, lab in [("now_3mo", "0-3 月"),
                           ("near_3_12mo", "3-12 月"),
                           ("mid_1_3yr", "1-3 年"),
                           ("long_3_10yr", "3-10 年")]:
                if th.get(k):
                    lines.append(f"- {lab}:{th[k]}")
            lines.append("")
        contradictions = b.get("contradictions") or []
        if contradictions:
            lines.append("**🌀 核心张力**:")
            for c in contradictions:
                lines.append(f"- {c}")
            lines.append("")
        cf = b.get("counterfactual_check")
        if cf:
            lines.append(f"**🔄 Counterfactual Check**:{cf}")
            lines.append("")
        et = b.get("evidence_threshold") or {}
        if et:
            verdict_emoji = {"sufficient": "✅", "borderline": "⚠️",
                             "insufficient": "🔴"}.get(et.get("verdict"), "⚪")
            lines.append(f"**📊 Evidence**:{verdict_emoji} 现有 {et.get('current_grade','?')} 级 "
                         f"vs 应需 {et.get('required_grade','?')} 级 — "
                         f"{et.get('verdict','?')} ({et.get('rationale','')})")
            lines.append("")
        nw = b.get("noise_warning") or []
        if nw:
            lines.append("**🔇 主动忽略的噪声**:")
            for n in nw:
                lines.append(f"- {n}")
        return "\n".join(lines)

    def _render_event(self, ev: TranscriptEvent) -> str:
        renderer = _RENDERERS.get(ev.type)
        if renderer:
            try:
                # Merge metadata so renderers can see provider/model without
                # the pipeline having to duplicate them in payload kwargs.
                merged = dict(ev.payload)
                if ev.provider and "provider" not in merged:
                    merged["provider"] = ev.provider
                if ev.model and "model" not in merged:
                    merged["model"] = ev.model
                return renderer(merged)
            except Exception as e:
                return f"## ⚠️ Render error for `{ev.type}`: {e}"
        return f"<!-- unhandled event type: {ev.type} -->"

    def _render_appendix(self) -> str:
        lines = ["---", "", "# Appendix:运行元数据", ""]
        lines.append("| 阶段 | Role | Provider | Model | Tokens (in/out) | Estimated cost |")
        lines.append("|---|---|---|---|---|---|")
        for ev in self.events:
            if ev.cost_usd is None:
                continue
            ti = ev.tokens_in or 0
            to = ev.tokens_out or 0
            role = ev.payload.get("role") or ev.payload.get("stage") or ev.type
            lines.append(
                f"| {ev.type} | {role} | {ev.provider or '?'} | "
                f"{ev.model or '?'} | {ti:,} / {to:,} | ${ev.cost_usd:.4f} |"
            )
        ti, to = self.total_tokens()
        lines.append(f"| **TOTAL** | | | | **{ti:,} / {to:,}** | **${self.total_cost():.4f}** |")
        lines.append("")
        lines.append(f"- 总时长:{self.latency_seconds():.0f}s")
        lines.append(f"- Soft budget 上限:${BUDGET_USD_LIMIT:.2f}")
        lines.append("- " + _pricing_note())
        lines.append(f"- Estimated budget 使用率:{self.total_cost()/BUDGET_USD_LIMIT*100:.0f}%")
        return "\n".join(lines)

    # ─── Vault ledger save ───────────────────────────────────────────────

    def save_to_vault(self, slug: str) -> Path:
        """Write full transcript to vault md file with frontmatter."""
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = LEDGER_DIR / f"{date_str}_{slug}.md"
        # Make unique if exists
        i = 2
        while path.exists():
            path = LEDGER_DIR / f"{date_str}_{slug}-{i}.md"
            i += 1
        # Frontmatter
        b = self.brief or {}
        intake = next((e.payload for e in self.events if e.type == "intake"), {})
        fm_lines = [
            "---",
            "type: delphi-judgment",
            f"date: {date_str}",
            f"question_type: {intake.get('question_type', 'unknown')}",
            f"confidence: {b.get('confidence', 'unknown')}",
            f"reversibility: {intake.get('reversibility', 'unknown')}",
            f"cost_of_being_wrong: {intake.get('cost_of_being_wrong', 'unknown')}",
        ]
        # v1.1: evidence threshold + delphi version into frontmatter
        et = (b.get("evidence_threshold") or {}) if b else {}
        if et.get("verdict"):
            fm_lines.append(f"evidence_verdict: {et['verdict']}")
            fm_lines.append(f"evidence_grade: {et.get('current_grade', 'unknown')}")
        fm_lines.append("delphi_version: v1.1")
        fm_lines.extend([
            "tags:",
            "  - delphi",
            "  - judgment",
        ])
        # Review trigger
        signals = b.get("validation_signals") or []
        if signals:
            fm_lines.append("review_at: 6 weeks from date")
        fm_lines.append("---")
        path.write_text("\n".join(fm_lines) + "\n\n" + self.render(), encoding="utf-8")
        return path


# ─────────────────────────────────────────────────────────────────────────
# Per-event renderers — each event type knows how to render itself
# ─────────────────────────────────────────────────────────────────────────

def _render_intake(p: dict) -> str:
    lines = ["## Reframe"]
    lines.append("")
    lines.append(f"> **原问题**:{p.get('original', '?')}")
    lines.append("")
    lines.append(f"**Reframed**:{p.get('reframed', '?')}")
    lines.append("")
    lines.append(f"- **类型**:{p.get('question_type', '?')}")
    lines.append(f"- **可逆性**:{p.get('reversibility', '?')}")
    lines.append(f"- **错的代价**:{p.get('cost_of_being_wrong', '?')}")
    pref = p.get("user_preference")
    if pref:
        lines.append(f"- **用户隐含倾向**:{pref}")
    out = "\n".join(lines)

    # v1.1: KAU 4-tier boundary visible in transcript
    fb = p.get("fact_boundary") or {}
    if any(fb.get(k) for k in ["known", "assumed", "unknown", "speculative"]):
        out += "\n\n## 📋 KAU 事实边界\n"
        for tier, label, icon in [
            ("known", "Known(用户陈述)", "✅"),
            ("assumed", "Assumed(合理假设)", "⚠️"),
            ("unknown", "Unknown(关键缺失)", "❓"),
            ("speculative", "Speculative(推测)", "🚫"),
        ]:
            items = fb.get(tier) or []
            if items:
                out += f"\n**{icon} {label}**:\n"
                for it in items:
                    out += f"- {it}\n"

    # v1.1: time horizon relevance hints
    th = p.get("time_horizon_relevance") or {}
    if th:
        out += "\n\n## ⏱ Time Horizon 相关性(intake 预判)\n"
        for k, v in th.items():
            out += f"- **{k}**: {v}\n"

    out += "\n\n## 变量地图\n\n" + _render_variable_table(p.get("variables") or [])
    return out


def _render_variable_table(vars_: list) -> str:
    if not vars_:
        return "_(无显著变量)_"
    lines = ["| 变量 | Impact | Uncertainty | 描述 |"]
    lines.append("|---|---|---|---|")
    for v in vars_:
        lines.append(
            f"| {v.get('name', '?')} | {v.get('impact', '?')} | "
            f"{v.get('uncertainty', '?')} | {v.get('description', '')[:120]} |"
        )
    return "\n".join(lines)


def _render_honcho(p: dict) -> str:
    content = p.get("content", "_(no memory context)_")
    return f"## 🧠 Long-term Memory(用户长期视角)\n\n> {content}"


def _render_past_ledgers(p: dict) -> str:
    """v0.5 — past Delphi judgments on related questions."""
    matches = p.get("matches") or []
    if not matches:
        return ""
    lines = [
        "## 📚 过去 Ledger 引用",
        "",
        f"_找到 {len(matches)} 条相关历史判断(按相关度 × 时间衰减排序)。Judge 会被告知。_",
        "",
    ]
    for m in matches:
        lines.append(f"### {m.get('ledger_name', '?')}")
        lines.append(f"_age={m.get('age_days')}d · overlap={m.get('overlap_score')}"
                     f" · confidence={m.get('confidence', '?')}_")
        lines.append("")
        if m.get("judgment"):
            lines.append(f"**当时判断**:{m['judgment']}")
            lines.append("")
    return "\n".join(lines)


def _render_adaptive_bias(p: dict) -> str:
    """v0.4 — adaptive.py preference layer signal (auto-injected if topic data exists)."""
    direction = p.get("direction", "?")
    mag = p.get("magnitude", "?")
    topic = p.get("strongest_topic", "?")
    score = p.get("strongest_score", 0.0)
    matches = p.get("matches") or []
    lines = [
        "## ⚠️ Adaptive Bias 警告",
        "",
        f"_检测到用户在此话题上有历史 **{mag}{direction}** (score={score:+.2f},"
        f" topic=`{topic}`)。Judge 应对最终判断保留警觉:这个判断方向可能"
        f"是 motivated reasoning 而非证据驱动。Red Team 会被通知加大火力。_",
        "",
    ]
    if len(matches) > 1:
        match_strs = [f"`{m['topic']}`={m['score']:+.2f}" for m in matches[:5]]
        lines.append(f"_其他匹配({len(matches)} 个): {', '.join(match_strs)}_")
    return "\n".join(lines)


def _render_role_opening(p: dict) -> str:
    role = p.get("role", "?")
    icon = p.get("icon", "")
    provider = p.get("provider", "?")
    model = p.get("model", "?")
    content = p.get("content", "")
    return f"## {icon} {role} _(opening · {provider}/{model})_\n\n{content}"


def _render_cross_exam(p: dict) -> str:
    attacker = p.get("attacker", "?")
    target = p.get("target", "?")
    content = p.get("content", "")
    round_n = p.get("round", 1)
    note = "\n\n_Post-verdict review: does not revise the Decision Brief or action._" if round_n == 2 else ""
    return f"## 交叉质询 轮次 {round_n}: {attacker} → {target}{note}\n\n{content}"


def _render_red_team(p: dict) -> str:
    content = p.get("content", "")
    return (
        "## 🔴 Red Team(攻击用户而非问题)\n\n"
        "_直击 stated preference / identity / sunk cost / motivated reasoning_\n\n"
        f"{content}"
    )


def _render_judge(p: dict) -> str:
    content = p.get("content", "")
    return f"## 👨‍⚖️ 法官裁决\n\n{content}"


def _render_evidence_audit(p: dict) -> str:
    """v1.1 § 7.9 — Evidence Auditor evaluation."""
    return f"## 📊 Evidence Audit\n\n{p.get('content', '')}"


def _render_counterfactual(p: dict) -> str:
    """v1.1 § 7.15 — Counterfactual Baseline."""
    return f"## 🔄 Counterfactual Baseline\n\n{p.get('content', '')}"


def _render_skip(p: dict) -> str:
    reason = p.get("reason", "?")
    what = p.get("what", "?")
    return f"## ⏭ 跳过:{what}\n\n_理由:{reason}_"


def _render_budget_warning(p: dict) -> str:
    return (
        f"## ⚠️ Budget warning\n\n"
        f"已用 ${p.get('used_usd', 0):.3f} / ${BUDGET_USD_LIMIT:.2f} "
        f"({p.get('used_pct', 0):.0f}%)。{p.get('action', '')}"
    )


def _render_skeptic_dissent(p: dict) -> str:
    """v0.7 — multi-model Skeptic dissent."""
    if not p.get("should_surface"):
        # Quiet form: just a footnote-style blurb
        return (
            "## ✅ Skeptic 跨模型一致\n\n"
            f"_({p.get('alt_provider', '?')}/{p.get('alt_model', '?')} 跑了同一个 Skeptic prompt,"
            f"agreement_level={p.get('agreement_level', '?')}。两个模型都攻击同一假设,"
            f"无显著分歧。)_"
        )
    # Loud form: full dissent display
    lines = [
        "## ⚠️ Skeptic 跨模型分歧",
        "",
        f"_两个模型(Codex 主 / {p.get('alt_provider', '?')} 备)对同一问题给出**实质性不同**的反驳。"
        f"agreement_level=`{p.get('agreement_level', '?')}`。Judge 必须在 narrative 中显式回应这个分歧。_",
        "",
        f"**分歧总结**:{p.get('divergence_summary', '?')}",
        "",
    ]
    shared = p.get("shared_attacks") or []
    if shared:
        lines.append("**两边都提出的攻击**:")
        for s in shared:
            lines.append(f"- {s}")
        lines.append("")
    diverging = p.get("diverging_attacks") or []
    if diverging:
        lines.append("**只一边提出的攻击**:")
        for s in diverging:
            lines.append(f"- {s}")
        lines.append("")
    lines.append(f"### Skeptic 备线全文 ({p.get('alt_provider', '?')}/{p.get('alt_model', '?')})")
    lines.append("")
    lines.append(p.get("alt_content", "")[:1500])
    return "\n".join(lines)


def _render_drilldown(p: dict) -> str:
    """v0.6 — post-pipeline follow-up to a specific role."""
    role = p.get("role", "?")
    icon = p.get("icon", "")
    provider = p.get("provider", "?")
    model = p.get("model", "?")
    question = p.get("question", "?")
    content = p.get("content", "")
    return (
        f"## {icon} 下钻 → {role} _(drilldown · {provider}/{model})_\n\n"
        f"> **追问**:{question}\n\n"
        f"{content}"
    )


def _render_action(p: dict) -> str:
    lines = ["## 终幕:行动"]
    lines.append("")
    lines.append(f"### 一行下注\n\n> {p.get('one_line_bet', '?')}")
    lines.append("")
    fd = p.get("first_domino") or {}
    lines.append("### 24h 第一步")
    lines.append(f"- **做什么**:{fd.get('action', '?')}")
    lines.append(f"- **成本**:{fd.get('cost', '?')}")
    lines.append(f"- **验证什么假设**:{fd.get('validates', '?')}")
    lines.append("")
    sigs = p.get("validation_signals") or []
    if sigs:
        lines.append("### 未来验证信号")
        lines.append("| 信号 | 方向 | 效果 |")
        lines.append("|---|---|---|")
        for s in sigs:
            lines.append(
                f"| {s.get('signal', '?')} | {s.get('direction', '?')} | "
                f"{s.get('effect', '?')} |"
            )
    return "\n".join(lines)


_RENDERERS: dict[str, Any] = {
    "intake": _render_intake,
    "memory_view": _render_honcho,
    "adaptive_bias": _render_adaptive_bias,  # v0.4
    "past_ledgers": _render_past_ledgers,    # v0.5
    "role_opening": _render_role_opening,
    "cross_exam": _render_cross_exam,
    "red_team": _render_red_team,
    "judge": _render_judge,
    "judge_repair": lambda p: f"<!-- judge_repair: ok={p.get('ok')} -->",  # silent in render
    "skip": _render_skip,
    "budget_warning": _render_budget_warning,
    "action": _render_action,
    "drilldown": _render_drilldown,                  # v0.6
    "skeptic_dissent": _render_skeptic_dissent,      # v0.7
    "evidence_audit": _render_evidence_audit,        # v1.1
    "counterfactual": _render_counterfactual,        # v1.1
}


# ─────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────

def _call_llm_for_role(
    role: str,
    system: str,
    user: str,
    *,
    transcript: DelphiTranscript,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    timeout: float | None = 90.0,
    retry_on_timeout: bool = False,
) -> tuple[str, int, int, float, str, str]:
    """One LLM call routed by role's provider config.

    Returns: (content, tokens_in, tokens_out, cost_usd, provider, model)
    Caller is responsible for emitting the transcript event with this data.

    timeout: per-call wall-clock cap. Default 90s — DeepSeek v4-pro reasoning
        models are slow on long contexts. Judge passes 180s.
    retry_on_timeout: if True, do exactly one retry on APITimeoutError.
        Used by Judge (most expensive call to lose).
    """
    provider, model = ROLE_PROVIDER.get(role, ROLE_PROVIDER["judge"])

    # v0.2: try primary, then optionally one same-provider retry on timeout,
    # then optionally one cross-provider failover. Track which path won.
    plan: list[tuple[str, str, str]] = [(provider, model, "primary")]
    if retry_on_timeout:
        plan.append((provider, model, "retry"))
    fallback = ROLE_FAILOVER.get((provider, model))
    if fallback:
        plan.append((*fallback, "failover"))

    last_err: Exception | None = None
    for prov, mdl, kind in plan:
        try:
            resp = ADAPTERS["llm"].call(
                provider=prov, model=mdl,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout if timeout is not None else 90.0,
            )
            content = resp.content
            # Strip <think> reasoning blocks if present (DeepSeek R1 / Gemini thinking mode)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            ti, to = resp.tokens_in, resp.tokens_out
            cost = _estimate_cost(prov, mdl, ti, to)
            if kind != "primary":
                _stream(role, f"{kind}-ok", provider=prov, model=mdl)
            return content, ti, to, cost, prov, mdl
        except Exception as e:
            last_err = e
            err_name = type(e).__name__
            # Retry only on timeout when next plan step is same-provider retry;
            # always proceed to failover on any error if a failover slot exists.
            if kind == "retry" and "Timeout" not in err_name:
                # Skip the retry slot if the error wasn't a timeout
                continue
            _stream(role, f"{kind}-fail",
                    err=err_name, msg=str(e)[:80])
            continue
    return (f"_(LLM call failed: {type(last_err).__name__}: {last_err})_",
            0, 0, 0.0, provider, model)


def _estimate_cost(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """Token-based estimate using explicit rates or labeled planning assumptions."""
    p_in, p_out = MODEL_PRICES.get(f"{provider}/{model}", FALLBACK_PRICES)
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def _query_past_ledgers(intake: dict, *, top_k: int = 3,
                        max_age_days: int = 180) -> list[dict]:
    """v0.5: search vault Delphi Ledger/ for related past judgments.

    Strategy (intentionally cheap — no embeddings, no vector search):
    1. List all .md files in LEDGER_DIR.
    2. For each: extract frontmatter + decision brief section + filename slug.
    3. Score by keyword overlap with intake (reframed text + variable names +
       question_type). zh-aware: split on punctuation, filter stop chars.
    4. Return top_k with ledger_path, brief_excerpt, judgment, confidence,
       overlap_score, age_days.

    Why no vectors: the vault has <100 ledgers for the foreseeable horizon;
    keyword overlap is robust + transparent + 0-cost. Vector search is a
    v1.x upgrade once the vault crosses 1000 entries.
    """
    if not LEDGER_DIR.exists():
        return []

    # Build query token set
    qparts: list[str] = []
    qparts.append(intake.get("reframed") or "")
    qparts.append(intake.get("question_type") or "")
    for v in intake.get("variables") or []:
        qparts.append(v.get("name") or "")
        qparts.append(v.get("description") or "")
    if intake.get("user_preference"):
        qparts.append(intake["user_preference"])
    query_text = " ".join(qparts).lower()
    # Tokenize: keep zh chunks (2-10 chars) + ascii words (3+ chars)
    query_tokens: set[str] = set()
    for tok in re.split(r"[\s,，。.!?！？:、/(){}\[\]<>\-_]+", query_text):
        tok = tok.strip()
        if not tok:
            continue
        # Skip pure numerals + super common words
        if tok.isdigit() or tok in {"the", "and", "or", "to", "for", "of",
                                    "is", "a", "in", "on", "with", "vs",
                                    "还是", "应该", "我", "的", "了", "是", "在"}:
            continue
        if 2 <= len(tok) <= 30:
            query_tokens.add(tok)
    if not query_tokens:
        return []

    now = datetime.now()
    candidates: list[tuple[float, dict]] = []
    for path in LEDGER_DIR.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        # age in days from filename date prefix YYYY-MM-DD_*.md
        age_days = 9999
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})_", path.name)
        if m:
            try:
                d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                age_days = (now - d).days
            except Exception:
                pass
        if age_days > max_age_days:
            continue

        # Skip self-match: the ledger we're about to write is older but
        # we don't yet know its name; this filter catches identical-question
        # repeat runs.
        text_lower = text.lower()

        # Keyword overlap score
        hits = sum(1 for t in query_tokens if t in text_lower)
        if hits < 2:  # need at least 2 token hits to count as related
            continue
        overlap_score = hits / max(len(query_tokens), 1)

        # Recency weight: 1.0 today, 0.5 at 90 days
        recency_weight = max(0.1, 1.0 - (age_days / 180))
        final_score = overlap_score * recency_weight

        # Extract brief excerpt + judgment from the file
        judgment = ""
        confidence = ""
        brief_excerpt = ""
        m2 = re.search(r"\*\*判断\*\*:(.+?)(?:\n\n|\*\*Confidence|\Z)",
                       text, flags=re.DOTALL)
        if m2:
            judgment = m2.group(1).strip()[:300]
        m3 = re.search(r"confidence:\s*(\w+)", text)
        if m3:
            confidence = m3.group(1)
        # Brief = full Decision Brief section (capped)
        m4 = re.search(r"## 🎯 Decision Brief\n(.*?)(?:\n---|\n# 完整辩论)",
                       text, flags=re.DOTALL)
        if m4:
            brief_excerpt = m4.group(1).strip()[:1000]

        candidates.append((final_score, {
            "ledger_path": str(path),
            "ledger_name": path.name,
            "age_days": age_days,
            "overlap_hits": hits,
            "overlap_score": round(overlap_score, 3),
            "final_score": round(final_score, 3),
            "judgment": judgment,
            "confidence": confidence,
            "brief_excerpt": brief_excerpt,
        }))

    candidates.sort(key=lambda kv: kv[0], reverse=True)
    return [c[1] for c in candidates[:top_k]]


def _query_adaptive_bias(intake: dict) -> dict | None:
    """Check the bias adapter for historical user bias on this question's topic.

    Default (NoOpBiasAdapter) returns None — no bias signal is injected.
    To enable, override:
        ADAPTERS["bias"] = MyAdaptivePreferenceAdapter()

    Adapter contract: see BiasAdapter Protocol. Return shape (used by Judge +
    Red Team prompt injection) at minimum:
        {"strongest_topic": str, "strongest_score": float (-1..1),
         "direction": "正向偏好"|"负向偏好", "magnitude": "强|中|弱",
         "matches": [{"topic": str, "score": float}, ...]}
    """
    try:
        signal = ADAPTERS["bias"].query(intake)
    except Exception as e:
        _stream("adaptive_bias", "error",
                err=type(e).__name__, msg=str(e)[:80])
        return None
    return signal


def _query_memory(question: str) -> str | None:
    """Query the memory adapter for the user's long-term context.

    The default adapter (NoOpMemoryAdapter) returns None. To enable, set:
        ADAPTERS["memory"] = HonchoMemoryAdapter(...)
    or write your own per the MemoryAdapter Protocol.
    """
    try:
        view = ADAPTERS["memory"].query(question)
        if not view:
            return None
        # Strip <think> reasoning blocks if present (some models leak these)
        cleaned = re.sub(r"<think>.*?</think>", "", str(view),
                         flags=re.DOTALL).strip()
        return cleaned[:1500] if cleaned else None
    except Exception as e:
        return f"_(memory adapter error: {type(e).__name__}: {e})_"


def _slugify(text: str, max_len: int = 50) -> str:
    """Make a filesystem-safe slug from question text."""
    # Keep CJK + ascii alphanumeric, replace whitespace with -
    out = re.sub(r"[\s/\\:*?\"<>|]+", "-", text.strip())
    out = re.sub(r"-+", "-", out).strip("-")
    return out[:max_len] or "delphi"


# ─────────────────────────────────────────────────────────────────────────
# Role prompts — system messages per role
# ─────────────────────────────────────────────────────────────────────────
# Design notes:
# - All in zh-CN with English code identifiers (matches Hermes house style).
# - Each prompt enforces:
#     (a) no hedging ("可能 / 或许" → ban),
#     (b) JSON-where-asked, prose-where-asked,
#     (c) frequency framing not numeric % (per design doc).
# - Adversarial roles (skeptic/red_team) have explicit "not your job to be
#   nice" framing to counter LLM agreeableness.

INTAKE_SYSTEM = """你是 Delphi Energy 的接单官。你的任务:把用户的原问题清洗成一个可被多角色辩论的命题,并提取关键变量、判断类型、可逆性、错的代价。不做实际推理,只做分类。

v0.9 + v1.1 强化:把所有"事实陈述"按 4 档分级(KAU 完整 boundary)。这是后续 Judge 防编造、Evidence Auditor 评级的基础。

输出严格 JSON,字段:
{
  "reframed": "重新陈述的问题(去掉情绪,保留原意)",
  "question_type": "decision | belief | strategy | identity | unclear",
  "reversibility": "low | medium | high",
  "cost_of_being_wrong": "low | medium | high | catastrophic",
  "user_preference": "若用户的措辞透露出隐含倾向,直接说出来;否则 null",

  "fact_boundary": {
    "known": ["用户在问题中明确陈述、可验证的事实(数字、版本、当前状态、时间点)"],
    "assumed": ["合理但用户没明说的假设(如行业常识、用户身份推断、市场常态)"],
    "unknown": ["对此判断重要但 user 没给、roles 也无法验证的信息(关键缺失)"],
    "speculative": ["纯想象 / 未来推测 / 弱依据的内容,如果在问题里出现就标出"]
  },

  "user_asserted_facts": ["DEPRECATED — 等于 fact_boundary.known,保留向后兼容,直接复制"],

  "variables": [
    {
      "name": "...",
      "impact": "high|medium|low",
      "uncertainty": "high|medium|low",
      "description": "一句话",
      "verifiability": "weak | medium | strong",
      "drilldown_priority": true
    }
  ],

  "time_horizon_relevance": {
    "now_3mo": "是否在 0-3 月内有 material 影响(yes/no/maybe)+ 1 句解释",
    "near_3_12mo": "3-12 月 (yes/no/maybe) + 1 句",
    "mid_1_3yr": "1-3 年 (yes/no/maybe) + 1 句",
    "long_3_10yr": "3-10 年 (yes/no/maybe) + 1 句"
  }
}

只输出 JSON,不解释。"""

ADVOCATE_SYSTEM = """你是 Advocate(支持方)。你的任务:为下方问题中"做 / 相信 / 推进"这一选项构建最强论证。

要求:
- 直接陈述,不 hedge。不说"可能 / 或许 / 也许"。
- 给出 3 个最有力的支持论点,每个论点说清楚:核心机制 + 一条具体证据或类比。
- 最后一句明确:"如果只能用一句话说服法官,我会说:______"
- 不预测对手会怎么反驳,那是 Skeptic 的活。
- 输出 markdown,不超过 400 字。"""

SKEPTIC_SYSTEM = """你是 Skeptic(反对方)。你不是来"平衡观点"的,你是来证明 Advocate 错了。你的工作不是讨好,是穿透。

要求:
- 找出 Advocate 论证里最薄弱的一个假设(structural assumption),指名说出来。
- 给出 3 个最强的反对理由,每个论点说:核心机制 + 一个 base rate / 历史对照 / 反例。
- 不要罗列"风险清单",罗列没用,要直击 Advocate 立论的根。
- 最后一句:"如果 Advocate 是对的,那必须假设 ______ 成立 — 而我认为这个假设站不住,因为 ______"
- 输出 markdown,不超过 400 字。"""

REALIST_SYSTEM = """你是 Realist(现实主义者)。你不站任一边。你的任务是揭示"做 vs 不做"在现实操作层会卡在哪里。

要求:
- 列出 3 个落地约束(操作 / 时间 / 资源 / 已有承诺冲突)。
- 每个约束给出:具体形式 + 通常的破解方式或"无解"。
- 不评价对错,只描述摩擦。
- 输出 markdown,不超过 350 字。"""

LONG_TERMIST_SYSTEM = """你是 Long-Termist(长期主义者)。你只看 5-10 年视角。

要求:
- 用一句话回答:5 年后这个决定看起来会是什么样?
- 列出 2-3 个长期复利或长期诅咒(compounding gains / compounding curses),指出哪个被低估了。
- 指出此决定与用户长期方向是否一致。如果不一致,直接说"这与 ______ 冲突"。
- 输出 markdown,不超过 300 字。"""

GAME_THEORIST_SYSTEM = """你是 Game Theorist(博弈论者)。你只看一件事:这个决定不是在真空里做的——还有谁是玩家,他们的反应函数是什么,均衡在哪里。

要求:
- 列出 2-4 个关键玩家(可以是:竞争对手、用户、投资人、监管者、平台方、未来的自己、市场情绪)。每个玩家给一句:他们的目标 + 他们看到用户做这个动作后的最佳反应。
- 找出至少一个 second-order effect(用户没注意到的、由对手反应引发的下一轮影响)。
- 如果有 dominant strategy(无论对手怎么动,某条路径都更优),直接指出。如果是 prisoner's dilemma 或 coordination problem,也直接命名。
- 最后一句:"用户在这个博弈里的位置是 ______(先手/后手/被动/有信息优势/有信息劣势),这个位置最适合的策略是 ______"
- 输出 markdown,不超过 350 字。
- 如果这个问题里没有真正的对手或反应函数(纯个人决定 / 内部决定),直接说"未识别显著博弈结构"并给一句解释,不要硬凑。"""

# v1.1 — 3 new roles per design § 7.7

BLACK_SWAN_SCOUT_SYSTEM = """你是 Black Swan Scout(黑天鹅侦察)。你只关心一件事:这个判断里被默认成"不会发生"的低概率高影响事件,有哪些?

要求:
- 列出 2-3 个 low-probability, high-impact disruption 候选。每条说清:
  · 触发条件(什么事一旦发生,这个判断就被颠覆)
  · 概率 base rate(用频率词:"罕见/十年一次/历史上发生过 X 次",不要用百分比)
  · 影响幅度(轻伤 / 重伤 / 全盘推倒)
- 至少有 1 条必须是 user 当前 framing 里完全没考虑到的那种 — 不是显而易见的"市场崩盘",而是结构性盲点。
- 不要列日常风险(Skeptic 已经管),只列"如果发生用户 will be blindsided"的事件。
- 最后一句:"如果只能 hedge 一个黑天鹅,我会 hedge ______ — 因为它的 cost-benefit 比最不对称。"
- 不超过 350 字。如果这个问题真的没有有意义的黑天鹅(例:纯审美决定),直接说"未识别有意义黑天鹅"并给一句解释。"""

ANALOGIST_SYSTEM = """你是 Analogist(类比者)。你的活:找历史 / 跨行业 case,既给参照,也指出参照在哪里 break。

要求:
- 列出 2 个最强类比。每个类比说清:
  · 历史案例(具体公司/产品/事件)
  · 在哪个维度跟当前问题相似(机制层面,不是表面)
  · **类比 break point** —— 这个类比在哪里失效(必须给出,不准只夸不损)
- 一个类比是同行业,一个是跨行业(扩大视野)。
- 不要用"硅谷常说" / "据报道" / "听说" 这类无来源话术 — 用具体可验证的案例。
- 最后一句:"如果用户只参考一个类比,应该选 ______,理由 ______。"
- 不超过 350 字。
- 如果这个问题没有有意义的历史类比(例:你是真新型 problem),直接说"无强类比,这是 first-of-kind problem"并给一句为什么。"""

EVIDENCE_AUDITOR_SYSTEM = """你是 Evidence Auditor(证据审计员)。你的活:把其他 role 的发言里所有"事实主张"挑出来,按 A/B/C/D 4 档评级,然后给一个总评。

评级标准(per design § 7.9):
- **A 级 — Hard Evidence**:审计财报、官方数据、生产数字、政策文件、签字合同、可观察的具体 operational action
- **B 级 — Strong Signal**:executive 公开目标、招聘信号、渠道动作、供应链动作、本地合作、监管申请
- **C 级 — Weak Signal**:媒体报道、社交舆论、launch narrative、用户情绪、预订宣称、行业传闻
- **D 级 — Narrative**:创始人魅力、品牌氛围、ambition、analogy、想象的未来、市场情绪

要求:
- 从 Advocate / Skeptic / Realist / Long-Termist / Game Theorist / Black Swan / Analogist 的发言里,挑 5-8 个最关键的"事实主张",逐条评级。
- 每条:`[X 级] role 说了什么 → 它的支撑实际上是什么级别的`
- 然后总评:"这场辩论的整体证据等级是 ___,主要是 ___ 级证据,___ 级证据缺位。"
- 不要重新论证问题,只评 evidence quality。
- 不超过 400 字。"""

RED_TEAM_SYSTEM = """你是 Red Team。你不攻击问题,你攻击提问者本人。

你的工作:找出用户在这个问题上正在自欺的地方。可能是 sunk cost、identity protection、stated-preference vs revealed-preference 不一致、motivated reasoning、avoiding 一个更深问题。

要求:
- 不安抚。不"理解"用户。直接指出。
- 给出 1-2 个最尖锐的诊断,每个用一句话定性 + 一句证据(从用户措辞 / 记忆层视角中拿)。
- 最后一句:"用户没问的、但更应该问的真问题是:______"
- 输出 markdown,不超过 300 字。
- 如果你判断没有自欺迹象,直接说"未发现明显自欺",并给一句解释。不要为了写而写。"""

CROSS_EXAM_SYSTEM = """你是质询官。下方是 Advocate 的开篇陈述。你要扮演 Skeptic,提出 1-2 个最致命的追问 — 不是泛泛的"那如果 X 怎么办",而是针对 Advocate 论证里的某个具体环节。

要求:
- 用问句形式,直接刺进 Advocate 论证的关节。
- 每个追问要附一句"如果回答是 ___,那 Advocate 立论就垮了"。
- 输出 markdown,不超过 250 字。"""

JUDGE_SYSTEM = """你是法官(Judge)。你已经看过 Advocate / Skeptic / Realist / Long-Termist / Game Theorist / Red Team / Black Swan Scout / Analogist 全部发言,以及 Evidence Auditor 的证据评级、Counterfactual 的对比基准、Memory 提供的用户长期视角。

**反编造硬规则**:你不允许在裁决里编造任何"细节"。具体说:
- 任何具体数字、版本号、人名、产品名、时间点、市场份额、对手动作 — 必须能追溯到 (a) 用户问题原文中明确陈述, (b) Memory 视角, (c) Past Ledger, 或 (d) 某个 role 的发言。
- 如果你在判断中需要用到一个无法追溯的事实, 用条件式表述("如果 X 是真,则...")而不是断言式("X 是真,所以...")。
- 如果某个 role 引用了一个你也无法追溯到 user/memory/past_ledger 的事实, 在 narrative 中标记 `[推测:role 引用了 ___]`。
- 你可以用普遍的 base rate / 类比, 但必须用 "类似情况下通常..." 这类语言, 不能伪装成具体事实。

你的任务,严格按这个顺序:

1. **Pre-Mortem**: 假设 6 个月后这个决定彻底失败了。最可能的失败路径是什么?用 2-3 句描述。
2. **Time Horizon Split** (v1.1 design § 7.4): 给出 4 个时间窗口分别的核心结论,每个 1 句:
   - now_3mo: 0-3 月内最关键的事是什么?
   - near_3_12mo: 3-12 月会浮现什么?
   - mid_1_3yr: 1-3 年的中期判断?
   - long_3_10yr: 3-10 年的长期方向?
3. **Contradiction Map** (v1.1 design § 7.14): 列出 2-4 个核心张力 — 即"这个判断成立必须同时容忍"的内部矛盾(例:"短期 hype 强 vs 长期信任未建")。
4. **Counterfactual Check** (v1.1 design § 7.15): 评论 Counterfactual 给出的备选 — 你的判断在跟最强备选比较后是否仍然成立?哪种情况下会翻盘?
5. **Evidence Threshold Check** (v1.1 design § 7.10): 根据 Evidence Auditor 的评级 + reversibility/cost_of_being_wrong, 判断证据是否够用。规则:
   - low-risk reversible:C 级证据可以试
   - medium-risk reversible:B/C 优先
   - high-risk irreversible:必须 A/B
   - 当前问题证据等级 vs 应需等级,直接说够 / 不够 / 临界
6. **Noise Warning** (v1.1 design § 7.19): 指出你在裁决中**主动忽略**了哪些噪声(例:社交媒体喊单、对手 PR 公关、个别 anecdote、用户情绪)。1-3 条。
7. **裁决**: 这个决定 / 这个判断该往哪个方向走?直接说,不 hedge。
8. **Confidence**: 用 frequency framing,不要给数字百分比。 例如"在类似的 10 个案子里,这个判断会对 7 次"或"我对这个论证的信心是 medium — 因为 X 还没证伪"。 取值 low / medium / high。
9. **一行下注 (one_line_bet)**: 用一句话,把判断变成一个可被未来检验的赌注。例如"我赌 3 个月内 X 会发生,如果不发生我错了"。
10. **Biggest Risk**: 这个判断最可能错在哪里?一句话。
11. **24h First Domino**: 未来 24 小时里,用户能做的最小成本动作是什么?这个动作必须能验证某个关键假设。给出 {action, cost, validates}。
12. **What would change my mind**: 列出 2-3 个具体的、可观察的信号 — 出现这些信号我就改判。
13. **Validation Signals**: 列出 2-3 个未来可观察的指标,用于 6 周后回头看这个判断对不对。

输出严格 JSON,字段:
{
  "pre_mortem": "失败叙事,2-3 句",
  "time_horizon": {
    "now_3mo": "1 句",
    "near_3_12mo": "1 句",
    "mid_1_3yr": "1 句",
    "long_3_10yr": "1 句"
  },
  "contradictions": ["张力 1", "张力 2", "张力 3"],
  "counterfactual_check": "1-2 句:跟最强备选比较后判断是否仍成立",
  "evidence_threshold": {
    "current_grade": "A | B | C | D | mixed",
    "required_grade": "A | B | C | D",
    "verdict": "sufficient | borderline | insufficient",
    "rationale": "1 句"
  },
  "noise_warning": ["主动忽略的噪声 1", "..."],
  "judgment": "明确的裁决,1-2 句",
  "confidence": "low | medium | high",
  "confidence_framing": "frequency-style 解释,1 句",
  "one_line_bet": "一行下注",
  "biggest_risk": "一句话",
  "first_domino": {"action": "...", "cost": "...", "validates": "..."},
  "what_would_change_my_mind": ["信号 1", "信号 2"],
  "validation_signals": [{"signal": "...", "direction": "up|down|appears", "effect": "..."}],
  "narrative": "一段散文式裁决书,300-500 字,把上面所有点串成一段可读的判决文 — 这一段会作为最终面向用户的决定段落"
}

只输出 JSON。"""


# ─────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator — emits events as it goes
# ─────────────────────────────────────────────────────────────────────────

def _safe_json_load(s: str) -> dict | None:
    """Tolerant JSON extract from LLM output (handles ```json fences, leading text)."""
    if not s:
        return None
    # Strip code fences
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", s, flags=re.DOTALL)
    if m:
        s = m.group(1)
    # Reject valid non-object JSON before attempting prose extraction.
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        pass
    else:
        return parsed if isinstance(parsed, dict) else None
    # Find first {…} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _is_lite_path(intake: dict) -> bool:
    """v0.2: Reversibility gate — short-circuit pipeline for low-stakes Q.

    Triggers when reversibility=high AND cost_of_being_wrong in {low, medium}.
    Skips: Long-Termist opening, Cross-Exam, Red Team.
    Keeps: Intake, Memory, Advocate + Skeptic + Realist openings, Judge.

    Rationale: a "should I switch from Cursor to Claude Code" question doesn't
    need 2-3 minutes of red-team. The full pipeline is meant for hard, hard-
    to-undo decisions; gate spares cost + latency on cheap-to-undo ones.
    """
    rev = (intake.get("reversibility") or "").lower()
    cow = (intake.get("cost_of_being_wrong") or "").lower()
    return rev == "high" and cow in {"low", "medium"}


def _budget_check_or_warn(transcript: DelphiTranscript) -> bool:
    """Return False if budget exhausted (caller should bail). Emits warning event at 80%."""
    used = transcript.total_cost()
    pct = used / BUDGET_USD_LIMIT * 100
    if used >= BUDGET_USD_LIMIT:
        transcript.emit(
            "budget_warning",
            used_usd=used, used_pct=pct,
            action="预算耗尽,提前进入 Judge 阶段并跳过剩余非必需 stage。",
        )
        return False
    if pct >= 80:
        # Soft warning, but allow continuing
        transcript.emit(
            "budget_warning",
            used_usd=used, used_pct=pct,
            action="估算预算用至 80%,仅警告并继续；达到 100% 才跳过可选阶段。",
        )
    return True


def _stage_intake(question: str, transcript: DelphiTranscript) -> dict:
    """Intake: reframe + classify + extract variables. Emits 'intake' event."""
    _stream("intake", "start")
    user = f"用户原问题:{question}\n\n请输出严格 JSON。"
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "intake", INTAKE_SYSTEM, user,
        transcript=transcript, max_tokens=1200, temperature=0.3,
    )
    parsed = _safe_json_load(content) or {}
    fact_boundary = parsed.get("fact_boundary") or {}
    payload = {
        "original": question,
        "reframed": parsed.get("reframed", question),
        "question_type": parsed.get("question_type", "unclear"),
        "reversibility": parsed.get("reversibility", "unknown"),
        "cost_of_being_wrong": parsed.get("cost_of_being_wrong", "unknown"),
        "user_preference": parsed.get("user_preference"),
        # v1.1: full KAU 4-tier boundary (Known/Assumed/Unknown/Speculative)
        "fact_boundary": {
            "known": fact_boundary.get("known") or parsed.get("user_asserted_facts") or [],
            "assumed": fact_boundary.get("assumed") or [],
            "unknown": fact_boundary.get("unknown") or [],
            "speculative": fact_boundary.get("speculative") or [],
        },
        # legacy alias for v0.9 callers
        "user_asserted_facts": (fact_boundary.get("known") or
                                parsed.get("user_asserted_facts") or []),
        "variables": parsed.get("variables") or [],
        # v1.1: time horizon relevance map (per design § 7.4)
        "time_horizon_relevance": parsed.get("time_horizon_relevance") or {},
    }
    transcript.emit(
        "intake",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        **payload,
    )
    _stream("intake", "done",
            type=payload["question_type"],
            rev=payload["reversibility"],
            cost_wrong=payload["cost_of_being_wrong"],
            n_vars=len(payload["variables"]),
            usd=cost)
    return payload


def _stage_memory(question: str, transcript: DelphiTranscript) -> str | None:
    """Long-term memory query — user context from memory adapter (e.g. Honcho).

    No-op if no memory adapter is wired (default). Emits 'memory_view' event
    when the adapter returns content.
    """
    _stream("memory", "start")
    view = _query_memory(question)
    if view:
        transcript.emit("memory_view", content=view)
        _stream("memory", "done", chars=len(view))
    else:
        transcript.emit("skip", what="Memory query",
                        reason="no memory adapter configured (or empty result)")
        _stream("memory", "skipped", reason="no-content")
    return view


_ROLE_META = {
    # role_key: (display_name, icon, system_prompt, max_output_tokens)
    # Note: bumped vs initial draft after first real run truncated Realist
    # mid-sentence at 500. 350-字 prose target ≠ 350 tokens because zh chars
    # vary. Give comfortable headroom (700) — this is a token limit, not a dollar spend guarantee.
    "advocate":         ("Advocate", "🟢", ADVOCATE_SYSTEM, 700),
    "skeptic":          ("Skeptic", "🔴", SKEPTIC_SYSTEM, 700),
    "realist":          ("Realist", "⚙️", REALIST_SYSTEM, 700),
    "long_termist":     ("Long-Termist", "🌳", LONG_TERMIST_SYSTEM, 600),
    # v0.3 add: Game Theorist — covers "who else is a player here?". Skipped
    # on lite-path (most lite Q's are pure personal preference).
    "game_theorist":    ("Game Theorist", "♟️", GAME_THEORIST_SYSTEM, 700),
    # v1.1 add: Black Swan Scout + Analogist (per design § 7.7)
    # Skipped on lite path (these matter for high-stakes / strategy questions).
    "black_swan":       ("Black Swan Scout", "🦢", BLACK_SWAN_SCOUT_SYSTEM, 700),
    "analogist":        ("Analogist", "📚", ANALOGIST_SYSTEM, 700),
}


def _build_role_user_msg(intake: dict, memory_view: str | None) -> str:
    """Shared user message for all 4 opening-round roles."""
    parts = [
        f"**问题(已 reframe)**:{intake['reframed']}",
        f"**类型**:{intake['question_type']}",
        f"**可逆性**:{intake['reversibility']}",
        f"**错的代价**:{intake['cost_of_being_wrong']}",
    ]
    if intake.get("user_preference"):
        parts.append(f"**用户隐含倾向**:{intake['user_preference']}")
    vars_ = intake.get("variables") or []
    if vars_:
        parts.append("**关键变量**:")
        for v in vars_:
            parts.append(
                f"- {v.get('name')} (impact={v.get('impact')}, "
                f"uncertainty={v.get('uncertainty')}): {v.get('description', '')}"
            )
    if memory_view:
        parts.append("")
        parts.append(f"**长期记忆层提供的用户视角(参考但不必采纳)**:\n{memory_view}")
    parts.append("")
    parts.append("请按你的角色 system prompt 给出发言。")
    return "\n".join(parts)


def _stage_opening_round(intake: dict, memory_view: str | None,
                         transcript: DelphiTranscript) -> dict[str, str]:
    """Opening roles speak sequentially; lite mode uses three roles.

    Returns dict role → content for downstream cross-exam / judge consumption.
    """
    user_msg = _build_role_user_msg(intake, memory_view)
    openings: dict[str, str] = {}
    # v1.1: lite path skips heavy / strategic roles (kept Advocate / Skeptic / Realist only)
    LITE_SKIP = {"long_termist", "game_theorist", "black_swan", "analogist"}
    for role_key, (role_name, icon, system, max_tok) in _ROLE_META.items():
        if role_key in LITE_SKIP and _is_lite_path(intake):
            transcript.emit("skip", what=f"{role_name} opening",
                            reason=f"reversibility={intake['reversibility']}, "
                                   f"cost_of_being_wrong={intake['cost_of_being_wrong']}, "
                                   "lite-path")
            _stream(role_key, "skipped", reason="lite-path")
            continue
        if not _budget_check_or_warn(transcript):
            transcript.emit("skip", what=f"{role_name} opening",
                            reason="预算上限,跳过")
            _stream(role_key, "skipped", reason="budget")
            continue
        _stream(role_key, "start")
        content, ti, to, cost, prov, mdl = _call_llm_for_role(
            role_key, system, user_msg,
            transcript=transcript, max_tokens=max_tok, temperature=0.7,
        )
        openings[role_key] = content
        transcript.emit(
            "role_opening",
            tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
            role=role_name, icon=icon, content=content,
        )
        _stream(role_key, "done", tokens_out=to, usd=cost)
    return openings


def _stage_cross_exam(intake: dict, openings: dict[str, str],
                      transcript: DelphiTranscript) -> str | None:
    """One round of Skeptic-style cross-exam against Advocate."""
    # v0.2: reversibility gate — skip cross-exam on lite path
    if _is_lite_path(intake):
        transcript.emit("skip", what="Cross-exam",
                        reason=f"lite-path (rev={intake['reversibility']}, "
                               f"cost={intake['cost_of_being_wrong']})")
        _stream("cross_exam", "skipped", reason="lite-path")
        return None
    if not _budget_check_or_warn(transcript):
        transcript.emit("skip", what="Cross-exam", reason="预算耗尽")
        _stream("cross_exam", "skipped", reason="budget")
        return None
    advocate = openings.get("advocate")
    if not advocate:
        transcript.emit("skip", what="Cross-exam",
                        reason="Advocate 开篇缺失,无法质询")
        _stream("cross_exam", "skipped", reason="no-advocate")
        return None
    _stream("cross_exam", "start")
    user_msg = (
        f"**问题**:{intake['reframed']}\n\n"
        f"**Advocate 开篇陈述**:\n{advocate}\n\n"
        f"请扮演 Skeptic,提出 1-2 个最致命追问。"
    )
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "cross_exam", CROSS_EXAM_SYSTEM, user_msg,
        transcript=transcript, max_tokens=400, temperature=0.6,
    )
    transcript.emit(
        "cross_exam",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        attacker="Skeptic", target="Advocate", round=1, content=content,
    )
    _stream("cross_exam", "done", usd=cost)
    return content


def _stage_red_team(intake: dict, memory_view: str | None, openings: dict[str, str],
                    transcript: DelphiTranscript, *,
                    bias: dict | None = None) -> str | None:
    """Red Team attacks the user, not the question.

    v0.4: if adaptive bias is non-zero, prepend a "boost" instruction so Red
    Team uses bias direction as starting attack vector.
    """
    # v0.2: reversibility gate — skip Red Team on lite path
    if _is_lite_path(intake):
        transcript.emit("skip", what="Red Team",
                        reason=f"lite-path (rev={intake['reversibility']}, "
                               f"cost={intake['cost_of_being_wrong']})")
        _stream("red_team", "skipped", reason="lite-path")
        return None
    if not _budget_check_or_warn(transcript):
        transcript.emit("skip", what="Red Team", reason="预算耗尽")
        _stream("red_team", "skipped", reason="budget")
        return None
    _stream("red_team", "start")
    parts = [
        f"**用户原问题**:{intake['original']}",
        f"**Reframed**:{intake['reframed']}",
    ]
    if intake.get("user_preference"):
        parts.append(f"**用户隐含倾向**:{intake['user_preference']}")
    if memory_view:
        parts.append(f"\n**长期记忆视角**:\n{memory_view}")
    advocate_excerpt = (openings.get("advocate") or "")[:600]
    if advocate_excerpt:
        parts.append(f"\n**用户/Advocate 的论证(前 600 字)**:\n{advocate_excerpt}")
    if bias:
        parts.append(
            f"\n**⚠️ Adaptive Bias 警告**:用户在话题 `{bias['strongest_topic']}`"
            f" 上历史表现 **{bias['magnitude']}{bias['direction']}** "
            f"(score={bias['strongest_score']:+.2f})。请把这个 bias 当作首要"
            f"攻击向量 — 假设用户的当前论证 70% 概率是 motivated reasoning。"
        )
    parts.append("\n请按 Red Team system prompt 给出诊断。")
    user_msg = "\n".join(parts)
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "red_team", RED_TEAM_SYSTEM, user_msg,
        transcript=transcript, max_tokens=450, temperature=0.7,
    )
    transcript.emit(
        "red_team",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        content=content,
    )
    _stream("red_team", "done", usd=cost)
    return content


COUNTERFACTUAL_SYSTEM = """你是 Counterfactual Analyst(反事实分析师)。你的活:把"做这个"和"不做 / 换个做法"放在一起比。

要求:
- 给出 2-3 个真正的备选方案(alternative paths),不是稻草人。每个备选说清:
  · 具体形式(怎么做)
  · 跟当前方案比,优点
  · 跟当前方案比,劣势
- 然后回答两个问题:
  1. **当前方案是因为它真比备选更强,还是只因为它更熟悉/更先想到?**
  2. **是否存在更小、更可逆、可先测试的备选?** (如果有,具体什么样)
- 最后一句:"如果删掉当前方案这条路,用户的最佳备选是 ______ ,理由 ______。"
- 不超过 400 字。
- 不要列"什么都不做"作为唯一备选 — 那是 Skeptic 的活。"""


def _stage_evidence_audit(intake: dict, openings: dict[str, str],
                           cross_exam: str | None, red_team: str | None,
                           transcript: DelphiTranscript) -> str | None:
    """v1.1 § 7.9 — grade evidence quality of role outputs A/B/C/D."""
    if _is_lite_path(intake):
        transcript.emit("skip", what="Evidence Auditor",
                        reason="lite-path")
        _stream("evidence_audit", "skipped", reason="lite-path")
        return None
    if not _budget_check_or_warn(transcript):
        _stream("evidence_audit", "skipped", reason="budget")
        return None
    if not openings:
        _stream("evidence_audit", "skipped", reason="no-openings")
        return None
    _stream("evidence_audit", "start")
    parts = [f"**问题**:{intake['reframed']}", "", "---", ""]
    for role_key, (role_name, icon, _sys, _tok) in _ROLE_META.items():
        c = openings.get(role_key)
        if c:
            parts.append(f"### {icon} {role_name}\n{c}\n")
    if cross_exam:
        parts.append(f"### 交叉质询\n{cross_exam}\n")
    if red_team:
        parts.append(f"### Red Team\n{red_team}\n")
    parts.append("\n请按 Evidence Auditor system prompt 评级。")
    user_msg = "\n".join(parts)
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "evidence_audit", EVIDENCE_AUDITOR_SYSTEM, user_msg,
        transcript=transcript, max_tokens=900, temperature=0.3,
    )
    transcript.emit(
        "evidence_audit",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        content=content,
    )
    _stream("evidence_audit", "done", usd=cost)
    return content


def _stage_counterfactual(intake: dict, memory_view: str | None,
                          openings: dict[str, str],
                          transcript: DelphiTranscript) -> str | None:
    """v1.1 § 7.15 — generate 2-3 real alternatives + comparison."""
    if _is_lite_path(intake):
        transcript.emit("skip", what="Counterfactual Baseline",
                        reason="lite-path")
        _stream("counterfactual", "skipped", reason="lite-path")
        return None
    if not _budget_check_or_warn(transcript):
        _stream("counterfactual", "skipped", reason="budget")
        return None
    _stream("counterfactual", "start")
    parts = [
        f"**问题(reframed)**:{intake['reframed']}",
        f"**类型**:{intake['question_type']}",
        f"**用户隐含倾向**:{intake.get('user_preference') or '_(无)_'}",
    ]
    realist = openings.get("realist")
    if realist:
        parts.append(f"\n**Realist 已经识别的操作约束**(用作备选可行性参考):\n{realist[:600]}")
    if memory_view:
        parts.append(f"\n**长期记忆视角**:\n{memory_view[:500]}")
    parts.append("\n请按 Counterfactual Analyst system prompt 给出备选方案。")
    user_msg = "\n".join(parts)
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "counterfactual", COUNTERFACTUAL_SYSTEM, user_msg,
        transcript=transcript, max_tokens=700, temperature=0.6,
    )
    transcript.emit(
        "counterfactual",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        content=content,
    )
    _stream("counterfactual", "done", usd=cost)
    return content


def _stage_judge(intake: dict, memory_view: str | None,
                 openings: dict[str, str],
                 cross_exam: str | None, red_team: str | None,
                 transcript: DelphiTranscript, *,
                 bias: dict | None = None,
                 past: list[dict] | None = None,
                 dissent: dict | None = None,
                 evidence_audit: str | None = None,
                 counterfactual: str | None = None) -> dict | None:
    """Judge synthesizes everything; Pre-Mortem inline; produces brief JSON.

    v0.4: bias signal is injected as an explicit "discount this judgment by X"
    instruction so Judge doesn't just rubber-stamp Advocate.
    v0.5: past ledger summaries are injected as continuity check — Judge must
    address consistency or explicit divergence from past judgment.
    """
    parts = [
        f"**问题**:{intake['reframed']}",
        f"**类型**:{intake['question_type']} · 可逆性:{intake['reversibility']}"
        f" · 错的代价:{intake['cost_of_being_wrong']}",
    ]
    # v1.1: KAU 4-tier boundary (Known/Assumed/Unknown/Speculative) for full grounding
    fb = intake.get("fact_boundary") or {}
    known = fb.get("known") or []
    assumed = fb.get("assumed") or []
    unknown = fb.get("unknown") or []
    speculative = fb.get("speculative") or []
    if known or assumed or unknown or speculative:
        parts.append("\n**📋 KAU 事实边界(v1.1 § 7.11)**:")
        if known:
            parts.append("**✅ KNOWN(用户明确陈述,可放心引用)**:")
            for f in known:
                parts.append(f"- {f}")
        if assumed:
            parts.append("\n**⚠️ ASSUMED(合理假设但未验证 — 引用时标 [假设])**:")
            for f in assumed:
                parts.append(f"- {f}")
        if unknown:
            parts.append("\n**❓ UNKNOWN(对判断重要但缺失 — 在 narrative 里点出影响)**:")
            for f in unknown:
                parts.append(f"- {f}")
        if speculative:
            parts.append("\n**🚫 SPECULATIVE(纯推测 — 不要 act on)**:")
            for f in speculative:
                parts.append(f"- {f}")
    else:
        parts.append("\n**⚠️ 用户没有给出具体事实陈述。** 你的裁决必须停留在抽象层面,"
                     "不能编造细节。所有具体数字 / 版本 / 对手动作的引用必须用 '如果...则...' 条件式。")
    # v1.1: time horizon hint from intake
    th = intake.get("time_horizon_relevance") or {}
    if th:
        parts.append("\n**⏱ Time Horizon 相关性(intake 预判,你 judge 时给出每个窗口的结论)**:")
        for k, v in th.items():
            parts.append(f"- {k}: {v}")
    if memory_view:
        parts.append(f"\n**长期记忆视角**:\n{memory_view}")
    if bias:
        parts.append(
            f"\n**⚠️ Adaptive Bias 警告**:用户在话题 `{bias['strongest_topic']}` "
            f"上历史表现 **{bias['magnitude']}{bias['direction']}** "
            f"(score={bias['strongest_score']:+.2f})。请在裁决时显式 discount "
            f"用户隐含倾向方向的 confidence,并在 narrative 里说明你是否怀疑"
            f"这是 motivated reasoning。"
        )
    if past:
        parts.append("")
        parts.append("**📚 过去相关 Ledger 引用**(必须在 narrative 中显式回应:本次"
                     "判断与下方过去判断一致 / 不一致 / 部分一致,以及为什么):")
        for p in past:
            parts.append(f"- `{p['ledger_name']}` (age={p['age_days']}d, "
                         f"confidence={p.get('confidence', '?')}): "
                         f"{p.get('judgment', '?')[:200]}")
    parts.append("")
    parts.append("---")
    for role_key, (role_name, icon, _sys, _tok) in _ROLE_META.items():
        c = openings.get(role_key)
        if c:
            parts.append(f"\n### {icon} {role_name} 开篇\n{c}")
    if cross_exam:
        parts.append(f"\n### 交叉质询(Skeptic → Advocate)\n{cross_exam}")
    if red_team:
        parts.append(f"\n### 🔴 Red Team(攻击用户)\n{red_team}")
    # v1.1: counterfactual + evidence audit feed Judge directly
    if counterfactual:
        parts.append(f"\n### 🔄 Counterfactual Baseline(备选方案对比)\n{counterfactual}")
    if evidence_audit:
        parts.append(f"\n### 📊 Evidence Auditor(证据评级)\n{evidence_audit}")
    if dissent and dissent.get("should_surface"):
        parts.append(
            f"\n### ⚠️ Skeptic 跨模型分歧 (agreement={dissent.get('agreement_level', '?')})\n"
            f"{dissent.get('divergence_summary', '')}\n\n"
            f"**Skeptic 备线**({dissent.get('alt_provider', '?')}/{dissent.get('alt_model', '?')}):\n"
            f"{dissent.get('alt_content', '')[:1000]}\n\n"
            f"**你必须在 narrative 显式回应这个分歧。**"
        )
    parts.append("\n---")
    parts.append("\n请按 Judge system prompt 输出严格 JSON。")
    user_msg = "\n".join(parts)
    _stream("judge", "start", ctx_chars=len(user_msg))
    # Judge is the most expensive call to lose: long context, heavy reasoning,
    # 3k+ output for v1.1 (added 4 directives + 4 brief fields). Bump timeout
    # to 240s and retry once on timeout.
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "judge", JUDGE_SYSTEM, user_msg,
        transcript=transcript, max_tokens=3500, temperature=0.4,
        timeout=240.0, retry_on_timeout=True,
    )
    parsed = _safe_json_load(content)
    # v0.2: JSON repair retry — Judge is too valuable to lose to JSON syntax.
    repair_cost = 0.0
    if not parsed and content and not content.startswith("_(LLM call failed"):
        _stream("judge", "json-fail-retry-repair", chars=len(content))
        repair_user = (
            "Your previous output below was supposed to be valid JSON matching "
            "the schema in your system prompt, but it didn't parse. Please "
            "re-emit ONLY the JSON object (no fences, no preamble), preserving "
            "all your original judgments and content. If a field was missing "
            "in the original, fill it in based on what you wrote.\n\n"
            "---\n"
            f"{content}\n"
            "---\n\n"
            "Output only the JSON object now."
        )
        repair_content, rti, rto, rcost, _, _ = _call_llm_for_role(
            "judge", JUDGE_SYSTEM, repair_user,
            transcript=transcript, max_tokens=2200, temperature=0.0,
            timeout=120.0,
        )
        repair_cost = rcost
        parsed = _safe_json_load(repair_content)
        # Add the repair call to transcript ledger so cost is tracked
        transcript.emit(
            "judge_repair",
            tokens_in=rti, tokens_out=rto, cost_usd=rcost,
            provider=prov, model=mdl,
            ok=bool(parsed),
        )
        if parsed:
            content = repair_content  # use repaired text for fallback render
    if not parsed:
        # Final fallback: emit raw content so user sees something
        transcript.emit(
            "judge",
            tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
            content=content + "\n\n_(⚠️ Judge 输出未能解析为 JSON,以上为原始文本)_",
        )
        _stream("judge", "fail", reason="json-unparseable-after-repair")
        return None
    transcript.emit(
        "judge",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        content=parsed.get("narrative") or content,
    )
    transcript.brief = parsed
    _stream("judge", "done",
            confidence=parsed.get("confidence", "?"),
            usd=cost + repair_cost)
    return parsed


def _stage_action(brief: dict, transcript: DelphiTranscript) -> None:
    """Promote the action data from brief into a standalone 'action' event."""
    if not brief:
        return
    transcript.emit(
        "action",
        one_line_bet=brief.get("one_line_bet", "?"),
        first_domino=brief.get("first_domino") or {},
        validation_signals=brief.get("validation_signals") or [],
    )


DISSENT_JUDGE_SYSTEM = """你是 dissent 检测员。下面是两个 Skeptic 角色对同一问题给出的反对论证(分别由不同 LLM 跑出)。判断它们是否实质性分歧。

输出严格 JSON:
{
  "agreement_level": "agree | partial | disagree",
  "shared_attacks": ["两边都提出的攻击点 1", "..."],
  "diverging_attacks": ["只一边提出的攻击点 1", "..."],
  "structural_assumption_match": true,
  "divergence_summary": "1-2 句话:他们在哪一点上不一致,这种不一致意味着什么",
  "should_surface": true
}

规则:
- 若两边攻击同一 structural assumption + 给出类似论证 → agree, should_surface=false
- 若两边攻击不同 structural assumption → disagree, should_surface=true
- 部分重叠但有显著分歧 → partial, should_surface=true
只输出 JSON。"""


def _stage_multi_model_skeptic(intake: dict, memory_view: str | None,
                                user_msg_skeptic: str,
                                primary_skeptic_content: str,
                                transcript: DelphiTranscript) -> dict | None:
    """v0.7: dual-model Skeptic dissent check on high-stakes questions.

    Strategy: primary Skeptic ran on Codex (per ROLE_PROVIDER) and we have
    that content. Now run a SECOND Skeptic on DeepSeek for a different
    cognitive style. Compare via dissent-judge LLM. If structural divergence,
    emit a separate event for Judge to weigh.

    Gated to cost_of_being_wrong in {high, catastrophic} to keep typical
    runs cheap.
    """
    if intake.get("cost_of_being_wrong", "").lower() not in {"high", "catastrophic"}:
        _stream("multi_skeptic", "skipped", reason="low/med stakes")
        return None
    if not _budget_check_or_warn(transcript):
        _stream("multi_skeptic", "skipped", reason="budget")
        return None

    _stream("multi_skeptic", "start")
    # Run alt Skeptic on a different (provider, model) than the primary so
    # we get a genuinely different cognitive lens. Pick from env or use a
    # sensible default heuristic. If you want deeper control, set:
    #   DELPHI_SKEPTIC_ALT_PROVIDER, DELPHI_SKEPTIC_ALT_MODEL
    primary_provider, primary_model = ROLE_PROVIDER["skeptic"]
    alt_provider = os.environ.get(
        "DELPHI_SKEPTIC_ALT_PROVIDER", primary_provider)
    # Default heuristic: use a different model from the same provider
    # (e.g. gpt-4o vs gpt-4o-mini); user can override.
    alt_model = os.environ.get(
        "DELPHI_SKEPTIC_ALT_MODEL",
        "gpt-4o" if primary_model == "gpt-4o-mini" else primary_model,
    )
    try:
        resp = ADAPTERS["llm"].call(
            provider=alt_provider, model=alt_model,
            messages=[
                {"role": "system", "content": SKEPTIC_SYSTEM},
                {"role": "user", "content": user_msg_skeptic},
            ],
            max_tokens=700, temperature=0.7, timeout=120.0,
        )
        alt_content = re.sub(r"<think>.*?</think>", "", resp.content,
                             flags=re.DOTALL).strip()
        ti, to = resp.tokens_in, resp.tokens_out
        cost = _estimate_cost(alt_provider, alt_model, ti, to)
    except Exception as e:
        _stream("multi_skeptic", "alt-fail", err=type(e).__name__)
        return None

    # Now ask a dissent-judge to compare
    dissent_user = (
        f"**Skeptic A({primary_provider} / {primary_model})**:\n{primary_skeptic_content}\n\n"
        f"---\n\n"
        f"**Skeptic B({alt_provider} / {alt_model})**:\n{alt_content}\n\n"
        f"判断他们的分歧程度,严格 JSON。"
    )
    judge_content, jti, jto, jcost, jprov, jmdl = _call_llm_for_role(
        "judge", DISSENT_JUDGE_SYSTEM, dissent_user,
        transcript=transcript, max_tokens=800, temperature=0.2,
    )
    parsed = _safe_json_load(judge_content) or {}

    payload = {
        "alt_provider": alt_provider,
        "alt_model": alt_model,
        "alt_content": alt_content,
        "agreement_level": parsed.get("agreement_level", "?"),
        "shared_attacks": parsed.get("shared_attacks") or [],
        "diverging_attacks": parsed.get("diverging_attacks") or [],
        "divergence_summary": parsed.get("divergence_summary", ""),
        "should_surface": parsed.get("should_surface", False),
    }
    transcript.emit(
        "skeptic_dissent",
        tokens_in=ti + jti, tokens_out=to + jto,
        cost_usd=cost + jcost,
        provider=f"{alt_provider}+{jprov}", model=f"{alt_model}+{jmdl}",
        llm_calls=2,
        **payload,
    )
    _stream("multi_skeptic", "done",
            agreement=payload["agreement_level"],
            surface=payload["should_surface"],
            usd=cost + jcost)
    return payload


def _stage_round_2_cross_exam(intake: dict, brief: dict | None,
                              openings: dict[str, str],
                              cross_exam_round_1: str | None,
                              transcript: DelphiTranscript) -> str | None:
    """Supplementary post-verdict review; does not revise the brief or action.

    Logic:
    - Only runs on standard path (not lite) and when Judge produced a brief
      with a substantial pre_mortem field.
    - Skeptic attacks the failure mode named by Judge's Pre-Mortem,
      targeting Advocate again — but now armed with what Judge revealed.
    - Skipped if Pre-Mortem is too short to constitute a "new" attack vector
      or if the Pre-Mortem theme was already covered in round 1 (cheap check).
    """
    if _is_lite_path(intake):
        return None
    if not brief:
        return None
    pre_mortem = (brief.get("pre_mortem") or "").strip()
    if len(pre_mortem) < 50:
        _stream("cross_exam_r2", "skipped", reason="pre-mortem too short")
        return None
    # Cheap heuristic: if pre-mortem core nouns already appear in round 1
    # cross-exam text, skip.
    if cross_exam_round_1:
        # Compare 4-char chunks (zh-aware-ish)
        r1_lower = cross_exam_round_1.lower()
        pm_chunks = re.findall(r".{4}", pre_mortem.lower())
        overlap = sum(1 for c in pm_chunks if c in r1_lower)
        if overlap > len(pm_chunks) * 0.4:
            _stream("cross_exam_r2", "skipped",
                    reason="pre-mortem theme already in round 1")
            return None
    if not _budget_check_or_warn(transcript):
        _stream("cross_exam_r2", "skipped", reason="budget")
        return None
    advocate = openings.get("advocate")
    if not advocate:
        _stream("cross_exam_r2", "skipped", reason="no-advocate")
        return None

    _stream("cross_exam_r2", "start")
    user_msg = (
        f"**问题**:{intake['reframed']}\n\n"
        f"**Advocate 原开篇**:\n{advocate[:800]}\n\n"
        f"**Round 1 你已问过的追问(避免重复)**:\n"
        f"{(cross_exam_round_1 or '_(无)_')[:600]}\n\n"
        f"**⚠️ Judge 在 Pre-Mortem 中暴露的失败叙事**:\n{pre_mortem}\n\n"
        f"现在你拿到了 Judge 的 Pre-Mortem,这是 Round 1 你不知道的新信息。"
        f"针对这个失败路径,提出 1-2 个 Round 2 致命追问 — "
        f"必须是 Round 1 没问过的新角度。"
    )
    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        "cross_exam", CROSS_EXAM_SYSTEM, user_msg,
        transcript=transcript, max_tokens=400, temperature=0.6,
    )
    transcript.emit(
        "cross_exam",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        attacker="Skeptic", target="Advocate", round=2,
        trigger="pre-mortem", content=content,
    )
    _stream("cross_exam_r2", "done", usd=cost)
    return content


def _stage_drilldown(role_key: str, drill_question: str,
                     transcript: DelphiTranscript) -> str | None:
    """v0.6: post-pipeline follow-up to a specific role.

    Gives the role its original system prompt + the full transcript so far +
    the user's pointed question. Use case: "Skeptic, if I had 1 month of
    runway instead of 3, would your attack still hold?"
    """
    role_key = role_key.lower().strip()
    # Map common aliases
    aliases = {
        "long-termist": "long_termist", "longtermist": "long_termist",
        "game-theorist": "game_theorist", "gt": "game_theorist",
        "redteam": "red_team", "red-team": "red_team",
    }
    role_key = aliases.get(role_key, role_key)

    if role_key in _ROLE_META:
        role_name, icon, system, _max_tok = _ROLE_META[role_key]
    elif role_key in {"judge", "red_team"}:
        # Special-case roles outside _ROLE_META
        if role_key == "judge":
            role_name, icon, system = "Judge", "👨‍⚖️", JUDGE_SYSTEM
        else:
            role_name, icon, system = "Red Team", "🔴", RED_TEAM_SYSTEM
    else:
        valid = list(_ROLE_META.keys()) + ["judge", "red_team"]
        transcript.emit("skip", what=f"Drilldown {role_key}",
                        reason=f"unknown role; valid: {valid}")
        _stream("drilldown", "skipped", reason=f"unknown-role:{role_key}")
        return None

    if not _budget_check_or_warn(transcript):
        transcript.emit("skip", what=f"Drilldown {role_name}",
                        reason="预算耗尽")
        _stream("drilldown", "skipped", reason="budget")
        return None

    _stream("drilldown", f"start role={role_key}",
            chars=len(drill_question))

    # Build context: include the role's own opening (if any) + relevant other
    # role outputs + brief, then the user's drill question
    intake = next((e.payload for e in transcript.events if e.type == "intake"), {})
    user_parts = [
        f"**问题**:{intake.get('reframed', '?')}",
        "",
        "**前轮你已经说过的**:",
    ]
    # find this role's opening if present
    own_opening = None
    for ev in transcript.events:
        if ev.type == "role_opening":
            if ev.payload.get("role", "").lower().replace(" ", "_").replace("-", "_") == role_key:
                own_opening = ev.payload.get("content", "")
                break
    if own_opening:
        user_parts.append(own_opening)
    else:
        user_parts.append("_(此角色本次没有开篇,以下是你的首次发言机会)_")
    # include brief (Judge's main verdict) so role has full context
    if transcript.brief:
        user_parts.append("")
        user_parts.append("**当前 Judge 给出的判断**:")
        user_parts.append(f"- 判断:{transcript.brief.get('judgment', '?')}")
        user_parts.append(f"- Confidence:{transcript.brief.get('confidence', '?')}")
        user_parts.append(f"- 一行下注:{transcript.brief.get('one_line_bet', '?')}")
    user_parts.append("")
    user_parts.append(f"**用户对你的追问**:{drill_question}")
    user_parts.append("")
    user_parts.append("请按你原本角色 system prompt 的语气和方法回应。"
                      "如果追问让你改变此前立场,直接说 — 这才是有用的反馈。"
                      "如果追问没让你改变立场,直接说为什么。不超过 350 字。")
    user_msg = "\n".join(user_parts)

    content, ti, to, cost, prov, mdl = _call_llm_for_role(
        role_key, system, user_msg,
        transcript=transcript, max_tokens=600, temperature=0.6,
    )
    transcript.emit(
        "drilldown",
        tokens_in=ti, tokens_out=to, cost_usd=cost, provider=prov, model=mdl,
        role=role_name, icon=icon,
        question=drill_question, content=content,
    )
    _stream("drilldown", "done", role=role_key, usd=cost)
    return content


def run_delphi(question: str, *, dry_run: bool = False,
               drills: list[tuple[str, str]] | None = None) -> DelphiTranscript:
    """Top-level orchestrator.

    drills: list of (role_key, drill_question). Each runs after the main
    pipeline completes (so the role can react to the Judge's verdict).
    """
    global _STREAM_T0
    _STREAM_T0 = time.time()
    transcript = DelphiTranscript(question)

    if dry_run:
        # Render structure preview without LLM calls
        transcript.emit(
            "intake",
            original=question, reframed=f"[DRY-RUN] {question}",
            question_type="unclear", reversibility="unknown",
            cost_of_being_wrong="unknown", user_preference=None,
            variables=[
                {"name": "DRY_RUN_VAR", "impact": "high", "uncertainty": "high",
                 "description": "Dry-run placeholder"},
            ],
        )
        transcript.emit("memory_view", content="_(dry-run, memory skipped)_")
        for role_key, (role_name, icon, _sys, _tok) in _ROLE_META.items():
            transcript.emit("role_opening",
                            role=role_name, icon=icon, provider="dry-run",
                            model="dry-run",
                            content=f"_(dry-run · {role_name} would speak here)_")
        transcript.emit("cross_exam", attacker="Skeptic", target="Advocate",
                        round=1, content="_(dry-run · cross-exam placeholder)_")
        transcript.emit("red_team", content="_(dry-run · red team placeholder)_")
        # v1.1: dry-run placeholders for new stages
        transcript.emit("counterfactual",
                        content="_(dry-run · counterfactual placeholder)_")
        transcript.emit("evidence_audit",
                        content="_(dry-run · evidence audit placeholder)_")
        transcript.brief = {
            "judgment": "[DRY-RUN] no real judgment",
            "confidence": "low",
            "confidence_framing": "dry-run mode",
            "one_line_bet": "[DRY-RUN]",
            "biggest_risk": "[DRY-RUN]",
            "first_domino": {"action": "[DRY-RUN]", "cost": "$0", "validates": "nothing"},
            "what_would_change_my_mind": ["[DRY-RUN]"],
            "validation_signals": [],
            "narrative": "Dry-run — no real LLM calls.",
        }
        transcript.emit("judge", content=transcript.brief["narrative"])
        _stage_action(transcript.brief, transcript)
        return transcript

    # Real pipeline
    _valid_amount(BUDGET_USD_LIMIT, "DELPHI_BUDGET_USD", positive=True)
    _stream("pricing", "estimate-only", basis=_pricing_note())
    _stream("pipeline", "start", chars=len(question))
    intake = _stage_intake(question, transcript)
    if _is_lite_path(intake):
        _stream("path", "lite", reason="high-rev + low/med-cost")
        transcript.emit("skip", what="(meta) Heavy stages",
                        reason="Lite path — Long-Termist, Game Theorist, Cross-Exam, Red Team 全跳过")
    else:
        _stream("path", "standard")
    # v0.4: adaptive bias check (cheap, no LLM call)
    bias = _query_adaptive_bias(intake)
    if bias:
        transcript.emit("adaptive_bias", **bias)
        _stream("adaptive_bias", "found",
                topic=bias["strongest_topic"],
                score=bias["strongest_score"])
    else:
        _stream("adaptive_bias", "no-signal")
    # v0.5: past ledger lookup (cheap, no LLM call, file glob + keyword match)
    past = _query_past_ledgers(intake)
    if past:
        transcript.emit("past_ledgers", matches=past)
        _stream("past_ledgers", "found", n=len(past),
                top_score=past[0]["final_score"] if past else 0)
    else:
        _stream("past_ledgers", "no-match")
    memory_view = _stage_memory(intake["reframed"], transcript)
    openings = _stage_opening_round(intake, memory_view, transcript)
    cross_exam = _stage_cross_exam(intake, openings, transcript)
    red_team = _stage_red_team(intake, memory_view, openings, transcript, bias=bias)
    # v0.7: dual-model Skeptic dissent on high-stakes only
    dissent = None
    if openings.get("skeptic"):
        dissent = _stage_multi_model_skeptic(
            intake, memory_view,
            user_msg_skeptic=_build_role_user_msg(intake, memory_view),
            primary_skeptic_content=openings["skeptic"],
            transcript=transcript,
        )
    # v1.1: Counterfactual + Evidence Auditor — both consume openings/red_team,
    # both feed Judge. Skipped on lite path (budget reasons).
    counterfactual = _stage_counterfactual(intake, memory_view, openings, transcript)
    evidence_audit = _stage_evidence_audit(intake, openings, cross_exam,
                                           red_team, transcript)
    brief = _stage_judge(intake, memory_view, openings, cross_exam, red_team,
                         transcript, bias=bias, past=past, dissent=dissent,
                         counterfactual=counterfactual,
                         evidence_audit=evidence_audit)
    # v0.8: round-2 cross-exam triggered by Judge's Pre-Mortem
    if brief:
        _stage_round_2_cross_exam(intake, brief, openings, cross_exam, transcript)
        _stage_action(brief, transcript)
    # v0.6: post-pipeline drilldowns (run after Judge so role can react to verdict)
    for role_key, drill_q in (drills or []):
        _stage_drilldown(role_key, drill_q, transcript)
    _stream("pipeline", "done",
            usd=transcript.total_cost(),
            calls=transcript.llm_call_count(),
            secs=transcript.latency_seconds())
    return transcript


# ─────────────────────────────────────────────────────────────────────────
# Discord brief push
# ─────────────────────────────────────────────────────────────────────────

def _notify(brief_md: str, ledger_path: Path | None) -> dict:
    """Push decision brief to the active notifier (default no-op).

    Override ADAPTERS["notifier"] for Discord, Slack, etc. The default
    DiscordWebhookNotifierAdapter activates if DELPHI_DISCORD_WEBHOOK_URL
    is set in the environment.
    """
    try:
        return ADAPTERS["notifier"].push(
            brief_md,
            str(ledger_path) if ledger_path else None,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────
# Workflow contract: describe + main
# ─────────────────────────────────────────────────────────────────────────

def describe() -> dict:
    """Return Delphi metadata as a JSON-serializable dict.

    Useful for agent frameworks that want to register Delphi as a tool.
    Print via `python delphi.py --describe`.
    """
    return {
        "name": WORKFLOW_NAME,
        "version": "1.1",
        "license": "MIT",
        "summary": (
            "Delphi Energy: multi-role adversarial judgment pipeline. Takes a "
            "question, runs 7 reasoning roles, cross-exam, red team, "
            "counterfactual, evidence audit, and a model-based Judge with "
            "Pre-Mortem, Time Horizon split, Contradiction Map, Evidence "
            "Threshold check, Noise Warning, Confidence (frequency framing), "
            "One-Line Bet, and 24h First Domino. Outputs a Decision Brief + "
            "full audit trail markdown ledger."
        ),
        "inputs": {
            "question": {"type": "string", "required": True,
                         "desc": "The question to judge (quote it)"},
        },
        "outputs": {
            "ledger": {"type": "file", "path_pattern":
                       str(LEDGER_DIR / "YYYY-MM-DD_<slug>.md")},
            "stdout_brief": {"type": "string"},
            "notifier_push": {"type": "external (Discord/Slack/etc.)"},
        },
        "args": [
            {"flag": "question", "type": "positional",
             "desc": "Question to judge (quote it)"},
            {"flag": "--no-notify", "type": "bool",
             "desc": "Skip notifier push (Discord/etc), brief still goes to stdout/ledger"},
            {"flag": "--no-vault", "type": "bool",
             "desc": "Skip ledger write, stdout only"},
            {"flag": "--no-stream", "type": "bool",
             "desc": "Suppress per-stage progress lines on stderr"},
            {"flag": "--dry-run", "type": "bool",
             "desc": "Render structure preview without LLM calls"},
            {"flag": "--describe", "type": "bool",
             "desc": "Print this metadata as JSON and exit"},
            {"flag": "--drill ROLE:QUESTION", "type": "list",
             "desc": "After pipeline, run a follow-up question to one role. "
                     "Roles: advocate, skeptic, realist, long_termist, "
                     "game_theorist, black_swan, analogist, red_team, judge. "
                     "Repeatable."},
        ],
        "cost_estimate": {
            "basis": _pricing_note(),
            "soft_budget_usd": BUDGET_USD_LIMIT,
            "benchmarked": False,
            "baseline_calls_without_retries_or_optional_review": {"lite": 5, "standard": 13, "high_stakes": 15},
        },
        "adapters": {
            "llm": type(ADAPTERS["llm"]).__name__,
            "memory": type(ADAPTERS["memory"]).__name__,
            "bias": type(ADAPTERS["bias"]).__name__,
            "notifier": type(ADAPTERS["notifier"]).__name__,
        },
        "design_doc": "https://github.com/j1374483500-dot/delphi-energy/blob/main/DESIGN.md",
    }


def _print_brief_to_stdout(transcript: DelphiTranscript) -> None:
    print(_pricing_note())
    print("\n" + "=" * 60)
    print(transcript.render_brief_only())
    print("=" * 60)
    print(f"\n💰 Estimated cost: ${transcript.total_cost():.4f}"
          f" · ⏱ {transcript.latency_seconds():.1f}s"
          f" · 📞 {transcript.llm_call_count()} LLM calls")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="delphi",
        description="Delphi Energy — sequential adversarial judgment prototype",
    )
    parser.add_argument("question", nargs="?",
                        help="Question to judge (quoted)")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip notifier push (Discord/etc.), still write ledger + stdout")
    parser.add_argument("--no-vault", action="store_true",
                        help="Skip ledger file write, stdout only")
    parser.add_argument("--no-stream", action="store_true",
                        help="Suppress per-stage progress lines on stderr")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render structure preview, no LLM calls")
    parser.add_argument("--describe", action="store_true",
                        help="Print Delphi metadata as JSON, exit")
    parser.add_argument("--drill", action="append", default=[],
                        metavar="ROLE:QUESTION",
                        help="After pipeline finishes, run a follow-up question "
                             "to one role. Format: 'skeptic:if runway were 1mo?'. "
                             "Repeatable. Valid roles: advocate, skeptic, realist, "
                             "long_termist, game_theorist, black_swan, analogist, "
                             "red_team, judge.")
    args = parser.parse_args()

    # Parse drill specs
    drills: list[tuple[str, str]] = []
    for spec in args.drill or []:
        if ":" not in spec:
            parser.error(f"--drill must be ROLE:QUESTION, got: {spec!r}")
        role_key, _, q = spec.partition(":")
        drills.append((role_key.strip(), q.strip()))

    # Wire stream toggle
    global _STREAM_ENABLED
    _STREAM_ENABLED = not args.no_stream

    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0

    if not args.question:
        parser.error("question is required (or use --describe / --dry-run)")

    transcript = run_delphi(args.question, dry_run=args.dry_run, drills=drills)
    exit_code = 0

    if args.dry_run:
        _print_brief_to_stdout(transcript)
        return 0

    # Save ledger
    ledger_path: Path | None = None
    if not args.no_vault:
        try:
            slug = _slugify(transcript._reframed_or_raw())
            ledger_path = transcript.save_to_vault(slug)
        except Exception as e:
            print(f"[delphi] ledger write failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            exit_code = 1

    # Notifier (Discord webhook by default if env set, otherwise no-op)
    if not args.no_notify:
        push = _notify(transcript.render_brief_only(), ledger_path)
        if not push.get("ok") and push.get("error"):
            # Don't fail the run on notifier error — just warn
            print(f"[delphi] notifier push failed: {push.get('error')}",
                  file=sys.stderr)

    _print_brief_to_stdout(transcript)
    if ledger_path:
        print(f"\n📝 Full transcript: {ledger_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
