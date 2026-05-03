"""openclaw_adapter.py — template for integrating Delphi as an OpenClaw skill/tool.

NOTE: this file is a template. Replace the OpenClaw-specific imports and
register-tool calls with the actual OpenClaw API. The pattern shown here is
the canonical "expose Delphi as a callable tool" approach that should map
cleanly to most agent frameworks.

What this file does:
    1. Wires Delphi's adapter slots so the LLM call goes through OpenClaw's
       LLM provider (instead of Delphi's default OpenAI client).
    2. Exposes a `delphi_judgment(question)` tool function that the OpenClaw
       agent can invoke.
    3. Returns the structured brief + path to the full audit ledger.

Integration sketch:
    from openclaw import register_tool, llm_call  # actual API may differ

    register_tool(
        name="delphi_judgment",
        description="Run Delphi Energy on an important question",
        fn=delphi_judgment,
    )
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from delphi import ADAPTERS, LLMResponse, run_delphi  # noqa: E402


class OpenClawLLMAdapter:
    """Routes Delphi LLM calls through OpenClaw's LLM provider.

    Replace the body of `call()` with however OpenClaw exposes LLM calls
    (e.g. `openclaw.llm.complete(...)`, `agent.chat(...)`, etc.).
    """
    def call(self, *, provider: str, model: str, messages: list[dict],
             max_tokens: int, temperature: float, timeout: float
             ) -> LLMResponse:
        # === REPLACE THIS BLOCK with OpenClaw's actual LLM call ==========
        # Pseudocode:
        #
        #   from openclaw import llm
        #   resp = llm.complete(
        #       model=model, messages=messages, max_tokens=max_tokens,
        #       temperature=temperature, timeout=timeout,
        #   )
        #   return LLMResponse(
        #       content=resp.text,
        #       tokens_in=resp.usage.input_tokens,
        #       tokens_out=resp.usage.output_tokens,
        #   )
        # ===================================================================
        raise NotImplementedError(
            "Replace OpenClawLLMAdapter.call body with OpenClaw's LLM API"
        )


def delphi_judgment(question: str, *, drills: list[tuple[str, str]] | None = None
                    ) -> dict:
    """Tool function — call from OpenClaw agent.

    Args:
        question: the question to judge
        drills:   optional [(role, follow_up_q), ...] for post-pipeline drilldowns

    Returns:
        {
          "brief_md":     Decision Brief markdown (Discord-friendly),
          "transcript":   full audit trail markdown,
          "ledger_path":  path to written ledger file (or None if vault disabled),
          "judgment":     the verdict string,
          "confidence":   "low" | "medium" | "high",
          "first_domino": {"action": ..., "cost": ..., "validates": ...},
          "cost_usd":     float,
          "wall_seconds": float,
          "llm_calls":    int,
        }
    """
    # Wire the OpenClaw LLM adapter (overrides default OpenAI)
    ADAPTERS["llm"] = OpenClawLLMAdapter()
    # If OpenClaw provides memory / preference layers, wire those too:
    # ADAPTERS["memory"] = OpenClawMemoryAdapter()
    # ADAPTERS["bias"] = OpenClawPreferenceAdapter()

    transcript = run_delphi(question, drills=drills or [])
    brief = transcript.brief or {}

    # Save ledger
    from delphi import _slugify  # noqa: E402
    try:
        slug = _slugify(transcript._reframed_or_raw())
        ledger_path = str(transcript.save_to_vault(slug))
    except Exception:
        ledger_path = None

    return {
        "brief_md":     transcript.render_brief_only(),
        "transcript":   transcript.render(),
        "ledger_path":  ledger_path,
        "judgment":     brief.get("judgment", ""),
        "confidence":   brief.get("confidence", ""),
        "first_domino": brief.get("first_domino") or {},
        "cost_usd":     round(transcript.total_cost(), 4),
        "wall_seconds": round(transcript.latency_seconds(), 1),
        "llm_calls":    len([e for e in transcript.events if e.cost_usd]),
    }


if __name__ == "__main__":
    # Smoke test (will fail with NotImplementedError until you fill in the
    # OpenClaw LLM call above)
    result = delphi_judgment("Should our team adopt Rust for the next service?")
    print(result["brief_md"])
