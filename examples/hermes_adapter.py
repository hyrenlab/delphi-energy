"""hermes_adapter.py — integrate Delphi as a Hermes skill.

Hermes is the framework Delphi was originally built for. This file shows the
adapter wiring that makes the OS Delphi version play nicely with a running
Hermes instance:

  - LLM calls go through Hermes' `agent.auxiliary_client.call_llm` (which has
    multi-provider routing, budget tracking, failover already built in).
  - Memory comes from Honcho dialectic at the local Hermes Honcho server.
  - Notifier uses Hermes' Discord helper.
  - Ledger writes to the Obsidian vault at the path Hermes expects.

Setup:
    # From a machine with Hermes installed
    export PYTHONPATH=$HOME/.hermes/hermes-agent:$HOME/.hermes/scripts
    python examples/hermes_adapter.py

This is exactly the wiring used in the original (non-OS) Delphi inside
Hermes. Use it as a reference for how to write deep integrations.
"""
import os
import re
import sys
from pathlib import Path

# Hermes paths (adjust if your Hermes lives elsewhere)
HERMES_HOME = Path.home() / ".hermes"
sys.path.insert(0, str(HERMES_HOME / "hermes-agent"))
sys.path.insert(0, str(HERMES_HOME / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from delphi import ADAPTERS, LLMResponse, run_delphi  # noqa: E402


class HermesLLMAdapter:
    """Routes through Hermes' multi-provider auxiliary_client.

    Hermes already does provider routing, budget tracking, and failover at
    the call site; we just translate the response shape.
    """
    def call(self, *, provider: str, model: str, messages: list[dict],
             max_tokens: int, temperature: float, timeout: float
             ) -> LLMResponse:
        from agent.auxiliary_client import call_llm  # type: ignore
        resp = call_llm(
            provider=provider, model=model,
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


class HermesHonchoMemoryAdapter:
    """Honcho dialectic at the local Hermes-bundled Honcho server."""
    def __init__(self, peer_id: str = "user"):
        self.peer_id = peer_id

    def query(self, question: str) -> str | None:
        try:
            from honcho import Honcho  # type: ignore
        except ImportError:
            return None
        try:
            client = Honcho(
                api_key="local-noauth",
                base_url="http://localhost:8000",
                workspace_id="hermes",
                timeout=60.0,
            )
            prompt = (
                f"Based on the user's last 30 days, is the question below "
                f"consistent with their long-term direction? Where might they "
                f"be self-deceiving? 2-3 sentences, direct.\n\nQuestion: {question}"
            )
            ans = client.peer(self.peer_id).chat(prompt, reasoning_level="low")
            if not ans:
                return None
            cleaned = re.sub(r"<think>.*?</think>", "", str(ans),
                             flags=re.DOTALL).strip()
            return cleaned[:1500] if cleaned else None
        except Exception as e:
            return f"_(Honcho error: {e})_"


class HermesAdaptiveBiasAdapter:
    """Hermes' adaptive.py preference layer — surfaces historical user bias
    on the question's topic so Judge / Red Team can amplify."""
    def query(self, intake: dict) -> dict | None:
        try:
            from adaptive import weight_for, get_map  # type: ignore
        except ImportError:
            return None
        m = get_map() or {}
        topic_map = (m.get("weights") or {}).get("topic") or {}
        if not topic_map:
            return None

        candidates: set[str] = set()
        if intake.get("question_type"):
            candidates.add(intake["question_type"])
        for v in intake.get("variables") or []:
            if v.get("name"):
                candidates.add(v["name"])
        reframed = intake.get("reframed") or ""
        for tok in re.split(r"[\s,，。.!?！？:、/(){}\[\]<>]+", reframed):
            tok = tok.strip()
            if 2 <= len(tok) <= 30:
                candidates.add(tok)

        best_topic, best_score, matches = None, 0.0, []
        for cand in candidates:
            if cand not in topic_map:
                continue
            w = weight_for("topic", cand)
            if w == 0.0:
                continue
            matches.append({"topic": cand, "score": w})
            if abs(w) > abs(best_score):
                best_score, best_topic = w, cand
        if not matches:
            return None
        return {
            "n_topics_checked": len(candidates),
            "n_topics_matched": len(matches),
            "strongest_topic": best_topic,
            "strongest_score": best_score,
            "matches": matches,
            "direction": "正向偏好" if best_score > 0 else "负向偏好",
            "magnitude": ("强" if abs(best_score) > 0.5
                          else "中" if abs(best_score) > 0.2 else "弱"),
        }


class HermesDiscordNotifierAdapter:
    """Reuse Hermes' Discord helper for the brief push."""
    def push(self, brief_md: str, ledger_path: str | None) -> dict:
        try:
            from agent.discord_notifier import send_discord_message  # type: ignore
        except Exception:
            try:
                from discord_helpers import post_to_discord as send_discord_message  # type: ignore
            except Exception as e:
                return {"ok": False, "error": f"no Discord helper: {e}"}
        msg = brief_md
        if ledger_path:
            msg += f"\n\n📝 Full transcript: `{Path(ledger_path).name}`"
        try:
            send_discord_message(msg)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def setup_hermes_adapters(peer_id: str = "user") -> None:
    """Wire all Hermes-flavored adapters in one call."""
    ADAPTERS["llm"] = HermesLLMAdapter()
    ADAPTERS["memory"] = HermesHonchoMemoryAdapter(peer_id=peer_id)
    ADAPTERS["bias"] = HermesAdaptiveBiasAdapter()
    ADAPTERS["notifier"] = HermesDiscordNotifierAdapter()


if __name__ == "__main__":
    setup_hermes_adapters()
    # Re-route ledger to the user's Hermes vault path if desired
    # (default OS path is ./ledger/; uncomment to override)
    # import delphi
    # delphi.LEDGER_DIR = Path.home() / "Vault" / "Delphi Ledger"

    transcript = run_delphi(
        "Should I push Phase 4 to 100% tonight or pause to validate stability?"
    )
    print(transcript.render_brief_only())
