"""multi_provider.py — route different roles to different LLM providers.

The default OpenAILLMAdapter sends every role to the same OpenAI-compatible
endpoint. For better diversity / cost control, you can override the adapter
to route by (provider, model) tuple.

Use case: Skeptic + Red Team benefit from a "less agreeable" model than
Judge / Advocate. This example routes adversarial roles to Anthropic (Claude
Sonnet) while keeping synthesis on OpenAI (GPT-4o).

Setup:
    pip install openai anthropic
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/multi_provider.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from delphi import (  # noqa: E402
    ADAPTERS, LLMResponse, ROLE_PROVIDER, run_delphi,
)


class MultiProviderLLMAdapter:
    """Dispatches based on the `provider` arg the pipeline passes in.

    Lazy-initializes each provider's client on first use.
    """
    def __init__(self):
        self._openai = None
        self._anthropic = None

    def _ensure_openai(self):
        if self._openai is not None:
            return
        from openai import OpenAI  # type: ignore
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _ensure_anthropic(self):
        if self._anthropic is not None:
            return
        from anthropic import Anthropic  # type: ignore
        self._anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(self, *, provider: str, model: str, messages: list[dict],
             max_tokens: int, temperature: float, timeout: float
             ) -> LLMResponse:
        if provider == "openai":
            self._ensure_openai()
            r = self._openai.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                timeout=timeout,
            )
            content = r.choices[0].message.content or ""
            usage = getattr(r, "usage", None)
            ti = getattr(usage, "prompt_tokens", 0) if usage else 0
            to = getattr(usage, "completion_tokens", 0) if usage else 0
            return LLMResponse(content=content, tokens_in=ti, tokens_out=to)

        if provider == "anthropic":
            self._ensure_anthropic()
            # Convert OpenAI-style messages → Anthropic
            system_msg = ""
            user_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_msgs.append({"role": m["role"], "content": m["content"]})
            r = self._anthropic.messages.create(
                model=model,
                system=system_msg,
                messages=user_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            content = r.content[0].text if r.content else ""
            return LLMResponse(
                content=content,
                tokens_in=r.usage.input_tokens,
                tokens_out=r.usage.output_tokens,
            )

        raise RuntimeError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # Wire multi-provider adapter
    ADAPTERS["llm"] = MultiProviderLLMAdapter()

    # Re-route adversarial roles to Anthropic (less agreeable than GPT-4o)
    ROLE_PROVIDER.update({
        "skeptic":       ("anthropic", "claude-sonnet-4-5"),
        "red_team":      ("anthropic", "claude-sonnet-4-5"),
        "game_theorist": ("anthropic", "claude-sonnet-4-5"),
        "black_swan":    ("anthropic", "claude-sonnet-4-5"),
        "cross_exam":    ("anthropic", "claude-sonnet-4-5"),
        # Judge stays on OpenAI for synthesis
        "judge":         ("openai", "gpt-4o"),
        # Cheap roles stay on mini
        "intake":        ("openai", "gpt-4o-mini"),
    })

    transcript = run_delphi(
        "Should our team adopt Rust for the next backend service we build?"
    )
    print(transcript.render_brief_only())
