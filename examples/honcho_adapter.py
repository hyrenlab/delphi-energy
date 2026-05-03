"""honcho_adapter.py — wire Honcho (https://honcho.dev) as the memory adapter.

Honcho is an OS personal-memory layer for AI agents. It stores conversations
and exposes a "dialectic" query that returns user-grounded long-term context.

Setup:
    pip install honcho
    # Run a local Honcho server (see honcho docs)
    export DELPHI_HONCHO_BASE_URL=http://localhost:8000
    export DELPHI_HONCHO_PEER_ID=your_user_id   # whatever you call yourself
    OPENAI_API_KEY=sk-... python examples/honcho_adapter.py

Effect: the "Long-term Memory" section of every Delphi run will show what
Honcho thinks about whether the question aligns with your long-term direction
and where you might be self-deceiving.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from delphi import ADAPTERS, run_delphi  # noqa: E402


class HonchoMemoryAdapter:
    """Memory adapter backed by a local Honcho server.

    Honcho's `dialectic` capability returns a free-text answer to a prompt
    grounded in the user's long-term conversation history.
    """

    def __init__(self, *, base_url: str | None = None,
                 peer_id: str | None = None,
                 api_key: str = "local-noauth",
                 workspace_id: str = "default",
                 timeout: float = 60.0):
        self.base_url = base_url or os.environ.get(
            "DELPHI_HONCHO_BASE_URL", "http://localhost:8000")
        self.peer_id = peer_id or os.environ.get(
            "DELPHI_HONCHO_PEER_ID", "user")
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.timeout = timeout

    def query(self, question: str) -> str | None:
        try:
            from honcho import Honcho  # type: ignore
        except ImportError:
            return None
        try:
            client = Honcho(
                api_key=self.api_key,
                base_url=self.base_url,
                workspace_id=self.workspace_id,
                timeout=self.timeout,
            )
            prompt = (
                f"Based on the user's last 30 days of conversation and notes, "
                f"is the question below consistent with their long-term "
                f"direction? Where might they be self-deceiving? Reply in "
                f"2-3 sentences, direct, no hedging.\n\nQuestion: {question}"
            )
            ans = client.peer(self.peer_id).chat(prompt, reasoning_level="low")
            if not ans:
                return None
            cleaned = re.sub(r"<think>.*?</think>", "", str(ans),
                             flags=re.DOTALL).strip()
            return cleaned[:1500] if cleaned else None
        except Exception as e:
            return f"_(Honcho error: {e})_"


if __name__ == "__main__":
    # Wire the adapter
    ADAPTERS["memory"] = HonchoMemoryAdapter()

    transcript = run_delphi(
        "Should I keep building this side project for another 6 months, "
        "or call it a learning experience and refocus?"
    )
    print(transcript.render_brief_only())
