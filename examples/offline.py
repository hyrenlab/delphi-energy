"""Run the actual sequential pipeline with scripted, synthetic responses.

No SDK, API keys, network, local-history reads, ledger writes, or notifications.
This verifies plumbing, not model quality. Run: python3 examples/offline.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import delphi

QUESTION = "Should our fictional reading club try a new meeting format?"
INTAKE = {
    "reframed": QUESTION, "question_type": "decision",
    "reversibility": "high", "cost_of_being_wrong": "low",
    "user_asserted_facts": ["This is a fictional offline example."],
    "variables": [],
}
BRIEF = {
    "judgment": "[SCRIPTED FIXTURE] Try one meeting and collect feedback.",
    "confidence": "low", "confidence_framing": "Not a calibrated prediction.",
    "one_line_bet": "A small reversible trial can inform the next meeting.",
    "biggest_risk": "Scripted responses provide no real evidence.",
    "first_domino": {"action": "Draft a fictional agenda.", "cost": "No model charges in this fixture.", "validates": "Nothing: offline fixture only."},
    "what_would_change_my_mind": ["Actual participant feedback."],
    "validation_signals": [],
    "narrative": "Synthetic example: no real decision or external fact verification.",
}


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected LLM call beyond the fixture")
        content = self.responses.pop(0)
        return delphi.LLMResponse(content, tokens_in=100, tokens_out=50)


def run_example():
    llm = ScriptedLLM([
        json.dumps(INTAKE),
        "[SCRIPTED Advocate] A one-meeting experiment is reversible.",
        "[SCRIPTED Skeptic] The preferences of attendees are unknown.",
        "[SCRIPTED Realist] Ask attendees before selecting a format.",
        json.dumps(BRIEF),
    ])
    # Empty temporary ledger directory prevents reading personal history.
    with tempfile.TemporaryDirectory() as directory, patch.multiple(
        delphi, LEDGER_DIR=Path(directory), _STREAM_ENABLED=False,
        BUDGET_USD_LIMIT=1.0, MODEL_PRICES={}, FALLBACK_PRICES=(1.0, 5.0),
    ), patch.dict(delphi.ADAPTERS, {
        "llm": llm, "memory": delphi.NoOpMemoryAdapter(),
        "bias": delphi.NoOpBiasAdapter(), "notifier": delphi.NoOpNotifierAdapter(),
    }):
        transcript = delphi.run_delphi(QUESTION)
        assert len(llm.calls) == 5 and not llm.responses
        print("# Offline scripted pipeline example\n")
        print("No model was called. Token counts and cost estimates below are synthetic.\n")
        print(transcript.render())


if __name__ == "__main__":
    run_example()
