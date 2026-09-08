"""basic.py — minimal Python usage of Delphi Energy.

Run from project root:
    OPENAI_API_KEY=sk-... python examples/basic.py
"""
import sys
from pathlib import Path

# Make `delphi.py` importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from delphi import run_delphi  # noqa: E402


if __name__ == "__main__":
    question = (
        "I'm 30, work at a stable job, have $80k saved. Should I quit and "
        "take 6 months to try building a startup, or keep the job and "
        "build nights/weekends?"
    )
    transcript = run_delphi(question)
    print(transcript.render_brief_only())
    print("\nFull transcript (not saved):\n" + transcript.render())
    print(f"Estimated cost: ${transcript.total_cost():.4f}")
