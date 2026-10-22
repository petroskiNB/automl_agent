"""One audit of a saved run through API; no training, no prediction changes."""
import argparse
import os
from pathlib import Path

from solution import review_experiments, reviewer_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    task = run.parent.parent / "task.md"
    if not (run / "research/state.json").is_file() or not task.is_file():
        parser.error("Expected an existing tasks/<task>/agent_runs/<run> directory")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        parser.error("Set OPENAI_API_KEY and OPENAI_BASE_URL in this terminal")
    if (task.parent / ".agent.lock").exists():
        parser.error("Stop the active agent before running a separate audit")
    from openai import OpenAI
    text = task.read_text(encoding="utf-8-sig")
    print(f"Reviewer input: {len(reviewer_context(run, text))} characters. No training.", flush=True)
    with OpenAI(max_retries=0) as client:
        print(review_experiments(client, run, text, 180), flush=True)


if __name__ == "__main__":
    main()
