"""Compile a specialized decision model with STROT.

Install the package with the local runtime and an LLM extra, then set a key:

    pip install -e ".[local,openai]"
    export OPENAI_API_KEY=...

    python3 examples/end_to_end.py

STROT designs a typed question set from your one-line intent, labels a sample of
your real data with the LLM teacher, distills a calibrated head per question onto
a frozen encoder, and returns a model that decides locally on CPU - no LLM call
at inference, $0 per decision.
"""

from __future__ import annotations

import json

import strot

# A representative sample of the live data stream (bring your own real records).
SUPPORT_TICKETS = [
    {"ticket_id": 101, "channel": "email", "body": "I was double charged on my invoice", "account_tier": "pro"},
    {"ticket_id": 102, "channel": "chat", "body": "App crashes when exporting a report", "account_tier": "free"},
    {"ticket_id": 103, "channel": "email", "body": "Please delete my account and all data", "account_tier": "pro"},
]


def main() -> None:
    # 1. Design the decision schema from a one-line intent. Review/edit before use.
    questions = strot.questions(SUPPORT_TICKETS, "triage incoming support tickets")
    print("Designed questions:\n", json.dumps(questions, indent=2))

    # 2. Compile a decision model: the teacher labels the sample, a calibrated
    #    head per question is distilled onto the frozen encoder.
    model = strot.decide(SUPPORT_TICKETS, questions)

    # 3. Decide - locally, on CPU.
    print("\nDecision for ticket 101:\n", json.dumps(model(SUPPORT_TICKETS[0]), indent=2, default=str))

    # 4. Ship one self-contained model file; reload with strot.load_model(...).
    path = model.save("ticket-triage")
    print("\nSaved model:", path)


if __name__ == "__main__":
    main()
