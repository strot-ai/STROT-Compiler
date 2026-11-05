"""Compile and run a ticket router with strot.run().

From the repo root, install the core package with pip install -e .
Or create and activate a .venv with uv:

    uv venv --python 3.14 .venv && uv pip install -e . && . .venv/bin/activate

After installation, choose a compile-time LLM:

    # Offline stub. No API key or Julia needed:
    python3 examples/end_to_end.py --compile-only

    # Codex: run codex login and sign in with ChatGPT first:
    codex login status
    python3 examples/end_to_end.py --llm codex-cli/gpt-6-luna --compile-only

    # Codex SDK: install the codex-sdk extra and reuse the same login:
    python3 examples/end_to_end.py --llm codex-sdk/gpt-6-luna --compile-only

    # Claude Code: run claude and /login with your subscription first:
    python3 examples/end_to_end.py --llm claude-cli/claude-haiku-5-5 --compile-only

To run the offline example in an isolated environment with uvx:

    uvx --python 3.14 --from . python examples/end_to_end.py --compile-only

Omit /<model> to use the CLI default. Check which login the CLI uses.
Other credentials can override a saved subscription login.
Your account controls model access and usage limits. Codex must support
--ignore-user-config and --ephemeral. Claude must support --safe-mode.
Codex shell commands and web search are disabled during synthesis.

Without --llm, the example uses the offline stub. Use a separate --state
directory to compare LLMs. Changing --llm alone does not clear the saved code.
This simple example uses reasoning effort "none" for an explicit Luna model.
Use --reasoning-effort medium for harder tasks. Other models keep their
configured effort unless you set this flag.

The --compile-only flag saves code without loading Julia. Review that code
before use. To run the tickets, install strot-ai[local], run
strot.prefetch_nlu(install=True), and omit --compile-only.

Official CLI documentation:
https://developers.openai.com/codex/auth
https://code.claude.com/docs/en/authentication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import strot

# ---------------------------------------------------------------------- #
# 1. A representative sample of the live data stream                      #
# ---------------------------------------------------------------------- #

SUPPORT_TICKETS = [
    {"ticket_id": 101, "created_at": "2026-10-01", "channel": "email", "body": "I was double charged on my invoice", "account_tier": "pro"},
    {"ticket_id": 102, "created_at": "2026-10-02", "channel": "chat", "body": "App crashes when exporting a report", "account_tier": "free"},
    {"ticket_id": 103, "created_at": "2026-10-03", "channel": "email", "body": "Please delete my account and all data", "account_tier": "pro"},
]

GOAL = "Triage incoming support tickets into the correct resolution workflow."

ACTION_REGISTRY = {
    "billing_dispute": "Refund checks, invoice corrections, double-charge investigations",
    "technical_defect": "Crashes, bugs, and broken product behavior",
    "account_lifecycle": "Account deletion, data export, GDPR requests",
}


def stub_synthesizer(prompt: str) -> str:
    """Offline synthesizer: returns paths that follow the compiler contract."""
    return '''```python
ROUTING_LABELS = {
    "billing_dispute": "billing invoice double charge refund",
    "technical_defect": "app crash bug broken export",
    "account_lifecycle": "account deletion personal data privacy",
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))


def execute_billing_dispute(record: dict) -> dict:
    try:
        return {"queue": "billing", "priority": "high" if record.get("account_tier") == "pro" else "normal", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "billing", "error": str(exc)}


def execute_technical_defect(record: dict) -> dict:
    try:
        return {"queue": "engineering", "needs_repro": True, "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "engineering", "error": str(exc)}


def execute_account_lifecycle(record: dict) -> dict:
    try:
        return {"queue": "trust_and_safety", "legal_hold_check": True, "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "trust_and_safety", "error": str(exc)}
```'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="stub", help="Synthesizer spec: stub, codex-cli, claude-cli, or provider/model")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        help="Override effort for an OpenAI or Codex synthesizer")
    parser.add_argument("--compile-only", action="store_true", help="Save without loading Julia")
    parser.add_argument("--state", type=Path, default=Path(".strot/end_to_end"))
    args = parser.parse_args()
    synthesizer = stub_synthesizer if args.llm == "stub" else strot.llm_client(
        args.llm, reasoning_effort=args.reasoning_effort)
    result = strot.run(
        data=SUPPORT_TICKETS,
        query=GOAL,
        action_registry=ACTION_REGISTRY,
        llm=synthesizer,
        state=args.state,
        execute=not args.compile_only,
    )
    print(json.dumps(result.summary(), default=str))
    print(f"Standalone asset: {result.artifact_path}")
    for outcome in result.outcomes:
        print(json.dumps(outcome, default=str))
    for fallback in result.fallbacks:
        print(json.dumps(fallback, default=str))


if __name__ == "__main__":
    main()
