# Contributing

Thanks for your interest in STROT. A few conventions keep the project easy to
review and maintain.

## Development setup

The core is pure standard library; the heavy pieces (the local model, hosted
clients, the synthesizer) come in through extras.

```bash
git clone https://github.com/strot-ai/STROT-Compiler
cd STROT-Compiler
pip install -e ".[anthropic]"   # add [local] for the on-device model, [hosted] for OpenAI
```

## Running the tests

```bash
pip install pytest
pytest
```

Tests that need the local model or an LLM client are skipped unless the matching
extra is installed. Run the suite before opening a PR, and add tests for any new
behavior.

## Pull requests

- Keep each PR to one logical change. A feature, a bug fix, and a refactor
  belong in separate PRs — it keeps review fast and merges clean.
- Branch off the latest `main`, and rebase if it moves under you.
- Don't bundle unrelated reformatting or dependency bumps with a feature.
- Say what changed and why, and note how you tested it.

Architecture and internals are documented in [DEVELOPERS.md](DEVELOPERS.md).

## Questions

Open an issue, or email lee@strot-ai.xyz.
