# OpenDecisions

### Open decision models, compiled from your data — $0 at inference.

[![PyPI](https://img.shields.io/pypi/v/strot-ai?color=7c3aed&label=pip%20install%20strot-ai)](https://pypi.org/project/strot-ai/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

OpenDecisions distills an agent's judgment into a **specialized decision model** —
a calibrated head on a frozen encoder, trained on *your* real data — that then
makes every decision **locally, on CPU, at $0**: no API calls, no GPU, no
per-record cost. **You (the agent) are the teacher: you design the questions and
label a small sample. It runs no LLM and holds no API key** — it only embeds,
fits, and bakes.

> The package is **`strot-ai`** (the STROT compiler engine): `pip install
> strot-ai`, `import strot`, `python -m strot.mcp`. OpenDecisions is the product;
> STROT is the method it implements.

> The core (a head on a frozen `gte-modernbert` encoder) is benchmark-validated on
> Banking77 (93% full-data, 87% at 10 examples/class). Drive it from your agent
> (Claude, Codex, ChatGPT) over MCP, or from the Python API. No keys.

```bash
pip install "strot-ai[local]"
```

```python
import strot

# You design the typed questions (see `strot playbook`), and label a small sample
# yourself — your own decisions, or human ground truth. OpenDecisions distills the rest.
questions = {"is_bug": {"type": "predicate", "instructions": "mentions a crash or error?"}}
labels    = [{"is_bug": True}, {"is_bug": False}, ...]   # one per record, same order

# Compile a decision model — no LLM, no key. A calibrated head per question.
model = strot.compile(tickets, questions, labels)

# Decide — locally, on CPU, $0 per call.
model(tickets[0])
# {"predicates": {...}, "scores": {...}, "choices": {...}}  probabilities + calibrated confidence

# Ship one self-contained file; reload anywhere.
model.dispatch("ticket-triage")
strot.load_model("ticket-triage")
```

No LLM runs inside OpenDecisions, at compile time or after. Production is a self-contained
model file: the local model answers every record, **nothing is generated at
runtime**, and every output is bounded typed JSON with a calibrated confidence.

## Typed questions

A decision config maps a name to a typed question — `predicate`, `choice`, or
`score`:

```python
questions = {
    "is_bug_report":      {"type": "predicate", "instructions": "mentions crashes or errors?"},
    "customer_sentiment": {"type": "score", "instructions": "how upset is the customer?",
                           "levels": [{"label": "angry"}, {"label": "neutral"}, {"label": "happy"}]},
    "routing_bucket":     {"type": "choice", "choices": [
                              {"value": "billing",     "description": "charges, invoices, refunds"},
                              {"value": "engineering", "description": "crashes, bugs, defects"}]},
}
```

Pass that to `strot.compile(data, questions, labels)`. You design the config —
the agent's job (see `strot playbook`). Each answer carries probabilities and a
calibrated confidence.

## Two tiers

`strot.compile(data, questions, labels, mode=...)` picks how hard it works. Both
distill from the labels you bring:

| mode | what it is | when |
|---|---|---|
| **`head`** *(default)* | a calibrated logistic head per question | the accuracy of training on your records |
| **`deep`** | a shared top-N LoRA adapter over all questions, then refit heads | representation-bound tasks a frozen head can't crack |

## How it works

Two phases: the labeling (the agent's judgment) happens **once**; the cheap
deterministic part runs on every record forever.

- **Compile once.** You design the questions and label a small sample; OpenDecisions
  projects each record to text, embeds it on the frozen encoder, fits a calibrated
  head per question, and bakes a standalone artifact. No LLM, no network.
  Content-addressed: different labels or a schema change is a fresh model.
- **Decide forever.** The local model projects a record, embeds it once, and
  every question head answers — typed JSON, bounded, with calibrated confidence.
  No LLM, ~tens of ms on CPU.
- **Refuses what it hasn't seen.** A drift gate (per-class centroids + a distance
  threshold, fit at compile time) returns a refusal for records unlike anything
  in training rather than guessing on them.
- **Recompile, don't increment.** Every compile is a *fresh fit* from the frozen
  base on the current labels — never a warm-start from prior weights. A model's
  weights are a pure function of its own `(data, questions, labels, mode)`, so
  retraining on accumulated data can only gain quality; it is immune to
  catastrophic forgetting by construction.

## Data & privacy

The labels are **yours** — the agent's own decisions, or human ground truth; OpenDecisions
never invents data. It holds no key and **sends records nowhere**: it embeds
them locally. The only PII exposure is whatever *you* feed your own agent while
labeling — so `strot.pii_scan(data)` is an advisory regex scan (email, SSN, card,
IBAN, IP/MAC/phone) that flags what a record carries, and the real protection is
**which agent you point at the data**: a local/open-weight agent keeps it in your
infra.

## Install

```bash
pip install strot-ai                # core + the MCP server
pip install "strot-ai[local]"       # + torch/transformers (compile + serve a model)
```

`[local]` is all you need to compile and serve. The MCP server ships in the core
install (`python -m strot.mcp`). No API keys, ever.

## Drive it from an agent (MCP)

The OpenDecisions MCP server lets an agent (Claude, Codex, ChatGPT, or a production
automation agent) compile, run, evaluate, and keep decision models healthy with
tools instead of code. The agent designs the questions and labels the sample; the
tools do the deterministic rest.

```bash
pip install "strot-ai[local]"
python -m strot.mcp          # runs the MCP server over stdio
```

Tools cover the whole lifecycle — `inspect_data`, `pii_scan`,
`compile_start`/`compile_status` (you pass the labels), `decide`, `evaluate_model`,
`read_traces`, `drift_check`, `cluster_rejections`, `recompile`, `set_status`
(circuit breaker) — and a `strot_compiler` prompt (the playbook) teaches the agent
how to use them. Even simpler for code-running agents: `strot init` drops a skill
(`SKILL.md` + a compile script) into your project.

Point any MCP client at `python -m strot.mcp`. For Claude Desktop, add to
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "strot": { "command": "python", "args": ["-m", "strot.mcp"] }
  }
}
```

## Watcher — learn from an agent's traces

An observed agent's `(input → decision)` pairs **are labels** — real, free, and
in-distribution. The watcher turns them into a local understudy that takes the
confident cases at $0 and defers the rest back to the agent:

```python
records, labels = strot.ingest_traces("agent_traces.jsonl")   # plain dicts or OTel spans
questions = strot.infer_questions(labels)                      # recover the schema
model = strot.compile(records, questions, labels)              # distill — no LLM
```

No labels to collect — the decisions the agent already made are the supervision.
It clones the agent's *policy* (not truth), for typed decisions only. `strot init`
installs the full watcher playbook as a skill; see `WATCHER.md`.

Bake the encoder into your image for offline production:

```dockerfile
RUN pip install "strot-ai[local]" \
 && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/encoder')"
ENV STROT_NLU_CHECKPOINT=/opt/encoder
```

## Docs

[DEVELOPERS.md](DEVELOPERS.md) covers the architecture — the compile path, the
`head`/`deep` tiers, calibration & the drift gate, the lifecycle primitives
(`evaluate`, traces, `drift_check`, `cluster_rejections`), the **watcher** (learn
from an agent's traces), and the MCP agent surface.

## Contributing & contact

Pull requests welcome — keep them focused, branch off the latest `main`, and run
the suite first. See [CONTRIBUTING.md](CONTRIBUTING.md). Questions or ideas: open
an issue or email **lee@strot-ai.xyz**.

Built by Amit Rath — [LinkedIn](https://www.linkedin.com/in/amirath/).

## License

MIT. The base encoder
([Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base))
is Apache-2.0. Implements ideas from STROT (arXiv:2505.01636).
