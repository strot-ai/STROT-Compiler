# STROT

### A decision compiler

**Compile System-2 reasoning into System-1 decisions — $0 at inference.**

[![PyPI](https://img.shields.io/pypi/v/strot-ai?color=7c3aed&label=pip%20install%20strot-ai)](https://pypi.org/project/strot-ai/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

STROT distills an agent's judgment into a **specialized decision model** — a
calibrated head on a frozen encoder, trained on *your* real data — that then makes
every decision **locally, on CPU, at $0**: no API calls, no GPU, no per-record
cost. **You (the agent) are the teacher: you design the questions and label a
small sample. STROT runs no LLM and holds no API key** — it only embeds, fits, and
bakes.

> The core (a head on a frozen `gte-modernbert` encoder) is benchmark-validated on
> Banking77 (93% full-data, 87% at 10 examples/class). Drive it from your agent
> (Claude, Codex, ChatGPT) over MCP, or from the Python API. No keys.

```bash
pip install "strot-ai[local]"
```

```python
import strot

# You design the typed questions (see `strot playbook`), and label a small sample
# yourself — your own decisions, or human ground truth. STROT distills the rest.
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

No LLM runs inside STROT, at compile time or after. Production is a self-contained
model file: the local model answers every record, **nothing is generated at
runtime**, and every output is bounded typed JSON with a calibrated confidence.
No labels to bring yet? `strot.compile(data, questions, mode="lite")` is a
zero-shot tier that needs none.

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

## Three tiers

`strot.compile(data, questions, labels, mode=...)` picks how hard it works:

| mode | what it is | needs | when |
|---|---|---|---|
| **`lite`** | zero-shot prototype heads (embed each option's description, cosine-match) | **no labels**, instant | try a decision immediately, $0, bring no data |
| **`head`** *(default)* | a calibrated logistic head per question, distilled from your labels | you label a small sample | the accuracy of training on your records |
| **`deep`** | a shared top-N LoRA adapter over all questions, then refit heads | more labeled records (`[local]`) | representation-bound tasks a frozen head can't crack |

The agent labels the sample for `head`/`deep`; `lite` needs nothing. `lite` is a
rough zero-shot *floor* for a quick look — accuracy comes from `head`'s labels,
and **predicates in particular need labels** (a single zero-shot prototype barely
separates). Reach for `head` for anything real.

## How it works

Two phases: the labeling (the agent's judgment) happens **once**; the cheap
deterministic part runs on every record forever.

- **Compile once.** You design the questions and label a small sample; STROT
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

The labels are **yours** — the agent's own decisions, or human ground truth; STROT
never invents data. STROT holds no key and **sends records nowhere**: it embeds
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

STROT's MCP server lets an agent (Claude, Codex, ChatGPT, or a production
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

Bake the encoder into your image for offline production:

```dockerfile
RUN pip install "strot-ai[local]" \
 && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/encoder')"
ENV STROT_NLU_CHECKPOINT=/opt/encoder
```

## Docs

[DEVELOPERS.md](DEVELOPERS.md) covers the architecture — the compile path, the
`lite`/`head`/`deep` tiers, calibration & the drift gate, the lifecycle
primitives (`evaluate`, traces, `drift_check`, `cluster_rejections`), and the MCP
agent surface.

## Contributing & contact

Pull requests welcome — keep them focused, branch off the latest `main`, and run
the suite first. See [CONTRIBUTING.md](CONTRIBUTING.md). Questions or ideas: open
an issue or email **lee@strot-ai.xyz**.

Built by Amit Rath — [LinkedIn](https://www.linkedin.com/in/amirath/).

## License

MIT. The base encoder
([Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base))
is Apache-2.0. Implements ideas from STROT (arXiv:2505.01636).
