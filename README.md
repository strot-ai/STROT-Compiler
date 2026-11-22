# STROT

### A decision compiler

**Compile System-2 reasoning into System-1 decisions — $0 at inference.**

[![PyPI](https://img.shields.io/pypi/v/strot-ai?color=7c3aed&label=pip%20install%20strot-ai)](https://pypi.org/project/strot-ai/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

An LLM reasons through your task once, at compile time. STROT distills that
judgment into a **specialized decision model** — a calibrated head on a frozen
encoder, trained on *your* real data — that then makes every decision **locally,
on CPU, at $0**: no API calls, no GPU, no per-record cost.

> **1.0 — the stable `decide` API.** The core (a head on a frozen
> `gte-modernbert` encoder) is benchmark-validated on Banking77 (93% full-data,
> 87% at 10 examples/class). Validation breadth is still growing — bring your own
> data and tell us how it does.

```bash
pip install "strot-ai[local,anthropic]"
export ANTHROPIC_API_KEY=...
```

```python
import strot

# 1. Design the decision schema from a one-line intent (review/edit it).
questions = strot.questions(tickets, "triage support tickets",
                            llm="anthropic/claude-opus-4-8")

# 2. Compile a decision model — the LLM teacher labels your sample, a calibrated
#    head per question is distilled onto the frozen encoder.
model = strot.decide(tickets, questions, llm="anthropic/claude-opus-4-8")

# 3. Decide — locally, on CPU, $0 per call.
model(tickets[0])
# {"predicates": {...}, "scores": {...}, "choices": {...}}  probabilities + calibrated confidence

# 4. Ship one self-contained file; reload anywhere.
model.save("ticket-triage")
strot.load_model("ticket-triage")
```

The LLM runs only at compile time. After that, production is a self-contained
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

Pass that to `strot.decide(data, questions)`, or hand `decide` a **bare intent
string** and it designs the schema for you first (`result.questions` shows what
it chose). Each answer carries probabilities and a calibrated confidence.

## Three tiers

`strot.decide(data, questions, mode=...)` picks how hard it works:

| mode | what it is | needs | when |
|---|---|---|---|
| **`lite`** | zero-shot prototype heads (embed each option's description, cosine-match) | no labels, instant | try a decision immediately, $0, bring no data |
| **`head`** *(default)* | a calibrated logistic head per question, distilled from teacher labels | the teacher labels your sample | the accuracy of training on your records |
| **`deep`** | a shared top-N LoRA adapter over all questions, then refit heads | the `[finetune]` extra + more records | representation-bound tasks a frozen head can't crack |

`warm=True` serves `lite` instantly while `head` trains in the background, then
hot-swaps it in — "zero-shot that grows up."

## How it works

Two phases: the expensive, non-deterministic one happens **once**; the cheap one
runs on every record forever.

- **Compile once.** The LLM designs the question schema (optional) and writes the
  record→text projection; the teacher labels a PII-masked sample; a calibrated
  head per question is trained on the frozen encoder's embeddings and baked into
  a standalone `.py` artifact. Content-addressed: a schema change triggers exactly
  one recompile, nothing else does.
- **Decide forever.** The local model projects a record, embeds it once, and
  every question head answers — typed JSON, bounded, with calibrated confidence.
  No LLM, no network, ~tens of ms on CPU.
- **Refuses what it hasn't seen.** A drift gate (per-class centroids + a distance
  threshold, fit at compile time) returns a refusal for records unlike anything
  in training rather than guessing on them.
- **Recompile, don't increment.** Every compile is a *fresh fit* from the frozen
  base on the current data — never a warm-start from prior weights. A model's
  weights are a pure function of its own `(data, questions, mode)`, so retraining
  on accumulated data can only gain quality; it is immune to catastrophic
  forgetting by construction.

## Data & privacy

The teacher LLM is used as a **labeler of your real records**, never as a bulk
synthetic generator. Before any record reaches the teacher it is **PII-masked at
that boundary**: a regex pre-pass guards format-defined tokens (email, SSN, card,
API keys, IBAN, IP/MAC/phone), and GLiNER (the `[pii]` extra) masks names,
addresses, and orgs. Masking is compile-time only.

We measured synthetic data and it hurt both accuracy and the confident slice — it
comes out too clean and leaks the label. The policy is **real-first**: label real
data; handle class imbalance by labeling more real minority-class records.

## Install

```bash
pip install strot-ai                # core (pure stdlib)
pip install "strot-ai[local]"       # + torch/transformers (compile + serve a model)
pip install "strot-ai[finetune]"    # + torch/transformers, for mode="deep"
pip install "strot-ai[pii]"         # + GLiNER masking at the teacher boundary
pip install "strot-ai[openai]"      # + openai, teacher via llm="openai/<model>"
pip install "strot-ai[anthropic]"   # + anthropic, teacher via llm="anthropic/<model>"
```

Extras install optional packages; they do not sign you in. Set the matching API
key and pass `llm="<provider>/<model>"`. Compile-time only — serving needs no key.

Bake the encoder into your image for offline production:

```dockerfile
RUN pip install "strot-ai[local]" \
 && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/encoder')"
ENV STROT_NLU_CHECKPOINT=/opt/encoder
```

## Contributing & contact

Pull requests welcome — keep them focused, branch off the latest `main`, and run
the suite first. See [CONTRIBUTING.md](CONTRIBUTING.md). Questions or ideas: open
an issue or email **lee@strot-ai.xyz**.

Built by Amit Rath — [LinkedIn](https://www.linkedin.com/in/amirath/).

## License

MIT. The base encoder
([Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base))
is Apache-2.0. Implements ideas from STROT (arXiv:2505.01636).
