# STROT

### The open decision system

**Compile System 2 reasoning into System 1 decisions — $0 at inference.**

**LOW LATENCY · ULTRAFAST · CHEAP & FREE · 0 TOKENS · LOW CARBON · CPU INFERENCE ONLY**

[![PyPI](https://img.shields.io/pypi/v/strot-ai?color=7c3aed&label=pip%20install%20strot-ai)](https://pypi.org/project/strot-ai/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97-live%20demo-ffce00)](https://huggingface.co/spaces/STROT-ai/STROT-AI-Compiler)

**▶ [Try the live demo](https://huggingface.co/spaces/STROT-ai/STROT-AI-Compiler)** — decisions in your browser, no install, key-free.

An LLM reasons through your task once, at compile time. Every decision after runs
on a small local model on **CPU** — no API calls, no GPU, no per-record cost:
**faster, cheaper, and a fraction of the energy** of calling an LLM on every record.

STROT compiles a decision task into a standalone pipeline. You describe the
intent once (a query plus the set of actions it can take), an LLM compiles that
into a frozen Python artifact, and from then on a small local model makes the
actual decisions — on CPU, no API calls at runtime, bounded typed-JSON output.

I built it because getting a small local model to reliably *decide* things
(route this ticket, classify this text, score this record) is fiddly work, and
accuracy swings hard depending on how you phrase the labels. STROT moves that
work to compile time and freezes the result, the way a compiler moves work off
the hot path.

```bash
pip install strot-ai
```

```python
import strot

result = strot.run(
    tickets,
    query="Triage incoming support tickets into the correct workflow.",
    action_registry={
        "billing_dispute":  "Refunds, invoice corrections, double charges",
        "technical_defect": "Crashes, bugs, broken behavior",
    },
    llm="anthropic/claude-opus-4-8",   # used once, on first compile only
)

for o in result.outcomes:
    print(o["action"], round(o["confidence"], 2))
# billing_dispute 0.97
# technical_defect 0.94
```

The LLM runs once, when the artifact is first compiled. After that, production
is a self-contained Python file: a small local model arbitrates every record,
nothing is generated at runtime, and every output is bounded and carries a
calibrated confidence. By default it dispatches every record and lets you decide
what to do with the low-confidence tail; set an accuracy or risk target and it
abstains on records below the bar instead of guessing.

To compile with a subscription CLI, use
`llm="claude-cli/claude-opus-4-8"` or `llm="codex-cli/gpt-6-luna"`.
Sign in through `claude` or `codex login` first. Omit `/<model>` to use the
CLI default. Model access and usage limits depend on your account.
The CLI only runs at compile time. See [the example](examples/end_to_end.py).
For a CLI outside PATH, pass a client such as
`llm=strot.codex_cli_llm_client(cli="/path/to/codex", model="gpt-6-luna")`.
For Claude, use `llm=strot.claude_cli_llm_client(cli="/path/to/claude")`.

## How it works

Two phases. The expensive, nondeterministic one happens once; the cheap one runs
on every record forever.

```mermaid
flowchart TB
    subgraph compile["COMPILE — once per intent + schema"]
        direction LR
        A["intent + data sample"] --> B["synthesize<br/>labels · projection · handlers"]
        B --> C["golden gate<br/>(must pass your known answers)"]
        C --> D["teacher labels a sample<br/>(once)"]
        D --> E["fit confidence threshold"]
        E --> F[["frozen .py artifact"]]
    end
    subgraph runtime["RUNTIME — every record"]
        direction LR
        F --> G["project record"]
        G --> H["decision model<br/>144M · CPU · ~60ms"]
        H --> I["typed JSON decision<br/>+ calibrated confidence"]
        I -.->|target set:<br/>below the bar| K["abstain → fallbacks"]
    end
    compile --> runtime
```

The compiler doesn't trust its first guess at the labels. It scores candidate
labelings against a teacher-labeled sample and keeps whichever one actually
measures best, and refuses to promote an artifact that fails your golden
records. By default the battery dispatches every record with a calibrated
confidence; give it an accuracy or risk target and it also fits the threshold
that decides which records to abstain on.

## Two ways to use it

`run()` routes each record to an action and executes the compiled handler:

```python
result = strot.run(records, query=..., action_registry={...}, llm=...)
```

`decide()` answers typed questions per record (choice / score / boolean):

```python
result = strot.decide(records, questions={
    "is_bug_report":      {"type": "boolean", "instructions": "mentions crashes/errors?"},
    "customer_sentiment": {"type": "score", "min": 1, "max": 5, "instructions": "1 angry, 5 happy"},
    "routing_bucket":     {"type": "choice", "options": ["billing", "engineering", "spam"]},
})
# {"booleans": {...}, "scores": {...}, "choices": {...}}  with probabilities + confidence
```

## Numbers

Real model (SupersonicLabs/Julia-1, 144M, CPU), real datasets, run in clean
Docker containers. The benchmark scripts are in [`benchmarks/`](benchmarks/),
losses included.

| Benchmark | Result |
|---|---|
| Grounded routing (398 records) | 92.5% compiled vs 74.8% naive — the compiler is worth +18pp on the same model |
| Semantic text, unseen labels (AG News) | 86.7% compiled vs 72.7% naive (+14pp) |
| Typed decisions vs a leading hosted decision API's own example | near-identical (sentiment 1.79 vs 1.8) |
| Latency / marginal cost / LLM calls per record | ~60ms · $0 · 0 |

### On real risk narratives

Record-level unstructured text, the kind cyber / financial / insurance triage
actually runs on. Public ground-truth datasets, classes balanced so accuracy is
interpretable, baselines shown. STROT never sees the test labels.
([`benchmarks/risk_narratives.py`](benchmarks/risk_narratives.py))

| Task — 6-class, balanced | Random | Majority | Naive | Compiled | Abstains |
|---|---|---|---|---|---|
| Cyber — [VERIS/VCDB](https://github.com/vz-risk/VCDB) incident note → attack action | 16.7% | 16.7% | 36.8% | 39.5% | 8/114 |
| Finance — [CFPB](https://www.consumerfinance.gov/data-research/consumer-complaints/) complaint → product | 16.7% | 26.4% | 54.2% | 61.1% | 5/72 |

Both clear their baselines and the compiler lifts accuracy on the same 144M
model; the abstains come from running these with a target set. These are honest
proof the architecture works on messy real text, not production underwriting
accuracy. For that you'd put a task-trained specialist in the `nlu=` slot; the
compiler, calibration, and gate don't change.

Single-seed numbers, so read them as ranges, not guarantees. The frozen tier
lands around the low-60s to low-70s; turning on the fine-tune tier
(`optimize=True` with a target) pushes the hard representation-bound tasks into
the low-to-mid-70s. The real win isn't the headline accuracy — it's coverage on
the confident slice (cov@90), which is what the calibration and the fine-tune
ensemble move most. (And a measured non-result: synthesizing training data with
the LLM hurt both numbers, which is why STROT only labels real data — see
[Data](#data).)

## What's actually going on under the hood

- The optimizer compiles once; the battery decides forever. The compiled
  artifact is content-addressed; a schema change triggers exactly one recompile,
  nothing else does. There is no runtime loop: compile is a single pass, and the
  battery never retrains itself.
- Cheapest tier that clears your bar. The optimizer keeps the cheapest rung that
  meets your target on held-out real records: a zero-shot route, then a logistic
  head distilled onto the frozen encoder. Both are frozen tiers and cover most
  tasks — routing and tool-selection rarely leave them.
- Fine-tune once, opt-in. For representation-bound tasks the frozen tiers can't
  crack, pass `optimize=True` together with a target (`target_accuracy` or
  `target_risk`). Only when the frozen tier misses that target on the real
  measurement fold does the compiler fine-tune — once — a light cached-lower +
  top-N LoRA on ModernBERT (the `[finetune]` extra). It trains a small K=3
  ensemble of diverse adapters and averages them at inference; the measured
  winner is kept. Without a target, `optimize=True` is a no-op. Already built an
  adapter? Pass it as `reuse_adapter=` to adopt it after a quick validation on
  this task's fold and skip training entirely.
- No abstention by default; abstention is opt-in. With no target set the battery
  never hard-abstains: it dispatches every record and publishes the confidence
  (calibrated if a threshold was fitted), and the caller decides what to do with
  the low-confidence tail. Set `target_risk` (split conformal) or
  `target_accuracy` (an empirical dispatch cutoff) and the battery starts
  abstaining on records below the bar. Abstained records come back in
  `result.fallbacks` for you to re-run however you like — STROT does not loop.
- Golden records are a promotion gate, not a hint. A battery that fails your
  known answers never serves.
- Calibration is temperature scaling plus split conformal prediction, fitted at
  compile time. When a target is set, dispatched decisions are wrong at most α of
  all records (marginal), and the confident slice — coverage at your target — is
  the product metric, not blanket accuracy.
- Nothing is generated at runtime. Output is selection and computation over your
  registry, so there's no structural hallucination to guard against.
- The decision model is a slot (`nlu=`): local Julia-1 zero-shot by default,
  ModernBERT for the fine-tune tier, a hosted API, or any specialist you drop in.

### Data

The teacher LLM is used as a labeler of real records (distillation), never as a
bulk synthetic generator. Before any record reaches the teacher it is PII-
masked at that boundary. GLiNER is the default masker for names, addresses, and
orgs (it needs the `[pii]` extra); a thin regex pre-pass runs ahead of it to
guard format-defined tokens (email, SSN, Luhn card, API keys, IBAN, IP/MAC/
phone). Without the `[pii]` extra it falls back to regex-only with a warning.
Masking is compile-time only.

Synthetic data is a last resort, not a default. We measured it: LLM-generated
synthetic records hurt — roughly −2pp accuracy and −17pp cov@90 on the cyber
task — because they come out too clean and tend to leak the label. So the policy
is real-first: label real data, and handle class imbalance by pulling and
labeling more real minority-class records (or oversampling / weighting the real
minority). Reach for synthetic only to seed a near-empty class, never to "scale
up."

## Install

```bash
pip install strot-ai             # core: pure stdlib
pip install "strot-ai[local]"     # + local model dependencies (torch/transformers)
pip install "strot-ai[finetune]"  # + torch/transformers, for the opt-in fine-tune tier
pip install "strot-ai[pii]"       # + GLiNER NER masking at the teacher boundary
pip install "strot-ai[hosted]"    # + openai, for API synthesis or runtime routing
pip install "strot-ai[anthropic]" # + anthropic, for Anthropic API synthesis
```

Extras install optional Python packages. They do not select a provider or sign you in.
For OpenAI API synthesis, install `[hosted]`, set `OPENAI_API_KEY`, and pass
`llm="openai/gpt-4o"`. For Anthropic API synthesis, install `[anthropic]`, set
`ANTHROPIC_API_KEY`, and pass `llm="anthropic/claude-opus-4-8"`.
Subscription CLI synthesis needs the CLI and its subscription login. It needs
neither the `[hosted]` nor the `[anthropic]` extra. Check which login the CLI uses.
Other credentials can override a saved subscription login.

Bake the local model into your image for offline production:

```dockerfile
RUN pip install strot-ai[local] \
 && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/julia-1', install=True)"
ENV STROT_NLU_CHECKPOINT=/opt/julia-1
```

(`julia` on PyPI is an unrelated package — don't install it. The decision
model's inference code is pulled from its Hugging Face repo by `prefetch_nlu`.)

## Docs

[DEVELOPERS.md](DEVELOPERS.md) covers the architecture, the compile lifecycle,
calibration semantics and their caveats, deployment, model swapping, and the
full API.

## Contributing & contact

Contributions are welcome — keep pull requests focused on one change, branch off
the latest `main`, and run the test suite first. See [CONTRIBUTING.md](CONTRIBUTING.md).

Questions, ideas, or problems: open an issue or email **lee@strot-ai.xyz**.

Built by Amit Rath. Follow me on [LinkedIn](https://www.linkedin.com/in/amirath/).

## License

MIT. The default decision model is Apache-2.0
([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)).
Implements ideas from STROT (arXiv:2505.01636).
