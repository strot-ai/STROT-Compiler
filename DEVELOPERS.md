# STROT — developer guide

STROT is a **decision compiler**. You (the agent) design a typed question set and
label a small sample of your real data; a calibrated head is distilled onto a
frozen encoder, and the result is a self-contained model that decides every record
after that **locally, on CPU, at $0**. **STROT runs no LLM and holds no API key** —
it only embeds, fits, and bakes. Output is bounded, type-safe JSON with calibrated
confidence — nothing is generated at runtime.

> You are the teacher. STROT is deterministic.

This guide covers the architecture, the runtime artifact, the lifecycle
primitives, and the MCP agent surface. For the quickstart, see the
[README](README.md).

## Compile path — `strot.compile(data, questions, labels)`

One deterministic pass, run once per `(questions, schema, labels, projection,
mode)` and cached. **No LLM, no network:**

1. **Schema** — `introspect_schema` profiles the sample (fields, types,
   cardinality, sample values).
2. **Projection** — a `route_context(record) -> text` selects the text to embed.
   The default concatenates the record's text fields; you may pass a custom one.
3. **Labels** — you bring them: one `{question: answer}` dict per record (an
   agent's own decisions, or human ground truth).
4. **Distill** — a calibrated logistic head per question is trained on the frozen
   encoder's embeddings of the projected text.
5. **Guards** — a temperature is fit per head (honest confidence) and a drift/OOD
   gate is fit (per-class centroids + distance threshold).
6. **Bake** — everything is written into a standalone `.py`/`.strot` artifact:
   the projection, the per-question heads (weights as literals), the calibration,
   the OOD gate, and (for `deep`) the LoRA adapter.

**Recompile, don't increment** (invariant): every compile is a *fresh fit* from
the frozen base on the current data/labels — never a warm-start from prior
weights. A model's weights are a pure function of its own `(data, questions,
labels, projection, mode)`, so retraining on accumulated data is forgetting-free.

## Tiers — `mode=`

Both tiers emit the same artifact type (a `DecisionModel`); only how the heads
are fitted differs. Both share one token window — `compile(..., max_length=256)`
(the default; ModernBERT allows up to 8192). It's baked into the artifact so
inference uses the same window, and head/deep always share it so they never
diverge. Raise it for long records (cost scales with length; `deep` especially):

| mode | heads | needs | when |
|---|---|---|---|
| `head` *(default)* | a calibrated logistic head per question, distilled from your labels | you label the sample | the accuracy of training on your records |
| `deep` | a shared top-N LoRA adapter over all questions, then refit heads | `[local]` + more labeled records | representation-bound tasks a frozen head can't crack |

## The runtime artifact (`.strot`)

A compiled model is a self-contained Python module (`CompiledDecider`) baking the
projection + `QUESTION_HEADS` (+ `ADAPTER` for `deep`, + `OOD`). At runtime per
record it: projects → embeds once with the frozen encoder → applies each head
(`softmax((W·x + b) / T)`) → emits:

```json
{"predicates": {name: {"type": "predicate", "probability": 0.98}},
 "scores":     {name: {"score": "angry", "probabilities": {...}, "confidence": 0.91}},
 "choices":    {name: {"choice": "engineering", "probabilities": {...}, "confidence": 0.97}}}
```

The **drift gate** refuses records unlike the training set up front
(`{"refusal": true, "reason": "ood"}`). Serving needs `strot-ai[local]`
(torch/transformers); it loads the shared base encoder by id (or
`STROT_NLU_CHECKPOINT`). No LLM, no network, ~tens of ms on CPU.

## The decision model

`compile()` returns a `DecisionModel`:

```python
model = strot.compile(data, questions, labels)
model(record)                    # -> typed decision JSON
model.questions                  # the config it answers
model.trace("traces.jsonl")      # opt-in: log every decision (the agent's eyes)
model.dispatch("triage")         # put into service: one self-contained triage.strot
strot.load_model("triage")       # reload a dispatched model
```

`dispatch()` is the ship verb.

## Calibration & refusal

- **Temperature** per head, fit in **logit space** on a held-out fold
  (`calibrate.fit_temperature_logits`) — fit on raw logits, not softmaxed probs
  (a saturated 0/1 can't be rescaled). Confidence is honest, not pinned at 1.0.
- **`confidence_threshold`** — an opt-in per-question refusal floor, applied
  post-hoc (changing it never forces a recompile).
- **Drift/OOD gate** — fit at compile time; refuses out-of-distribution records.

## Lifecycle primitives (torch-free to *read*)

The signals a maintainer (or an agent) runs on — all pure-Python, so you can
assess a model's health without loading it:

- **`strot.evaluate(model, labeled)`** / **`score_decisions(...)`** — accuracy,
  coverage, calibration (ECE) per question. The learning signal.
- **`strot.read_traces(path)`** / **`summarize_traces(...)`** — volume, refusal
  rate, confidence distribution, value mix. The monitoring signal.
- **`strot.drift_check(traces)`** — recent-vs-baseline drift + a
  `recommend_recompile` verdict.
- **`strot.cluster_rejections(traces)`** — cluster the refused records (top terms
  + samples) to surface the gap to close.
- **`strot.inspect_data(data)`** — profile raw input before compiling.
- **`strot.pii_scan(data)`** — advisory regex scan of structured PII in a sample.

## Watcher — learn from an agent's traces

An observed agent's `(input -> decision)` pairs are labels. Two torch-free
helpers turn a trace stream into a compile, so the understudy reuses the normal
`compile` path:

- **`strot.ingest_traces(traces, input_key, decision_key)`** — `(records, labels)`
  from a list / JSONL / JSON file of `{input, decision}`, or OpenTelemetry spans
  whose `attributes` carry those keys (stringified JSON is decoded).
- **`strot.infer_questions(labels)`** — recover a typed config from observed
  decisions (all-boolean -> `predicate`, else `choice` over observed values;
  `score` is never inferred — order can't be recovered from observation).

```python
records, labels = strot.ingest_traces("agent_traces.jsonl")
model = strot.compile(records, strot.infer_questions(labels), labels)
```

Then shadow (`evaluate` for fidelity-to-the-agent), take over confidence-gated
(call the understudy before the agent; defer low-confidence / OOD), and recompile
on drift. It clones the agent's *policy*, not truth, for typed decisions only. The
full methodology ships as the `WATCHER.md` skill (`strot init`).

## MCP — drive it from an agent

The MCP server ships in the core install (`python -m strot.mcp`, SDK 2.x) — it is
the primary interface. The capability layer (`strot.mcp.tools`) is SDK-free and
torch-free for the read tools; `strot.mcp.server` registers them. Tools:

`list_models` · `model_info` · `inspect_data` · `pii_scan` ·
`compile_start` / `compile_status` (you pass the labels; async — a compile runs
minutes) · `decide` · `evaluate_model` · `score` · `read_traces` · `drift_check` ·
`cluster_rejections` · `recompile` · `set_status` / `model_status` (circuit
breaker) · `budget_status`.

A **`strot_compiler` prompt** (the playbook) teaches the agent the methodology:
inspect → pii → design questions → label → compile(tier) → verify → deploy →
monitor → recompile → gatekeep. The agent brings the intelligence (design +
labeling); the server does no LLM work of its own. For code-running agents,
`strot init` installs the same playbook as a skill + a bundled compile script.

## Data & privacy

- **Labels are yours** — the agent's own decisions, or human ground truth (the
  strongest signal). STROT never invents data.
- **STROT sends records nowhere** — it embeds them locally on the frozen encoder.
  There is no teacher boundary and no masking: raw text in a local embedder is not
  a leak. The only PII exposure is whatever *you* feed your own agent while
  labeling, which STROT cannot see.
- **`pii_scan` is advisory** — a zero-dep regex scan (email, SSN, card, IBAN,
  IP/MAC/phone) that flags what a record carries so the agent can decide before
  feeding it to its own LLM. The real protection is **placement**: a
  local/open-weight agent keeps data in your infra.

## Base encoder

The frozen base is `Alibaba-NLP/gte-modernbert-base` (150M), validated on
Banking77 (93% full-data, 87% @10/class, 82% @5/class). Not caller-selectable.
Bake it into an image for offline/air-gapped serving with
`strot.prefetch_nlu(local_dir=...)` + `STROT_NLU_CHECKPOINT`.

## Development

```bash
poetry install --extras all
poetry run pytest -q          # offline; the encoder is stubbed, so no torch needed
```

The core is stdlib + the MCP SDK; torch/transformers come only via `[local]`.
CI runs the suite on the core install across Python versions.

## Reference

Implements ideas from STROT (Structured Task Reasoning and Output Transformation,
[arXiv:2505.01636](https://arxiv.org/abs/2505.01636)). MIT licensed; the base
encoder is Apache-2.0. Source: <https://github.com/strot-ai/OpenDecisions>.
