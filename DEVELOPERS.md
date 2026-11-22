# STROT — developer guide

STROT is a **decision compiler**. You bring a sample of your real data and a
typed question; an LLM teacher labels the sample once, a calibrated head is
distilled onto a frozen encoder, and the result is a self-contained model that
decides every record after that **locally, on CPU, at $0**. Output is bounded,
type-safe JSON with calibrated confidence — nothing is generated at runtime.

> Reason once, at compile time. Decide forever, locally.

This guide covers the architecture, the runtime artifact, the lifecycle
primitives, and the MCP agent surface. For the quickstart, see the
[README](README.md).

## Compile path — `strot.decide(data, questions)`

One deterministic pass, run once per `(questions, schema, mode)` and cached:

1. **Schema** — `introspect_schema` profiles the sample (fields, types,
   cardinality, sample values).
2. **Synthesis** — the LLM designs the typed question set (if you passed a bare
   intent) and writes the **state projection** `route_context(record) -> text`
   (which fields to embed), validated with a bounded self-correction loop.
3. **Teacher labels** — the LLM answers every question for every **PII-masked**
   sampled record (see *Data & privacy*).
4. **Distill** — a calibrated logistic head per question is trained on the frozen
   encoder's embeddings of the projected text.
5. **Guards** — a temperature is fit per head (honest confidence) and a drift/OOD
   gate is fit (per-class centroids + distance threshold).
6. **Bake** — everything is written into a standalone `.py`/`.strot` artifact:
   the projection, the per-question heads (weights as literals), the calibration,
   the OOD gate, and (for `deep`) the LoRA adapter.

**Recompile, don't increment** (invariant): every compile is a *fresh fit* from
the frozen base on the current data — never a warm-start from prior weights. A
model's weights are a pure function of its own `(data, questions, mode)`, so
retraining on accumulated data is forgetting-free. There is no runtime training.

## Tiers — `mode=`

All three tiers emit the same artifact type (a `DecisionModel`); only how the
heads are fitted differs:

| mode | heads | needs | when |
|---|---|---|---|
| `lite` | prototype embeddings of each option's description (cosine) | no labels | try instantly, $0, bring no data |
| `head` *(default)* | a calibrated logistic head per question, distilled from teacher labels | the teacher labels your sample | the accuracy of training on your records |
| `deep` | a shared top-N LoRA adapter over all questions, then refit heads | `[finetune]` extra + more records | representation-bound tasks a frozen head can't crack |

`warm=True` serves `lite` instantly while `head` trains in a background thread,
then hot-swaps it in.

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

`decide()` returns a `DecisionModel`:

```python
model = strot.decide(data, questions)
model(record)                    # -> typed decision JSON
model.questions                  # the config it answers
model.tier                       # "lite" | "head"
model.trace("traces.jsonl")      # opt-in: log every decision (the teacher's eyes)
model.dispatch("triage")         # put into service: one self-contained triage.strot
strot.load_model("triage")       # reload a dispatched model
```

`dispatch()` is the ship verb (`save()` is a back-compat alias).

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
- **`strot.pii_scan(data)`** — what the teacher-boundary masker would redact.

## MCP — drive it from an agent

`strot-ai[mcp]` exposes the compiler as an MCP server (`python -m strot.mcp`,
SDK 2.x). The capability layer (`strot.mcp.tools`) is SDK-free and torch-free for
the read tools; `strot.mcp.server` registers them. Tools:

`list_models` · `model_info` · `inspect_data` · `pii_scan` · `design_questions` ·
`compile_start` / `compile_status` (async — a compile runs minutes) · `decide` ·
`evaluate_model` · `score` · `read_traces` · `drift_check` · `cluster_rejections` ·
`recompile` · `set_status` / `model_status` (circuit breaker) · `budget_status`.

A **`strot_compiler` prompt** (the playbook) teaches the agent the methodology:
inspect → pii → design → compile(tier) → verify → deploy → monitor → recompile →
gatekeep. Guardrails: an async-compile budget (`STROT_MCP_COMPILE_BUDGET`), and
`decide` honors a paused model (the circuit breaker). The server does no LLM work
of its own beyond what the compile tools pass through.

## Data & privacy

- The teacher LLM is a **labeler of your real records**, never a bulk synthetic
  generator. Synthetic data measured *worse* (too clean, leaks the label) — the
  policy is real-first.
- **PII is masked at the teacher boundary**: a regex pre-pass (email, SSN, card,
  API keys, IBAN, IP/MAC/phone) plus GLiNER (`[pii]` extra) for names/addresses.
  Compile-time only; the served model never sees raw PII.
- **Labels from any source.** Distillation (the teacher labels for you) is the
  default *when you have no labels*. If you already have labels — human ground
  truth, or an existing agent's decisions — those are valid training data too
  (human labels are the strongest signal). Distillation is the fallback, not a
  restriction.

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

The core is pure stdlib; torch/transformers come only via `[local]`/`[finetune]`.
CI runs the suite on the core install across Python versions.

## Reference

Implements ideas from STROT (Structured Task Reasoning and Output Transformation,
[arXiv:2505.01636](https://arxiv.org/abs/2505.01636)). MIT licensed; the base
encoder is Apache-2.0. Source: <https://github.com/strot-ai/STROT-Compiler>.
