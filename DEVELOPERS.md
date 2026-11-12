# STROT Developer Guide

This guide covers the architecture, the compile lifecycle, the decision-model tier ladder, calibration semantics (including the honest caveats), PII masking at the teacher boundary, batteries, deployment patterns, and how to extend STROT. For the quickstart, see the [README](README.md).

## Contents

1. [Architecture](#architecture)
2. [The compile lifecycle](#the-compile-lifecycle)
3. [The tier ladder](#the-tier-ladder)
4. [The artifact (battery)](#the-artifact-battery)
5. [Golden records](#golden-records)
6. [Calibration, abstention, and guarantees](#calibration-abstention-and-guarantees)
7. [PII masking at the teacher boundary](#pii-masking-at-the-teacher-boundary)
8. [Distributing batteries](#distributing-batteries)
9. [API reference](#api-reference)
10. [Deployment patterns](#deployment-patterns)
11. [Swapping the decision model](#swapping-the-decision-model)
12. [LLM clients](#llm-clients)
13. [Episode log](#episode-log)
14. [Testing your integration](#testing-your-integration)
15. [Versioning and cache busting](#versioning-and-cache-busting)

---

## Architecture

```
src/strot/
├── jit.py        run() / decide(), battery cache, episode logging
├── compiler.py   StrotCompiler: prompts, validation/repair loop, golden gate,
│                 teacher labeling, tier escalation, calibration fitting
├── calibrate.py  temperature scaling + split conformal + class bias +
│                 selective cutoff (pure Python)
├── distill.py    frozen-encoder embedder, logistic head, three-way split
├── finetune.py   compile-time distillation (cached-lower + top-N LoRA, K-ensemble)
├── pii.py        PII masking at the teacher boundary (regex pre-pass + GLiNER)
├── battery.py    data-only battery bundles: export / load / list / info
├── templates.py  hand-written runtime code rendered into batteries
├── router.py     runtime strategies (local / distilled / fine-tuned / hosted),
│                 OOD centroid guard
├── schema.py     SchemaProfiler: verified schema from the sample
├── prefetch.py   build-time model baking (prefetch_nlu)
└── exceptions.py
```

Design invariant: **intelligence at compile time, determinism at runtime.** The LLM, the gates, the tier search, and all statistical fitting run once per configuration; the battery that serves traffic is frozen, self-contained, and never updates itself. Even the fine-tune tier is tuned once, just-in-time on the first compile, and then frozen — it is never tuned per inference, and there is no runtime loop.

The core package is pure stdlib. Heavy dependencies arrive only through extras, and only on the boxes that need them.

## The compile lifecycle

A cache miss in `run()`/`decide()` triggers this sequence:

1. **Schema profiling** — the sample (first `sample_size` records) is introspected into a verified schema: field names, semantic types, null rates, example values. This grounds every prompt.
2. **Synthesis** — the LLM produces the configuration contract: routing labels, state projection, handlers (`run()`), or expanded criteria/rubrics (`decide()`).
3. **Validation / repair loop** — each candidate is checked: AST parse, exact contract (function names, label keys match the registry), and a security lint (ML packages plus stdlib I/O / network / process / dynamic-execution imports and calls are rejected). Failures feed back verbatim; the LLM retries within `max_retries`. **The lint is validation, not a sandbox**: the golden gate `exec()`s candidates on the compile box, so compiling fully untrusted intent requires external isolation (container, locked-down user, resource limits).
4. **Golden gate** — if `golden=` is provided, the assembled artifact is *executed* against your known-answer records. Any mismatch is a validation failure: the error (expected vs got, with confidence) feeds the repair loop. Only an artifact for which every expectation holds is promoted.
5. **Teacher labeling** — if `target_risk=`, `target_accuracy=`, `measure_labels=`, `distill=`, or `optimize=` is set, the LLM annotates the real sample records once (chunked into batches of 50, one call per batch, to keep the exact-count contract reliable at any sample size). Every record is **PII-masked before it reaches the teacher** — see [PII masking](#pii-masking-at-the-teacher-boundary). The label indices come back against the original records.
6. **Measured label search** — if `measure_labels=` is on, the synthesized routing labels and the bare registry keys are both scored on the teacher-labeled measure fold through the actual artifact; the winner is baked in (measurement instead of guessing — synthesis wins when integration is the bottleneck and loses when the keys are already optimal). A winner that breaks the goldens is rejected, keeping the synthesized labels; scores land in `label_search` metadata.
7. **Tier search** — the optimizer picks the cheapest decision tier that clears the target on a held-out real fold: the zero-shot route, then a distilled logistic head, then (opt-in) a fine-tuned adapter. See [The tier ladder](#the-tier-ladder). Only tiers above zero-shot run here, and only when `distill=`/`optimize=` ask for them.
8. **Calibration** — if `target_risk=` is set, temperature and a conformal quantile are fitted on the winning tier's output distributions and baked in; if `target_accuracy=` is set, temperature + per-class bias + a dispatch cutoff are fitted instead. The golden gate re-runs on the calibrated battery.
9. **Persistence** — the battery (`battery_<slug>_<hash>.py`) and its metadata (`battery_<slug>_<hash>.json`) land in the `state` directory; the compile is appended to `episodes.jsonl`. The slug is a human-readable task label from `name=` or the query (cosmetic only — the hash is the cache key).

Every later call with the same cache key loads the battery and executes — zero LLM involvement.

## The tier ladder

The decision layer is a ladder of three rungs, cheapest first. The optimizer keeps **the cheapest rung that clears the target** on a held-out fold of real, teacher-labeled records. Most tasks stay on the frozen rungs and never touch the fine-tune tier.

| Tier | What it is | Extra | Weights baked | Engaged by |
|---|---|---|---|---|
| **Zero-shot route** | the decision model scores the synthesized routing labels directly — no training | `[local]` (or hosted) | none | always the baseline |
| **Distilled head** | a logistic head on the **frozen** encoder's embeddings | `[local]` | `HEAD_WEIGHTS` / `HEAD_BIAS` | `distill=True` |
| **JIT Tuning** (fine-tune tier) | cached-lower + top-N LoRA on ModernBERT, a small K-ensemble of diverse adapters | `[finetune]` | `ADAPTER` / `ADAPTER_CONFIG` | `optimize=True` **and** a missed target |

**The frozen rungs.** With `distill=True`, a logistic head is trained on the frozen encoder's pooled embeddings (train fold) and kept **only if it beats the zero-shot route** on the held-out measure fold. The encoder weights never move; this caps at the frozen representation ceiling. For representation-bound tasks — where the decision-critical distinction is not linearly present in the pooled embedding — no head crosses that ceiling; only changing weights does.

**JIT Tuning (the fine-tune tier).** Technically this is *compile-time distillation*: the only honest framing is that it is tuned **just-in-time on the first compile and then frozen** — never per inference, never in a runtime loop. It is the one rung that changes weights, so it is opt-in and heavy (`strot.finetune`, the `[finetune]` extra = torch + transformers). It escalates only when **all** of these hold:

- `optimize=True`, and
- an accuracy/risk target is set (today this means `target_accuracy=`), and
- the best frozen rung (zero-shot or distilled) **misses that target on the real measure fold**.

Without a target, `optimize=True` is a no-op (`model_search["optimize"] == "skipped_no_target"`); when the frozen tier already clears the bar it is also skipped (`"skipped_frozen_cleared_bar"`). When it does fire, it fine-tunes **once**: the lower encoder layers are frozen and their activations cached, only the top `N` layers (default 4, the measured knee) carry a trainable LoRA adapter, and a small ensemble of `K` diverse adapters (default 3, varied by seed) is trained and **softmax-averaged at inference** — the same average the compile-time judge scores. The trained adapter(s) win only if they beat the frozen baseline on the measure fold and pass the golden gate; the winner lands in `model_search["winner"] == "fine_tuned"`. `K` and `N` are internal knobs (`FineTuneConfig`), never on the public API. A missing `[finetune]` extra does not crash the compile — it **degrades gracefully to the frozen tier** and records why (`model_search["optimize"] == "unavailable"`).

Fold hygiene is strict once `distill=`/`optimize=` is on: teacher-labeled records are split three ways (train / measure / calibrate) so training, judging, and calibration never share data. This needs more records — `optimize`/`distill` require `sample_size` ≥ 48 and reward 100+.

**`reuse_adapter=`** — adopt a pre-built adapter instead of training. Pass a prior compile's baked `ADAPTER` / `ADAPTER_CONFIG` as `{"adapters": [...], "config": {...}}`. When `optimize` is on and the frozen tier misses the bar, the provided battery is **validated on this task's own measure fold** (no training); if it clears the target, beats the frozen baseline, and passes the golden gate, it is adopted and training is skipped (`model_search["optimize"] == "reused"`). A validation miss — or a different label space, or a missing `[finetune]` extra — falls through to normal training, never fatal. Calibration is always re-fit on this task's own calibrate fold, never reused from the old battery. `reuse_adapter` is part of the cache key (a reused compile keys differently from a trained one) and is ignored when `optimize` is off.

## The artifact (battery)

The compiled artifact is a **battery**: one standalone Python file with zero `strot` imports, written to the `state` directory as `battery_<slug>_<hash>.py` (the slug comes from `name=` or the query; the hash is the cache key). Anatomy:

```python
NLU_MODEL = "SupersonicLabs/Julia-1"     # decision model baked at compile
CONFIDENCE_THRESHOLD = 0.4               # legacy threshold (uncalibrated path)
TEMPERATURE = 0.25                       # fitted (1.0 if uncalibrated)
CONFORMAL_Q = 5e-06                      # fitted (None if uncalibrated)
CLASS_BIAS = [...]                       # fitted per-class bias (target_accuracy)
DISPATCH_CUTOFF = 0.7                    # fitted cutoff (target_accuracy)
CALIBRATION = {...}                      # the full fitted conformal report
POLICY = {...}                           # the full fitted dispatch policy
OOD = {...}                              # drift guard: centroids + threshold
DATASET_SCHEMA = {...}                   # the schema it was compiled against

ROUTING_LABELS = {...}                   # synthesized: what the model sees
HEAD_WEIGHTS / HEAD_BIAS = ...           # distilled tier only
ADAPTER / ADAPTER_CONFIG = ...           # fine-tune tier only
def route_context(record): ...           # synthesized: state projection
def execute_<action>(record): ...        # synthesized handlers (run() only)

class _RuntimeRouter: ...                # from reviewed templates
class CompiledPipeline:                  # run() batteries
    def process(self, record) -> dict    # {"action", "confidence", "result"}
class CompiledDecider:                   # decide() batteries
    def process(self, record) -> dict    # {"booleans", "scores", "choices"}
```

Which literals appear depends on the winning tier: a distilled battery carries `HEAD_WEIGHTS`/`HEAD_BIAS`, a fine-tuned one carries `ADAPTER`/`ADAPTER_CONFIG`, a zero-shot one neither. The runtime block is selected to match (`_RuntimeRouter` is the local, distilled, fine-tuned, or hosted template).

Batteries are deployable alone: copy the `.py` to any box with the runtime deps (see [Deployment](#deployment-patterns)) and instantiate the class directly. `python battery_<slug>_<hash>.py < records.jsonl` also works as a CLI filter. For shipping the decision as *data* (weights + config, no code) instead of the whole `.py`, see [Distributing batteries](#distributing-batteries).

## Golden records

```python
golden=[{"record": {...}, "expect": "billing_dispute"}, ...]           # run()
golden=[{"record": {...}, "expect": {"team": "engineering",           # decide()
                                     "is_bug": True,
                                     "sentiment": [1, 2.5]}}, ...]
```

- `run()`: `expect` is the action key.
- `decide()`: `expect` maps question names to — choice: the option; boolean: `True`/`False` (checked against probability ≥ 0.5); score: a number (±0.5 tolerance) or `[low, high]` range.

Semantics worth knowing:

- Goldens are validated **before any LLM call** — malformed goldens fail fast and free.
- The gate runs **inside** the repair loop: a failing candidate costs one retry, and the specific failures are shown to the LLM.
- Goldens are part of the **cache key**: changing them forces a re-gated recompile.
- The gate needs the decision runtime **on the compile box** (it executes the artifact). A missing runtime raises immediately rather than burning retries.
- 3–5 goldens covering one clear example per action is the practical sweet spot. They are your regression suite against compile variance — two compiles of the same config can differ in quality, and the gate is what makes that invisible to production.

## Calibration, abstention, and guarantees

**The default: no abstention.** With neither `target_risk=` nor `target_accuracy=` set, the battery dispatches **every** record. It attaches a calibrated confidence to each decision but never hard-abstains on the calibrated path — the raw argmax and its probability pass through unchanged. (The legacy uncalibrated `confidence_threshold` still gates below `0.40`, but that path is being superseded by the fitted policies below.) Set a target to turn abstention on; abstained records land in `result.fallbacks` for the caller to re-run however it likes — there is no runtime retry loop inside the battery.

Two mutually exclusive ways to set a bar:

- **`target_risk=`** — a split-conformal guarantee (below).
- **`target_accuracy=`** — an empirical dispatch policy (further below).

### `target_risk`: split conformal

`target_risk=0.1` buys you a fitted, quantified dispatch policy. What actually happens:

1. The LLM teacher labels your real sample records (annotation of real data — never synthetic generation; conformal validity requires calibration points from the same distribution as runtime traffic).
2. The labeled points are **split into disjoint folds**: temperature is fitted on one, the conformal quantile on the other. This is required for validity — a score transform fitted on the quantile's own calibration points makes the scores non-exchangeable with fresh records.
3. **Temperature scaling** (fold A): one scalar fitted by NLL so the model's probabilities mean what they say. Falls back to T=1 below 4 points.
4. **Split conformal quantile** (fold B) at `alpha = target_risk`, over nonconformity scores `1 − p_T(true label)`, rounded conservatively upward.
5. At runtime the artifact forms the prediction set `{label : p_T(label) ≥ 1 − q}` and **dispatches only on a singleton**; empty or multi-label sets raise `FallbackException`.

**Read the guarantee precisely.** Split conformal targets **marginal coverage of at least 1−α under exchangeability**, with abstention on non-singleton prediction sets. Consequence: the rate of wrong dispatches **over all incoming records** is ≤ α. The error rate **conditional on dispatching** is *not* bounded by α — it is bounded by α / dispatch-rate, which the artifact reports as `conditional_error_bound_estimate` in its `CALIBRATION` metadata. At a 90% dispatch rate the two nearly coincide; at a 20% dispatch rate the conditional rate may reach 5α. Phrase claims as "wrong dispatches ≤ α of all records", never "≤ α of dispatched records."

**Honest caveats, surfaced in `CALIBRATION` metadata:**

- `quantile_granularity = 1/(n_conformal+1)`: fold splitting halves the effective calibration set, so a 25-record sample yields ~12 conformal points (granularity ~8%). Want α=0.01? You need ≥200 labeled sample records (`sample_size=200`).
- `temperature` pinned at the grid edge (0.25 or 4.0) means the calibration set was too easy or too hard to be informative — inspect it.
- Teacher labels are LLM annotations: imperfect labels loosen the real guarantee. Production feedback (resolved fallbacks, corrections) is a better calibration source once you have it — refit by recompiling with it as goldens/sample.
- The guarantee holds for traffic that looks like the sample. Distribution shift voids it — which is one more reason the schema fingerprint busts the cache on drift.

### `target_accuracy`: empirical dispatch policy

`target_accuracy=` fits an empirical selective-prediction policy instead of a conformal one (the two are mutually exclusive — setting both raises). On the teacher-labeled calibrate fold it fits temperature + per-class bias corrections (on one disjoint half) and a confidence **dispatch cutoff** (on the other) such that records auto-decided above the cutoff meet the accuracy target; below-cutoff records abstain and surface in `result.fallbacks`. The fitted knobs and estimated tier-1 coverage/accuracy are reported in `POLICY` metadata.

This is an **empirical fit, not a finite-sample guarantee** — there is no coverage theorem behind the number. It also **optimizes accuracy**, which is the wrong objective for imbalanced tasks where minority recall matters: it will trade recall for accuracy. Prefer `target_risk` when you need a defensible bound.

### The OOD / drift guard

When the distilled tier wins **and** `optimize=True`, the compiler also fits an out-of-distribution guard on the train-fold frozen embeddings: per-class centroids plus a high percentile (95th) of in-distribution nearest-centroid distances, baked as the `OOD` literal. At runtime a record farther than that threshold from every centroid is refused up front, before routing. A plain `distill=True` compile (no `optimize`) bakes an empty `OOD` literal, which the template treats as a no-op — so distilled runtime abstention is unchanged unless you opt in.

## PII masking at the teacher boundary

The cloud LLM is confined to compile time and never sees production traffic — but at compile time one boundary *does* hand real sample records to the cloud teacher: the labeling call that turns records into supervision (`strot.pii`). STROT masks at exactly that boundary. **Before any record reaches the teacher, it is masked**; the label indices the teacher returns apply to the original, unmasked records. The runtime stays entirely mask-free.

The default masker (`pii.default_maskers()`) is a two-stage pipeline:

1. **A regex pre-pass** for format-defined tokens regex can catch with near-zero false positives: email, SSN, Luhn-validated credit card, common secret/credential shapes (AWS keys, JWTs, `sk-`/`ghp_`/`xox…` provider keys, `Bearer` tokens), IBAN, IP (v4/v6), MAC, and phone (checked last so the more specific digit patterns claim their spans first).
2. **GLiNER** (`urchade/gliner_medium-v2.1`), a zero-shot NER model, for the linguistic PII regex cannot catch: person names, addresses, organizations, locations. This is the **default** masker and carries the headline coverage.

Regex runs *first* on purpose: running NER first can tag part of an email domain as an organization and leave a mangled `jane@[ORG]` that the email regex can no longer match, leaking the local part.

GLiNER needs the **`[pii]` extra** (torch + the gliner model, loaded once and cached). Without it, `default_maskers()` **warns and degrades to the regex pre-pass alone** — base installs still compile, but names/addresses are not masked. None of this is a compliance guarantee; deployments can pass their own `maskers=` list (a stronger NER/PII model, an allowlist projection in `route_context`, etc.) to `pii.mask_records`.

## Distributing batteries

A battery normally ships as its self-contained `.py` (handlers + projection + runtime, all inlined). For sharing the **decision** without the code, `strot.battery` offers a parallel **data-only** representation (`strot.export_bundle`, `strot.load`, `strot.list_batteries`, `strot.battery_info`).

A battery here is a **data-only classifier**: given a projected context string it emits a label, a calibrated confidence, the full class distribution, and whether it abstained — reproducing `CompiledPipeline`'s *decision* (within floating-point tolerance) but **not** handler dispatch. The synthesized `execute_*` handlers and `route_context` projection are app-side code, not shippable data, so they stay in your application; `Battery.decide(context)` takes the already-projected context the decision model sees.

**`export_bundle(asset, directory)`** writes a bundle — a directory with two files:

- `battery.json` — pure JSON (torch-free to read): task metadata, routing labels, fitted calibration (temperature / conformal quantile / class bias / dispatch cutoff / policy), optional OOD guard, and a schema fingerprint.
- `weights.safetensors` — the numeric weights: the distilled head (`HEAD_WEIGHTS`/`HEAD_BIAS`) or the adapter tensors (`ADAPTER` members + serving config).

The bundle data is captured by **reading the compiled asset's module-level literals** (`ast.literal_eval`, never `exec`), so it is torch-free to produce and cannot drift from the baked `.py`. Only `distilled` / `fine_tuned` runtimes are exportable — a zero-shot local or hosted battery has no shippable weights, so distribute it as the compiled `.py`.

**`load(name_or_path)`** returns a `Battery`. It disambiguates by the argument: an existing local bundle directory loads straight off disk; anything else is treated as a **bare battery name** and resolved from the Hugging Face **dataset** repo `strot-ai/batteries` (pinned by default so a bare name never resolves somewhere untrusted; retarget with `repo=` or the `STROT_BATTERY_REPO` env var, arg winning over env). `list_batteries()` reads the repo's root `manifest.json` (or a local catalog directory), and `battery_info(name)` reads one `battery.json` — both torch-free, needing only `huggingface_hub` on the hub path. The torch-backed base model loads lazily on the first `decide()`, resolving the base checkpoint the same way the runtime does (`STROT_NLU_CHECKPOINT` → local dir → HF download).

## API reference

### `strot.run(data, query, action_registry, ...)`

| Param | Default | Meaning |
|---|---|---|
| `data` | — | DataFrame-like, list of dicts, or dict of lists |
| `query` | — | the goal, natural language |
| `action_registry` | — | ordered `{key: description}`; the bounded decision space |
| `llm` | `"openai/gpt-4o"` | synthesizer spec or callable; resolved **only on cache miss** |
| `nlu` | `SupersonicLabs/Julia-1` | decision model baked into the artifact |
| `state` | `".strot"` | artifact store directory |
| `sample_size` | 25 | records used for schema, synthesis grounding, teacher labeling |
| `max_retries` | 3 | repair-loop budget |
| `confidence_threshold` | 0.40 | fallback bar (uncalibrated artifacts) |
| `golden` | None | promotion gate records |
| `target_risk` | None | α for the fitted split-conformal guarantee (abstains on non-singleton prediction sets) |
| `target_accuracy` | None | empirical dispatch policy (mutually exclusive with `target_risk`): fit temperature + per-class bias corrections and a confidence cutoff on held-out folds so auto-decided records meet the target; below-cutoff records abstain into `fallbacks`. NOT a finite-sample guarantee. **Optimizes accuracy** — wrong objective for imbalanced tasks where minority recall matters (it will trade recall for accuracy) |
| `measure_labels` | None | score synthesized labels vs bare registry keys on the teacher-labeled measure fold, keep the winner; defaults ON when `target_risk` is set |
| `distill` | False | the distilled tier: train a logistic head on the frozen encoder's embeddings from teacher-labeled records; baked in only if it beats zero-shot on the held-out measure fold (three-way split: train/measure/calibrate). Needs `sample_size` ≥ 48 (100+ recommended), a **local** decision model, and `[local]` deps on the compile box |
| `optimize` | False | permit the **fine-tune tier** (JIT Tuning) above the frozen rungs. No-op without an accuracy target; only escalates when the frozen tier misses that target on the measure fold, then trains a cached-lower + top-N LoRA K-ensemble once at compile. Opt-in and heavy: needs a local decision model and the `[finetune]` extra (degrades gracefully to the frozen tier if missing). Implies distill-style fold hygiene (`sample_size` ≥ 48, 100+ recommended) |
| `reuse_adapter` | None | adopt a prior compile's baked adapter (`{"adapters": [...], "config": {...}}`) instead of training: validated on this task's measure fold, adopted if it clears the bar, else falls through to training. Part of the cache key; ignored when `optimize` is off |
| `name` | None | cosmetic label seeding the battery filename slug (`battery_<slug>_<hash>.py`); auto-derived from the query when omitted. NOT part of the cache key |
| `execute` | True | False = warm the cache only (no runtime needed) |

Returns `RunResult`: `outcomes` (list of `{index, record, action, confidence, result}`), `fallbacks` (`{index, record, confidence, reason}` — records that abstained), `pipeline` (live, for streaming single records), `cache_hit`, `cache_key`, `attempts`, `artifact_path`, `summary()`.

### `strot.decide(data, questions, ...)`

`questions` is a typed-question config (dict or JSON string; a wrapper object with a `"questions"` key is accepted):

```python
{"<name>": {"type": "choice",  "options": [...]},
 "<name>": {"type": "score",   "min": 1, "max": 5, "instructions": "..."},
 "<name>": {"type": "boolean", "instructions": "..."}}
```

Score spans must cover 2–20 levels; choice takes 2–20 options (the decision model's native range). Each outcome carries `decision = {"booleans": ..., "scores": ..., "choices": ...}` with probabilities and confidences. Shares `llm`, `nlu` (local only for now), `state`, `sample_size`, `golden`, `name`, `execute` semantics with `run()`. `optimize` and `reuse_adapter` are accepted for API parity and participate in the cache key, but decide mode has no routing-tier escalation, so they are otherwise inert here.

### `strot.prefetch_nlu(model=..., local_dir=..., install=False)`

Downloads the decision-model checkpoint (and with `install=True`, pip-installs its bundled inference package). For image build time.

## Deployment patterns

**CI compile (recommended default).** Compile in CI with a representative fixture, review the battery diff like any code change, ship the frozen `.py`. Prod needs only the runtime deps — the battery never changes underneath you.

**Prod compile (self-provisioning).** Call `run()`/`decide()` in the service with a persistent `state` volume. First boot compiles against live data; every restart is a cache hit. Treat an unexpected recompile as an alertable event, and always use `golden=` — in unattended compiles the gate is the only reviewer.

**Runtime dependency matrix:**

| Box | Needs |
|---|---|
| Compile-only (`execute=False`) | `strot-ai` + LLM client (+ `[local]` runtime if using `golden`/`target_risk`/`target_accuracy`; `[pii]` for name/address masking; `[finetune]` if `optimize=True`) |
| Prod, zero-shot / distilled local tier | battery + `[local]` deps + baked checkpoint (`STROT_NLU_CHECKPOINT`) |
| Prod, fine-tune tier (`ADAPTER` baked) | battery + `[finetune]` deps + the ModernBERT base checkpoint |
| Prod, hosted decision model | battery + `openai` + provider key |

The fine-tune and distilled tiers run at compile on the compile box, but serving a fine-tuned battery needs the `[finetune]` extra (it reconstructs the cached-lower + top-N LoRA model to score contexts). Checkpoint resolution in batteries: `STROT_NLU_CHECKPOINT` env → `NLU_MODEL` as a local directory → Hugging Face download (cached).

## Swapping the decision model

The `nlu=` slot resolves by name:

- Local checkpoint path or HF repo id → `LocalEncoderStrategy` (native typed-decision API)
- `openai/<model>`, `anthropic/<model>`, `azure/<model>`, `gemini/<model>` → `HostedApiStrategy` (structured outputs)

Two further strategies are selected by the winning **tier**, not the `nlu=` name: the distilled tier serves the frozen encoder plus the saved logistic head, and the fine-tune tier serves the base encoder plus the saved adapter(s) via `FineTunedAdapterStrategy` (softmax-averaging a K-adapter ensemble). `distill=`/`optimize=` require a **local** `nlu` — the distilled/fine-tune tiers need encoder embeddings; a hosted `nlu` with either flag raises.

To add a new local engine, implement the strategy contract — `select(state, question, options) -> RouteDecision(index, option, confidence, distribution)` — in `router.py` and the corresponding template block. The compiler, gates, calibration, and cache are model-agnostic and apply unchanged; conformal calibration in particular fits **any** probabilistic selector.

## LLM clients

`llm=` accepts any `(prompt: str) -> str` callable, or a lazy spec string:

| Spec | Client |
|---|---|
| `"anthropic/claude-opus-4-8"` | Anthropic API (`[anthropic]` extra, `ANTHROPIC_API_KEY`) |
| `"openai/gpt-4o"` or bare model | OpenAI API (`[hosted]` extra, `OPENAI_API_KEY`) |
| `"codex-cli"` / `"codex-cli/<model>"` | Codex CLI with saved login. Run `codex login` for subscription access. No SDK required. |
| `"claude-cli"` / `"claude-cli/<model>"` | local Claude Code CLI as proxy (no key, no SDK) |

Specs resolve **only on a cache miss** — a warm production box never needs the LLM dependency or key.

Both CLIs run in an isolated temporary directory and keep CLI authentication.
Codex skips user configuration and disables project instructions, shell commands,
and web search. Claude uses `--safe-mode` to skip customizations and disables
built-in tools and session persistence. See the
[Codex configuration reference](https://developers.openai.com/codex/config-reference)
and [Claude CLI reference](https://code.claude.com/docs/en/cli-reference).

## Episode log

Every compile appends one JSON line to `<state>/episodes.jsonl`:

```json
{"ts": "...", "kind": "run", "status": "promoted", "cache_key": "...",
 "query": "...", "actions": [...], "golden_count": 3, "target_risk": 0.1,
 "attempts": 1, "calibration": {...}, "label_search": {...},
 "model_search": {...}, "policy": {...}}
```

`model_search` records the tier search (zero-shot vs distilled vs fine-tuned accuracies on the measure fold, the winner, the ensemble size, and why the fine-tune tier did or did not escalate); `policy` records the fitted `target_accuracy` dispatch policy; `label_search` records the measured-labeling scores. Any of these is `null` when its feature was not engaged.

Failures log too (`status: "failed"`, with the error). Logging is best-effort and never breaks a compile. Purpose: an audit trail today; the training dataset for compile-policy learning (supervised, later RL) tomorrow. Keep it.

## Testing your integration

Unit-test without the 550 MB model by faking the decision package (see `tests/test_jit.py` for the full pattern):

```python
monkeypatch.setitem(sys.modules, "julia", fake_julia_module)   # your fake engine
monkeypatch.setenv("STROT_NLU_CHECKPOINT", str(tmp_path))      # skip downloads
```

and script the LLM with a deterministic callable. The repo's `make_paths()` helper shows the exact synthesis contract a fake must satisfy. For integration tests against the real model, bake the checkpoint into a Docker image with `prefetch_nlu` (the repo's test images build in under a minute once the heavy layers are cached).

To test the real Julia model without downloads, use an installed inference package and a cached checkpoint:

```bash
STROT_TEST_CHECKPOINT=/path/to/Julia-1 python -m pytest -q tests/test_quality.py
```

This test uses nine ticket cases and a fixed compiler response. It does not test live LLM output or general model accuracy.

## Versioning and cache busting

The battery cache key hashes: package version, mode, query/config, registry, decision model, question, threshold, schema fingerprint (field names + types — not values), goldens, `target_risk`, `measure_labels`, `distill`, `target_accuracy`, `optimize`, and a compact projection of `reuse_adapter` (its config + adapter count, never the raw tensors). The battery filename slug (`name=`) is **not** in the key. Consequences:

- Changing any of those = exactly one recompile. Changing none = cache hit forever.
- Schema **drift** (new/removed/retyped fields) busts the key automatically; value changes don't.
- Package upgrades bust all caches on purpose (templates may have changed).
- Artifacts are content-addressed: name artifacts by key, keep the previous one for rollback, and alert on unexpected recompiles in prod.

Loaded modules are keyed by the full cache key and canonical artifact path.
File metadata changes trigger a reload. Concurrent loads share a completed module.
