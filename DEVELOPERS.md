# STROT Developer Guide

This guide covers the architecture, the compile lifecycle, calibration semantics (including the honest caveats), deployment patterns, and how to extend STROT. For the quickstart, see the [README](README.md).

## Contents

1. [Architecture](#architecture)
2. [The compile lifecycle](#the-compile-lifecycle)
3. [The artifact](#the-artifact)
4. [Golden records](#golden-records)
5. [Calibration and guarantees](#calibration-and-guarantees)
6. [API reference](#api-reference)
7. [Deployment patterns](#deployment-patterns)
8. [Swapping the decision model](#swapping-the-decision-model)
9. [LLM clients](#llm-clients)
10. [Episode log](#episode-log)
11. [Testing your integration](#testing-your-integration)
12. [Versioning and cache busting](#versioning-and-cache-busting)

---

## Architecture

```
src/strot/
├── jit.py        run() / decide(), artifact cache, episode logging
├── compiler.py   StrotCompiler: prompts, validation/repair loop,
│                 golden gate, teacher labeling, calibration fitting
├── calibrate.py  temperature scaling + split conformal (pure Python)
├── templates.py  hand-written runtime code rendered into artifacts
├── router.py     runtime strategies (local decision model / hosted API)
├── schema.py     SchemaProfiler: verified schema from the sample
├── prefetch.py   build-time model baking (prefetch_nlu)
└── exceptions.py
```

Design invariant: **intelligence at compile time, determinism at runtime.** The LLM, the gates, and all statistical fitting run once per configuration; the artifact that serves traffic is frozen, self-contained, and never updates itself.

The core package is pure stdlib. Heavy dependencies arrive only through extras, and only on the boxes that need them.

## The compile lifecycle

A cache miss in `run()`/`decide()` triggers this sequence:

1. **Schema profiling** — the sample (first `sample_size` records) is introspected into a verified schema: field names, semantic types, null rates, example values. This grounds every prompt.
2. **Synthesis** — the LLM produces the configuration contract: routing labels, state projection, handlers (`run()`), or expanded criteria/rubrics (`decide()`).
3. **Validation / repair loop** — each candidate is checked: AST parse, exact contract (function names, label keys match the registry), and a security lint (ML packages plus stdlib I/O / network / process / dynamic-execution imports and calls are rejected). Failures feed back verbatim; the LLM retries within `max_retries`. **The lint is validation, not a sandbox**: the golden gate `exec()`s candidates on the compile box, so compiling fully untrusted intent requires external isolation (container, locked-down user, resource limits).
4. **Golden gate** — if `golden=` is provided, the assembled artifact is *executed* against your known-answer records. Any mismatch is a validation failure: the error (expected vs got, with confidence) feeds the repair loop. Only an artifact for which every expectation holds is promoted.
5. **Teacher labeling** — if `target_risk=` or `measure_labels=` is set, the LLM annotates the real sample records once (one call, shared by both features).
6. **Measured label search** — the synthesized routing labels and the bare registry keys are both scored on the teacher-labeled sample through the actual artifact; the winner is baked in (measurement instead of guessing — benchmarks showed synthesis wins when integration is the bottleneck and loses when the keys are already optimal). A winner that breaks the goldens is rejected, keeping the synthesized labels; scores land in `label_search` metadata.
7. **Calibration** — if `target_risk=` is set, temperature and a conformal quantile are fitted on the final artifact's output distributions and baked in. The golden gate re-runs on the calibrated artifact.
8. **Persistence** — the artifact (`strot_<key>.py`) and its metadata (`strot_<key>.json`) land in the `state` directory; the compile is appended to `episodes.jsonl`.

Every later call with the same cache key loads the artifact and executes — zero LLM involvement.

## The artifact

A compiled artifact is one standalone Python file with zero `strot` imports. Anatomy:

```python
NLU_MODEL = "SupersonicLabs/Julia-1"     # decision model baked at compile
CONFIDENCE_THRESHOLD = 0.4               # legacy threshold (uncalibrated path)
TEMPERATURE = 0.25                       # fitted (1.0 if uncalibrated)
CONFORMAL_Q = 5e-06                      # fitted (None if uncalibrated)
CALIBRATION = {...}                      # the full fitted report
DATASET_SCHEMA = {...}                   # the schema it was compiled against

ROUTING_LABELS = {...}                   # synthesized: what the model sees
def route_context(record): ...           # synthesized: state projection
def execute_<action>(record): ...        # synthesized handlers (run() only)

class _RuntimeRouter: ...                # from reviewed templates
class CompiledPipeline:                  # run() artifacts
    def process(self, record) -> dict    # {"action", "confidence", "result"}
class CompiledDecider:                   # decide() artifacts
    def process(self, record) -> dict    # {"booleans", "scores", "choices"}
```

Artifacts are deployable alone: copy the `.py` to any box with the runtime deps (see [Deployment](#deployment-patterns)) and instantiate the class directly. `python strot_<key>.py < records.jsonl` also works as a CLI filter.

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

## Calibration and guarantees

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

## API reference

### `strot.run(data, query, action_registry, ...)`

| Param | Default | Meaning |
|---|---|---|
| `data` | — | DataFrame-like, list of dicts, or dict of lists |
| `query` | — | the goal, natural language |
| `action_registry` | — | ordered `{key: description}`; the bounded decision space |
| `llm` | `"openai/gpt-6-luna"` | synthesizer spec or callable; resolved **only on cache miss** |
| `nlu` | `SupersonicLabs/Julia-1` | decision model baked into the artifact |
| `state` | `".strot"` | artifact store directory |
| `sample_size` | 25 | records used for schema, synthesis grounding, teacher labeling |
| `max_retries` | 3 | repair-loop budget |
| `confidence_threshold` | 0.40 | fallback bar (uncalibrated artifacts) |
| `golden` | None | promotion gate records |
| `target_risk` | None | α for the fitted conformal guarantee |
| `measure_labels` | None | score synthesized labels vs bare registry keys on the teacher-labeled sample, keep the winner; defaults ON when `target_risk` is set |
| `distill` | False | train a logistic head on the frozen encoder's embeddings from teacher-labeled records; baked in only if it beats zero-shot on a held-out fold (three-way split: train/measure/calibrate). Needs `sample_size` ≥ 24 (100+ recommended) and `[local]` deps on the compile box |
| `target_accuracy` | None | empirical dispatch policy (mutually exclusive with `target_risk`): fit temperature + per-class bias corrections and a confidence cutoff on held-out folds so auto-decided records meet the target; below-cutoff records abstain or escalate. NOT a finite-sample guarantee. **Optimizes accuracy** — wrong objective for imbalanced tasks where minority recall matters (it will trade recall for accuracy) |
| `execute` | True | False = warm the cache only (no runtime needed) |

Returns `RunResult`: `outcomes` (list of `{index, record, action, confidence, result}`), `fallbacks` (`{index, record, confidence, reason}`), `pipeline` (live, for streaming single records), `cache_hit`, `attempts`, `artifact_path`, `summary()`.

### `strot.decide(data, questions, ...)`

`questions` is a typed-question config (dict or JSON string; a wrapper object with a `"questions"` key is accepted):

```python
{"<name>": {"type": "choice",  "options": [...]},
 "<name>": {"type": "score",   "min": 1, "max": 5, "instructions": "..."},
 "<name>": {"type": "boolean", "instructions": "..."}}
```

Score spans must cover 2–20 levels; choice takes 2–20 options (the decision model's native range). Each outcome carries `decision = {"booleans": ..., "scores": ..., "choices": ...}` with probabilities and confidences. Shares `llm`, `nlu` (local only for now), `state`, `sample_size`, `golden`, `execute` semantics with `run()`.

### `strot.prefetch_nlu(model=..., local_dir=..., install=False)`

Downloads the decision-model checkpoint (and with `install=True`, pip-installs its bundled inference package). For image build time.

## Deployment patterns

**CI compile (recommended default).** Compile in CI with a representative fixture, review the artifact diff like any code change, ship the frozen `.py`. Prod needs only the runtime deps — the artifact never changes underneath you.

**Prod compile (self-provisioning).** Call `run()`/`decide()` in the service with a persistent `state` volume. First boot compiles against live data; every restart is a cache hit. Treat an unexpected recompile as an alertable event, and always use `golden=` — in unattended compiles the gate is the only reviewer.

**Runtime dependency matrix:**

| Box | Needs |
|---|---|
| Compile-only (`execute=False`) | `strot-ai` + LLM client (+ runtime if using `golden`/`target_risk`) |
| Prod, local decision model | artifact + `[local]` deps + baked checkpoint (`STROT_NLU_CHECKPOINT`) |
| Prod, hosted decision model | artifact + `openai` + provider key |

Checkpoint resolution order in artifacts: `STROT_NLU_CHECKPOINT` env → `NLU_MODEL` as a local directory → Hugging Face download (cached).

## Swapping the decision model

The `nlu=` slot resolves by name:

- Local checkpoint path or HF repo id → `LocalEncoderStrategy` (native typed-decision API)
- `openai/<model>`, `anthropic/<model>`, `azure/<model>`, `gemini/<model>` → `HostedApiStrategy` (structured outputs)

To add a new local engine, implement the strategy contract — `select(state, question, options) -> RouteDecision(index, option, confidence, distribution)` — in `router.py` and the corresponding template block. The compiler, gates, calibration, and cache are model-agnostic and apply unchanged; conformal calibration in particular fits **any** probabilistic selector.

## LLM clients

`llm=` accepts any `(prompt: str) -> str` callable, or a lazy spec string:

| Spec | Client |
|---|---|
| `"anthropic/claude-haiku-5-5"` | Anthropic API (`[anthropic]` extra, `ANTHROPIC_API_KEY`) |
| `"openai/gpt-6-luna"` or bare model | OpenAI API (`[hosted]` extra, `OPENAI_API_KEY`) |
| `"codex-cli"` / `"codex-cli/<model>"` | Codex CLI with saved login. Run `codex login` for subscription access. No SDK required. |
| `"codex-sdk"` / `"codex-sdk/<model>"` | Codex Python SDK (`[codex-sdk]` extra), with saved Codex login. |
| `"claude-cli"` / `"claude-cli/<model>"` | local Claude Code CLI as proxy (no key, no SDK) |

Specs resolve **only on a cache miss** — a warm production box never needs the LLM dependency or key.

For SDK calls, STROT selects the newest verified CLI from `PATH`, standard locations, or desktop apps. If none meets the SDK's pinned minimum version, STROT uses the bundle. Verification uses OpenAI signatures on macOS/Windows and online release hashes on Linux. Explicit `cli=` skips verification. Without the bundle, a compatible CLI is required. Dependency checks still report the bundle as missing.

Each Codex SDK synthesis call starts and closes its runtime. `openai_codex.CodexError` exceptions pass through unchanged. Empty or incomplete replies raise `CompilationError`.

OpenAI synthesis defaults to Luna with `none` effort. Set `reasoning_effort` on `strot.llm_client` or an OpenAI/Codex factory to change it. Other models keep provider defaults. Anthropic synthesis defaults to Haiku 5.5. The Anthropic and Claude CLI clients reject explicit effort. With reasoning enabled, OpenAI calls omit `temperature`.

Hosted Luna routing defaults to `low`. For OpenAI routing, pass `reasoning_effort` to `HostedApiStrategy` or set `STROT_NLU_REASONING_EFFORT` before startup. The argument overrides the environment.

`strot.llm_client(spec, **hooks)` accepts the same specs as `llm=`. It creates and reuses a client on first use. Custom callables keep their own callbacks.

All factories accept the callbacks below. Client calls return text. Unsupported callbacks are ignored. Callbacks run on the caller's thread and must return promptly. Errors pass through unchanged. Result fields vary by provider.

Call `handle.interrupt()` from `on_turn` to request cancellation. Cancelled turns and expired response waits raise `CompilationError` without a compiler retry.

SDK response waits default to 600 seconds. Set `timeout=60` on `strot.codex_sdk_llm_client` to change the limit. Expiry closes the runtime. Startup, callbacks, and cleanup can outlast the limit.

| Client | `on_result` | `on_event` | `on_turn` |
|---|---|---|---|
| Codex SDK | Native result, usage, duration | Typed notification | Active turn handle |
| OpenAI / Anthropic API | Native API response | Unused | Unused |
| Raw CLIs | Unused | Unused | Unused |

```python
llm = strot.llm_client("codex-sdk/gpt-6-luna", on_result=lambda result: print(result.usage))
result = strot.run(records, query=goal, action_registry=actions, llm=llm)
```

Changing `llm` or reasoning effort does not clear cached artifacts. Use a fresh `state` directory to compare compiles or recalibrate.

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
 "attempts": 1, "calibration": {...}}
```

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

The artifact cache key hashes: package version, query/config, registry, decision model, question, threshold, schema fingerprint (field names + types — not values), goldens, and target_risk. Consequences:

- Changing any of those = exactly one recompile. Changing none = cache hit forever.
- Schema **drift** (new/removed/retyped fields) busts the key automatically; value changes don't.
- Package upgrades bust all caches on purpose (templates may have changed).
- Artifacts are content-addressed: name artifacts by key, keep the previous one for rollback, and alert on unexpected recompiles in prod.

Loaded modules are keyed by the full cache key and canonical artifact path.
File metadata changes trigger a reload. Concurrent loads share a completed module.
