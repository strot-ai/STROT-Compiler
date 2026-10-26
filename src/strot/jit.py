"""The production JIT front door: :func:`strot.run`.

STROT as a *production* compiler, not a dev tool. Developers set only the
intent (a natural-language query plus an action registry); the compile event
happens wherever :func:`run` is first called — typically in production — and
the resulting asset is persisted to a content-addressed artifact store. Every
subsequent call (and every restart sharing the ``state`` directory) is a pure
cache hit: no LLM, no API key, just deterministic execution.

    key = hash(query + schema fingerprint + registry + nlu + version)
    ├─ artifact exists  ──▶ load compiled code, execute          (every call)
    └─ cache miss       ──▶ LLM synthesizes, validate, persist   (once)

The schema fingerprint is part of the key, so schema drift in the live data
automatically misses the cache and triggers exactly one recompile against
the new reality.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from threading import RLock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .compiler import (
    CompiledAsset,
    LLMClient,
    LLMHook,
    StrotCompiler,
    anthropic_llm_client,
    claude_cli_llm_client,
    codex_cli_llm_client,
    codex_sdk_llm_client,
    openai_llm_client,
)
from .exceptions import RouterConfigurationError, SchemaIntrospectionError
from .router import CONFIDENCE_THRESHOLD, DEFAULT_NLU_MODEL
from .schema import DatasetSchema, introspect_schema

#: ``llm`` parameter: a client callable, or a lazy spec string resolved only
#: on a cache miss ("openai/gpt-6-luna", "gpt-6-luna", "codex-cli", "claude-cli/sonnet").
LLMSpec = Union[str, LLMClient]

_KEY_PREFIX_LEN = 16
_ARTIFACT_LOCK = RLock()


@dataclass
class RunResult:
    """Outcome of one :func:`run` call over a dataset."""

    outcomes: List[Dict[str, Any]]
    fallbacks: List[Dict[str, Any]]
    artifact_path: Path
    cache_hit: bool
    cache_key: str
    pipeline: Any = None  #: live CompiledPipeline for streaming further records
    attempts: int = 0  #: synthesizer attempts (0 on a cache hit)
    mode: str = "decision"

    def summary(self) -> Dict[str, Any]:
        if self.mode == "decide":
            return {
                "mode": "decide",
                "processed": len(self.outcomes),
                "cache_hit": self.cache_hit,
                "artifact": str(self.artifact_path),
            }
        per_action: Dict[str, int] = {}
        for outcome in self.outcomes:
            per_action[outcome["action"]] = per_action.get(outcome["action"], 0) + 1
        return {
            "processed": len(self.outcomes),
            "fallbacks": len(self.fallbacks),
            "actions": per_action,
            "cache_hit": self.cache_hit,
            "artifact": str(self.artifact_path),
        }


def as_records(data: Any) -> List[Dict[str, Any]]:
    """Normalize a DataFrame-like, dict-of-lists, or list-of-dicts into records."""
    if hasattr(data, "to_dict") and not isinstance(data, Mapping):
        return [dict(r) for r in data.to_dict(orient="records")]
    if isinstance(data, Mapping):
        if any(not isinstance(v, (list, tuple)) for v in data.values()):
            raise SchemaIntrospectionError(
                "Column-oriented data must map each header to a sequence of values."
            )
        columns = {str(k): list(v) for k, v in data.items()}
        if not columns:
            raise SchemaIntrospectionError("Dataset contains no columns.")
        length = max(len(v) for v in columns.values())
        return [
            {k: (v[i] if i < len(v) else None) for k, v in columns.items()}
            for i in range(length)
        ]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        records = list(data)
        if not records:
            raise SchemaIntrospectionError("Dataset is empty; cannot derive a schema.")
        if not all(isinstance(r, Mapping) for r in records):
            raise SchemaIntrospectionError(
                "Record-oriented data must be a sequence of mappings (dict rows)."
            )
        return [dict(r) for r in records]
    raise SchemaIntrospectionError(
        f"Unsupported data type {type(data).__name__!r}; "
        "expected a DataFrame-like object, list of dicts, or dict of lists."
    )


def schema_fingerprint(schema: DatasetSchema) -> List[List[str]]:
    """Structural identity of a schema: names and types, not sample values.

    Cardinality, null rates, and sample values vary batch to batch and must
    not churn the cache key; a changed column set or type is real drift and
    must miss the cache.
    """
    return sorted(
        [f.name, f.semantic_type, f.python_type] for f in schema.fields
    )


def cache_key(
    query: str,
    action_registry: Optional[Mapping[str, str]],
    nlu: str,
    schema: DatasetSchema,
    question: Optional[str],
    confidence_threshold: float,
    mode: str = "decision",
    golden: Optional[Any] = None,
    target_risk: Optional[float] = None,
    measure_labels: bool = False,
    distill: bool = False,
    target_accuracy: Optional[float] = None,
) -> str:
    from . import __version__

    payload = {
        "strot": __version__,
        "mode": mode,
        "query": query,
        "registry": list(action_registry.items()) if action_registry else [],
        "nlu": nlu,
        "question": question,
        "threshold": confidence_threshold,
        "schema": schema_fingerprint(schema),
        "golden": golden,
        "target_risk": target_risk,
        "measure_labels": measure_labels,
        "distill": distill,
        "target_accuracy": target_accuracy,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def llm_client(
    spec: LLMSpec, *, reasoning_effort: Optional[str] = None,
    on_result: Optional[LLMHook] = None,
    on_event: Optional[LLMHook] = None, on_turn: Optional[LLMHook] = None,
) -> LLMClient:
    """Create a lazy synthesizer with shared hooks. Custom callables keep their own hooks."""
    client, lock = None, RLock()

    def _call(prompt: str) -> str:
        nonlocal client
        with lock:
            if client is None:
                client = _resolve_llm(spec, reasoning_effort=reasoning_effort,
                                      on_result=on_result, on_event=on_event, on_turn=on_turn)
        return client(prompt)

    return _call


def _resolve_llm(llm: LLMSpec, reasoning_effort: Optional[str] = None, **hooks: Any) -> LLMClient:
    """Resolve a synthesizer spec or return a custom callable."""
    if callable(llm):
        return llm
    if isinstance(llm, str) and llm:
        if reasoning_effort is not None:
            hooks["reasoning_effort"] = reasoning_effort
        provider, slash, model = llm.partition("/")
        if provider == "anthropic":
            return anthropic_llm_client(model, **hooks) if model else anthropic_llm_client(**hooks)
        if provider in ("claude-cli", "claude-code"):
            return claude_cli_llm_client(model=model or None, **hooks)

        if provider == "openai":
            return openai_llm_client(model, **hooks) if model else openai_llm_client(**hooks)
        if provider == "codex-cli":
            return codex_cli_llm_client(model=model or None, **hooks)
        if provider == "codex-sdk":
            return codex_sdk_llm_client(model=model or None, **hooks)

        if not slash:  # bare model name, e.g. "gpt-6-luna"
            return openai_llm_client(llm, **hooks)
    raise RouterConfigurationError(
        f"Cannot resolve llm spec {llm!r}; pass a callable, 'anthropic/<model>', "
        "'openai/<model>', a bare OpenAI model name, 'codex-cli[/<model>]', "
        "'codex-sdk[/<model>]', or 'claude-cli[/<model>]'."
    )


def _artifact_paths(state: Union[str, Path], key: str) -> "tuple[Path, Path]":
    state_dir = Path(state)
    stem = f"strot_{key[:_KEY_PREFIX_LEN]}"
    return state_dir / f"{stem}.py", state_dir / f"{stem}.json"


def _artifact_module_name(path: Path, key: str) -> str:
    location = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"strot_compiled_{key}_{location}"


def _load_artifact(path: Path, key: str) -> Any:
    with _ARTIFACT_LOCK:
        module_name = _artifact_module_name(path, key)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
            cached = sys.modules.get(module_name)
            if cached is not None and getattr(cached, "__strot_source_stamp__", None) == stamp:
                return cached
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # Read source directly so a fast recompile cannot reuse stale bytecode.
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise RouterConfigurationError(
                f"Compiled artifact {path} failed to load: {exc}. The local runtime "
                "needs strot-ai[local] (torch/transformers); the hosted runtime "
                "needs strot-ai[hosted] (openai)."
            ) from exc
        module.__strot_source_stamp__ = stamp
        return module


def _log_episode(state: Union[str, Path], payload: Dict[str, Any]) -> None:
    """Append one compile episode to the state dir's episodes.jsonl.

    Every compile - success or failure - is a (configuration, outcome) pair:
    the raw material for future supervised/RL training of the compile step.
    Logging is best-effort and never breaks a compile.
    """
    from datetime import datetime, timezone

    try:
        state_dir = Path(state)
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with (state_dir / "episodes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


def run(
    data: Any,
    query: str,
    action_registry: Mapping[str, str],
    llm: LLMSpec = "openai/gpt-6-luna",
    nlu: str = DEFAULT_NLU_MODEL,
    state: Union[str, Path] = ".strot",
    sample_size: int = 25,
    max_retries: int = 3,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    question: Optional[str] = None,
    execute: bool = True,
    golden: Optional[List[Mapping[str, Any]]] = None,
    target_risk: Optional[float] = None,
    measure_labels: Optional[bool] = None,
    distill: bool = False,
    target_accuracy: Optional[float] = None,
) -> RunResult:
    """Process ``data`` against ``query``, JIT-compiling the pipeline on first use.

    STROT is a decision system — the LLM is confined to compile time. At
    runtime the final arbiter is the ML decision model (``nlu``): output is
    bounded, type-safe JSON (action, probabilities, confidence) with the
    compiled handlers executing the chosen path.

    Args:
        data: The dataset — DataFrame-like, list of dict records, or dict of
            column lists. The first ``sample_size`` records also serve as the
            synthesis/validation sample on a cache miss.
        query: The functional goal in natural language (the paper's ``goal``).
        action_registry: Ordered mapping of action key -> description; defines
            the bounded decision space enforced on the ML decision model.
        llm: Synthesizer spec or callable; resolved lazily, only on cache miss.
            ``codex-cli[/<model>]`` and ``claude-cli[/<model>]`` use CLI
            authentication. API clients need provider credentials.
            ``codex-sdk[/<model>]`` uses the optional Codex Python SDK.
        nlu: Runtime decision model baked into the asset (local or hosted).
        state: Artifact-store directory — the persisted "compiled flag + code".
        sample_size: Records used for schema profiling and synthesis grounding.
        max_retries: Feedback-driven synthesis budget on a cache miss.
        confidence_threshold: Routing confidence below this raises the
            compiled asset's FallbackException instead of executing a path.
        question: Optional override of the routing question baked into the asset.
        execute: If False, only ensure the artifact exists (warm the cache)
            without loading the runtime — useful on boxes without the runtime
            extras installed.
        golden: Optional known-answer records
            (``[{"record": {...}, "expect": "<action key>"}, ...]``) — the
            promotion gate. Candidate artifacts are executed against them
            inside the compile repair loop; only an artifact for which every
            expectation holds is persisted and served. Requires the decision
            runtime on the compile box.
        target_risk: Optional miscoverage level ``alpha`` (e.g. 0.05). Fits
            temperature scaling + a split-conformal threshold from an
            LLM-teacher-labeled sample at compile time (disjoint folds
            preserve conformal validity); the artifact then dispatches only
            on singleton prediction sets. The guarantee is MARGINAL: wrong
            dispatches are <= alpha of ALL records under exchangeability;
            conditional-on-dispatch error is bounded by alpha/dispatch_rate
            (estimated in the calibration metadata). Requires the decision
            runtime on the compile box.
        measure_labels: Score the synthesized routing labels against the bare
            registry keys on the teacher-labeled sample and bake the winner -
            measurement instead of guessing. Defaults to ON whenever
            target_risk is set (the teacher labels are already paid for);
            pass True to enable it alone (triggers one teacher call) or
            False to disable.
        distill: Compile-time distillation - train a logistic head on the
            frozen encoder's embeddings from teacher-labeled records and bake
            it in if it beats the zero-shot router on a held-out fold. Use a
            larger sample_size (100+); requires the [local] runtime deps on
            the compile box.
        target_accuracy: Empirical dispatch policy (mutually exclusive with
            target_risk): temperature + per-class bias corrections and a
            dispatch cutoff are fitted on held-out teacher-labeled folds so
            auto-decided records meet the accuracy target; below-cutoff
            records abstain. An empirical fit, NOT a finite-sample
            guarantee; reported in POLICY metadata.
    """
    if not action_registry:
        raise RouterConfigurationError(
            "run() requires an action_registry (the bounded decision space "
            "enforced on the ML decision model)."
        )

    records = as_records(data)
    sample = records[:sample_size]
    schema = introspect_schema(sample)
    if golden is not None:
        StrotCompiler.validate_golden_run(golden, action_registry)
    effective_measure = (target_risk is not None) if measure_labels is None else measure_labels
    key = cache_key(
        query, action_registry, nlu, schema, question, confidence_threshold,
        golden=golden, target_risk=target_risk, measure_labels=effective_measure,
        distill=distill, target_accuracy=target_accuracy,
    )
    artifact_path, meta_path = _artifact_paths(state, key)

    cache_hit = artifact_path.exists()
    attempts = 0
    if not cache_hit:
        compiler = StrotCompiler(
            nlu_model=nlu,
            llm_client=_resolve_llm(llm),
            max_retries=max_retries,
            confidence_threshold=confidence_threshold,
        )
        try:
            asset: CompiledAsset = compiler.compile(
                sample=sample,
                goal=query,
                action_registry=action_registry,
                question=question,
                golden=golden,
                target_risk=target_risk,
                measure_labels=effective_measure,
                distill=distill,
                target_accuracy=target_accuracy,
            )
        except Exception as exc:
            _log_episode(state, {
                "kind": "run", "status": "failed", "cache_key": key,
                "query": query, "actions": list(action_registry),
                "golden_count": len(golden or []), "target_risk": target_risk,
                "error": str(exc)[:500],
            })
            raise
        _log_episode(state, {
            "kind": "run", "status": "promoted", "cache_key": key,
            "query": query, "actions": list(action_registry),
            "golden_count": len(golden or []), "target_risk": target_risk,
            "attempts": asset.attempts, "calibration": asset.calibration,
            "label_search": asset.label_search, "model_search": asset.model_search,
            "policy": asset.policy,
        })
        attempts = asset.attempts
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with _ARTIFACT_LOCK:
            asset.save(artifact_path)
            sys.modules.pop(_artifact_module_name(artifact_path, key), None)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_key": key,
                    "query": query,
                    "nlu": nlu,
                    "runtime_kind": asset.runtime_kind,
                    "attempts": asset.attempts,
                    "action_registry": dict(action_registry),
                    "schema": schema_fingerprint(schema),
                    "calibration": asset.calibration,
                    "label_search": asset.label_search,
                    "model_search": asset.model_search,
                    "policy": asset.policy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    result = RunResult(
        outcomes=[],
        fallbacks=[],
        artifact_path=artifact_path,
        cache_hit=cache_hit,
        cache_key=key,
        attempts=attempts,
    )
    if not execute:
        return result

    module = _load_artifact(artifact_path, key)
    result.pipeline = module.CompiledPipeline()
    for index, record in enumerate(records):
        try:
            outcome = result.pipeline.process(record)
            result.outcomes.append({"index": index, "record": record, **outcome})
        except module.FallbackException as exc:
            result.fallbacks.append(
                {
                    "index": index,
                    "record": record,
                    "confidence": exc.confidence,
                    "reason": str(exc),
                }
            )
    return result


def decide(
    data: Any,
    questions: Union[str, Mapping[str, Any]],
    llm: LLMSpec = "openai/gpt-6-luna",
    nlu: str = DEFAULT_NLU_MODEL,
    state: Union[str, Path] = ".strot",
    sample_size: int = 25,
    max_retries: int = 3,
    execute: bool = True,
    golden: Optional[List[Mapping[str, Any]]] = None,
) -> RunResult:
    """Answer a typed-question decision config over every record.

    ``questions`` is a typed-question config (a dict or JSON string; a full config
    object with a top-level ``"questions"`` key is also accepted)::

        {
          "is_bug_report": {"type": "boolean", "instructions": "..."},
          "customer_sentiment": {"type": "score", "min": 1, "max": 5,
                                  "instructions": "..."},
          "routing_bucket": {"type": "choice",
                              "options": ["billing_team", "engineering_team"]},
        }

    Each outcome's ``decision`` is typed decision JSON::

        {"booleans": {name: {"type": "boolean", "probability": 0.99}},
         "scores": {name: {"value": 1.8, "confidence": 0.89}},
         "choices": {name: {"selected": ..., "probabilities": {...},
                             "confidence": 0.94}}}

    The compile phase (once per config + schema) synthesizes choice criteria,
    score rubrics, boolean case descriptions, and the state projection; at
    runtime the local ML decision model answers all questions per record in
    one batched pass. Output is bounded, type-safe JSON - nothing generated.
    """
    if isinstance(questions, str):
        questions = json.loads(questions)
    if isinstance(questions, Mapping) and isinstance(questions.get("questions"), Mapping):
        questions = questions["questions"]  # accept a full config object with a questions key
    StrotCompiler.validate_questions_config(questions)
    if golden is not None:
        StrotCompiler.validate_golden_decide(golden, questions)

    records = as_records(data)
    sample = records[:sample_size]
    schema = introspect_schema(sample)
    canonical_config = json.dumps(questions, sort_keys=True)
    key = cache_key(
        canonical_config, None, nlu, schema, None, 0.0, mode="decide",
        golden=golden,
    )
    artifact_path, meta_path = _artifact_paths(state, key)

    cache_hit = artifact_path.exists()
    attempts = 0
    if not cache_hit:
        compiler = StrotCompiler(
            nlu_model=nlu,
            llm_client=_resolve_llm(llm),
            max_retries=max_retries,
        )
        try:
            asset = compiler.compile_decide(
                sample=sample, questions=questions, golden=golden
            )
        except Exception as exc:
            _log_episode(state, {
                "kind": "decide", "status": "failed", "cache_key": key,
                "questions": list(questions), "golden_count": len(golden or []),
                "error": str(exc)[:500],
            })
            raise
        _log_episode(state, {
            "kind": "decide", "status": "promoted", "cache_key": key,
            "questions": list(questions), "golden_count": len(golden or []),
            "attempts": asset.attempts,
        })
        attempts = asset.attempts
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with _ARTIFACT_LOCK:
            asset.save(artifact_path)
            sys.modules.pop(_artifact_module_name(artifact_path, key), None)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_key": key,
                    "mode": "decide",
                    "questions": dict(questions),
                    "nlu": nlu,
                    "attempts": asset.attempts,
                    "schema": schema_fingerprint(schema),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    result = RunResult(
        outcomes=[],
        fallbacks=[],
        artifact_path=artifact_path,
        cache_hit=cache_hit,
        cache_key=key,
        attempts=attempts,
        mode="decide",
    )
    if not execute:
        return result

    module = _load_artifact(artifact_path, key)
    result.pipeline = module.CompiledDecider()
    for index, record in enumerate(records):
        result.outcomes.append(
            {"index": index, "record": record, "decision": result.pipeline.process(record)}
        )
    return result
