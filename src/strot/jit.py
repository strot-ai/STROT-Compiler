"""The JIT front door: :func:`strot.questions` and :func:`strot.decide`.

STROT as a *production* compiler, not a dev tool. You provide a one-line intent
(or a reviewed typed-question config) plus a sample of your real data; the
compile event happens the first time :func:`decide` is called and the resulting
model (the frozen, self-contained decision unit the compiler emits) is persisted
to a content-addressed store. Every subsequent call (and every restart sharing
the ``state`` directory) is a pure cache hit: no LLM, no API key, just the local
model.

    key = hash(questions + schema fingerprint + nlu + mode + version)
    ├─ model exists   ──▶ load compiled code, decide            (every call)
    └─ cache miss     ──▶ LLM designs/labels, train, persist    (once)

On disk a compiled model is a pair ``battery_<slug>_<hash>.py`` / ``.json``: the
slug is a human-readable task label (from ``name`` or auto-derived) and is purely
cosmetic — the content hash, not the slug, is the cache key.

The schema fingerprint is part of the key, so schema drift in the live data
automatically misses the cache and triggers exactly one recompile against the
new reality.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from threading import RLock, Thread
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .compiler import (
    CompiledAsset,
    LLMClient,
    StrotCompiler,
    anthropic_llm_client,
    claude_cli_llm_client,
    codex_cli_llm_client,
    openai_llm_client,
)
from .exceptions import RouterConfigurationError, SchemaIntrospectionError
from .constants import CONFIDENCE_THRESHOLD, DEFAULT_NLU_MODEL
from .schema import DatasetSchema, introspect_schema

#: ``llm`` parameter: a client callable, or a lazy spec string resolved only
#: on a cache miss ("openai/gpt-4o", "gpt-4o", "codex-cli", "claude-cli/sonnet").
LLMSpec = Union[str, LLMClient]

_KEY_PREFIX_LEN = 16
_ARTIFACT_LOCK = RLock()
_SLUG_MAX_LEN = 24


class DecisionModel:
    """A compiled, self-contained decision model.

    Returned by :func:`decide` and reloaded by :func:`load_model`. It is
    callable - ``model(record)`` returns the typed decision for one record -
    and saveable - ``model.save("name")`` writes the whole model to one
    ``.strot`` file (heads baked in; the encoder downloads by id at runtime)::

        model = strot.decide(tickets, "triage support tickets")
        model(tickets[0])            # -> {"predicates": {...}, "choices": {...}, ...}
        model.save("ticket-triage")  # -> Path("ticket-triage.strot")
        same = strot.load_model("ticket-triage")
    """

    def __init__(
        self,
        artifact_path: Union[str, Path],
        key: str,
        questions: Optional[Dict[str, Any]] = None,
        confidence_threshold: Optional[float] = None,
        pipeline: Any = None,
        tier: str = "head",
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.questions = questions  #: the typed question config this model answers
        self.confidence_threshold = confidence_threshold
        self.tier = tier  #: "lite" | "head" - which tier is serving right now
        self._key = key
        self._pipeline = pipeline
        self._lock = RLock()
        self._thread: Optional[Thread] = None  #: warm-start background upgrade

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            module = _load_artifact(self.artifact_path, self._key)
            self._pipeline = module.CompiledDecider()
        return self._pipeline

    def __call__(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Decide one record: returns typed decision JSON (locally, on CPU)."""
        with self._lock:
            decision = self._ensure_pipeline().process(record)
        if self.confidence_threshold is not None:
            decision = _apply_refusals(decision, self.confidence_threshold)
        return decision

    def _swap_to(
        self, artifact_path: Union[str, Path], key: str,
        questions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Hot-swap the backing artifact (warm-start: lite -> trained head)."""
        with self._lock:
            self.artifact_path = Path(artifact_path)
            self._key = key
            self._pipeline = None  # lazy-reload the upgraded head on next call
            if questions is not None:
                self.questions = questions
            self.tier = "head"

    def wait(self, timeout: Optional[float] = None) -> "DecisionModel":
        """Block until a warm-start upgrade finishes (no-op if not warm)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self

    def save(self, path: Union[str, Path]) -> Path:
        """Write the self-contained model to ``path`` (``.strot`` if no suffix)."""
        dest = Path(path)
        if dest.suffix == "":
            dest = dest.with_suffix(".strot")
        if dest.parent != Path("."):
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.artifact_path.read_bytes())
        return dest


def load_model(path: Union[str, Path]) -> DecisionModel:
    """Load a decision model saved by :meth:`DecisionModel.save`.

    ``path`` is the ``.strot`` file (the suffix is optional). The returned model
    is callable on a record and re-saveable. Running it needs ``strot-ai[local]``
    (torch/transformers) to load the encoder.
    """
    p = Path(path)
    if p.suffix == "" and not p.exists():
        p = p.with_suffix(".strot")
    if not p.exists():
        raise FileNotFoundError(f"No decision model at {str(p)!r}.")
    key = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return DecisionModel(artifact_path=p, key=key)


@dataclass
class RunResult:
    """Outcome of one :func:`run` call over a dataset.

    For a :func:`decide` outcome this is also the decision model: it is callable
    (``result(record)``) and saveable (``result.save("name")``), delegating to a
    :class:`DecisionModel` over the same compiled artifact.
    """

    outcomes: List[Dict[str, Any]]
    fallbacks: List[Dict[str, Any]]
    artifact_path: Path
    cache_hit: bool
    cache_key: str
    pipeline: Any = None  #: live CompiledPipeline for streaming further records
    attempts: int = 0  #: synthesizer attempts (0 on a cache hit)
    mode: str = "decision"
    questions: Optional[Dict[str, Any]] = None  #: the config decided on (handy when decide() designed it from a bare intent)
    confidence_threshold: Optional[float] = None  #: refusal policy carried onto the callable model
    _model: Any = field(default=None, compare=False, repr=False)

    @property
    def model(self) -> DecisionModel:
        """The compiled artifact as a reusable, saveable :class:`DecisionModel`."""
        if self._model is None:
            self._model = DecisionModel(
                self.artifact_path, self.cache_key, self.questions,
                self.confidence_threshold, pipeline=self.pipeline,
            )
        return self._model

    def __call__(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Decide one further record against the compiled model."""
        return self.model(record)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the compiled model to one ``.strot`` file; returns the path."""
        return self.model.save(path)

    @property
    def tier(self) -> str:
        """Which tier is serving now - ``"lite"`` until a warm upgrade swaps in ``"head"``."""
        return self.model.tier

    def wait(self, timeout: Optional[float] = None) -> "RunResult":
        """Block until a warm-start head upgrade finishes (no-op if not warm)."""
        self.model.wait(timeout)
        return self

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


def _reuse_adapter_key(reuse_adapter: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """A stable, compact cache-key projection of a reuse-adapter battery.

    The raw adapters are MB of float tensors - unsuitable for a cache key. The
    battery's identity for keying is its config (base_model, cut, n_layers,
    lora_rank, lora_alpha, max_length, n_classes) plus how many adapters it
    carries; two batteries with the same config and width compile the same
    adopt-or-train decision. ``None`` when no battery is provided.
    """
    if reuse_adapter is None:
        return None
    return {
        "config": reuse_adapter.get("config"),
        "n_adapters": len(reuse_adapter.get("adapters") or []),
    }


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
    optimize: bool = False,
    reuse_adapter: Optional[Mapping[str, Any]] = None,
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
        "optimize": optimize,
        # A reused-adapter compile must key differently from a trained one.
        # Hash a stable projection (the adapter config + battery size), never
        # the raw float tensors, so the key is small and deterministic.
        "reuse_adapter": _reuse_adapter_key(reuse_adapter),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_llm(llm: LLMSpec) -> LLMClient:
    """Resolve the synthesizer client. Called ONLY on a cache miss, so a warm
    production box never needs the LLM dependency or its API key."""
    if callable(llm):
        return llm
    if isinstance(llm, str) and llm:
        provider, slash, model = llm.partition("/")
        if provider in ("claude-cli", "claude-code"):
            return claude_cli_llm_client(model=model or None)
        if provider == "codex-cli":
            return codex_cli_llm_client(model=model or None)
        if provider == "anthropic":
            return anthropic_llm_client(model) if model else anthropic_llm_client()
        if provider == "openai":
            return openai_llm_client(model)
        if not slash:  # bare model name, e.g. "gpt-4o"
            return openai_llm_client(llm)
    raise RouterConfigurationError(
        f"Cannot resolve llm spec {llm!r}; pass a callable, 'anthropic/<model>', "
        "'openai/<model>', a bare OpenAI model name, 'codex-cli[/<model>]', "
        "or 'claude-cli[/<model>]'."
    )


def _battery_slug(name: Optional[str], query: str) -> str:
    """A short, filesystem-safe task label for the battery filename.

    Cosmetic only: it rides in the ``.py`` / ``.json`` stem so a human can tell
    one battery from another on disk, but it is NOT part of the cache key (that
    stays a pure content address). Prefer an explicit ``name``; otherwise derive
    from the first meaningful words of the query. Lowercased, ``[a-z0-9-]`` only,
    hyphens collapsed/trimmed, capped at ~24 chars, ``"task"`` if nothing survives.
    """
    source = name if name else query
    slug = re.sub(r"[^a-z0-9]+", "-", (source or "").lower()).strip("-")
    slug = slug[:_SLUG_MAX_LEN].strip("-")
    return slug or "task"


def _artifact_paths(state: Union[str, Path], slug: str, key: str) -> "tuple[Path, Path]":
    state_dir = Path(state)
    stem = f"battery_{slug}_{key[:_KEY_PREFIX_LEN]}"
    return state_dir / f"{stem}.py", state_dir / f"{stem}.json"


def _battery_module_name(path: Path, key: str) -> str:
    location = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"strot_compiled_{key}_{location}"


def _load_artifact(path: Path, key: str) -> Any:
    with _ARTIFACT_LOCK:
        module_name = _battery_module_name(path, key)
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
                f"Compiled model {path} failed to load: {exc}. Serving a decision "
                "model needs strot-ai[local] (torch/transformers)."
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


def _apply_refusals(decision: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Replace below-threshold answers with ``{"refusal": True}``, in place.

    A post-hoc policy over an already-produced decision: it never touches the
    compiled battery or the wire shapes, only swaps a whole answer for a
    refusal when the model is not sure enough. Confidence per type is the
    ``confidence`` field for scores and choices; a predicate carries a
    probability instead, so its confidence is ``max(p, 1 - p)`` - how sure it
    is either way. A single global threshold; a per-question override could be
    layered on later if a config ever needs it.
    """
    for group, answers in decision.items():
        for qname, answer in answers.items():
            if group == "predicates":
                p = answer["probability"]
                confidence = max(p, 1.0 - p)
            else:
                confidence = answer["confidence"]
            if confidence < threshold:
                answers[qname] = {"refusal": True}
    return decision


def questions(
    data: Any,
    intent: str,
    llm: LLMSpec = "openai/gpt-4o",
    sample_size: int = 25,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Intake: design a typed question set from a one-line decision intent.

    The LLM reads ``intent`` and the data schema and designs a small
    ``{name: {type, ...}}`` config (predicate / choice / score). It designs the
    decision schema only - it does not label, train, or invent data. Review and
    edit the result, then pass it to :func:`decide`::

        questions = strot.questions(tickets, "triage support tickets")
        # ...author reviews/edits questions...
        strot.decide(tickets, questions)
    """
    return _design_questions(data, intent, llm, sample_size, max_retries)


def _design_questions(
    data: Any,
    intent: str,
    llm: LLMSpec,
    sample_size: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Shared schema-design step: LLM proposes a typed question set from an intent.

    Used both by the public :func:`questions` intake and by :func:`decide` when
    it is handed a bare intent instead of a config.
    """
    sample = as_records(data)[:sample_size]
    compiler = StrotCompiler(
        nlu_model=DEFAULT_NLU_MODEL,
        llm_client=_resolve_llm(llm),
        max_retries=max_retries,
    )
    return compiler.propose_questions(sample, intent)


def decide(
    data: Any,
    questions: Union[str, Mapping[str, Any]],
    llm: LLMSpec = "openai/gpt-4o",
    state: Union[str, Path] = ".strot",
    sample_size: int = 25,
    max_retries: int = 3,
    execute: bool = True,
    golden: Optional[List[Mapping[str, Any]]] = None,
    confidence_threshold: Optional[float] = None,
    name: Optional[str] = None,
    mode: str = "head",
    warm: bool = False,
) -> RunResult:
    """Answer a typed-question decision config over every record.

    ``questions`` is either a **typed-question config** or a **bare intent**:

    * A config - a dict or JSON string; a full config object with a top-level
      ``"questions"`` key is also accepted. This is the reproducible path: you
      hold the schema, review it, and version-control it.
    * A plain-language **intent** string (anything that is not JSON) - the LLM
      designs the schema for you first, then compiles. This is the AI-first
      quick path: ``strot.decide(data, "triage support tickets")``. The schema
      it chose is returned on ``result.questions`` so you can inspect it; once
      you like it, pin it via :func:`questions` for a reproducible build.

    A config looks like::

        {
          "is_bug_report": {"type": "predicate", "instructions": "..."},
          "customer_sentiment": {"type": "score", "instructions": "...",
                                  "levels": [
                                      {"label": "angry", "description": "..."},
                                      {"label": "neutral", "description": "..."},
                                      {"label": "happy", "description": "..."}]},
          "routing_bucket": {"type": "choice", "choices": [
              {"value": "billing_team", "description": "payments, invoices, refunds"},
              {"value": "engineering_team", "description": "crashes, bugs, defects"}]},
        }

    A ``"choice"`` value may be a string or a boolean; each choice's
    ``"description"`` is optional and guides the compile-phase criteria. The
    bare ``"options": [value, ...]`` list is still accepted as a deprecated
    alias (values only, no descriptions).

    A ``"score"`` carries 2-10 ordered ``"levels"`` (lowest to highest), each a
    ``{"label": ..., "description": ...}`` mapping (description optional); the
    answer is the best-matching level's label. The bare ``"min"``/``"max"``
    integer range is still accepted as a deprecated alias (one level per
    integer, the integers as labels).

    A ``"predicate"`` question returns a probability (0-1 likelihood the
    statement is true); ``"boolean"`` is still accepted as a deprecated alias.

    Each outcome's ``decision`` is typed decision JSON::

        {"predicates": {name: {"type": "predicate", "probability": 0.99}},
         "scores": {name: {"score": "neutral", "probabilities": {...},
                           "confidence": 0.89}},
         "choices": {name: {"choice": ..., "probabilities": {...},
                             "confidence": 0.94}}}

    The compile phase (once per config + schema) synthesizes choice criteria,
    score rubrics, predicate case descriptions, and the state projection; at
    runtime the local ML decision model answers all questions per record in
    one batched pass. Output is bounded, type-safe JSON - nothing generated.

    The base encoder is a STROT decision - not a caller choice - so there is no
    ``nlu``/``model`` argument; questions compile to one distilled head each.

    ``confidence_threshold`` is an opt-in per-question refusal policy: when set,
    any answer whose confidence falls below it is replaced with
    ``{"refusal": True}`` while the other questions still answer. It is a
    post-hoc view applied after the compiled battery runs - NOT part of the
    battery or the cache key, so changing it never forces a recompile. With the
    default ``None``, nothing is ever refused (dispatch-by-default).

    ``name`` is an optional label for the compiled battery. It seeds the on-disk
    filename slug (``battery_<slug>_<hash>.py``); when omitted the slug is
    auto-derived from the question names. Cosmetic only — not part of the cache
    key.

    ``mode`` picks the tier. ``"lite"`` is the local $0 zero-shot floor: each
    class becomes a prototype embedding of its description, scored by cosine - no
    teacher pass, no training, instant. ``"head"`` (default) distills a calibrated
    logistic head per question from teacher labels on your data. ``"deep"`` adds a
    shared top-N LoRA adapter trained over all questions (needs the ``[finetune]``
    extra and more records), then refits + calibrates the heads on the adapted
    embedding - the ceiling tier for when a frozen head is representation-bound.

    ``warm`` starts the model serving immediately with the ``lite`` tier (local,
    $0) and trains the ``head`` in a background thread, hot-swapping it in when
    ready. ``result(record)`` never blocks: it answers from ``lite`` until the
    head lands, then from the head. ``result.tier`` reports which is serving and
    ``result.wait()`` blocks until the upgrade finishes. The caller must have
    enough records to train the head, same as ``mode="head"``.
    """
    if warm:
        # Resolve a bare intent once, then share the config across both tiers so
        # the background head reuses the lite floor's schema (no re-design).
        if isinstance(questions, str):
            try:
                json.loads(questions)
            except (json.JSONDecodeError, ValueError):
                questions = _design_questions(data, questions, llm, sample_size, max_retries)
        common = dict(
            llm=llm, state=state, sample_size=sample_size, max_retries=max_retries,
            golden=golden, confidence_threshold=confidence_threshold, name=name,
        )
        floor = decide(data, questions, execute=execute, mode="lite", warm=False, **common)
        model = floor.model
        model.tier = "lite"

        def _upgrade_to_head() -> None:
            try:
                trained = decide(data, questions, execute=False, mode="head",
                                 warm=False, **common)
                model._swap_to(
                    trained.artifact_path, trained.cache_key, trained.questions
                )
            except Exception as exc:  # keep lite serving; never crash the thread
                _log_episode(state, {
                    "kind": "decide", "status": "warm_upgrade_failed",
                    "error": str(exc)[:500],
                })

        thread = Thread(target=_upgrade_to_head, name="strot-warm-head", daemon=True)
        model._thread = thread
        thread.start()
        return floor

    designed_from_intent = None
    if isinstance(questions, str):
        try:
            parsed = json.loads(questions)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping):
            questions = parsed  # a JSON config string
        else:
            # Bare intent prose: let the LLM design the schema (AI-first quick path).
            # Reproducible production use should pin the result via strot.questions().
            designed_from_intent = questions
            questions = _design_questions(data, questions, llm, sample_size, max_retries)
    if isinstance(questions, Mapping) and isinstance(questions.get("questions"), Mapping):
        questions = questions["questions"]  # accept a full config object with a questions key
    StrotCompiler.validate_questions_config(questions)
    if golden is not None:
        StrotCompiler.validate_golden_decide(golden, questions)

    records = as_records(data)
    sample = records[:sample_size]
    schema = introspect_schema(sample)
    if mode not in ("lite", "head", "deep"):
        raise RouterConfigurationError(
            f"decide mode must be 'lite', 'head', or 'deep' (got {mode!r})."
        )
    canonical_config = json.dumps(questions, sort_keys=True)
    nlu = DEFAULT_NLU_MODEL  # STROT owns the base encoder; it is not a caller choice
    # Fold the tier into the cache key so lite/head never collide. "head" keeps
    # the bare "decide" key so existing head caches stay valid.
    key_mode = "decide" if mode == "head" else f"decide-{mode}"
    key = cache_key(
        canonical_config, None, nlu, schema, None, 0.0, mode=key_mode, golden=golden,
    )
    # The canonical config is JSON, so auto-slug off the question names instead.
    slug = _battery_slug(name, " ".join(questions))
    artifact_path, meta_path = _artifact_paths(state, slug, key)

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
                sample=sample, questions=questions, golden=golden, mode=mode
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
            "attempts": asset.attempts, "designed_from_intent": designed_from_intent,
        })
        attempts = asset.attempts
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with _ARTIFACT_LOCK:
            asset.save(artifact_path)
            sys.modules.pop(_battery_module_name(artifact_path, key), None)
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
        questions=dict(questions),
        confidence_threshold=confidence_threshold,
    )
    if not execute:
        return result

    module = _load_artifact(artifact_path, key)
    result.pipeline = module.CompiledDecider()
    for index, record in enumerate(records):
        decision = result.pipeline.process(record)
        if confidence_threshold is not None:
            decision = _apply_refusals(decision, confidence_threshold)
        result.outcomes.append(
            {"index": index, "record": record, "decision": decision}
        )
    return result
