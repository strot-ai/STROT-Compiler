"""The JIT front door: :func:`strot.compile` and :func:`strot.load_model`.

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
import importlib.machinery
import importlib.util
import json
import re
import sys
from threading import RLock, Thread
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .compiler import CompiledAsset, StrotCompiler
from .exceptions import RouterConfigurationError, SchemaIntrospectionError
from .constants import CONFIDENCE_THRESHOLD, DEFAULT_NLU_MODEL, NLU_MAX_LENGTH
from .schema import DatasetSchema, introspect_schema

_KEY_PREFIX_LEN = 16
_ARTIFACT_LOCK = RLock()
_SLUG_MAX_LEN = 24


class DecisionModel:
    """A compiled, self-contained decision model.

    Returned by :func:`compile` and reloaded by :func:`load_model`. It is
    callable - ``model(record)`` returns the typed decision for one record - and
    dispatchable - ``model.dispatch("name")`` puts it into service as one
    ``.strot`` file (heads baked in; the encoder downloads by id at runtime)::

        model = strot.compile(tickets, questions, labels)
        model(tickets[0])                # -> {"predicates": {...}, "choices": {...}, ...}
        model.dispatch("ticket-triage")  # -> Path("ticket-triage.strot")
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
        self._trace_path: Optional[Path] = None
        self._trace_record = False

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            module = _load_artifact(self.artifact_path, self._key)
            self._pipeline = module.CompiledDecider()
        return self._pipeline

    def trace(self, path: Union[str, Path], include_record: bool = False) -> "DecisionModel":
        """Log every decision to a JSONL file (the teacher's eyes); returns self.

        Each call appends ``{ts, answers: {q: {value, confidence, refused}}}``.
        ``include_record=True`` also logs the input record - off by default
        (privacy). Read it back with :func:`strot.read_traces` /
        :func:`strot.summarize_traces`.
        """
        self._trace_path = Path(path)
        self._trace_record = include_record
        return self

    def __call__(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Decide one record: returns typed decision JSON (locally, on CPU)."""
        with self._lock:
            decision = self._ensure_pipeline().process(record)
        if self.confidence_threshold is not None:
            decision = _apply_refusals(decision, self.confidence_threshold)
        if self._trace_path is not None:
            self._log_trace(record, decision)
        return decision

    def _log_trace(self, record: Mapping[str, Any], decision: Dict[str, Any]) -> None:
        # Best-effort; a trace write never breaks a decision.
        try:
            from datetime import datetime, timezone

            from .trace import summarize_decision

            entry: Dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "answers": summarize_decision(decision),
            }
            if self._trace_record:
                entry["record"] = dict(record)
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

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

    def dispatch(self, path: Union[str, Path]) -> Path:
        """Put the model into service: write the self-contained model to ``path``
        (``.strot`` if no suffix) - the one file you ship and reload with
        :func:`strot.load_model`."""
        dest = Path(path)
        if dest.suffix == "":
            dest = dest.with_suffix(".strot")
        if dest.parent != Path("."):
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.artifact_path.read_bytes())
        return dest

    #: Back-compat alias for :meth:`dispatch`.
    save = dispatch


def load_model(path: Union[str, Path]) -> DecisionModel:
    """Load a decision model saved by :meth:`DecisionModel.dispatch`.

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
    (``result(record)``) and dispatchable (``result.dispatch("name")``), delegating to a
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

    def dispatch(self, path: Union[str, Path]) -> Path:
        """Put the compiled model into service: one self-contained ``.strot`` file."""
        return self.model.dispatch(path)

    #: Back-compat alias for :meth:`dispatch`.
    save = dispatch

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


def cache_key(
    query: str,
    nlu: str,
    schema: DatasetSchema,
    question: Optional[str],
    confidence_threshold: float,
    mode: str = "decision",
    golden: Optional[Any] = None,
) -> str:
    """A content address for a compiled model: same inputs -> same key -> cache
    hit. ``question`` carries the labels+projection fingerprint (see compile)."""
    from . import __version__

    payload = {
        "strot": __version__,
        "mode": mode,
        "query": query,
        "nlu": nlu,
        "question": question,
        "threshold": confidence_threshold,
        "schema": schema_fingerprint(schema),
        "golden": golden,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()




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
            # Load with an explicit source loader so any extension works - a
            # saved model is `<name>.strot`, which importlib would not recognize
            # from the suffix alone (spec_from_file_location returns None).
            loader = importlib.machinery.SourceFileLoader(module_name, str(path))
            spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
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


def train(
    data: Any,
    questions: Union[str, Mapping[str, Any]],
    labels: Optional[Sequence[Mapping[str, Any]]] = None,
    projection: Optional[str] = None,
    state: Union[str, Path] = ".strot",
    golden: Optional[List[Mapping[str, Any]]] = None,
    confidence_threshold: Optional[float] = None,
    name: Optional[str] = None,
    mode: str = "head",
    max_length: int = NLU_MAX_LENGTH,
    modality: str = "text",
    encoder: Optional[str] = None,
    image_field: Optional[str] = None,
    execute: bool = True,
) -> RunResult:
    """Compile a decision model - the deterministic, LLM-free door (``strot.compile``).

    STROT runs no LLM and holds no key. ``mode="head"`` (default) distills a
    calibrated head per question; ``mode="deep"`` adds a shared LoRA adapter. Both
    distill from ``labels`` you bring: a list of ``{question: answer}`` dicts, one
    per record in ``data`` order - an agent's own decisions, or human ground truth
    (the strongest signal).

    ``questions`` is a typed-question config (a dict, or a JSON string). Designing
    one from a bare intent is the agent's job (see the STROT skill), so a bare
    intent is not accepted here. ``projection`` is optional ``route_context``
    source; the default concatenates the record's text fields.

    ``modality="image"`` (head tier only) embeds each record's ``image_field``
    (a path / URL / bytes / PIL image) with a frozen vision ``encoder`` instead
    of the text encoder; the heads, OOD gate, and decision JSON are identical.

    ``recompile, don't increment`` holds: the baked weights are a pure function of
    ``(data, questions, labels, projection, mode, modality, encoder)``.
    """
    if isinstance(questions, str):
        try:
            parsed = json.loads(questions)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not isinstance(parsed, Mapping):
            raise RouterConfigurationError(
                "compile needs a typed-question config (dict or JSON string). "
                "Designing one from a bare intent is the agent's job - see the "
                "STROT skill (run `strot playbook`)."
            )
        questions = parsed
    if isinstance(questions, Mapping) and isinstance(questions.get("questions"), Mapping):
        questions = questions["questions"]  # accept a full config object
    StrotCompiler.validate_questions_config(questions)
    if golden is not None:
        StrotCompiler.validate_golden_decide(golden, questions)
    if mode not in ("head", "deep"):
        raise RouterConfigurationError(
            f"mode must be 'head' or 'deep' (got {mode!r})."
        )
    if modality not in ("text", "image"):
        raise RouterConfigurationError(
            f"modality must be 'text' or 'image' (got {modality!r})."
        )

    records = as_records(data)
    if labels is None:
        raise RouterConfigurationError(
            f"mode={mode!r} needs labels: one {{question: answer}} dict per record, "
            "in data order."
        )
    labels = [dict(row) for row in labels]
    if len(labels) != len(records):
        raise RouterConfigurationError(
            f"compile needs one label dict per record: got {len(labels)} labels "
            f"for {len(records)} records."
        )
    schema = introspect_schema(records)
    canonical_config = json.dumps(questions, sort_keys=True)
    if modality == "image":
        from .constants import DEFAULT_IMAGE_ENCODER

        nlu = encoder or DEFAULT_IMAGE_ENCODER
    else:
        nlu = DEFAULT_NLU_MODEL
    # Labels + projection + the token window + modality are inputs, so they bust
    # the cache: a fresh fit on any of them is a different model
    # (recompile-don't-increment).
    labels_fp = hashlib.sha256(
        (json.dumps(labels, sort_keys=True, default=str) + "\n"
         + (projection or "") + "\n" + str(int(max_length)) + "\n"
         + modality + "\n" + (image_field or ""))
        .encode("utf-8")
    ).hexdigest()
    key_mode = "compile" if mode == "head" else f"compile-{mode}"
    key = cache_key(
        canonical_config, nlu, schema, labels_fp, 0.0, mode=key_mode, golden=golden,
    )
    slug = _battery_slug(name, " ".join(questions))
    artifact_path, meta_path = _artifact_paths(state, slug, key)

    cache_hit = artifact_path.exists()
    attempts = 0
    if not cache_hit:
        compiler = StrotCompiler(nlu_model=nlu)
        try:
            asset = compiler.compile_model(
                sample=records, questions=questions, labels=labels,
                projection=projection, golden=golden, mode=mode,
                max_length=max_length, modality=modality, encoder=encoder,
                image_field=image_field,
            )
        except Exception as exc:
            _log_episode(state, {
                "kind": "compile", "status": "failed", "cache_key": key,
                "questions": list(questions), "label_count": len(labels or []),
                "error": str(exc)[:500],
            })
            raise
        _log_episode(state, {
            "kind": "compile", "status": "promoted", "cache_key": key,
            "questions": list(questions), "label_count": len(labels or []),
            "attempts": asset.attempts,
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
                    "mode": "compile",
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
