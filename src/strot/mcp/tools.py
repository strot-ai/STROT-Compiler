"""Capability functions behind the STROT MCP server.

Deliberately import-light: the discovery tools (:func:`list_models`,
:func:`model_info`) read a compiled model's baked literals with ``ast`` - no
torch, no LLM, no model load - so an agent can survey the whole roster for free.
The compile/inference tools import ``strot`` lazily and need the ``[local]`` extra.
"""

from __future__ import annotations

import ast
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

Records = List[Dict[str, Any]]
DataArg = Union[str, Records]


def _as_records(data: DataArg) -> Records:
    """Accept inline records or a path to a JSON file of records."""
    if isinstance(data, str):
        return json.loads(Path(data).read_text(encoding="utf-8"))
    return list(data)


def _module_literal(source: str, name: str) -> Any:
    """Literal-eval a top-level ``name = ...`` / ``name: T = ...`` assignment."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign):
            target = next((getattr(t, "id", None) for t in node.targets), None)
        if target == name:
            try:
                return ast.literal_eval(node.value)
            except Exception:
                return None
    return None


def read_model_meta(path: Union[str, Path]) -> Dict[str, Any]:
    """Torch-free metadata for a compiled model (``.strot`` or ``battery_*.py``).

    Reads the baked ``QUESTIONS_CONFIG`` / ``NLU_MODEL`` literals (and the
    optional ``.json`` sidecar) without loading the encoder.
    """
    p = Path(path)
    meta: Dict[str, Any] = {}
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    source = p.read_text(encoding="utf-8", errors="ignore")
    questions = meta.get("questions") or _module_literal(source, "QUESTIONS_CONFIG")
    return {
        "path": str(p),
        "name": p.stem,
        "questions": list(questions) if isinstance(questions, dict) else None,
        "encoder": _module_literal(source, "NLU_MODEL") or meta.get("nlu"),
        "mode": meta.get("mode"),
        "has_adapter": "ADAPTER_CONFIG" in source,  # a deep (LoRA) model
        "has_drift_gate": "OOD" in source and "centroids" in source,
    }


def list_models(root: Union[str, Path] = ".") -> List[Dict[str, Any]]:
    """List compiled decision models under ``root`` (torch-free inventory)."""
    base = Path(root)
    if not base.exists():
        return []
    files = sorted(set(base.rglob("*.strot")) | set(base.rglob("battery_*.py")))
    return [read_model_meta(f) for f in files]


def model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """One compiled model's metadata, without loading it."""
    return read_model_meta(path)


def inspect_data(data: DataArg, sample_size: int = 25) -> Dict[str, Any]:
    """Profile raw input - fields, types, cardinality, sample values (torch-free).
    The agent's view of the data before it plans a compilation."""
    import strot

    return strot.inspect_data(_as_records(data), sample_size=sample_size)


def pii_scan(data: DataArg, sample_size: int = 25) -> Dict[str, Any]:
    """Detect structured PII in a sample (torch-free regex, advisory) - what a
    record carries before you feed it to your own LLM."""
    import strot

    return strot.pii_scan(_as_records(data), sample_size=sample_size)


def compile_model(
    data: DataArg,
    questions: Union[str, Dict[str, Any]],
    labels: Optional[List[Dict[str, Any]]] = None,
    mode: str = "head",
    projection: Optional[str] = None,
    state: str = ".strot",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile a decision model - the LLM-free door (needs the ``[local]`` extra).

    STROT holds no key and calls no LLM: YOU (the agent) design the questions and
    label the sample, then hand them here. ``labels`` is one ``{question: answer}``
    dict per record in ``data`` order (an agent's decisions or human ground
    truth); ``projection`` is optional
    ``route_context`` source (default concatenates text fields). Returns the path.
    """
    import strot

    result = strot.compile(
        _as_records(data), questions, labels, projection=projection, mode=mode,
        state=state, execute=False, name=name,
    )
    path = result.dispatch(name) if name else result.artifact_path
    return {
        "model_path": str(path),
        "mode": mode,
        "questions": list(result.questions or {}),
        "cache_hit": result.cache_hit,
    }


# --- Compile budget: a simple guardrail so an agent can't run away compiling ---
# A per-process cap on compiles started (compiles cost minutes of compute).
# Set STROT_MCP_COMPILE_BUDGET=<int>; unset = unlimited.
_compiles_started = 0
_budget_lock = threading.Lock()


def _compile_cap() -> Optional[int]:
    raw = os.environ.get("STROT_MCP_COMPILE_BUDGET")
    try:
        return int(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def budget_status() -> Dict[str, Any]:
    """How many compiles have run this session, and the cap (if any)."""
    cap = _compile_cap()
    with _budget_lock:
        started = _compiles_started
    return {"compiles_started": started, "cap": cap,
            "remaining": (cap - started) if cap is not None else None}


# --- Async compile: compiles run minutes (download + label + train), too long
# for one tool call, so start returns a job id the agent polls with status. ---
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def compile_start(
    data: DataArg,
    questions: Union[str, Dict[str, Any]],
    labels: Optional[List[Dict[str, Any]]] = None,
    mode: str = "head",
    projection: Optional[str] = None,
    state: str = ".strot",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a compile in the background; returns a ``job_id`` to poll with
    :func:`compile_status`. YOU bring the labels (no LLM). Embedding a large sample runs minutes, so it is off-thread. Honors the
    compile cap (a compute guard). Needs the ``[local]`` extra.
    """
    cap = _compile_cap()
    with _budget_lock:
        global _compiles_started
        if cap is not None and _compiles_started >= cap:
            return {"status": "refused",
                    "reason": f"compile budget exhausted ({_compiles_started}/{cap})"}
        _compiles_started += 1

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "result": None, "error": None}

    def _run() -> None:
        try:
            result = compile_model(data, questions, labels, mode=mode,
                                   projection=projection, state=state, name=name)
            with _jobs_lock:
                _jobs[job_id].update(status="done", result=result)
        except Exception as exc:  # surface the error to the poller, never crash the thread
            with _jobs_lock:
                _jobs[job_id].update(status="error", error=str(exc)[:500])

    threading.Thread(target=_run, name=f"strot-compile-{job_id}", daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def compile_status(job_id: str) -> Dict[str, Any]:
    """Poll a compile job: ``running`` | ``done`` (with ``result``) | ``error``."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "unknown"}
    return {"job_id": job_id, **job}


def decide_record(model_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Run a compiled model on one record -> typed decision JSON (needs ``[local]``).

    Honors the circuit breaker: a paused model returns ``{"paused": true, ...}``
    instead of deciding (for calls routed through this tool).
    """
    status = model_status(model_path)
    if status.get("status") == "paused":
        return {"paused": True, "reason": status.get("reason")}
    import strot

    return strot.load_model(model_path)(record)


# --- Serving state: a model's status flag (the gatekeeper's circuit breaker) ---
# STROT is a compiler + inference function, not a runtime, so the status is a
# sidecar flag. decide_record (the MCP inference path) honors it; direct
# model(record) calls in your own serving code should check model_status too.


def _status_path(model_path: str) -> Path:
    return Path(str(model_path) + ".status.json")


def set_status(model_path: str, status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Set a model's serving status: ``active`` or ``paused`` (circuit breaker).

    Pausing a drifting/broken model stops decide_record from serving it; ``active``
    resumes. Also the promote/rollback primitive - activate the version you trust.
    """
    from datetime import datetime, timezone

    if status not in ("active", "paused"):
        return {"error": "status must be 'active' or 'paused'"}
    payload = {"status": status, "reason": reason,
               "ts": datetime.now(timezone.utc).isoformat()}
    _status_path(model_path).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def model_status(model_path: str) -> Dict[str, Any]:
    """Read a model's serving status (defaults to ``active`` if never set)."""
    sp = _status_path(model_path)
    if sp.exists():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"status": "active", "reason": None}


def _questions_config(model_path: str) -> Any:
    """Read a compiled model's full question config (the baked clean config)."""
    source = Path(model_path).read_text(encoding="utf-8", errors="ignore")
    return _module_literal(source, "QUESTIONS_CONFIG")


def recompile(
    model_path: str,
    data: DataArg,
    labels: Optional[List[Dict[str, Any]]] = None,
    mode: str = "head",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompile a model on fresh/accumulated data, reusing its question config.

    A fresh distillation (recompile-don't-increment), started async. YOU re-label
    the new data (no LLM) and pass ``labels``; Defaults
    to a new ``<name>-recompiled`` artifact so the agent can evaluate it before
    promoting with :func:`set_status`. Needs the ``[local]`` extra.
    """
    cfg = _questions_config(model_path)
    if not isinstance(cfg, dict) or not cfg:
        return {"status": "error",
                "error": f"could not read a question config from {model_path!r}"}
    orig = Path(model_path)
    target = name or str(orig.with_name(orig.stem + "-recompiled"))  # same dir as the original
    return compile_start(data, cfg, labels, mode=mode, name=target)


def cluster_rejections(traces_path: str, top_terms: int = 10, max_samples: int = 20) -> Dict[str, Any]:
    """Cluster a model's refused records to surface the gap to close (torch-free)."""
    import strot

    return strot.cluster_rejections(traces_path, top_terms=top_terms, max_samples=max_samples)


def drift_check(traces_path: str, refusal_threshold: float = 0.2, confidence_floor: float = 0.5) -> Dict[str, Any]:
    """Drift signal from a model's traces + a recompile recommendation (torch-free)."""
    import strot

    return strot.drift_check(traces_path, refusal_threshold=refusal_threshold, confidence_floor=confidence_floor)


def evaluate_model(
    model_path: str,
    labeled: List[Dict[str, Any]],
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a compiled model over labeled records and score it (needs ``[local]``).

    ``labeled`` is ``[{"record": {...}, "expect": {question: value}}, ...]``.
    Returns accuracy / coverage / calibration (ECE) per question + overall.
    """
    import strot

    return strot.evaluate(model_path, labeled, confidence_threshold=confidence_threshold)


def score(
    decisions: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Score *already-produced* decisions against expected answers (torch-free)."""
    import strot

    return strot.score_decisions(decisions, expected, confidence_threshold=confidence_threshold)


def read_traces(path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Summarize a model's decision traces - volume, refusal rate, confidence,
    value mix per question (torch-free; the teacher's monitoring signal)."""
    import strot

    return strot.summarize_traces(strot.read_traces(path, limit=limit))
