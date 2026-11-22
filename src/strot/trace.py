"""Decision traces - a model's per-record decisions over time (the teacher's eyes).

A model logs one compact entry per decision when tracing is enabled
(:meth:`strot.DecisionModel.trace`). :func:`read_traces` loads them back and
:func:`summarize_traces` aggregates the monitoring signal - volume, refusal rate,
confidence distribution, and value mix per question - all pure and torch-free, so
drift/health can be read without loading the model.

Entry shape (one JSON object per line)::

    {"ts": "...", "answers": {"is_bug": {"value": true, "confidence": 0.9, "refused": false},
                              "team":   {"value": "billing", "confidence": 0.97, "refused": false}}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "it", "for",
    "was", "with", "this", "that", "please", "my", "me", "at", "be", "have",
    "when", "but", "not", "are", "you", "your", "from", "has", "can", "will",
}


def summarize_decision(decision: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Compact ``{question: {value, confidence, refused}}`` from a decision dict."""
    out: Dict[str, Dict[str, Any]] = {}
    for group in ("predicates", "scores", "choices"):
        for q, answer in (decision.get(group) or {}).items():
            if answer.get("refusal"):
                out[q] = {"value": None, "confidence": None, "refused": True,
                          "reason": answer.get("reason")}
            elif group == "predicates":
                p = float(answer["probability"])
                out[q] = {"value": p >= 0.5, "confidence": round(max(p, 1.0 - p), 4), "refused": False}
            else:
                key = "score" if group == "scores" else "choice"
                out[q] = {"value": answer[key],
                          "confidence": round(float(answer.get("confidence", 1.0)), 4),
                          "refused": False}
    return out


def read_traces(path: Union[str, Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load decision-trace entries from a JSONL file (newest ``limit`` if set)."""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:] if limit else entries


def _percentile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize_traces(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate the monitoring signal per question (pure, torch-free).

    Per question: count, refusal rate, confidence mean/min/p10, and the value
    mix (most common first) - the raw material for drift / health checks.
    """
    per_q: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        for q, a in (entry.get("answers") or {}).items():
            s = per_q.setdefault(q, {"n": 0, "refused": 0, "confidences": [], "values": {}})
            s["n"] += 1
            if a.get("refused"):
                s["refused"] += 1
                continue
            if a.get("confidence") is not None:
                s["confidences"].append(float(a["confidence"]))
            key = str(a.get("value"))
            s["values"][key] = s["values"].get(key, 0) + 1

    out: Dict[str, Any] = {}
    for q, s in per_q.items():
        confs = sorted(s["confidences"])
        out[q] = {
            "n": s["n"],
            "refusal_rate": round(s["refused"] / s["n"], 4) if s["n"] else 0.0,
            "confidence": {
                "mean": round(sum(confs) / len(confs), 4) if confs else None,
                "min": confs[0] if confs else None,
                "p10": _percentile(confs, 0.1),
            },
            "values": dict(sorted(s["values"].items(), key=lambda kv: -kv[1])),
        }
    return {"per_question": out, "n": len(entries)}


def _entries(traces: Union[str, Path, Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    """Accept a path to a trace file or an already-loaded list of entries."""
    if isinstance(traces, (str, Path)):
        return read_traces(traces)
    return list(traces)


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _record_text(record: Any) -> str:
    if isinstance(record, Mapping):
        return " ".join(str(v) for v in record.values())
    return str(record)


def cluster_rejections(
    traces: Union[str, Path, Sequence[Mapping[str, Any]]],
    top_terms: int = 10,
    max_samples: int = 20,
) -> Dict[str, Any]:
    """Surface what the *refused* records are about (the gap to close).

    Returns the rejection rate, the most common terms across refused records,
    and sample records - the raw signal the agent reasons over to decide what to
    recompile. Needs traces logged with ``include_record=True`` to see the
    inputs; otherwise it can count refusals but not characterize them. Torch-free.
    """
    entries = _entries(traces)
    rejected = [e for e in entries
                if any(a.get("refused") for a in (e.get("answers") or {}).values())]
    with_record = [e for e in rejected if "record" in e]

    counts: Dict[str, int] = {}
    for e in with_record:
        for tok in set(_tokens(_record_text(e["record"]))):
            counts[tok] = counts.get(tok, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:top_terms]

    result: Dict[str, Any] = {
        "n_total": len(entries),
        "n_rejected": len(rejected),
        "rejection_rate": round(len(rejected) / len(entries), 4) if entries else 0.0,
        "top_terms": [{"term": t, "count": c} for t, c in top],
        "samples": [e["record"] for e in with_record[:max_samples]],
    }
    if rejected and not with_record:
        result["note"] = ("records not logged; enable trace(path, include_record=True) "
                          "to diagnose what is being rejected")
    return result


def _window_stats(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = refused = 0
    confs: List[float] = []
    for e in entries:
        for a in (e.get("answers") or {}).values():
            n += 1
            if a.get("refused"):
                refused += 1
            elif a.get("confidence") is not None:
                confs.append(float(a["confidence"]))
    return {
        "n": n,
        "refusal_rate": round(refused / n, 4) if n else 0.0,
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
    }


def drift_check(
    traces: Union[str, Path, Sequence[Mapping[str, Any]]],
    refusal_threshold: float = 0.2,
    confidence_floor: float = 0.5,
) -> Dict[str, Any]:
    """Drift signal from traces + a ``recommend_recompile`` verdict (torch-free).

    Splits the trace into an earlier baseline and a recent window; recommends a
    recompile if the recent refusal rate is high, confidence has sagged below the
    floor, or refusals have risen materially vs the baseline.
    """
    entries = _entries(traces)
    if not entries:
        return {"n": 0, "recommend_recompile": False, "reasons": ["no traces"]}
    half = len(entries) // 2
    baseline = _window_stats(entries[:half]) if half else None
    recent = _window_stats(entries[half:])

    reasons: List[str] = []
    if recent["refusal_rate"] > refusal_threshold:
        reasons.append(f"refusal rate {recent['refusal_rate']} > {refusal_threshold}")
    if recent["mean_confidence"] is not None and recent["mean_confidence"] < confidence_floor:
        reasons.append(f"mean confidence {recent['mean_confidence']} < {confidence_floor}")
    if baseline and recent["refusal_rate"] - baseline["refusal_rate"] >= 0.1:
        reasons.append(
            f"refusal rate rose {baseline['refusal_rate']} -> {recent['refusal_rate']}"
        )
    return {
        "n": len(entries),
        "recent": recent,
        "baseline": baseline,
        "recommend_recompile": bool(reasons),
        "reasons": reasons,
    }
