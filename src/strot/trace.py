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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


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
