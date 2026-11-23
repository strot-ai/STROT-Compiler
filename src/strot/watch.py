"""The watcher: learn a local model from another agent's own decisions.

An observed agent's ``(input -> decision)`` pairs **are labels** — real, free,
and in-distribution. :func:`ingest_traces` turns an agent's decision traces into
``(records, labels)`` and :func:`infer_questions` recovers the decision schema
from what the agent actually did, so the whole watcher loop is:

    observe agent -> ingest_traces -> infer_questions -> compile -> shadow -> take over

No LLM runs here: the decisions have already been made, so they are the labels.
STROT distills them into a $0 local understudy that takes the confident cases and
defers the rest back to the agent. The understudy clones the agent's *policy*, not
truth — it is capped at the agent's quality — and only for typed/discrete
decisions (predicate / choice), not free-form generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

Trace = Mapping[str, Any]
TraceSource = Union[str, Path, Sequence[Trace]]


def _load(traces: TraceSource) -> List[Dict[str, Any]]:
    """Accept a list of traces, or a path to a JSON array / JSONL file."""
    if isinstance(traces, (str, Path)):
        text = Path(traces).read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text[0] == "[":
            return list(json.loads(text))
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [dict(t) for t in traces]


def _span_attr(item: Mapping[str, Any], key: str) -> Any:
    """Pull ``key`` from an OTel-style span's attributes (list- or dict-shaped).

    OTLP/JSON carries attributes as ``[{"key": ..., "value": {"stringValue": ...}}]``;
    a flat ``{key: value}`` attributes dict is also accepted.
    """
    attrs = item.get("attributes")
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, Mapping) and a.get("key") == key:
                value = a.get("value")
                if isinstance(value, Mapping):
                    # {"stringValue": "..."} / {"intValue": 1} / ... — take the payload.
                    return next(iter(value.values()), None)
                return value
        return None
    if isinstance(attrs, Mapping):
        return attrs.get(key)
    return None


def _extract(item: Mapping[str, Any], key: str) -> Any:
    """Top-level key wins; otherwise look inside OTel span attributes."""
    if key in item:
        return item[key]
    return _span_attr(item, key)


def _coerce(value: Any) -> Any:
    """JSON-decode a string payload (OTel attributes are often stringified)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def ingest_traces(
    traces: TraceSource,
    input_key: str = "input",
    decision_key: str = "decision",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn an agent's decision traces into ``(records, labels)`` for :func:`compile`.

    Each trace is one observed ``(input -> decision)``. ``traces`` is a list of
    dicts, or a path to a JSON array / JSONL file (one trace per line). A trace is
    either ``{input_key: {record}, decision_key: {question: answer}}`` or an OTel
    span whose ``attributes`` carry those keys (stringified JSON is decoded).
    Traces whose input or decision is not a dict are skipped.

    Returns ``(records, labels)`` aligned one-to-one, ready to hand straight to
    ``strot.compile(records, questions, labels)``.
    """
    records: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    for item in _load(traces):
        record = _coerce(_extract(item, input_key))
        decision = _coerce(_extract(item, decision_key))
        if isinstance(record, Mapping) and isinstance(decision, Mapping):
            records.append(dict(record))
            labels.append(dict(decision))
    return records, labels


def infer_questions(labels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recover a typed-question config from observed decisions.

    For each decision key, collects the values the agent produced: all-boolean ->
    a ``predicate``; otherwise a ``choice`` over the observed values (in first-seen
    order). ``score`` is never inferred — order can't be recovered from
    observation, so ordinal decisions come back as a ``choice`` (pass an explicit
    config if you need the ordering). Review the result before compiling.
    """
    seen: Dict[str, List[Any]] = {}
    for row in labels:
        for key, value in row.items():
            bucket = seen.setdefault(key, [])
            if value not in bucket:
                bucket.append(value)

    questions: Dict[str, Any] = {}
    for key, values in seen.items():
        if values and all(isinstance(v, bool) for v in values):
            questions[key] = {
                "type": "predicate",
                "instructions": f"does the record indicate {key.replace('_', ' ')}?",
            }
        else:
            questions[key] = {
                "type": "choice",
                "instructions": f"which {key.replace('_', ' ')}?",
                "choices": [{"value": v} for v in values],
            }
    return questions
