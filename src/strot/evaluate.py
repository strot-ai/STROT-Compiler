"""Measure a decision model against known answers - the learning signal.

Two layers, mirroring the compiler split:

* :func:`score_decisions` - pure, torch-free metrics over *already-produced*
  decisions + expected answers. Accuracy, coverage (fraction answered vs
  refused), and a calibration gap (ECE) per question. Testable with no model.
* :func:`evaluate` - the convenience: run a model over labeled records, then
  score. Needs the model loaded (the ``[local]`` extra).

``labeled`` is the same shape as ``golden``::

    [{"record": {...}, "expect": {question_name: value}}, ...]

where ``value`` is a bool (predicate), a choice value, or a score label.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union


def _extract(decision: Mapping[str, Any], question: str):
    """Return ``(kind, predicted, confidence, refused)`` for a question, or
    ``(None, None, None, None)`` if the decision has no answer for it."""
    for group, kind in (("predicates", "predicate"), ("scores", "score"), ("choices", "choice")):
        answers = decision.get(group) or {}
        if question in answers:
            answer = answers[question]
            if answer.get("refusal"):
                return kind, None, None, True
            if kind == "predicate":
                p = float(answer["probability"])
                return kind, (p >= 0.5), max(p, 1.0 - p), False
            key = "score" if kind == "score" else "choice"
            return kind, answer[key], float(answer.get("confidence", 1.0)), False
    return None, None, None, None


def _is_correct(kind: str, predicted: Any, expected: Any) -> bool:
    if kind == "predicate":
        return bool(predicted) == bool(expected)
    if kind == "score":
        return str(predicted) == str(expected)
    return predicted == expected


def _ece(points: Sequence["tuple[float, bool]"], bins: int) -> Optional[float]:
    """Expected calibration error over ``(confidence, correct)`` pairs."""
    if not points:
        return None
    total = len(points)
    gap = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in points if (c > lo or b == 0) and c <= hi]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        gap += abs(avg_conf - acc) * (len(bucket) / total)
    return round(gap, 4)


def score_decisions(
    decisions: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    confidence_threshold: Optional[float] = None,
    ece_bins: int = 10,
) -> Dict[str, Any]:
    """Score model decisions against expected answers (pure, torch-free).

    With ``confidence_threshold`` set, any answer below it counts as *not
    answered* (so accuracy is measured on the confident slice and coverage drops
    - the risk/coverage tradeoff).
    """
    if len(decisions) != len(expected):
        raise ValueError("decisions and expected must be the same length")

    per_q: Dict[str, Dict[str, Any]] = {}
    cal: Dict[str, List["tuple[float, bool]"]] = {}
    for decision, exp_row in zip(decisions, expected):
        for question, exp_val in exp_row.items():
            stats = per_q.setdefault(question, {"total": 0, "answered": 0, "correct": 0})
            cal.setdefault(question, [])
            stats["total"] += 1
            kind, predicted, confidence, refused = _extract(decision, question)
            if kind is None or refused:
                continue  # question missing, or model refused -> unanswered
            if confidence_threshold is not None and confidence < confidence_threshold:
                continue  # below the bar -> abstain, counts against coverage
            stats["answered"] += 1
            correct = _is_correct(kind, predicted, exp_val)
            if correct:
                stats["correct"] += 1
            cal[question].append((float(confidence), bool(correct)))

    questions = {}
    tot = ans = cor = 0
    for q, s in per_q.items():
        tot += s["total"]; ans += s["answered"]; cor += s["correct"]
        questions[q] = {
            "accuracy": round(s["correct"] / s["answered"], 4) if s["answered"] else None,
            "coverage": round(s["answered"] / s["total"], 4) if s["total"] else None,
            "ece": _ece(cal[q], ece_bins),
            "n": s["total"],
        }
    return {
        "per_question": questions,
        "overall": {
            "accuracy": round(cor / ans, 4) if ans else None,
            "coverage": round(ans / tot, 4) if tot else None,
            "n": len(decisions),
        },
    }


def _resolve_model(model: Any) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    from pathlib import Path

    if isinstance(model, (str, Path)):
        from .jit import load_model

        return load_model(model)
    if callable(model):
        return model
    raise TypeError("model must be a DecisionModel, a callable, or a path to a .strot file")


def evaluate(
    model: Any,
    labeled: Sequence[Mapping[str, Any]],
    confidence_threshold: Optional[float] = None,
    ece_bins: int = 10,
) -> Dict[str, Any]:
    """Run ``model`` over labeled records and score it.

    ``model`` is a :class:`~strot.DecisionModel`, any callable ``record ->
    decision``, or a path to a saved ``.strot`` file. ``labeled`` is a list of
    ``{"record": ..., "expect": {question: value}}``.
    """
    decide = _resolve_model(model)
    decisions = [decide(dict(row["record"])) for row in labeled]
    expected = [row["expect"] for row in labeled]
    return score_decisions(decisions, expected, confidence_threshold, ece_bins)
