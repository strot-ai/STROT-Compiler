"""Compile-time statistical calibration: fitted state for compiled artifacts.

Two pieces of standard, defensible machinery, fitted once at compile time
from teacher-labeled sample records and baked into the artifact as constants:

* **Temperature scaling** (Guo et al., 2017) — one scalar ``T`` fitted by
  minimizing negative log-likelihood, turning the decision model's raw
  probabilities into calibrated ones. Applied to probabilities as
  ``p_i^(1/T)`` renormalized (equivalent to scaling logits).
* **Split conformal prediction** — a quantile ``q`` of nonconformity scores
  (``1 - p_T(true label)``) at miscoverage level ``alpha``. At runtime the
  artifact forms the prediction set ``{label : p_T(label) >= 1 - q}`` and
  dispatches only when that set is a singleton; otherwise it falls back.

**Read the guarantee precisely.** Split conformal targets *marginal*
coverage: under exchangeability, the true label lands in the prediction set
with probability >= 1 - alpha. A wrong singleton dispatch implies the true
label was outside the set, so the rate of wrong dispatches **over all
incoming records** is bounded by alpha. The error rate *conditional on
dispatching* is NOT bounded by alpha — it is bounded by
``alpha / dispatch_rate`` (reported as an estimate in the calibration
metadata). At a 90% dispatch rate the two nearly coincide; at a 20%
dispatch rate the conditional rate may be 5x alpha.

**Validity requires fold splitting.** The temperature transform must be
fixed independently of the conformal calibration points, so
:func:`fit_split_calibration` fits ``T`` on one fold and the quantile on a
disjoint fold. Fitting both on the same points (as naive implementations
do) makes the scores non-exchangeable with fresh records and voids the
finite-sample guarantee.

Everything here is pure Python on top of probabilities the runtime already
produces — no torch, no new dependencies, honest small-sample caveats: with
``n`` calibration points the quantile granularity is ``1/(n+1)``, so tiny
samples give coarse guarantees. The compiler surfaces ``n`` in the artifact
metadata for exactly that reason.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

#: Search grid for the temperature scalar (log-spaced, inclusive bounds).
_T_MIN, _T_MAX, _T_STEPS = 0.25, 4.0, 60


def apply_temperature(probs: Sequence[float], temperature: float) -> List[float]:
    """Rescale a probability vector by ``p^(1/T)`` and renormalize."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    floor = 1e-12
    powered = [max(p, floor) ** (1.0 / temperature) for p in probs]
    total = sum(powered)
    return [p / total for p in powered]


def fit_temperature(
    distributions: Sequence[Sequence[float]], true_indices: Sequence[int]
) -> float:
    """Fit the NLL-minimizing temperature over a labeled calibration set."""
    if len(distributions) != len(true_indices) or not distributions:
        raise ValueError("need equal, non-empty distributions and labels")
    best_t, best_nll = 1.0, float("inf")
    for step in range(_T_STEPS + 1):
        t = _T_MIN * (_T_MAX / _T_MIN) ** (step / _T_STEPS)
        nll = 0.0
        for probs, true_index in zip(distributions, true_indices):
            calibrated = apply_temperature(probs, t)
            nll -= math.log(max(calibrated[true_index], 1e-12))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return round(best_t, 4)


def softmax_logits(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Numerically-stable softmax of ``logits / temperature``."""
    scaled = [z / temperature for z in logits]
    peak = max(scaled)
    exps = [math.exp(z - peak) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def fit_temperature_logits(
    logits: Sequence[Sequence[float]], true_indices: Sequence[int]
) -> float:
    """Temperature scaling in logit space: the NLL-minimizing ``T`` for
    ``softmax(logits / T)``.

    Unlike :func:`fit_temperature` (which rescales already-softmaxed
    probabilities and so cannot recover a saturated 0/1 vector), this fits on
    the raw pre-softmax logits, where a sharp head still carries the magnitude
    the temperature acts on.
    """
    if len(logits) != len(true_indices) or not logits:
        raise ValueError("need equal, non-empty logits and labels")
    best_t, best_nll = 1.0, float("inf")
    for step in range(_T_STEPS + 1):
        t = _T_MIN * (_T_MAX / _T_MIN) ** (step / _T_STEPS)
        nll = 0.0
        for row, true_index in zip(logits, true_indices):
            nll -= math.log(max(softmax_logits(row, t)[true_index], 1e-12))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return round(best_t, 4)


def conformal_quantile(
    distributions: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    temperature: float,
    alpha: float,
) -> float:
    """Split-conformal quantile of nonconformity scores at level ``alpha``.

    Score per calibration point: ``1 - p_T(true label)``. Returns the
    ``ceil((n+1)(1-alpha))/n`` empirical quantile (capped at 1.0), the
    standard finite-sample-valid choice.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = sorted(
        1.0 - apply_temperature(probs, temperature)[true_index]
        for probs, true_index in zip(distributions, true_indices)
    )
    n = len(scores)
    if n == 0:
        raise ValueError("need at least one calibration point")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0  # too few points for this alpha: everything conformal
    # Round UP at 1e-6: conservative (never shrinks the prediction set below
    # the fitted guarantee) and immune to float-boundary dispatch flakiness.
    return min(1.0, math.ceil(scores[rank - 1] * 1e6) / 1e6)


def prediction_set(
    probs: Sequence[float], temperature: float, quantile: float
) -> Tuple[List[int], List[float]]:
    """Indices whose calibrated probability clears the conformal bar."""
    calibrated = apply_temperature(probs, temperature)
    included = [i for i, p in enumerate(calibrated) if p >= 1.0 - quantile]
    return included, calibrated


def fit_split_calibration(
    distributions: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    alpha: float,
) -> Dict[str, object]:
    """Validity-preserving calibration: disjoint folds for T and the quantile.

    Even-indexed points fit the temperature; odd-indexed points supply the
    conformal quantile. With fewer than 4 temperature points, T falls back
    to 1.0 (an uninformed fit is worse than none). With a single labeled
    point overall, everything goes to the quantile fold.

    Returns the full calibration report to bake into the artifact,
    including honest fold sizes and the estimated conditional error bound.
    """
    if len(distributions) != len(true_indices) or not distributions:
        raise ValueError("need equal, non-empty distributions and labels")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    t_dists, t_true = list(distributions[0::2]), list(true_indices[0::2])
    c_dists, c_true = list(distributions[1::2]), list(true_indices[1::2])
    if not c_dists:  # single labeled point: quantile fold takes it
        c_dists, c_true = list(distributions), list(true_indices)
        t_dists, t_true = [], []
    temperature = fit_temperature(t_dists, t_true) if len(t_dists) >= 4 else 1.0
    quantile = conformal_quantile(c_dists, c_true, temperature, alpha)

    n_conformal = len(c_dists)
    singleton = correct_singleton = 0
    for probs, true_index in zip(c_dists, c_true):
        included, _ = prediction_set(probs, temperature, quantile)
        if len(included) == 1:
            singleton += 1
            if included[0] == true_index:
                correct_singleton += 1
    dispatch_rate = singleton / n_conformal if n_conformal else 0.0
    return {
        "n_calibration": len(distributions),
        "n_temperature_fold": len(t_dists),
        "n_conformal_fold": n_conformal,
        "alpha": alpha,
        "temperature": temperature,
        "conformal_quantile": quantile,
        "quantile_granularity": round(1.0 / (n_conformal + 1), 4),
        "dispatch_rate_on_calibration": round(dispatch_rate, 4),
        "dispatch_accuracy_on_calibration": (
            round(correct_singleton / singleton, 4) if singleton else None
        ),
        "guarantee": (
            "marginal: P(dispatch AND wrong) <= alpha over exchangeable "
            "records; non-singleton prediction sets abstain"
        ),
        "conditional_error_bound_estimate": (
            round(min(1.0, alpha / dispatch_rate), 4) if dispatch_rate > 0 else None
        ),
    }


def calibration_report(
    distributions: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    temperature: float,
    quantile: float,
    alpha: float,
) -> Dict[str, object]:
    """Summary baked into artifact metadata: honest, inspectable numbers."""
    n = len(distributions)
    singleton = correct_singleton = 0
    for probs, true_index in zip(distributions, true_indices):
        included, _ = prediction_set(probs, temperature, quantile)
        if len(included) == 1:
            singleton += 1
            if included[0] == true_index:
                correct_singleton += 1
    return {
        "n_calibration": n,
        "alpha": alpha,
        "temperature": temperature,
        "conformal_quantile": quantile,
        "quantile_granularity": round(1.0 / (n + 1), 4),
        "dispatch_rate_on_calibration": round(singleton / n, 4) if n else 0.0,
        "dispatch_accuracy_on_calibration": (
            round(correct_singleton / singleton, 4) if singleton else None
        ),
    }


def fit_class_bias(
    distributions: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    temperature: float,
    epochs: int = 200,
    learning_rate: float = 0.5,
) -> List[float]:
    """Per-class log-bias correction fitted by NLL (vector scaling, bias part).

    Corrects systematic class skew (e.g. chronic over-flagging of one class)
    that a single temperature cannot: calibrated probabilities become
    ``p_i^(1/T) * exp(b_i)`` renormalized. Pure Python, fitted once at
    compile time on held-out labeled points.
    """
    if not distributions or len(distributions) != len(true_indices):
        raise ValueError("need equal, non-empty distributions and labels")
    n_classes = len(distributions[0])
    bias = [0.0] * n_classes
    n = len(distributions)
    for _ in range(epochs):
        grad = [0.0] * n_classes
        for probs, true_index in zip(distributions, true_indices):
            corrected = apply_correction(probs, temperature, bias)
            for c in range(n_classes):
                grad[c] += corrected[c] - (1.0 if c == true_index else 0.0)
        for c in range(n_classes):
            bias[c] -= learning_rate * grad[c] / n
    center = sum(bias) / n_classes
    return [round(b - center, 6) for b in bias]


def apply_correction(
    probs: Sequence[float], temperature: float, bias: Optional[Sequence[float]] = None
) -> List[float]:
    """Temperature + per-class bias correction, renormalized."""
    floor = 1e-12
    scores = [
        math.log(max(p, floor)) / temperature + (bias[i] if bias else 0.0)
        for i, p in enumerate(probs)
    ]
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def selective_cutoff(
    confidences: Sequence[float],
    correct: Sequence[bool],
    target_accuracy: float,
    min_points: int = 8,
) -> Optional[float]:
    """Smallest confidence cutoff whose empirical selective accuracy meets the
    target, maximizing coverage. None = no cutoff qualifies (abstain tier).

    Requires at least ``min_points`` above the cutoff so a lucky handful
    cannot fake the target; this is an empirical policy fit, not a
    finite-sample guarantee - report it as such.
    """
    if not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")
    pairs = sorted(zip(confidences, correct), key=lambda x: x[0])
    best: Optional[float] = None
    for i in range(len(pairs)):
        cutoff = pairs[i][0]
        kept = pairs[i:]
        if len(kept) < min_points:
            break
        accuracy = sum(1 for _, ok in kept if ok) / len(kept)
        if accuracy >= target_accuracy:
            best = round(cutoff, 6)
            break  # smallest qualifying cutoff = maximum coverage
    return best
