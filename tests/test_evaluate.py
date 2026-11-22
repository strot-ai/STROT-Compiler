"""Tests for strot.evaluate.score_decisions - the pure, torch-free metrics."""

import pytest

from strot.evaluate import score_decisions


def _pred(name, p):
    return {"predicates": {name: {"type": "predicate", "probability": p}}}


def _choice(name, value, conf):
    return {"choices": {name: {"choice": value, "probabilities": {}, "confidence": conf}}}


def _score(name, label, conf):
    return {"scores": {name: {"score": label, "probabilities": {}, "confidence": conf}}}


def _refuse(group, name):
    return {group: {name: {"refusal": True, "reason": "ood"}}}


def test_predicate_accuracy_and_coverage():
    decisions = [_pred("is_bug", 0.9), _pred("is_bug", 0.2)]
    expected = [{"is_bug": True}, {"is_bug": False}]  # 0.9->True, 0.2->False, both right
    r = score_decisions(decisions, expected)
    q = r["per_question"]["is_bug"]
    assert q["accuracy"] == 1.0 and q["coverage"] == 1.0 and q["n"] == 2
    assert r["overall"]["accuracy"] == 1.0


def test_wrong_prediction_lowers_accuracy():
    decisions = [_pred("is_bug", 0.9), _pred("is_bug", 0.9)]
    expected = [{"is_bug": True}, {"is_bug": False}]  # second is wrong
    assert score_decisions(decisions, expected)["per_question"]["is_bug"]["accuracy"] == 0.5


def test_refusal_drops_coverage_not_accuracy():
    decisions = [_choice("team", "billing", 0.9), _refuse("choices", "team")]
    expected = [{"team": "billing"}, {"team": "engineering"}]
    q = score_decisions(decisions, expected)["per_question"]["team"]
    assert q["coverage"] == 0.5  # one answered, one refused
    assert q["accuracy"] == 1.0  # the answered one was correct


def test_confidence_threshold_abstains_below_bar():
    decisions = [_choice("team", "billing", 0.9), _choice("team", "billing", 0.3)]
    expected = [{"team": "billing"}, {"team": "billing"}]
    # no threshold -> both answered
    assert score_decisions(decisions, expected)["per_question"]["team"]["coverage"] == 1.0
    # threshold 0.5 -> the 0.3 abstains
    gated = score_decisions(decisions, expected, confidence_threshold=0.5)["per_question"]["team"]
    assert gated["coverage"] == 0.5 and gated["accuracy"] == 1.0


def test_score_and_multi_question_overall():
    decisions = [
        {
            "predicates": {"a": {"type": "predicate", "probability": 0.9}},
            "choices": {"b": {"choice": "x", "probabilities": {}, "confidence": 0.9}},
            "scores": {"c": {"score": "angry", "probabilities": {}, "confidence": 0.8}},
        }
    ]
    expected = [{"a": True, "b": "x", "c": "angry"}]
    r = score_decisions(decisions, expected)
    assert set(r["per_question"]) == {"a", "b", "c"}
    assert r["overall"]["accuracy"] == 1.0 and r["overall"]["n"] == 1


def test_missing_question_counts_against_coverage():
    decisions = [{"predicates": {}, "scores": {}, "choices": {}}]  # model answered nothing
    expected = [{"team": "billing"}]
    q = score_decisions(decisions, expected)["per_question"]["team"]
    assert q["coverage"] == 0.0 and q["accuracy"] is None


def test_ece_is_reported():
    decisions = [_choice("team", "billing", 0.9), _choice("team", "engineering", 0.9)]
    expected = [{"team": "billing"}, {"team": "billing"}]  # one right, one wrong, both conf 0.9
    ece = score_decisions(decisions, expected)["per_question"]["team"]["ece"]
    assert isinstance(ece, float) and ece == pytest.approx(0.4)  # |0.9 conf - 0.5 acc|


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_decisions([_pred("a", 0.5)], [{"a": True}, {"a": False}])
