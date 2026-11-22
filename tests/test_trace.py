"""Tests for strot.trace - decision summaries, the reader/summarizer, and the
opt-in DecisionModel logging (all torch-free via a fake pipeline)."""

import json

import strot
from strot.trace import read_traces, summarize_decision, summarize_traces


def test_summarize_decision_all_types():
    decision = {
        "predicates": {"is_bug": {"type": "predicate", "probability": 0.9}},
        "choices": {"team": {"choice": "billing", "probabilities": {}, "confidence": 0.97}},
        "scores": {"sent": {"refusal": True, "reason": "ood"}},
    }
    s = summarize_decision(decision)
    assert s["is_bug"] == {"value": True, "confidence": 0.9, "refused": False}
    assert s["team"]["value"] == "billing" and s["team"]["confidence"] == 0.97
    assert s["sent"]["refused"] is True and s["sent"]["reason"] == "ood"


def test_read_and_summarize_traces(tmp_path):
    entries = [
        {"answers": {"team": {"value": "billing", "confidence": 0.9, "refused": False}}},
        {"answers": {"team": {"value": "engineering", "confidence": 0.7, "refused": False}}},
        {"answers": {"team": {"value": None, "confidence": None, "refused": True, "reason": "ood"}}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries))
    loaded = read_traces(p)
    assert len(loaded) == 3

    q = summarize_traces(loaded)["per_question"]["team"]
    assert q["n"] == 3
    assert q["refusal_rate"] == round(1 / 3, 4)
    assert q["confidence"]["mean"] == round((0.9 + 0.7) / 2, 4)
    assert q["values"] == {"billing": 1, "engineering": 1}  # refused not counted


def test_read_traces_missing_file_is_empty(tmp_path):
    assert read_traces(tmp_path / "nope.jsonl") == []


class _FakePipeline:
    def process(self, record):
        return {"choices": {"team": {"choice": record["t"], "probabilities": {}, "confidence": 0.9}}}


def test_decision_model_logs_traces(tmp_path):
    model = strot.DecisionModel(tmp_path / "m.strot", key="k", pipeline=_FakePipeline())
    tp = tmp_path / "traces.jsonl"
    assert model.trace(tp) is model  # chainable
    model({"t": "billing"})
    model({"t": "engineering"})

    entries = read_traces(tp)
    assert len(entries) == 2
    assert entries[0]["answers"]["team"]["value"] == "billing"
    assert "record" not in entries[0]  # record not logged by default
    assert summarize_traces(entries)["per_question"]["team"]["values"] == {"billing": 1, "engineering": 1}


def test_decision_model_trace_include_record(tmp_path):
    model = strot.DecisionModel(
        tmp_path / "m.strot", key="k", pipeline=_FakePipeline()
    ).trace(tmp_path / "t.jsonl", include_record=True)
    model({"t": "billing"})
    entry = read_traces(tmp_path / "t.jsonl")[0]
    assert entry["record"] == {"t": "billing"}
