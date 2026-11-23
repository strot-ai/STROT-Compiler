"""Tests for the watcher: turning an agent's decision traces into (records,
labels) and recovering the decision schema. All pure-Python, no torch, no LLM -
the observed decisions ARE the labels."""

import json

import pytest

from strot import ingest_traces, infer_questions


PLAIN = [
    {"input": {"notes": "attorney demand"}, "decision": {"attorney": True, "route": "lit"}},
    {"input": {"notes": "cracked windshield"}, "decision": {"attorney": False, "route": "fast"}},
]

# An OTLP/JSON-style span: attributes as a list of {key, value:{stringValue}}.
OTEL_SPAN = {
    "name": "agent.decide",
    "attributes": [
        {"key": "input", "value": {"stringValue": json.dumps({"notes": "lawyer letter"})}},
        {"key": "decision", "value": {"stringValue": json.dumps({"attorney": True, "route": "lit"})}},
    ],
}


class TestIngest:
    def test_plain_input_decision_pairs(self):
        records, labels = ingest_traces(PLAIN)
        assert records == [{"notes": "attorney demand"}, {"notes": "cracked windshield"}]
        assert labels == [{"attorney": True, "route": "lit"}, {"attorney": False, "route": "fast"}]

    def test_otel_span_attributes_with_stringified_json(self):
        records, labels = ingest_traces([OTEL_SPAN])
        assert records == [{"notes": "lawyer letter"}]
        assert labels == [{"attorney": True, "route": "lit"}]

    def test_mixed_sources_and_custom_keys(self):
        traces = [{"x": {"a": 1}, "y": {"q": True}}]
        records, labels = ingest_traces(traces, input_key="x", decision_key="y")
        assert records == [{"a": 1}] and labels == [{"q": True}]

    def test_malformed_traces_are_skipped(self):
        traces = [
            {"input": {"ok": 1}, "decision": {"q": True}},
            {"input": "not a dict", "decision": {"q": True}},  # skipped
            {"decision": {"q": True}},                          # no input -> skipped
        ]
        records, labels = ingest_traces(traces)
        assert records == [{"ok": 1}] and labels == [{"q": True}]

    def test_reads_jsonl_and_json_array(self, tmp_path):
        jsonl = tmp_path / "t.jsonl"
        jsonl.write_text("\n".join(json.dumps(t) for t in PLAIN), encoding="utf-8")
        r1, l1 = ingest_traces(str(jsonl))
        arr = tmp_path / "t.json"
        arr.write_text(json.dumps(PLAIN), encoding="utf-8")
        r2, l2 = ingest_traces(str(arr))
        assert r1 == r2 == [t["input"] for t in PLAIN]
        assert l1 == l2 == [t["decision"] for t in PLAIN]


class TestInferQuestions:
    def test_bool_values_become_a_predicate(self):
        q = infer_questions([{"attorney": True}, {"attorney": False}])
        assert q["attorney"]["type"] == "predicate" and q["attorney"]["instructions"]

    def test_string_values_become_a_choice_over_observed_options(self):
        q = infer_questions([{"route": "lit"}, {"route": "fast"}, {"route": "lit"}])
        assert q["route"]["type"] == "choice"
        assert [c["value"] for c in q["route"]["choices"]] == ["lit", "fast"]  # first-seen order

    def test_inferred_config_round_trips_through_validation(self):
        # The whole point: infer -> compile without touching it.
        from strot.compiler import StrotCompiler

        _, labels = ingest_traces(PLAIN)
        q = infer_questions(labels)
        StrotCompiler.validate_questions_config(q)  # must not raise
        # and the observed labels validate against the inferred schema
        err = StrotCompiler._check_decide_answers(labels, len(labels), q)
        assert err is None
