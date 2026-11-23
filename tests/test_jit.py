"""Tests for the strot.decide() JIT facade and its shared helpers.

Compiling a real decision model needs torch + an encoder, so these tests stub
the embedder (see the ``decide_encoder`` fixture) and assert on the baked
artifact's structure rather than live inference.
"""

import json
import sys

import pytest

import strot
from strot.exceptions import RouterConfigurationError, SchemaIntrospectionError

from strot.jit import _battery_slug, _load_artifact, as_records, cache_key, schema_fingerprint
from strot.schema import introspect_schema

SAMPLE = [
    {"ticket_id": 1, "body": "double billing charge", "tier": "pro"},
    {"ticket_id": 2, "body": "app crash defect", "tier": "free"},
]

REGISTRY = {
    "billing": "Billing disputes, charges, and refunds",
    "defect": "Crashes, bugs, and broken behavior",
}

QUERY = "Triage incoming tickets into the correct workflow."

class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.responses.pop(0)


class ExplodingLLM:
    """Proves laziness: must never be invoked on a cache hit."""

    def __call__(self, prompt):  # pragma: no cover - the point is it never runs
        raise AssertionError("LLM was called despite a warm cache.")


@pytest.fixture(autouse=True)
def _fresh_compiled_modules():
    # Artifact modules are cached in sys.modules under key-derived names; two
    # tests can share a cache key (same query/schema/mode) while expecting
    # different artifacts, so always start clean.
    for name in [n for n in sys.modules if n.startswith("strot_compiled_")]:
        del sys.modules[name]
    yield


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    # Point checkpoint resolution at a local dir so the compiled artifact never
    # tries a Hugging Face download during tests.
    monkeypatch.setenv("STROT_NLU_CHECKPOINT", str(tmp_path))


# --------------------------------------------------------------------- #
# Record normalization and cache keys                                    #
# --------------------------------------------------------------------- #


class TestAsRecords:
    def test_list_of_dicts_passthrough(self):
        assert as_records(SAMPLE) == SAMPLE

    def test_dict_of_lists_becomes_records(self):
        records = as_records({"a": [1, 2], "b": ["x", "y"]})
        assert records == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_dataframe_like_uses_to_dict(self):
        class FakeFrame:
            def to_dict(self, orient):
                assert orient == "records"
                return [{"a": 1}]

        assert as_records(FakeFrame()) == [{"a": 1}]

    def test_unsupported_type_raises(self):
        with pytest.raises(SchemaIntrospectionError):
            as_records("not a dataset")

    def test_empty_raises(self):
        with pytest.raises(SchemaIntrospectionError):
            as_records([])


class TestBatterySlug:
    def test_name_takes_precedence_and_is_slugified(self):
        assert _battery_slug("Ticket Router v2", QUERY) == "ticket-router-v2"

    def test_auto_slug_derived_from_query_when_no_name(self):
        assert _battery_slug(None, "Triage incoming tickets") == "triage-incoming-tickets"

    def test_slug_charset_and_collapse(self):
        assert _battery_slug("A/B  Test!! (beta)", "") == "a-b-test-beta"

    def test_slug_capped_and_trimmed(self):
        slug = _battery_slug("x" * 40, "")
        assert slug == "x" * 24 and not slug.endswith("-")

    def test_empty_falls_back_to_task(self):
        assert _battery_slug(None, "!!! @@@") == "task"
        assert _battery_slug("", "") == "task"


class TestCacheKey:
    def make_key(self, sample=SAMPLE, query=QUERY, question=None, nlu="gte"):
        schema = introspect_schema(sample)
        return cache_key(query, nlu, schema, question, 0.40)

    def test_deterministic(self):
        assert self.make_key() == self.make_key()

    def test_query_change_busts_cache(self):
        assert self.make_key() != self.make_key(query="A different goal entirely.")

    def test_question_change_busts_cache(self):
        # `question` carries the labels+projection fingerprint in compile().
        assert self.make_key(question="fp-a") != self.make_key(question="fp-b")

    def test_schema_drift_busts_cache(self):
        drifted = [dict(r, severity="high") for r in SAMPLE]
        assert self.make_key() != self.make_key(sample=drifted)

    def test_sample_values_do_not_churn_key(self):
        other_batch = [
            {"ticket_id": 9, "body": "totally different words", "tier": "enterprise"},
            {"ticket_id": 10, "body": "yet another body", "tier": "pro"},
        ]
        assert self.make_key() == self.make_key(sample=other_batch)

    def test_fingerprint_ignores_field_order(self):
        reordered = [
            {"tier": r["tier"], "body": r["body"], "ticket_id": r["ticket_id"]}
            for r in SAMPLE
        ]
        fp_a = schema_fingerprint(introspect_schema(SAMPLE))
        fp_b = schema_fingerprint(introspect_schema(reordered))
        assert fp_a == fp_b

    def test_mode_changes_key(self):
        schema = introspect_schema(SAMPLE)
        assert cache_key(QUERY, "gte", schema, None, 0.40, mode="compile") != \
            cache_key(QUERY, "gte", schema, None, 0.40, mode="compile-lite")


# --------------------------------------------------------------------- #
# The JIT loop: miss -> compile -> persist; hit -> execute only          #
# --------------------------------------------------------------------- #


DECIDE_CONFIG = {
    "is_bug": {"type": "predicate", "instructions": "Does this mention crashes or errors?"},
    "sentiment": {"type": "score",
                  "instructions": "angry is a complaint, happy is praise.",
                  "levels": [
                      {"label": "angry", "description": "angry complaint"},
                      {"label": "neutral", "description": "neutral request"},
                      {"label": "happy", "description": "happy praise"}]},
    "team": {"type": "choice", "choices": [
        {"value": "billing", "description": "charges, invoices, payments"},
        {"value": "engineering", "description": "crashes, bugs, defects"}]},
}

VALID_DECIDE = '''```python
QUESTIONS = {
    "is_bug": {"type": "predicate", "instructions": "Does this mention crashes or errors?",
               "criteria": {"false": "normal request no malfunction",
                            "true": "crash defect error malfunction"}},
    "sentiment": {"type": "score",
                  "instructions": "angry is a complaint, happy is praise.",
                  "rubric": [{"label": "angry", "description": "angry complaint charge dispute"},
                             {"label": "neutral", "description": "neutral request"},
                             {"label": "happy", "description": "happy praise"}]},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "billing charge payment",
                          "engineering": "crash defect bug"}},
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))
```'''

# Decide compiles head-first (one distilled head per question), so it needs
# >= MIN_DISTILL_RECORDS sample records and a teacher-labels pass.
DECIDE_SAMPLE = [
    {"ticket_id": i, "body": ("double billing charge" if i % 2 == 0 else "app crash defect"),
     "tier": "pro"}
    for i in range(28)
]


def _decide_embed(texts):
    # Deterministic separable features per context so train_head yields valid
    # per-question heads. Runtime output is not asserted (that needs a real
    # encoder) - the baked heads' structure is what these tests verify.
    return [[float("billing" in t or "charge" in t),
             float("crash" in t or "defect" in t), 1.0] for t in texts]


def _decide_teacher(config, n=28):
    """A valid ANSWERS code block: cycle each question's allowed answers."""
    rows = []
    for i in range(n):
        row = {}
        for name, spec in config.items():
            qtype = spec["type"]
            if qtype in ("predicate", "boolean"):
                row[name] = (i % 2 == 0)
            elif qtype == "choice":
                vals = ([c["value"] for c in spec["choices"]] if "choices" in spec
                        else list(spec["options"]))
                row[name] = vals[i % len(vals)]
            else:
                labels = ([lvl["label"] for lvl in spec["levels"]] if "levels" in spec
                          else [str(x) for x in range(spec["min"], spec["max"] + 1)])
                row[name] = labels[i % len(labels)]
        rows.append(row)
    return "```python\nANSWERS = " + repr(rows) + "\n```"


def _baked_literal(source, name):
    """Literal-eval a baked top-level annotated assignment (e.g. QUESTION_HEADS, OOD)."""
    import ast as _ast

    for node in _ast.parse(source).body:
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return _ast.literal_eval(node.value)
    return {}


def _baked_heads(source):
    """Literal-eval the QUESTION_HEADS baked into a compiled decide model."""
    return _baked_literal(source, "QUESTION_HEADS")


@pytest.fixture
def decide_encoder(monkeypatch):
    """Point decide's per-question head training at a deterministic fake
    embedder, and resolve the checkpoint locally (no download)."""
    import strot.distill as distill_mod
    monkeypatch.setattr(distill_mod, "make_encoder_embedder", lambda ckpt: _decide_embed)
