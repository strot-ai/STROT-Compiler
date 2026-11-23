"""Tests for strot.compile() - the LLM-free compile door.

train() shares decide()'s deterministic core (embed -> calibrated heads + OOD
gate -> bake) but the caller brings the labels instead of a teacher LLM. These
tests stub the embedder (same fake as the decide tests) and assert on the baked
artifact, and prove that no LLM is ever constructed or called on this path.
"""

import json

import pytest

import strot
from strot.compiler import DEFAULT_DECIDE_PROJECTION, StrotCompiler
from strot.exceptions import RouterConfigurationError

from test_jit import (
    DECIDE_CONFIG,
    DECIDE_SAMPLE,
    _baked_heads,
    _baked_literal,
    _decide_embed,
    decide_encoder,  # noqa: F401 - fixture
    fake_runtime,  # noqa: F401 - fixture
    _fresh_compiled_modules,  # noqa: F401 - autouse fixture
)


def _labels(config=DECIDE_CONFIG, n=28):
    """One label dict per record, cycling each question's allowed answers -
    the same shape decide()'s teacher produces, but the caller brings it."""
    rows = []
    for i in range(n):
        row = {}
        for name, spec in config.items():
            qtype = spec["type"]
            if qtype in ("predicate", "boolean"):
                row[name] = (i % 2 == 0)
            elif qtype == "choice":
                vals = [c["value"] for c in spec["choices"]]
                row[name] = vals[i % len(vals)]
            else:
                labels = [lvl["label"] for lvl in spec["levels"]]
                row[name] = labels[i % len(labels)]
        rows.append(row)
    return rows


class TestTrainCompiles:
    def test_train_builds_one_calibrated_head_per_question(
        self, tmp_path, fake_runtime, decide_encoder
    ):
        # The caller supplies labels; STROT distills the heads with no LLM.
        result = strot.compile(
            DECIDE_SAMPLE, DECIDE_CONFIG, _labels(), state=tmp_path, execute=False,
        )
        assert result.mode == "decide"  # same artifact type as decide()
        source = result.artifact_path.read_text()
        assert "class CompiledDecider" in source
        heads = _baked_heads(source)
        assert heads["is_bug"]["type"] == "predicate"
        assert heads["is_bug"]["classes"] == [False, True]
        assert heads["sentiment"]["classes"] == ["angry", "neutral", "happy"]
        assert heads["team"]["classes"] == ["billing", "engineering"]
        # Calibrated (held-out temperature), and an OOD gate was fit.
        assert all("temperature" in h for h in heads.values())
        assert _baked_literal(source, "OOD")

    def test_strot_has_no_llm_api(self):
        # The whole LLM surface is gone: no decide(), no questions(), no clients.
        # STROT is deterministic; the agent brings the labels.
        for gone in ("decide", "questions", "openai_llm_client", "anthropic_llm_client"):
            assert not hasattr(strot, gone), gone

    def test_train_model_is_dispatchable(self, tmp_path, fake_runtime, decide_encoder):
        # The model it builds is a normal DecisionModel: dispatch writes one
        # self-contained .strot (byte copy - no runtime/torch needed here, same
        # as the decide suite, which never executes live inference).
        result = strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, _labels(), state=tmp_path,
                             execute=False)
        path = result.model.dispatch(tmp_path / "triage")
        assert path.suffix == ".strot" and path.exists()
        assert "class CompiledDecider" in path.read_text()


class TestTrainProjection:
    def test_default_projection_concats_text_fields(self):
        ns = {}
        exec(compile(DEFAULT_DECIDE_PROJECTION, "<t>", "exec"), ns)
        text = ns["route_context"]({"body": "app crash defect", "ticket_id": 7, "tier": "pro"})
        assert "body: app crash defect" in text and "tier: pro" in text
        assert "ticket_id" not in text  # numeric id is not a text field

    def test_default_projection_falls_back_to_numbers_when_no_text(self):
        ns = {}
        exec(compile(DEFAULT_DECIDE_PROJECTION, "<t>", "exec"), ns)
        assert ns["route_context"]({"score": 42}) == "score: 42"

    def test_custom_projection_is_baked(self, tmp_path, fake_runtime, decide_encoder):
        projection = (
            "def route_context(record: dict) -> str:\n"
            "    return str(record.get('body', ''))\n"
        )
        result = strot.compile(
            DECIDE_SAMPLE, DECIDE_CONFIG, _labels(), projection=projection,
            state=tmp_path, execute=False,
        )
        assert "record.get('body', '')" in result.artifact_path.read_text()

    def test_unsafe_projection_is_rejected(self, tmp_path, fake_runtime, decide_encoder):
        projection = (
            "import os\n"
            "def route_context(record: dict) -> str:\n"
            "    return os.getcwd()\n"
        )
        with pytest.raises(RouterConfigurationError, match="projection is invalid"):
            strot.compile(
                DECIDE_SAMPLE, DECIDE_CONFIG, _labels(), projection=projection,
                state=tmp_path, execute=False,
            )


class TestTrainValidation:
    def test_label_count_must_match_records(self, tmp_path, fake_runtime, decide_encoder):
        with pytest.raises(RouterConfigurationError, match="one label dict per record"):
            strot.compile(
                DECIDE_SAMPLE, DECIDE_CONFIG, _labels(n=5), state=tmp_path, execute=False,
            )

    def test_bad_label_value_is_rejected(self, tmp_path, fake_runtime, decide_encoder):
        bad = _labels()
        bad[0]["team"] = "legal"  # not an allowed choice
        with pytest.raises(RouterConfigurationError, match="not an allowed choice"):
            strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, bad, state=tmp_path, execute=False)

    def test_bare_intent_is_refused(self, tmp_path, fake_runtime, decide_encoder):
        with pytest.raises(RouterConfigurationError, match="bare\n?.*intent|typed-question"):
            strot.compile(
                DECIDE_SAMPLE, "triage these tickets", _labels(),
                state=tmp_path, execute=False,
            )

    def test_only_head_and_deep_are_valid_modes(self, tmp_path, fake_runtime, decide_encoder):
        # lite was removed (zero-shot didn't discriminate); head/deep only.
        with pytest.raises(RouterConfigurationError, match="'head' or 'deep'"):
            strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, _labels(), mode="lite",
                          state=tmp_path, execute=False)

    def test_head_without_labels_is_refused(self, tmp_path, fake_runtime, decide_encoder):
        with pytest.raises(RouterConfigurationError, match="needs labels"):
            strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, mode="head",
                          state=tmp_path, execute=False)


class TestTrainCache:
    def test_same_labels_hit_cache(self, tmp_path, fake_runtime, decide_encoder):
        first = strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, _labels(),
                            state=tmp_path, execute=False)
        again = strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, _labels(),
                            state=tmp_path, execute=False)
        assert again.cache_hit and again.cache_key == first.cache_key

    def test_different_labels_recompile(self, tmp_path, fake_runtime, decide_encoder):
        first = strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, _labels(),
                            state=tmp_path, execute=False)
        flipped = _labels()
        flipped[0]["is_bug"] = not flipped[0]["is_bug"]
        other = strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, flipped,
                            state=tmp_path, execute=False)
        assert other.cache_key != first.cache_key

    def test_projection_change_recompiles(self, tmp_path, fake_runtime, decide_encoder):
        # The projection is an input: a different route_context is a fresh model.
        base = strot.compile(DECIDE_SAMPLE, DECIDE_CONFIG, _labels(),
                             state=tmp_path, execute=False)
        custom = strot.compile(
            DECIDE_SAMPLE, DECIDE_CONFIG, _labels(),
            projection="def route_context(record: dict) -> str:\n    return str(record.get('body',''))\n",
            state=tmp_path, execute=False,
        )
        assert custom.cache_key != base.cache_key
