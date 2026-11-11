"""Tests for strot.run(): the production JIT facade.

The compiled local artifact imports the Julia decision package at module
scope, so these tests install a minimal pure-Python fake of the ``julia``
module that routes by keyword overlap between the record state and the
option descriptions — enough to exercise the full compiled code path
(route -> threshold -> dispatch) without the real 550MB checkpoint.
"""

import importlib.util
import json
import math
import sys
import types

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

def make_paths(keys):
    """Contract-compliant synthesizer output for the given action keys.

    Labels are the bare keys so the fake keyword router matches them against
    the record body (e.g. body "double billing charge" -> label "billing").
    """
    labels = ", ".join(f'"{k}": "{k}"' for k in keys)
    handlers = "\n\n\n".join(
        f'def execute_{k}(record: dict) -> dict:\n'
        f'    try:\n'
        f'        return {{"queue": "{k}", "ticket_id": record.get("ticket_id")}}\n'
        f'    except Exception as exc:\n'
        f'        return {{"queue": "{k}", "error": str(exc)}}'
        for k in keys
    )
    return (
        "```python\n"
        f"ROUTING_LABELS = {{{labels}}}\n\n\n"
        "def route_context(record: dict) -> str:\n"
        '    return str(record.get("body", ""))\n\n\n'
        f"{handlers}\n```"
    )


VALID_PATHS = make_paths(["billing", "defect"])


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


# --------------------------------------------------------------------- #
# Fake julia package: keyword-overlap decision engine                    #
# --------------------------------------------------------------------- #


def _keyword_scores(state, options):
    """Score each option by word overlap with the serialized record state."""
    import re

    state_words = set(re.findall(r"[a-z]+", str(state).lower()))
    return [
        float(len(set(re.findall(r"[a-z]+", opt.lower())) & state_words))
        for opt in options
    ]


def _build_fake_julia():
    """A stand-in for supersonic-julia's `julia` module: same predict contract
    (rows of {state, question, options} -> [{index, probabilities}])."""
    julia_mod = types.ModuleType("julia")

    def _softmax_overlap(state, labels):
        exps = [math.exp(s * 4.0) for s in _keyword_scores(state, labels)]
        total = sum(exps)
        return [e / total for e in exps]

    class _FakeEngine:
        def predict(self, rows=None, questions=None, state=None):
            if questions is not None:  # typed path, mirrors julia/typed.py output
                answers = {}
                for qid, q in questions.items():
                    kind = q["type"]
                    if kind == "choice":
                        keys = list(q["criteria"])
                        labels = list(q["criteria"].values())
                    elif kind == "score":
                        labels = list(q["criteria"])
                        keys = [str(i) for i in range(len(labels))]
                    else:  # noul
                        keys = ["false", "true"]
                        criteria = q.get("criteria")
                        labels = [criteria["false"], criteria["true"]] if criteria else keys
                    probs = _softmax_overlap(state, labels)
                    answer = {"type": kind, "probabilities": dict(zip(keys, probs))}
                    if kind == "choice":
                        answer["choice"] = keys[max(range(len(probs)), key=probs.__getitem__)]
                    elif kind == "score":
                        answer["score"] = sum(i * p for i, p in enumerate(probs))
                    else:
                        answer["noul"] = probs[1]
                    if kind != "noul":
                        answer["max_probability"] = max(probs)
                    answers[qid] = answer
                return {"answers": answers}
            results = []
            for row in rows:
                probs = _softmax_overlap(row["state"], row["options"])
                results.append(
                    {
                        "index": max(range(len(probs)), key=probs.__getitem__),
                        "probabilities": probs,
                    }
                )
            return results

    julia_mod.load_model = lambda checkpoint, device="cpu", **kwargs: _FakeEngine()
    return julia_mod


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
    monkeypatch.setitem(sys.modules, "julia", _build_fake_julia())
    # Point checkpoint resolution at a local dir so the artifact never tries
    # a Hugging Face download during tests.
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
    def make_key(self, sample=SAMPLE, query=QUERY, registry=REGISTRY, nlu="SupersonicLabs/Julia-1"):
        schema = introspect_schema(sample)
        return cache_key(query, registry, nlu, schema, None, 0.40)

    def test_deterministic(self):
        assert self.make_key() == self.make_key()

    def test_query_change_busts_cache(self):
        assert self.make_key() != self.make_key(query="A different goal entirely.")

    def test_registry_change_busts_cache(self):
        bigger = dict(REGISTRY, refund="Refund processing")
        assert self.make_key() != self.make_key(registry=bigger)

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

    def test_optimize_flag_changes_key(self):
        schema = introspect_schema(SAMPLE)
        plain = cache_key(QUERY, REGISTRY, "SupersonicLabs/Julia-1", schema, None, 0.40)
        optimized = cache_key(QUERY, REGISTRY, "SupersonicLabs/Julia-1", schema,
                              None, 0.40, optimize=True)
        assert plain != optimized


# --------------------------------------------------------------------- #
# The JIT loop: miss -> compile -> persist; hit -> execute only          #
# --------------------------------------------------------------------- #


class TestRun:
    @pytest.mark.parametrize("alias", [False, True])
    def test_cache_refreshes_replaced_source_through_path_alias(self, tmp_path, alias):
        import os

        path = tmp_path / "asset.py"
        path.write_text("value = 'first'\n")
        first = _load_artifact(path, "replaced")
        alias = tmp_path / "directory" / ".." / "asset.py" if alias else path
        (tmp_path / "directory").mkdir()
        assert _load_artifact(alias, "replaced") is first
        replacement = tmp_path / "replacement.py"
        replacement.write_text("value = 'other'\n")
        stat = path.stat()
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(path)
        assert _load_artifact(alias, "replaced").value == "other"
        assert _load_artifact(path, "replaced").value == "other"

    def test_parallel_loads_wait_for_complete_module(self, monkeypatch, tmp_path):
        import builtins
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        from threading import Event

        started, release = Event(), Event()
        def pause():
            started.set()
            assert release.wait(5)
        monkeypatch.setattr(builtins, "_strot_pause", pause, raising=False)
        path = tmp_path / "asset.py"
        path.write_text("import builtins\nbuiltins._strot_pause()\nvalue = 1\n")
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(_load_artifact, path, "parallel")
            assert started.wait(5)
            second = pool.submit(_load_artifact, path, "parallel")
            try:
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.05)
            finally:
                release.set()
            assert first.result(timeout=5) is second.result(timeout=5)
            assert first.result().value == 1

    def test_failed_load_does_not_poison_cache(self, tmp_path):
        path = tmp_path / "asset.py"
        path.write_text("invalid syntax here\n")
        with pytest.raises(RouterConfigurationError, match="failed to load"):
            _load_artifact(path, "failed_load")
        path.write_text("value = 1\n")
        assert _load_artifact(path, "failed_load").value == 1

    def test_load_ignores_stale_bytecode(self, tmp_path):
        import os
        import py_compile

        path = tmp_path / "asset.py"
        path.write_text("value = 'first'\n")
        stamp = path.stat()
        py_compile.compile(str(path), doraise=True)
        path.write_text("value = 'other'\n")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert _load_artifact(path, "stale_bytecode").value == "other"

    def test_cache_miss_compiles_persists_and_executes(self, tmp_path, fake_runtime):
        llm = ScriptedLLM([VALID_PATHS])
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path)

        assert llm.calls == 1
        assert result.cache_hit is False
        assert result.attempts == 1
        assert result.artifact_path.exists()
        metadata = json.loads(
            result.artifact_path.with_suffix(".json").read_text()
        )
        assert metadata["cache_key"] == result.cache_key
        assert metadata["runtime_kind"] == "local"

        assert len(result.outcomes) == 2
        routed = {o["record"]["ticket_id"]: o["action"] for o in result.outcomes}
        assert routed == {1: "billing", 2: "defect"}

    def test_battery_filename_leads_with_auto_slug_from_query(self, tmp_path, fake_runtime):
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        stem = result.artifact_path.stem
        # battery_<slug>_<hash>: slug auto-derived from the query words.
        assert stem.startswith("battery_triage-incoming-tickets")
        assert stem.endswith(result.cache_key[:16])
        assert result.artifact_path.with_suffix(".json").exists()

    def test_name_seeds_the_battery_filename_slug(self, tmp_path, fake_runtime):
        result = strot.run(
            SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]),
            state=tmp_path, name="Ticket Router v2",
        )
        assert result.artifact_path.stem.startswith("battery_ticket-router-v2_")

    def test_name_is_cosmetic_only_cache_key_unchanged(self, tmp_path, fake_runtime):
        # The slug rides in the filename but is NOT part of the content key:
        # the same (query, schema, ...) under two names recompiles under a new
        # filename yet produces the identical cache key.
        a = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]),
                      state=tmp_path, name="alpha")
        b = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]),
                      state=tmp_path, name="beta")
        assert a.cache_key == b.cache_key
        assert a.artifact_path != b.artifact_path
        assert a.artifact_path.stem.startswith("battery_alpha_")
        assert b.artifact_path.stem.startswith("battery_beta_")

    def test_cache_hit_never_touches_the_llm(self, tmp_path, fake_runtime):
        strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path)
        assert result.cache_hit is True
        assert result.attempts == 0
        assert len(result.outcomes) == 2

    def test_state_directories_do_not_share_loaded_code(self, tmp_path, fake_runtime):
        for directory, queue in [("first", "billing"), ("second", "changed")]:
            paths = VALID_PATHS.replace('"queue": "billing"', f'"queue": "{queue}"')
            for llm in [ScriptedLLM([paths]), ExplodingLLM()]:
                result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path / directory)
                assert result.outcomes[0]["result"]["queue"] == queue

    @pytest.mark.parametrize("execute", [True, False])
    @pytest.mark.parametrize("mode", ["run", "decide"])
    def test_recompile_reloads_code(self, tmp_path, fake_runtime, execute, mode):
        call = getattr(strot, mode)
        args = (SAMPLE, QUERY, REGISTRY) if mode == "run" else (SAMPLE, DECIDE_CONFIG)
        source = VALID_PATHS if mode == "run" else VALID_DECIDE
        first = call(*args, llm=ScriptedLLM([source]), state=tmp_path)
        first.artifact_path.unlink()
        paths = source.replace('return str(record.get("body", ""))', 'return "app crash defect"')
        second = call(*args, llm=ScriptedLLM([paths]), state=tmp_path, execute=execute)
        assert not second.cache_hit
        warm = call(*args, llm=ExplodingLLM(), state=tmp_path)
        if mode == "run":
            assert warm.outcomes[0]["action"] == "defect"
        else:
            assert warm.outcomes[0]["decision"]["choices"]["team"]["selected"] == "engineering"

    def test_lazy_string_spec_not_resolved_on_hit(self, tmp_path, fake_runtime):
        strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        # openai is not installed here: resolving this spec would raise.
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm="openai/gpt-4o", state=tmp_path)
        assert result.cache_hit is True

    def test_schema_drift_triggers_one_recompile(self, tmp_path, fake_runtime):
        first = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        drifted = [dict(r, severity="high") for r in SAMPLE]
        second = strot.run(drifted, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        assert second.cache_hit is False
        assert second.cache_key != first.cache_key
        assert first.artifact_path.exists() and second.artifact_path.exists()

    def test_low_confidence_dispatched_with_confidence_without_target(self, tmp_path, fake_runtime):
        # Default (no target): STROT never hard-abstains. A low-confidence,
        # zero-overlap record (ties at ~0.33) is DISPATCHED with its low
        # confidence shown, not collected as a fallback. Hard abstention is
        # opt-in via target_risk / target_accuracy.
        registry3 = dict(REGISTRY, lifecycle="Account deletion and data export requests")
        paths3 = make_paths(["billing", "defect", "lifecycle"])
        with_unroutable = SAMPLE + [{"ticket_id": 3, "body": "zzz qqq", "tier": None}]
        result = strot.run(
            with_unroutable, QUERY, registry3, llm=ScriptedLLM([paths3]), state=tmp_path
        )
        assert len(result.outcomes) == 3  # everything dispatched, nothing refused
        assert len(result.fallbacks) == 0  # no abstain without a target
        unroutable = [o for o in result.outcomes if o["record"]["ticket_id"] == 3][0]
        assert unroutable["confidence"] < 0.40  # the low confidence is shown, not hidden

    def test_execute_false_warms_cache_without_runtime(self, tmp_path):
        # No fake torch here: loading the artifact would fail, proving the
        # warm-only path never touches the runtime.
        llm = ScriptedLLM([VALID_PATHS])
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, execute=False)
        assert result.artifact_path.exists()
        assert result.pipeline is None
        assert result.outcomes == []

    def test_golden_gate_promotes_passing_artifact(self, tmp_path, fake_runtime):
        golden = [
            {"record": SAMPLE[0], "expect": "billing"},
            {"record": SAMPLE[1], "expect": "defect"},
        ]
        llm = ScriptedLLM([VALID_PATHS])
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, golden=golden)
        assert result.attempts == 1
        assert len(result.outcomes) == 2

    def test_golden_gate_rejects_misrouting_artifact_and_repairs(self, tmp_path, fake_runtime):
        swapped = VALID_PATHS.replace(
            'ROUTING_LABELS = {"billing": "billing", "defect": "defect"}',
            'ROUTING_LABELS = {"billing": "defect", "defect": "billing"}',
        )
        llm = ScriptedLLM([swapped, VALID_PATHS])
        golden = [{"record": SAMPLE[0], "expect": "billing"}]
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, golden=golden)
        assert result.attempts == 2
        routed = {o["record"]["ticket_id"]: o["action"] for o in result.outcomes}
        assert routed == {1: "billing", 2: "defect"}

    def test_golden_validation_before_llm(self, tmp_path):
        golden = [{"record": SAMPLE[0], "expect": "not_an_action"}]
        with pytest.raises(RouterConfigurationError, match="expect"):
            strot.run(SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path, golden=golden)

    def test_golden_is_part_of_cache_key(self, tmp_path, fake_runtime):
        plain = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        golden = [{"record": SAMPLE[0], "expect": "billing"}]
        gated = strot.run(
            SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path, golden=golden
        )
        assert plain.cache_key != gated.cache_key

    def test_summary_counts_actions(self, tmp_path, fake_runtime):
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        summary = result.summary()
        assert summary["processed"] == 2
        assert summary["actions"] == {"billing": 1, "defect": 1}

    def test_unresolvable_llm_spec_raises(self, tmp_path):
        with pytest.raises(RouterConfigurationError):
            strot.run(SAMPLE, QUERY, REGISTRY, llm=12345, state=tmp_path)

    def test_anthropic_spec_requires_extra_when_missing(self, tmp_path):
        if importlib.util.find_spec("anthropic") is not None:
            pytest.skip("anthropic installed; factory would construct a real client")
        with pytest.raises(RouterConfigurationError, match="anthropic"):
            strot.run(SAMPLE, QUERY, REGISTRY, llm="anthropic/claude-opus-4-8", state=tmp_path)


class TestCalibration:
    SAMPLE3 = SAMPLE + [{"ticket_id": 3, "body": "zzz qqq", "tier": None}]
    TEACHER = '```python\nLABELS = ["billing", "defect", "billing"]\n```'

    def test_calibrated_artifact_ships_fitted_guarantee(self, tmp_path, fake_runtime):
        llm = ScriptedLLM([VALID_PATHS, self.TEACHER])
        result = strot.run(
            self.SAMPLE3, QUERY, REGISTRY, llm=llm, state=tmp_path, target_risk=0.5
        )
        assert llm.calls == 2  # one synthesis + one teacher labeling
        source = result.artifact_path.read_text()
        assert "CONFORMAL_Q = None" not in source  # fitted quantile baked in
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        calibration = meta["calibration"]
        assert calibration["n_calibration"] == 3
        assert calibration["alpha"] == 0.5
        assert calibration["temperature"] > 0
        # Clear records dispatch via singleton prediction sets...
        routed = {o["record"]["ticket_id"]: o["action"] for o in result.outcomes}
        assert routed == {1: "billing", 2: "defect"}
        # ...the ambiguous record abstains (set is not a singleton).
        assert [fb["record"]["ticket_id"] for fb in result.fallbacks] == [3]
        assert "Conformal prediction set" in result.fallbacks[0]["reason"]
        # Warm call: zero LLM, identical behavior.
        warm = strot.run(
            self.SAMPLE3, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path,
            target_risk=0.5,
        )
        assert warm.cache_hit is True
        assert len(warm.fallbacks) == 1

    def test_target_risk_is_part_of_cache_key(self, tmp_path, fake_runtime):
        plain = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        calibrated = strot.run(
            SAMPLE, QUERY, REGISTRY,
            llm=ScriptedLLM([VALID_PATHS, '```python\nLABELS = ["billing", "defect"]\n```']),
            state=tmp_path, target_risk=0.5,
        )
        assert plain.cache_key != calibrated.cache_key

    def test_invalid_target_risk_rejected(self, tmp_path):
        with pytest.raises(RouterConfigurationError, match="target_risk"):
            strot.run(SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path, target_risk=1.5)


class TestMeasuredLabels:
    TEACHER = '```python\nLABELS = ["billing", "defect"]\n```'

    def test_bad_synthesized_labels_lose_to_registry_keys(self, tmp_path, fake_runtime):
        # Synthesized labels match nothing in the records; bare keys match.
        bad_labels = VALID_PATHS.replace(
            'ROUTING_LABELS = {"billing": "billing", "defect": "defect"}',
            'ROUTING_LABELS = {"billing": "zzz", "defect": "qqq"}',
        )
        llm = ScriptedLLM([bad_labels, self.TEACHER])
        result = strot.run(
            SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, measure_labels=True
        )
        assert llm.calls == 2  # synthesis + one teacher call
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        assert meta["label_search"]["winner"] == "registry_keys"
        assert meta["label_search"]["scores"]["registry_keys"] > meta["label_search"]["scores"]["synthesized"]
        # The measured winner actually routes correctly at runtime.
        routed = {o["record"]["ticket_id"]: o["action"] for o in result.outcomes}
        assert routed == {1: "billing", 2: "defect"}
        assert "Measured label search winner" in result.artifact_path.read_text()

    def test_good_synthesized_labels_are_kept(self, tmp_path, fake_runtime):
        # Identical candidates (keys == synthesized): measurement short-circuits.
        llm = ScriptedLLM([VALID_PATHS, self.TEACHER])
        result = strot.run(
            SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, measure_labels=True
        )
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        assert meta["label_search"]["winner"] == "synthesized"
        assert "Measured label search winner" not in result.artifact_path.read_text()

    def test_measurement_defaults_on_with_target_risk(self, tmp_path, fake_runtime):
        llm = ScriptedLLM([VALID_PATHS, self.TEACHER])
        result = strot.run(
            SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, target_risk=0.5
        )
        assert llm.calls == 2  # measurement shares the single teacher call
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        assert meta["label_search"] is not None
        assert meta["calibration"] is not None

    def test_measure_flag_is_part_of_cache_key(self, tmp_path, fake_runtime):
        plain = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        measured = strot.run(
            SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS, self.TEACHER]),
            state=tmp_path, measure_labels=True,
        )
        assert plain.cache_key != measured.cache_key


class TestDistill:
    # 28 records (>= MIN_DISTILL_RECORDS), alternating classes.
    BIG_SAMPLE = [
        {"ticket_id": i, "body": ("double billing charge refund" if i % 2 == 0
                                  else "app crash defect bug"), "tier": "pro"}
        for i in range(28)
    ]
    TEACHER = ("```python\nLABELS = ["
               + ", ".join('"billing"' if i % 2 == 0 else '"defect"' for i in range(28))
               + "]\n```")

    @staticmethod
    def _keyword_embedder(texts):
        # Linearly separable 3-dim features: billing-ish, defect-ish, bias.
        return [
            [float("billing" in t or "charge" in t), float("crash" in t or "defect" in t), 1.0]
            for t in texts
        ]

    def test_distilled_head_wins_over_bad_zero_shot(self, tmp_path, fake_runtime, monkeypatch):
        import strot.distill as distill_mod
        monkeypatch.setattr(
            distill_mod, "make_encoder_embedder", lambda ckpt: self._keyword_embedder
        )
        bad_labels = VALID_PATHS.replace(
            'ROUTING_LABELS = {"billing": "billing", "defect": "defect"}',
            'ROUTING_LABELS = {"billing": "zzz", "defect": "qqq"}',
        )
        llm = ScriptedLLM([bad_labels, self.TEACHER])
        result = strot.run(
            self.BIG_SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path,
            sample_size=28, distill=True, execute=False,  # distilled artifact needs torch to run
        )
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["winner"] == "distilled"
        assert search["distilled"] > search["zero_shot"]
        assert search["n_train"] == 14 and search["n_measure"] == 7
        source = result.artifact_path.read_text()
        assert "HEAD_WEIGHTS" in source and "distilled" in meta["runtime_kind"]

    def test_zero_shot_kept_when_head_cannot_beat_it(self, tmp_path, fake_runtime, monkeypatch):
        import strot.distill as distill_mod
        monkeypatch.setattr(
            distill_mod, "make_encoder_embedder",
            lambda ckpt: (lambda texts: [[1.0, 1.0] for _ in texts]),  # useless embeddings
        )
        llm = ScriptedLLM([VALID_PATHS, self.TEACHER])
        result = strot.run(
            self.BIG_SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path,
            sample_size=28, distill=True,
        )
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        assert meta["model_search"]["winner"] == "zero_shot"
        assert "HEAD_WEIGHTS" not in result.artifact_path.read_text()
        # zero-shot artifact executes normally under the fake runtime
        assert len(result.outcomes) == 28

    def test_distill_requires_enough_records(self, tmp_path, fake_runtime):
        with pytest.raises(RouterConfigurationError, match="sample_size"):
            strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS, "x"]),
                      state=tmp_path, distill=True)

    def test_distill_is_part_of_cache_key(self, tmp_path, fake_runtime, monkeypatch):
        import strot.distill as distill_mod
        monkeypatch.setattr(
            distill_mod, "make_encoder_embedder", lambda ckpt: self._keyword_embedder
        )
        plain = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                          llm=ScriptedLLM([VALID_PATHS]), state=tmp_path, sample_size=28)
        distilled = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                              llm=ScriptedLLM([VALID_PATHS, self.TEACHER]),
                              state=tmp_path, sample_size=28, distill=True, execute=False)
        assert plain.cache_key != distilled.cache_key


class TestOptimize:
    """The optimize flag + fine-tune escalation (no torch: finetune_adapter is
    faked). Mirrors TestDistill's sample/teacher so folds behave identically."""

    BIG_SAMPLE = TestDistill.BIG_SAMPLE
    TEACHER = TestDistill.TEACHER

    # Labels the keyword router cannot match -> zero-shot misses on the measure
    # fold, so the frozen tier does not clear the bar and optimize escalates.
    BAD_LABELS = VALID_PATHS.replace(
        'ROUTING_LABELS = {"billing": "billing", "defect": "defect"}',
        'ROUTING_LABELS = {"billing": "zzz", "defect": "qqq"}',
    )

    @staticmethod
    def _fake_trainer(measure_acc, n_classes=2):
        """A fake finetune_adapter: emits per-measure-row distributions whose
        argmax hits ``measure_acc`` of the real measure-fold labels, so the
        ensemble's own recomputed accuracy (from the averaged rows) matches.
        Every member returns identical rows, so the K-way average is stable."""
        from strot.finetune import FineTuneResult

        def _trainer(texts, labels, n_cls, measure, calibrate, train, config=None):
            measure = list(measure)
            n_hit = round(measure_acc * len(measure))
            rows = []
            for pos, idx in enumerate(measure):
                true = int(labels[idx])
                # First n_hit rows peak on the true class; the rest peak wrong.
                peak = true if pos < n_hit else (true + 1) % n_cls
                rows.append([0.9 if c == peak else 0.1 / (n_cls - 1) for c in range(n_cls)])
            calib = [[0.8, 0.2], [0.3, 0.7]] * 3
            return FineTuneResult(
                measure_accuracy=measure_acc,
                measure_distributions=rows,
                calibrate_distributions=calib,
                adapter={
                    "__head_weight__": [[0.0]] * n_cls,
                    "__head_bias__": [0.0] * n_cls,
                },
                config={"base_model": "x", "cut": 0, "n_layers": 1, "lora_rank": 2,
                        "lora_alpha": 4, "max_length": 8, "n_classes": n_cls,
                        "device": "cpu"},
            )

        return _trainer

    def test_optimize_is_part_of_cache_key(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        monkeypatch.setattr(ft_mod, "finetune_adapter", self._fake_trainer(1.0))
        plain = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                          llm=ScriptedLLM([VALID_PATHS]), state=tmp_path, sample_size=28)
        optimized = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                              llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                              state=tmp_path, sample_size=28, optimize=True, execute=False)
        assert plain.cache_key != optimized.cache_key

    def test_optimize_without_target_is_a_noop(self, tmp_path, fake_runtime, monkeypatch):
        # Adaptive, not maximal: optimize=True WITHOUT a target_accuracy must
        # never fine-tune (there is no measured gap to close), even when the
        # frozen tier is imperfect (BAD_LABELS -> zero-shot misses).
        import strot.finetune as ft_mod

        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("finetune_adapter called though no target was set")

        monkeypatch.setattr(ft_mod, "finetune_adapter", _boom)
        monkeypatch.setattr(ft_mod, "finetune_ensemble", _boom)
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True, execute=False)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["optimize"] == "skipped_no_target"
        assert search["winner"] == "zero_shot"
        assert "ADAPTER" not in result.artifact_path.read_text()

    def test_frozen_clears_bar_is_a_noop(self, tmp_path, fake_runtime, monkeypatch):
        # VALID_PATHS -> keyword router scores 1.0 on the measure fold, so with
        # a target_accuracy of 0.9 the frozen tier CLEARS the bar and the
        # fine-tune tier must not be invoked.
        import strot.finetune as ft_mod

        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("finetune_adapter called though frozen cleared the bar")

        monkeypatch.setattr(ft_mod, "finetune_adapter", _boom)
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([VALID_PATHS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["optimize"] == "skipped_frozen_cleared_bar"
        assert search["winner"] == "zero_shot"
        assert "ADAPTER" not in result.artifact_path.read_text()

    def test_finetune_wins_when_it_beats_frozen(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        monkeypatch.setattr(ft_mod, "finetune_adapter", self._fake_trainer(0.95))
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9, execute=False)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["winner"] == "fine_tuned"
        # Ensemble judges on the member-averaged distributions: accuracy is
        # recomputed from those rows (>= the frozen baseline it must beat).
        assert search["fine_tuned"] > search["zero_shot"]  # frozen missed
        assert search["ensemble_size"] == 3  # default internal battery width
        assert len(search["member_accuracies"]) == 3
        source = result.artifact_path.read_text()
        # The list of adapters is baked (serving averages them).
        assert "ADAPTER = [" in source and "fine_tuned" in meta["runtime_kind"]

    def test_frozen_kept_when_finetune_cannot_beat_it(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        # Fine-tune accuracy no better than the (missed) zero-shot baseline.
        monkeypatch.setattr(ft_mod, "finetune_adapter", self._fake_trainer(0.0))
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9, execute=False)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        assert meta["model_search"]["winner"] == "zero_shot"
        assert "ADAPTER" not in result.artifact_path.read_text()

    def test_missing_extra_degrades_gracefully(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        from strot.exceptions import RouterConfigurationError as RCE

        def _no_extra(*a, **k):
            raise RCE("The fine-tune tier needs the finetune extra")

        monkeypatch.setattr(ft_mod, "finetune_adapter", _no_extra)
        # Must not crash: falls back to the frozen tier and records why.
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["optimize"] == "unavailable"
        assert "finetune extra" in search["optimize_error"]
        assert search["winner"] == "zero_shot"
        assert "ADAPTER" not in result.artifact_path.read_text()

    def test_optimize_requires_local_runtime(self, tmp_path):
        with pytest.raises(RouterConfigurationError, match="optimize requires"):
            strot.run(self.BIG_SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(),
                      nlu="openai/gpt-4o", state=tmp_path, sample_size=28, optimize=True)

    def test_optimize_requires_enough_records(self, tmp_path, fake_runtime):
        with pytest.raises(RouterConfigurationError, match="sample_size"):
            strot.run(SAMPLE, QUERY, REGISTRY,
                      llm=ScriptedLLM([VALID_PATHS, "x"]),
                      state=tmp_path, optimize=True)


class TestReuseAdapter:
    """Adopt a pre-built adapter battery before training (find/reuse first).

    No torch: strot.finetune.validate_adapter and finetune_ensemble are faked,
    so the compiler's adopt-or-train branch is exercised purely in Python.
    """

    BIG_SAMPLE = TestDistill.BIG_SAMPLE
    TEACHER = TestDistill.TEACHER
    BAD_LABELS = TestOptimize.BAD_LABELS

    #: A pre-built battery shaped like a prior EnsembleResult's adapters+config.
    REUSE = {
        "adapters": [{"__head_bias__": [0.0, 0.0]}, {"__head_bias__": [0.0, 0.0]}],
        "config": {"base_model": "x", "cut": 0, "n_layers": 1, "lora_rank": 2,
                   "lora_alpha": 4, "max_length": 8, "n_classes": 2, "device": "cpu"},
    }

    @staticmethod
    def _fake_validator(measure_acc, calib_row=None):
        """A fake validate_adapter returning an EnsembleResult whose averaged
        measure rows hit ``measure_acc`` of the real measure-fold labels. The
        calibrate rows are sized to the real calibrate fold; ``calib_row`` sets
        a distinctive per-row signature (default is a plain [0.8, 0.2])."""
        from strot.finetune import EnsembleResult

        def _validate(texts, labels, n_classes, measure, calibrate, adapters, config,
                      _loader=None):
            measure = list(measure)
            n_hit = round(measure_acc * len(measure))
            rows = []
            for pos, idx in enumerate(measure):
                true = int(labels[idx])
                peak = true if pos < n_hit else (true + 1) % n_classes
                rows.append([0.9 if c == peak else 0.1 / (n_classes - 1)
                             for c in range(n_classes)])
            row = calib_row or [0.8, 0.2]
            return EnsembleResult(
                measure_accuracy=measure_acc,
                measure_distributions=rows,
                calibrate_distributions=[list(row) for _ in calibrate],
                adapters=list(adapters),
                config=dict(config),
            )

        return _validate

    def test_adopts_when_validates_and_skips_training(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod

        def _never_trains(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("finetune_ensemble called though reuse validated")

        monkeypatch.setattr(ft_mod, "validate_adapter", self._fake_validator(0.95))
        monkeypatch.setattr(ft_mod, "finetune_ensemble", _never_trains)
        # target_accuracy=0.9 is the bar: frozen (BAD labels) misses it, the
        # reused battery (0.95) clears it -> adopt, skip training.
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9, reuse_adapter=self.REUSE, execute=False)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["optimize"] == "reused"
        assert search["winner"] == "fine_tuned"
        assert search["reused_measure_accuracy"] == 0.95
        assert search["ensemble_size"] == 2  # the provided battery width
        assert "fine_tuned" in meta["runtime_kind"]
        # The battery's adapters are baked, exactly like a trained winner.
        assert "ADAPTER = [" in result.artifact_path.read_text()

    def test_trains_when_reuse_fails_to_clear_bar(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        trained = {"called": False}

        monkeypatch.setattr(ft_mod, "validate_adapter", self._fake_validator(0.0))

        # Use the real ensemble over the faked member trainer so training runs.
        monkeypatch.setattr(ft_mod, "finetune_adapter", TestOptimize._fake_trainer(0.95))
        orig_ensemble = ft_mod.finetune_ensemble

        def _spy_ensemble(*a, **k):
            trained["called"] = True
            return orig_ensemble(*a, **k)

        monkeypatch.setattr(ft_mod, "finetune_ensemble", _spy_ensemble)
        # Bar 0.9: frozen misses it, the reused battery (0.0) also misses it ->
        # reuse rejected, fall through to training.
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9, reuse_adapter=self.REUSE, execute=False)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert trained["called"] is True  # fell through to training
        assert search["optimize_reuse"] == "rejected"
        assert search["reused_measure_accuracy"] == 0.0
        assert search["winner"] == "fine_tuned"  # the trained battery won

    def test_reuse_missing_extra_degrades_to_training(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        from strot.exceptions import RouterConfigurationError as RCE

        def _no_extra(*a, **k):
            raise RCE("The fine-tune tier needs the finetune extra")

        monkeypatch.setattr(ft_mod, "validate_adapter", _no_extra)
        monkeypatch.setattr(ft_mod, "finetune_adapter", TestOptimize._fake_trainer(0.95))
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           target_accuracy=0.9, reuse_adapter=self.REUSE, execute=False)
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        search = meta["model_search"]
        assert search["optimize_reuse"] == "unavailable"
        assert "finetune extra" in search["optimize_reuse_error"]
        assert search["winner"] == "fine_tuned"  # trained path still ran

    def test_calibration_refit_on_this_task_not_reused(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        from strot import calibrate as calibrate_mod

        # A distinctive calibrate signature so we can prove it drove the fit.
        sentinel_row = [0.73, 0.27]
        monkeypatch.setattr(
            ft_mod, "validate_adapter", self._fake_validator(0.95, calib_row=sentinel_row)
        )
        monkeypatch.setattr(ft_mod, "finetune_ensemble",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("should not train")))

        seen = {}
        orig_fit = calibrate_mod.fit_temperature

        def _spy_fit(dists, true_indices):
            seen["dists"] = dists
            return orig_fit(dists, true_indices)

        monkeypatch.setattr(calibrate_mod, "fit_temperature", _spy_fit)
        # target_accuracy=0.9: frozen (BAD labels) misses the bar, the reused
        # battery (0.95) clears it and is adopted; the policy fit then runs on
        # the battery's own calibrate-fold rows. measure_labels=False so the
        # label search can't rescue the BAD labels and clear the bar frozen.
        strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                  llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                  state=tmp_path, sample_size=28, optimize=True,
                  reuse_adapter=self.REUSE, target_accuracy=0.9,
                  measure_labels=False, execute=False)
        # Calibration ran on the adopted battery's own calibrate-fold rows
        # (scored on this task's calibrate fold), never skipped. Every row is
        # the sentinel signature -> the reused battery's rows drove the fit.
        assert seen["dists"] and all(r == sentinel_row for r in seen["dists"])

    def test_reuse_adapter_is_part_of_cache_key(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod
        monkeypatch.setattr(ft_mod, "finetune_adapter", TestOptimize._fake_trainer(0.95))
        monkeypatch.setattr(ft_mod, "validate_adapter", self._fake_validator(0.95))
        plain = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                          llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                          state=tmp_path, sample_size=28, optimize=True, execute=False)
        reused = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([self.BAD_LABELS, self.TEACHER]),
                           state=tmp_path, sample_size=28, optimize=True,
                           reuse_adapter=self.REUSE, execute=False)
        assert plain.cache_key != reused.cache_key

    def test_reuse_ignored_when_optimize_off(self, tmp_path, fake_runtime, monkeypatch):
        import strot.finetune as ft_mod

        def _boom(*a, **k):  # pragma: no cover - must never run without optimize
            raise AssertionError("validate_adapter called though optimize is off")

        monkeypatch.setattr(ft_mod, "validate_adapter", _boom)
        # optimize off: reuse_adapter is inert, no fine-tune tier touched.
        result = strot.run(self.BIG_SAMPLE, QUERY, REGISTRY,
                           llm=ScriptedLLM([VALID_PATHS]),
                           state=tmp_path, sample_size=28,
                           reuse_adapter=self.REUSE, execute=False)
        assert result.artifact_path.exists()


class TestTargetAccuracy:
    # 24 labeled records: 16 clear (separable by keyword), 8 ambiguous.
    SAMPLE24 = (
        [{"ticket_id": i, "body": "double billing charge refund" if i % 3 else "app crash defect bug",
          "tier": "pro"} for i in range(16)]
        + [{"ticket_id": 100 + i, "body": f"misc note {i}", "tier": None} for i in range(8)]
    )
    # Ambiguous records get genuinely 50/50 labels (alternating) so neither
    # argmax tie-breaking nor the fitted class bias can make them "correct":
    # they stay low-confidence and half-wrong, forcing the cutoff above them.
    TEACHER = ("```python\nLABELS = ["
               + ", ".join('"billing"' if i % 3 else '"defect"' for i in range(16))
               + ", " + ", ".join('"billing"' if i % 2 else '"defect"' for i in range(8))
               + "]\n```")

    def test_policy_fits_cutoff_and_dispatches_selectively(self, tmp_path, fake_runtime):
        llm = ScriptedLLM([VALID_PATHS, self.TEACHER])
        result = strot.run(
            self.SAMPLE24, QUERY, REGISTRY, llm=llm, state=tmp_path,
            sample_size=24, target_accuracy=0.9, measure_labels=False,
        )
        meta = json.loads(result.artifact_path.with_suffix(".json").read_text())
        policy = meta["policy"]
        assert policy["target_accuracy"] == 0.9
        assert policy["dispatch_cutoff"] is not None
        assert "NOT a finite-sample guarantee" in policy["note"]
        # Clear records dispatch locally; ambiguous ones (no keywords) abstain.
        assert len(result.outcomes) > 0
        ambiguous_ids = {100 + i for i in range(8)}
        fallback_ids = {f["record"]["ticket_id"] for f in result.fallbacks}
        assert ambiguous_ids <= fallback_ids

    def test_mutual_exclusion_with_target_risk(self, tmp_path):
        with pytest.raises(RouterConfigurationError, match="not both"):
            strot.run(SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path,
                      target_risk=0.1, target_accuracy=0.8)

    def test_target_accuracy_in_cache_key(self, tmp_path, fake_runtime):
        plain = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        policy = strot.run(
            self.SAMPLE24, QUERY, REGISTRY,
            llm=ScriptedLLM([VALID_PATHS, self.TEACHER]), state=tmp_path,
            sample_size=24, target_accuracy=0.9, measure_labels=False,
        )
        assert plain.cache_key != policy.cache_key


class TestEpisodeLog:
    def test_promoted_compile_logged_once(self, tmp_path, fake_runtime):
        strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        strot.run(SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path)  # warm: no episode
        lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
        assert len(lines) == 1
        episode = json.loads(lines[0])
        assert episode["kind"] == "run"
        assert episode["status"] == "promoted"
        assert episode["attempts"] == 1
        assert episode["actions"] == ["billing", "defect"]

    def test_failed_compile_logged(self, tmp_path, fake_runtime):
        with pytest.raises(Exception):
            strot.run(
                SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM(["nonsense"]),
                state=tmp_path, max_retries=1,
            )
        episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[-1])
        assert episode["status"] == "failed"
        assert "error" in episode


DECIDE_CONFIG = {
    "is_bug": {"type": "boolean", "instructions": "Does this mention crashes or errors?"},
    "sentiment": {"type": "score", "min": 1, "max": 3,
                  "instructions": "1 is angry complaint, 3 is happy praise."},
    "team": {"type": "choice", "options": ["billing", "engineering"]},
}

VALID_DECIDE = '''```python
QUESTIONS = {
    "is_bug": {"type": "boolean", "instructions": "Does this mention crashes or errors?",
               "criteria": {"false": "normal request no malfunction",
                            "true": "crash defect error malfunction"}},
    "sentiment": {"type": "score", "min": 1,
                  "instructions": "1 is angry complaint, 3 is happy praise.",
                  "rubric": ["angry complaint charge dispute", "neutral request", "happy praise"]},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "billing charge payment",
                          "engineering": "crash defect bug"}},
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))
```'''


class TestDecide:
    def test_decide_outputs_typed_decision_json(self, tmp_path, fake_runtime):
        llm = ScriptedLLM([VALID_DECIDE])
        result = strot.decide(SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path)
        assert llm.calls == 1
        assert result.mode == "decide"
        assert len(result.outcomes) == 2
        crash = result.outcomes[1]["decision"]  # body: "app crash defect"
        assert set(crash) == {"booleans", "scores", "choices"}
        assert crash["booleans"]["is_bug"]["type"] == "boolean"
        assert crash["booleans"]["is_bug"]["probability"] > 0.9
        assert crash["choices"]["team"]["selected"] == "engineering"
        assert set(crash["choices"]["team"]["probabilities"]) == {"billing", "engineering"}
        assert 0.0 <= crash["choices"]["team"]["confidence"] <= 1.0
        billing = result.outcomes[0]["decision"]  # body: "double billing charge"
        assert billing["choices"]["team"]["selected"] == "billing"
        assert 1.0 <= billing["scores"]["sentiment"]["value"] <= 3.0
        assert billing["scores"]["sentiment"]["value"] < 1.5  # angry complaint
        # Warm call: no LLM.
        warm = strot.decide(SAMPLE, DECIDE_CONFIG, llm=ExplodingLLM(), state=tmp_path)
        assert warm.cache_hit is True

    def test_decide_accepts_full_config_json_string(self, tmp_path, fake_runtime):
        config = json.dumps({"model": "decision-model", "state": "ignored",
                             "questions": DECIDE_CONFIG})
        result = strot.decide(SAMPLE, config, llm=ScriptedLLM([VALID_DECIDE]), state=tmp_path)
        assert len(result.outcomes) == 2

    def test_decide_validates_config_before_any_llm_call(self, tmp_path):
        bad = {"q": {"type": "score", "min": 1, "max": 99, "instructions": "x"}}
        with pytest.raises(RouterConfigurationError, match="2-20"):
            strot.decide(SAMPLE, bad, llm=ExplodingLLM(), state=tmp_path)

    def test_decide_golden_gate_rejects_and_repairs(self, tmp_path, fake_runtime):
        swapped = VALID_DECIDE.replace(
            '"billing": "billing charge payment"', '"billing": "crash defect bug"'
        ).replace(
            '"engineering": "crash defect bug"', '"engineering": "billing charge payment"'
        )
        llm = ScriptedLLM([swapped, VALID_DECIDE])
        golden = [{"record": SAMPLE[1], "expect": {"team": "engineering", "is_bug": True}}]
        result = strot.decide(SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path, golden=golden)
        assert result.attempts == 2
        assert result.outcomes[1]["decision"]["choices"]["team"]["selected"] == "engineering"

    def test_decide_golden_validation_before_llm(self, tmp_path):
        golden = [{"record": SAMPLE[0], "expect": {"nope": "billing"}}]
        with pytest.raises(RouterConfigurationError, match="unknown question"):
            strot.decide(SAMPLE, DECIDE_CONFIG, llm=ExplodingLLM(), state=tmp_path, golden=golden)

    def test_decide_expansion_errors_feed_repair_loop(self, tmp_path, fake_runtime):
        missing_rubric = VALID_DECIDE.replace(
            '"rubric": ["angry complaint charge dispute", "neutral request", "happy praise"]',
            '"rubric": ["angry", "happy"]',
        )
        llm = ScriptedLLM([missing_rubric, VALID_DECIDE])
        result = strot.decide(SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path)
        assert result.attempts == 2
