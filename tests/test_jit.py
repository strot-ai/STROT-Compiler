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


class TestQuestions:
    def test_proposes_a_validated_typed_question_set(self, tmp_path):
        # Intake: one-line intent -> a typed question config, ready for decide().
        proposed = '''```python
QUESTIONS = {
    "is_bug": {"type": "predicate", "instructions": "Is this a bug report?"},
    "team": {"type": "choice", "instructions": "Which team?",
             "choices": [{"value": "billing"}, {"value": "engineering"}]},
}
```'''
        questions = strot.questions(
            SAMPLE, "triage support tickets", llm=ScriptedLLM([proposed])
        )
        assert set(questions) == {"is_bug", "team"}
        assert questions["is_bug"]["type"] == "predicate"
        assert questions["team"]["type"] == "choice"
        assert [c["value"] for c in questions["team"]["choices"]] == ["billing", "engineering"]

    def test_invalid_proposal_feeds_the_repair_loop(self, tmp_path):
        bad = '```python\nQUESTIONS = {"q": {"type": "bogus", "instructions": "x"}}\n```'
        good = ('```python\nQUESTIONS = {"ok": {"type": "predicate", '
                '"instructions": "is it urgent?"}}\n```')
        questions = strot.questions(SAMPLE, "flag urgent", llm=ScriptedLLM([bad, good]))
        assert set(questions) == {"ok"}


class TestDecide:
    def test_decide_compiles_one_head_per_question(self, tmp_path, fake_runtime, decide_encoder):
        # Head-first: each typed question becomes a distilled head on the encoder
        # (predicate -> binary, choice -> N-way, score -> ordinal). The model runs
        # on a real encoder, so this verifies the baked heads, not runtime output.
        llm = ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(
            DECIDE_SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path,
            sample_size=28, execute=False,
        )
        assert result.mode == "decide"
        source = result.artifact_path.read_text()
        assert "class CompiledDecider" in source and "import julia" not in source
        heads = _baked_heads(source)
        assert heads["is_bug"]["type"] == "predicate" and heads["is_bug"]["classes"] == [False, True]
        assert heads["sentiment"]["type"] == "score"
        assert heads["sentiment"]["classes"] == ["angry", "neutral", "happy"]
        assert heads["team"]["type"] == "choice"
        assert heads["team"]["classes"] == ["billing", "engineering"]
        # Each head carries a compile-time-fitted temperature for honest confidence.
        assert all("temperature" in h for h in heads.values())
        # Warm call: no LLM.
        warm = strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG, llm=ExplodingLLM(),
                            state=tmp_path, sample_size=28, execute=False)
        assert warm.cache_hit is True

    def test_decide_lite_mode_builds_prototype_heads_without_a_teacher(
        self, tmp_path, fake_runtime, decide_encoder
    ):
        # lite = zero-shot: synthesize the projection, embed each class's
        # description as a prototype, bake. No teacher-labeling pass, so the LLM
        # script holds only the projection (one call, not two).
        result = strot.decide(
            DECIDE_SAMPLE, DECIDE_CONFIG, llm=ScriptedLLM([VALID_DECIDE]),
            state=tmp_path, sample_size=28, execute=False, mode="lite",
        )
        assert result.mode == "decide"
        heads = _baked_heads(result.artifact_path.read_text())
        assert set(heads) == {"is_bug", "sentiment", "team"}
        # choice/score: one L2-normalized prototype row per class, no training.
        assert heads["team"]["classes"] == ["billing", "engineering"]
        assert len(heads["team"]["weights"]) == 2
        assert len(heads["sentiment"]["weights"]) == 3
        for row in heads["team"]["weights"]:
            assert abs(sum(x * x for x in row) ** 0.5 - 1.0) < 1e-6  # unit prototypes
        # predicate: [-proto, +proto] -> a sigmoid around the statement.
        neg, pos = heads["is_bug"]["weights"]
        assert neg == [-x for x in pos]
        # lite is uncalibrated (no labels to fit a temperature).
        assert all(h["temperature"] == 1.0 for h in heads.values())

    def test_decide_lite_and_head_cache_separately(self, tmp_path, fake_runtime, decide_encoder):
        lite = strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG, llm=ScriptedLLM([VALID_DECIDE]),
                            state=tmp_path, sample_size=28, execute=False, mode="lite")
        head = strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG,
                            llm=ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)]),
                            state=tmp_path, sample_size=28, execute=False, mode="head")
        assert lite.cache_key != head.cache_key
        assert lite.artifact_path != head.artifact_path

    def test_decide_warm_serves_lite_then_upgrades_to_head(
        self, tmp_path, fake_runtime, decide_encoder
    ):
        # warm=True: lite serves instantly; the head trains in a background
        # thread and hot-swaps in. Scripts (consumed in order): lite projection,
        # then head projection + teacher. The lite call finishes before the
        # thread starts, so consumption is deterministic.
        llm = ScriptedLLM([VALID_DECIDE, VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(
            DECIDE_SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path,
            sample_size=28, execute=False, warm=True,
        )
        assert result.tier == "lite"
        lite_path = result.model.artifact_path

        result.wait(timeout=30)
        assert result.tier == "head"
        assert result.model.artifact_path != lite_path  # swapped to the trained head
        heads = _baked_heads(result.model.artifact_path.read_text())
        assert set(heads) == {"is_bug", "sentiment", "team"}

    def test_decide_head_bakes_an_ood_drift_gate(self, tmp_path, fake_runtime, decide_encoder):
        # head/deep fit a drift gate: per-class centroids (finest-grained question)
        # + a distance threshold, baked as OOD; the runtime refuses records unlike
        # the training set. Centroids use the richest question (sentiment, 3 levels).
        llm = ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path,
                              sample_size=28, execute=False, mode="head")
        source = result.artifact_path.read_text()
        ood = _baked_literal(source, "OOD")
        assert set(ood) == {"centroids", "threshold", "percentile", "basis"}
        assert ood["basis"] == "sentiment"  # the 3-level question, finest partition
        assert ood["percentile"] == 0.99 and ood["threshold"] >= 0.0
        assert len(ood["centroids"]) == 3  # one centroid per sentiment level
        assert "def _is_ood" in source  # the runtime gate is present

    def test_decide_lite_has_no_drift_gate(self, tmp_path, fake_runtime, decide_encoder):
        # lite has no record embeddings, so no centroids -> an empty OOD no-op.
        result = strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG, llm=ScriptedLLM([VALID_DECIDE]),
                              state=tmp_path, sample_size=28, execute=False, mode="lite")
        assert _baked_literal(result.artifact_path.read_text(), "OOD") == {}

    def test_decide_recompile_is_history_independent(self, tmp_path, fake_runtime, decide_encoder):
        # Invariant "recompile, don't increment": a model's weights are a pure
        # function of its own (data, questions, mode) - a fresh fit from the
        # frozen base, never a warm-start from prior weights. So an intervening
        # compile of a DIFFERENT model cannot perturb a later recompile. If a
        # warm-start path is ever added, this reproduction breaks.
        def _heads(data, state):
            llm = ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
            r = strot.decide(data, DECIDE_CONFIG, llm=llm, state=state,
                             sample_size=28, execute=False, mode="head")
            return _baked_heads(r.artifact_path.read_text())

        first = _heads(DECIDE_SAMPLE, tmp_path / "a1")
        _heads(list(reversed(DECIDE_SAMPLE)), tmp_path / "b")  # different model trains in between
        again = _heads(DECIDE_SAMPLE, tmp_path / "a2")          # fresh state -> a real recompile
        assert first == again  # byte-identical: no carried-over training state

    def test_decide_rejects_unknown_mode(self, tmp_path, fake_runtime, decide_encoder):
        with pytest.raises(strot.RouterConfigurationError):
            strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG, llm=ExplodingLLM(),
                        state=tmp_path, execute=False, mode="turbo")

    def test_decide_deep_bakes_a_shared_adapter_runtime(
        self, tmp_path, fake_runtime, decide_encoder, monkeypatch
    ):
        # deep = one shared LoRA adapter + refit heads. Fine-tuning needs torch,
        # so stub the trainer with a fake adapter + separable adapted vectors and
        # verify the wiring: the baked model carries ADAPTER/ADAPTER_CONFIG and the
        # LoRA-adapted decider runtime, with one head per question.
        import strot.finetune as ft

        def _fake_finetune_decide(contexts, q_labels, q_classes, train):
            return ft.DecideAdaptResult(
                adapter={"model.layers.21.attn.Wo.A": [[0.0], [0.0], [0.0]]},
                vectors=_decide_embed(contexts),
                config={"base_model": "answerdotai/ModernBERT-base", "cut": 18,
                        "n_layers": 22, "lora_rank": 16, "lora_alpha": 32,
                        "max_length": 128, "device": "cpu"},
            )

        monkeypatch.setattr(ft, "finetune_decide", _fake_finetune_decide)
        monkeypatch.setattr(ft, "MIN_FINETUNE_RECORDS", 1)

        llm = ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(
            DECIDE_SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path,
            sample_size=28, execute=False, mode="deep",
        )
        source = result.artifact_path.read_text()
        assert "ADAPTER_CONFIG" in source and "_LoRALinear" in source
        assert "class CompiledDecider" in source
        assert set(_baked_heads(source)) == {"is_bug", "sentiment", "team"}

    def test_decide_deep_needs_enough_records(self, tmp_path, fake_runtime, decide_encoder):
        # The real floor (MIN_FINETUNE_RECORDS) guards deep before any training.
        with pytest.raises(strot.RouterConfigurationError):
            strot.decide(DECIDE_SAMPLE[:10], DECIDE_CONFIG, llm=ExplodingLLM(),
                        state=tmp_path, sample_size=10, execute=False, mode="deep")

    def test_decide_designs_schema_from_a_bare_intent(self, tmp_path, fake_runtime, decide_encoder):
        # AI-first quick path: a plain-language intent (not JSON) makes decide()
        # design the schema first, then compile. The chosen config surfaces on
        # result.questions so the author can inspect/pin it.
        designed = '''```python
QUESTIONS = {
    "is_bug": {"type": "predicate", "instructions": "Does this mention crashes or errors?"},
    "sentiment": {"type": "score", "instructions": "angry is a complaint, happy is praise.",
                  "levels": [{"label": "angry", "description": "angry complaint"},
                             {"label": "neutral", "description": "neutral request"},
                             {"label": "happy", "description": "happy praise"}]},
    "team": {"type": "choice", "choices": [
        {"value": "billing", "description": "charges, invoices, payments"},
        {"value": "engineering", "description": "crashes, bugs, defects"}]},
}
```'''
        llm = ScriptedLLM([designed, VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(
            DECIDE_SAMPLE, "triage support tickets", llm=llm, state=tmp_path,
            sample_size=28, execute=False,
        )
        assert result.mode == "decide"
        assert set(result.questions) == {"is_bug", "sentiment", "team"}
        assert set(_baked_heads(result.artifact_path.read_text())) == {"is_bug", "sentiment", "team"}

    def test_decide_result_is_a_callable_saveable_model(self, tmp_path, fake_runtime, decide_encoder):
        # The decide() result is the model: callable + saveable to one .strot file,
        # reloadable with load_model. (Inference itself needs a real encoder, so
        # here we verify the save/load plumbing, not the baked head's output.)
        llm = ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(
            DECIDE_SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path,
            sample_size=28, execute=False,
        )
        assert callable(result) and callable(result.model)

        dest = result.save(tmp_path / "ticket-triage")
        assert dest == tmp_path / "ticket-triage.strot"
        assert dest.read_bytes() == result.artifact_path.read_bytes()  # self-contained copy

        reloaded = strot.load_model(tmp_path / "ticket-triage")  # suffix optional
        assert isinstance(reloaded, strot.DecisionModel)
        assert reloaded.artifact_path == dest

    def test_decision_model_call_delegates_and_applies_refusals(self, tmp_path):
        # __call__ runs the pipeline and applies the carried refusal threshold.
        class FakePipeline:
            def process(self, record):
                return {"choices": {"team": {"choice": "billing", "confidence": 0.4}}}

        raw = strot.DecisionModel(tmp_path / "m.strot", key="k", pipeline=FakePipeline())
        assert raw({"x": 1})["choices"]["team"]["choice"] == "billing"

        guarded = strot.DecisionModel(
            tmp_path / "m.strot", key="k", confidence_threshold=0.7, pipeline=FakePipeline()
        )
        assert guarded({"x": 1})["choices"]["team"] == {"refusal": True}

    def test_load_model_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            strot.load_model(tmp_path / "nope")

    def test_decide_warns_on_thin_class_coverage(self, tmp_path, fake_runtime, decide_encoder):
        # A head can't learn an outcome it never sees: if the teacher labels every
        # record "billing", the "engineering" class is empty and compile warns.
        config = {"team": {"type": "choice",
                           "choices": [{"value": "billing"}, {"value": "engineering"}]}}
        expansion = '''```python
QUESTIONS = {
    "team": {"type": "choice", "instructions": "which team",
             "criteria": {"billing": "billing", "engineering": "engineering"}},
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))
```'''
        teacher = "```python\nANSWERS = " + repr([{"team": "billing"}] * 28) + "\n```"
        with pytest.warns(UserWarning, match="thin coverage"):
            strot.decide(DECIDE_SAMPLE, config, llm=ScriptedLLM([expansion, teacher]),
                         state=tmp_path, sample_size=28, execute=False)

    def test_decide_accepts_deprecated_boolean_alias(self, tmp_path, fake_runtime, decide_encoder):
        # "boolean" is a deprecated alias for "predicate": it must still
        # compile, run, and produce the same probability output under the
        # "predicates" group.
        config = {
            "is_bug": {"type": "boolean",
                       "instructions": "Does this mention crashes or errors?"},
        }
        expansion = '''```python
QUESTIONS = {
    "is_bug": {"type": "boolean", "instructions": "Does this mention crashes or errors?",
               "criteria": {"false": "normal request no malfunction",
                            "true": "crash defect error malfunction"}},
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))
```'''
        result = strot.decide(DECIDE_SAMPLE, config,
                              llm=ScriptedLLM([expansion, _decide_teacher(config)]),
                              state=tmp_path, sample_size=28, execute=False)
        heads = _baked_heads(result.artifact_path.read_text())
        assert heads["is_bug"]["type"] == "predicate"
        assert heads["is_bug"]["classes"] == [False, True]

    def test_decide_accepts_deprecated_options_alias(self, tmp_path, fake_runtime, decide_encoder):
        # The bare "options" list (values only, no descriptions) is still
        # accepted as a deprecated alias for the richer "choices" form and
        # produces the same value-keyed choice output.
        config = {
            "team": {"type": "choice", "options": ["billing", "engineering"]},
        }
        expansion = '''```python
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "billing charge payment",
                          "engineering": "crash defect bug"}},
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))
```'''
        result = strot.decide(DECIDE_SAMPLE, config,
                              llm=ScriptedLLM([expansion, _decide_teacher(config)]),
                              state=tmp_path, sample_size=28, execute=False)
        heads = _baked_heads(result.artifact_path.read_text())
        assert heads["team"]["type"] == "choice"
        assert heads["team"]["classes"] == ["billing", "engineering"]

    def test_decide_accepts_full_config_json_string(self, tmp_path, fake_runtime, decide_encoder):
        config = json.dumps({"model": "decision-model", "state": "ignored",
                             "questions": DECIDE_CONFIG})
        result = strot.decide(DECIDE_SAMPLE, config,
                              llm=ScriptedLLM([VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)]),
                              state=tmp_path, sample_size=28, execute=False)
        assert set(_baked_heads(result.artifact_path.read_text())) == {"is_bug", "sentiment", "team"}

    def test_decide_validates_config_before_any_llm_call(self, tmp_path):
        bad = {"q": {"type": "score", "min": 1, "max": 99, "instructions": "x"}}
        with pytest.raises(RouterConfigurationError, match="2-10"):
            strot.decide(SAMPLE, bad, llm=ExplodingLLM(), state=tmp_path)

    def test_decide_accepts_deprecated_min_max_alias(self, tmp_path, fake_runtime, decide_encoder):
        # The bare "min"/"max" integer range is still accepted as a deprecated
        # alias for "levels": one level per integer, the integers as labels. It
        # must compile, run, and produce a labeled score under "scores".
        config = {
            "sentiment": {"type": "score", "min": 1, "max": 3,
                          "instructions": "1 is angry complaint, 3 is happy praise."},
        }
        expansion = '''```python
QUESTIONS = {
    "sentiment": {"type": "score",
                  "instructions": "1 is angry complaint, 3 is happy praise.",
                  "rubric": [{"label": "1", "description": "angry complaint charge dispute"},
                             {"label": "2", "description": "neutral request"},
                             {"label": "3", "description": "happy praise"}]},
}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))
```'''
        result = strot.decide(DECIDE_SAMPLE, config,
                              llm=ScriptedLLM([expansion, _decide_teacher(config)]),
                              state=tmp_path, sample_size=28, execute=False)
        heads = _baked_heads(result.artifact_path.read_text())
        assert heads["sentiment"]["type"] == "score"
        assert heads["sentiment"]["classes"] == ["1", "2", "3"]  # min/max -> integer labels

    @pytest.mark.skip(reason="decide golden gate runs the distilled model, which "
                             "needs a real local encoder (no fake); covered by the "
                             "routing golden-gate tests and a real-encoder e2e")
    def test_decide_golden_gate_rejects_and_repairs(self):
        pass

    def test_decide_golden_validation_before_llm(self, tmp_path):
        golden = [{"record": SAMPLE[0], "expect": {"nope": "billing"}}]
        with pytest.raises(RouterConfigurationError, match="unknown question"):
            strot.decide(SAMPLE, DECIDE_CONFIG, llm=ExplodingLLM(), state=tmp_path, golden=golden)

    def test_decide_expansion_errors_feed_repair_loop(self, tmp_path, fake_runtime, decide_encoder):
        missing_rubric = VALID_DECIDE.replace(
            '"rubric": [{"label": "angry", "description": "angry complaint charge dispute"},\n'
            '                             {"label": "neutral", "description": "neutral request"},\n'
            '                             {"label": "happy", "description": "happy praise"}]',
            '"rubric": [{"label": "angry", "description": "angry"}]',
        )
        llm = ScriptedLLM([missing_rubric, VALID_DECIDE, _decide_teacher(DECIDE_CONFIG)])
        result = strot.decide(DECIDE_SAMPLE, DECIDE_CONFIG, llm=llm, state=tmp_path,
                              sample_size=28, execute=False)
        assert result.attempts == 2

    def _decision(self):
        # Mixed-confidence decision: team is unsure, is_bug and sentiment are sure.
        return {
            "predicates": {"is_bug": {"type": "predicate", "probability": 0.95}},
            "scores": {"sentiment": {"score": "angry", "confidence": 0.9,
                                     "probabilities": {"angry": 0.9, "neutral": 0.05, "happy": 0.05}}},
            "choices": {"team": {"choice": "billing", "confidence": 0.55,
                                 "probabilities": {"billing": 0.55, "engineering": 0.45}}},
        }

    def test_confidence_threshold_refuses_only_low_confidence_answers(self):
        # Opt-in per-question refusal: unsure answers become {"refusal": True}
        # while confident ones still answer. predicate confidence = max(p, 1-p).
        from strot.jit import _apply_refusals

        out = _apply_refusals(self._decision(), 0.8)
        assert out["choices"]["team"] == {"refusal": True}  # 0.55 < 0.8
        assert out["predicates"]["is_bug"]["probability"] == 0.95  # 0.95 >= 0.8, survives
        assert out["scores"]["sentiment"]["score"] == "angry"  # 0.9 >= 0.8, survives

    def test_no_threshold_never_refuses(self):
        # A threshold of 0 refuses nothing (confidence is always >= 0); the
        # decide() caller skips the pass entirely when the threshold is None.
        from strot.jit import _apply_refusals

        out = _apply_refusals(self._decision(), 0.0)
        assert "refusal" not in out["choices"]["team"]
        assert "refusal" not in out["predicates"]["is_bug"]
        assert "refusal" not in out["scores"]["sentiment"]
