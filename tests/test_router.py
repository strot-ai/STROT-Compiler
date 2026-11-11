import json

import pytest

from strot.exceptions import FallbackException, RouterConfigurationError
from strot.router import (
    HostedApiStrategy,
    LocalEncoderStrategy,
    RouteDecision,
    Router,
    chunk_options,
    class_centroids,
    create_strategy,
    is_ood,
    nearest_centroid_distance,
    ood_threshold,
)


class TestFactory:
    def test_default_resolves_to_local(self):
        strategy = create_strategy()
        assert isinstance(strategy, LocalEncoderStrategy)
        assert strategy.nlu_model == "SupersonicLabs/Julia-1"

    @pytest.mark.parametrize("name", ["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"])
    def test_provider_prefixes_resolve_to_hosted(self, name):
        assert isinstance(create_strategy(name), HostedApiStrategy)

    def test_hosted_requires_provider_slash_model(self):
        with pytest.raises(RouterConfigurationError):
            HostedApiStrategy("openai")


class TestOptionTreeSplitter:
    def test_small_lists_stay_flat(self):
        assert chunk_options(list("abc")) == [["a", "b", "c"]]

    def test_splits_beyond_twenty_into_balanced_groups(self):
        options = [f"opt_{i}" for i in range(45)]
        groups = chunk_options(options)
        assert all(len(g) <= 20 for g in groups)
        assert [o for g in groups for o in g] == options

    def test_hierarchical_selection_preserves_global_index(self):
        options = [f"opt_{i}" for i in range(25)]

        class FakeLocal(LocalEncoderStrategy):
            def _select_flat(self, context, question, opts):
                # Always pick the last option of whatever tier it is given.
                return RouteDecision(
                    index=len(opts) - 1,
                    option=opts[-1],
                    confidence=0.9,
                    distribution={},
                )

        decision = FakeLocal().select("ctx", "which?", options)
        assert decision.option == "opt_24"
        assert decision.index == 24
        assert decision.confidence == pytest.approx(0.81)  # product of two tiers


class TestJulia1RowFormat:
    def test_row_matches_julia_decision_schema(self):
        row = LocalEncoderStrategy.build_row("ctx", "which?", ["a", "b"])
        assert row == {"state": "ctx", "question": "which?", "options": ["a", "b"]}

    def test_single_option_short_circuits_without_engine(self):
        decision = LocalEncoderStrategy().select("ctx", "which?", ["only"])
        assert decision.index == 0
        assert decision.confidence == 1.0


class _FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class _Msg:
            content = json.dumps(self.payload)

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeClient:
    def __init__(self, payload):
        self.chat = type("chat", (), {"completions": _FakeCompletions(payload)})()


class TestHostedStrategy:
    def make(self, payload):
        client = _FakeClient(payload)
        strategy = HostedApiStrategy("openai/gpt-4o-mini", client_factory=lambda: client)
        return strategy, client

    def test_selects_index_from_structured_output(self):
        strategy, client = self.make({"selected_index": 1, "confidence": 0.92})
        decision = strategy.select("ctx", "which?", ["a", "b", "c"])
        assert decision.option == "b"
        assert decision.confidence == pytest.approx(0.92)
        sent = client.chat.completions.last_kwargs
        assert sent["model"] == "gpt-4o-mini"
        assert sent["response_format"]["json_schema"]["strict"] is True

    def test_out_of_range_index_triggers_fallback(self):
        strategy, _ = self.make({"selected_index": 9, "confidence": 0.99})
        with pytest.raises(FallbackException):
            strategy.select("ctx", "which?", ["a", "b"])

    def test_missing_env_key_is_a_configuration_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        strategy = HostedApiStrategy("openai/gpt-4o-mini")
        with pytest.raises(RouterConfigurationError):
            strategy.select("ctx", "which?", ["a"])


class TestConfidenceEnforcement:
    class StubStrategy(LocalEncoderStrategy):
        def __init__(self, confidence):
            super().__init__()
            self._confidence = confidence

        def select(self, context, question, options):
            return RouteDecision(0, options[0], self._confidence, {})

    def test_low_confidence_raises_fallback_instead_of_executing(self):
        router = Router(self.StubStrategy(confidence=0.39))
        with pytest.raises(FallbackException) as excinfo:
            router.route("ctx", "which?", ["a", "b"])
        assert excinfo.value.threshold == pytest.approx(0.40)
        assert excinfo.value.confidence == pytest.approx(0.39)

    def test_confident_decision_passes_through(self):
        decision = Router(self.StubStrategy(confidence=0.41)).route("ctx", "q", ["a"])
        assert decision.option == "a"


class TestFineTunedAdapterStrategy:
    """Serving a compile-time fine-tuned battery. The real forward needs torch;
    the ensemble averaging + decision shaping are pure Python and faked here."""

    CONFIG = {"base_model": "x", "cut": 0, "n_layers": 1, "lora_rank": 2,
              "lora_alpha": 4, "max_length": 8, "n_classes": 2}
    ADAPTER = {"__head_weight__": [[0.0], [0.0]], "__head_bias__": [0.0, 0.0]}

    def _strategy(self, forwards):
        from strot.router import FineTunedAdapterStrategy

        strat = FineTunedAdapterStrategy([self.ADAPTER], self.CONFIG)
        strat._forwards = forwards  # inject fakes, skip the torch load
        return strat

    def test_empty_adapters_rejected(self):
        from strot.router import FineTunedAdapterStrategy

        with pytest.raises(RouterConfigurationError):
            FineTunedAdapterStrategy([], self.CONFIG)

    def test_single_adapter_argmax_and_confidence(self):
        strat = self._strategy([lambda texts: [[0.2, 0.8]]])
        decision = strat.select("ctx", "q", ["a", "b"])
        assert decision.index == 1 and decision.option == "b"
        assert decision.confidence == pytest.approx(0.8)
        assert decision.distribution == {"a": pytest.approx(0.2), "b": pytest.approx(0.8)}

    def test_ensemble_averages_softmax_rows(self):
        # Two adapters: [0.2,0.8] and [0.6,0.4] -> avg [0.4,0.6] -> argmax 1.
        strat = self._strategy([
            lambda texts: [[0.2, 0.8]],
            lambda texts: [[0.6, 0.4]],
        ])
        probs = strat.distributions(["ctx"])[0]
        assert probs == pytest.approx([0.4, 0.6])
        assert strat.select("ctx", "q", ["a", "b"]).index == 1


def test_finetuned_runtime_template_parses():
    import ast

    from strot.templates import FINETUNED_RUNTIME_TEMPLATE

    ast.parse(FINETUNED_RUNTIME_TEMPLATE.format(nlu_model="test"))


class TestOODDetection:
    """Drift / out-of-distribution decision logic (piece D), injected vectors."""

    def test_class_centroids_are_per_class_means(self):
        vectors = [[0.0, 0.0], [2.0, 0.0], [0.0, 10.0], [0.0, 12.0]]
        labels = [0, 0, 1, 1]
        centroids = class_centroids(vectors, labels, n_classes=2)
        assert centroids == [[1.0, 0.0], [0.0, 11.0]]

    def test_empty_class_gets_zero_centroid(self):
        centroids = class_centroids([[1.0, 1.0]], [0], n_classes=3)
        assert centroids[1] == [0.0, 0.0] and centroids[2] == [0.0, 0.0]

    def test_nearest_centroid_distance_picks_closest(self):
        centroids = [[0.0, 0.0], [10.0, 0.0]]
        assert nearest_centroid_distance([1.0, 0.0], centroids) == pytest.approx(1.0)
        assert nearest_centroid_distance([9.0, 0.0], centroids) == pytest.approx(1.0)

    def test_threshold_is_high_percentile_of_in_dist_distances(self):
        # all points sit on their centroid except one outlier at distance 5
        centroids = [[0.0, 0.0]]
        vectors = [[0.0, 0.0]] * 9 + [[5.0, 0.0]]
        # 95th percentile over 10 points -> the outlier distance (5.0)
        assert ood_threshold(vectors, centroids, percentile=0.95) == pytest.approx(5.0)

    def test_is_ood_flags_only_far_points(self):
        centroids = [[0.0, 0.0], [10.0, 0.0]]
        threshold = 2.0
        assert is_ood([0.5, 0.0], centroids, threshold) is False  # near class 0
        assert is_ood([10.5, 0.0], centroids, threshold) is False  # near class 1
        assert is_ood([5.0, 0.0], centroids, threshold) is True  # far from both

    def test_guards(self):
        with pytest.raises(ValueError):
            class_centroids([], [], 2)
        with pytest.raises(ValueError):
            nearest_centroid_distance([0.0], [])
        with pytest.raises(ValueError):
            ood_threshold([], [[0.0]])
        with pytest.raises(ValueError):
            ood_threshold([[0.0]], [[0.0]], percentile=1.5)


class TestOODRuntimeGate:
    """The distilled runtime template's OOD gate forces abstain (status 'ood')."""

    def _build_pipeline(self, monkeypatch, ood, embed_vector):
        import math
        from types import SimpleNamespace

        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        safetensors = pytest.importorskip("safetensors.torch")

        from strot import templates

        # Minimal fake encoder: pooled embedding is fixed to embed_vector.
        dim = len(embed_vector)
        seq = torch.tensor([[list(embed_vector), list(embed_vector)]], dtype=torch.float)

        class Encoder:
            def load_state_dict(self, *a, **k):
                pass

            def eval(self):
                pass

            def __call__(self, **kwargs):
                mask = kwargs["attention_mask"]
                return SimpleNamespace(
                    last_hidden_state=seq.expand(mask.shape[0], -1, -1)
                )

        tokenizer = lambda texts, **k: {"attention_mask": torch.ones(len(texts), 2)}
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _: None)
        monkeypatch.setattr(transformers.AutoModel, "from_config", lambda _: Encoder())
        monkeypatch.setattr(safetensors, "load_file", lambda _: {"encoder.x": torch.zeros(1)})
        monkeypatch.setenv("STROT_NLU_CHECKPOINT", "/tmp/ckpt")

        header = templates.ASSET_HEADER_TEMPLATE.format(
            nlu_model="test", runtime_kind="distilled", goal="g", threshold=0.4,
            temperature=repr(1.0), conformal_q=repr(None), calibration_literal=repr({}),
            class_bias=repr(None), dispatch_cutoff=repr(None), policy_literal=repr({}),
            ood_literal=repr(ood), schema_literal=repr({}),
        )
        runtime = templates.DISTILLED_RUNTIME_TEMPLATE.format(nlu_model="test")
        footer = templates.ASSET_FOOTER_TEMPLATE.format(
            generated_paths=(
                'ROUTING_LABELS = {"a": "alpha", "b": "beta"}\n\n\n'
                "def route_context(record):\n    return str(record)\n\n\n"
                "def execute_a(record):\n    return {'ok': 'a'}\n\n\n"
                "def execute_b(record):\n    return {'ok': 'b'}\n"
            ),
            registry_literal=repr({"a": "A", "b": "B"}),
            handlers_literal="{'a': execute_a, 'b': execute_b}",
            default_question="q?",
        )
        source = "\n\n".join([header, runtime, footer]) + "\n"
        ns = {}
        exec(compile(source, "<ood-asset>", "exec"), ns)
        # Head weights so class 0 wins when not ood.
        ns["HEAD_WEIGHTS"] = [[1.0] * dim, [0.0] * dim]
        ns["HEAD_BIAS"] = [0.0, 0.0]
        return ns

    def test_in_distribution_record_dispatches(self, monkeypatch):
        # embed vector sits on centroid -> distance 0 <= threshold -> dispatch
        ns = self._build_pipeline(
            monkeypatch,
            ood={"centroids": [[1.0, 0.0], [0.0, 1.0]], "threshold": 2.0},
            embed_vector=[1.0, 0.0],
        )
        out = ns["CompiledPipeline"]().process({"x": 1})
        assert "action" in out and out["action"] in ("a", "b")

    def test_out_of_distribution_record_abstains(self, monkeypatch):
        FallbackException = None
        ns = self._build_pipeline(
            monkeypatch,
            ood={"centroids": [[0.0, 0.0], [0.0, 0.0]], "threshold": 1.0},
            embed_vector=[9.0, 9.0],  # far from both centroids
        )
        with pytest.raises(ns["FallbackException"]) as excinfo:
            ns["CompiledPipeline"]().process({"x": 1})
        assert excinfo.value.status == "ood"
