"""Battery distribution round-trip: compiled .py decision == bundle decision.

The correctness gate for Stage 1 of the battery distribution. A compiled
battery's decision path (route/classify -> calibrated label + confidence +
abstention) must survive the trip through a data-only bundle unchanged: export
the bundle, reload it into a :class:`strot.battery.Battery`, and assert its
``decide`` reproduces the baked ``CompiledPipeline`` on the same records.

Two layers, clearly separated so it is obvious which assertions run vs skip:

* **Data-only (always runs, torch-free).** The decision *math* -
  :func:`strot.battery.apply_decision` - is asserted to match the compiled
  asset template's ``CompiledPipeline.process`` correction + abstention block
  verbatim, by running that block over a *canned* router distribution under
  faked torch modules (no real model). And ``battery.json`` is asserted to
  round-trip the labels + calibration faithfully, captured straight from the
  compiled source's literals. These are the gate that holds without the extra.

* **Torch-backed (skips without torch/transformers/safetensors).** The full
  compile -> export -> from_bundle -> decide round-trip over the real encoder
  runtime, matching the baked asset. Guarded with ``importorskip`` like the
  existing distill/finetune integration tests.
"""

import re
import sys
import types

import pytest

from strot import battery as bat
from strot.battery import (
    BUNDLE_FORMAT_VERSION,
    Battery,
    Calibration,
    apply_decision,
    battery_info,
    export_bundle,
    list_batteries,
    load,
)
from strot.compiler import StrotCompiler
from strot.schema import introspect_schema

SAMPLE = [
    {"ticket_id": 1, "body": "double charged", "tier": "pro"},
    {"ticket_id": 2, "body": "app crashes", "tier": "free"},
]
REGISTRY = {
    "billing": "Billing disputes and refunds",
    "defect": "Crashes and bugs",
}
VALID_PATHS = '''```python
ROUTING_LABELS = {"billing": "billing charge", "defect": "app crash"}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))


def execute_billing(record: dict) -> dict:
    return {"queue": "billing", "ticket_id": record.get("ticket_id")}


def execute_defect(record: dict) -> dict:
    return {"queue": "engineering", "ticket_id": record.get("ticket_id")}
```'''


def _code_body() -> str:
    return re.search(r"```python\s*\n(.*?)```", VALID_PATHS, re.DOTALL).group(1)


def _distilled_source(**assemble_kwargs) -> str:
    """Assemble a distilled asset with a known head baked in (no LLM, no torch).

    Only assembles source text; running it (below) is done under faked torch.
    """
    compiler = StrotCompiler(llm_client=lambda p: VALID_PATHS)
    schema = introspect_schema(SAMPLE)
    head_code = (
        _code_body()
        + "\n\n# Compile-time-distilled head\n"
        + "HEAD_WEIGHTS = [[0.1, 0.2], [0.3, 0.4]]\n"
        + "HEAD_BIAS = [0.0, 0.0]"
    )
    return compiler._assemble(
        schema, "triage tickets", dict(REGISTRY), head_code, None,
        runtime="distilled", **assemble_kwargs,
    )


class _CannedRouter:
    """A fake ``_RuntimeRouter`` returning a fixed distribution over the options.

    Lets the real ``CompiledPipeline.process`` decision block run with no torch
    and no model, so its output can be compared to :func:`apply_decision`.
    """

    def __init__(self, route_decision_cls, distribution):
        self._rd = route_decision_cls
        self._dist = distribution

    def select(self, context, question, options):
        probs = [self._dist[opt] for opt in options]
        index = max(range(len(options)), key=probs.__getitem__)
        return self._rd(
            index=index, option=options[index],
            confidence=probs[index], distribution=dict(zip(options, probs)),
        )


def _run_compiled_decision(source, distribution):
    """Run ``CompiledPipeline.process``'s decision block over a canned dist.

    Fakes the torch-family imports the distilled asset needs at module scope,
    execs the asset, swaps in a :class:`_CannedRouter`, and returns a normalized
    ``{label, confidence, abstained}`` so it can be compared to a Battery
    decision. No real torch is loaded.
    """
    fakes = {}
    for name in ("torch",):
        fakes[name] = types.ModuleType(name)
    st = types.ModuleType("safetensors")
    stt = types.ModuleType("safetensors.torch")
    stt.load_file = lambda p: {}
    fakes["safetensors"] = st
    fakes["safetensors.torch"] = stt
    tf = types.ModuleType("transformers")
    stub = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: None, from_config=lambda *a, **k: None
    )
    tf.AutoConfig = tf.AutoModel = tf.AutoTokenizer = stub
    fakes["transformers"] = tf

    saved = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        module = types.ModuleType("asset_under_test")
        exec(compile(source, "asset_under_test", "exec"), module.__dict__)
        options = module._ROUTING_OPTIONS
        pipeline = module.CompiledPipeline.__new__(module.CompiledPipeline)
        pipeline._router = _CannedRouter(module.RouteDecision, distribution)
        try:
            outcome = pipeline.process({"ticket_id": 1, "body": "x"})
            # The compiled asset reports the *action key*; map back to the label
            # the router scored so the comparison is label-space aligned.
            action_index = module._ACTION_KEYS.index(outcome["action"])
            return {
                "label": options[action_index],
                "confidence": outcome["confidence"],
                "abstained": False,
            }
        except module.FallbackException as exc:
            return {"confidence": exc.confidence, "abstained": True}
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# --------------------------------------------------------------------------- #
# Data-only: decision-math parity (ALWAYS RUNS, no torch)                      #
# --------------------------------------------------------------------------- #


CANNED = {"billing charge": 0.62, "app crash": 0.38}
LABELS = ["billing charge", "app crash"]


def _calibration_from_source(source) -> Calibration:
    lits = bat._read_literals(source)
    return Calibration(
        temperature=float(lits.get("TEMPERATURE", 1.0) or 1.0),
        conformal_q=lits.get("CONFORMAL_Q"),
        class_bias=lits.get("CLASS_BIAS"),
        dispatch_cutoff=lits.get("DISPATCH_CUTOFF"),
        confidence_threshold=float(lits.get("CONFIDENCE_THRESHOLD", 0.40)),
        calibration=lits.get("CALIBRATION") or {},
        policy=lits.get("POLICY") or {},
    )


def test_decision_math_matches_uncalibrated():
    # No target set -> raw argmax + raw confidence pass through, never abstain.
    source = _distilled_source()
    baked = _run_compiled_decision(source, CANNED)
    mine = apply_decision(CANNED, LABELS, _calibration_from_source(source))
    assert baked["abstained"] is False and mine.abstained is False
    assert mine.label == baked["label"]
    assert mine.confidence == pytest.approx(baked["confidence"])


def test_decision_math_matches_dispatch_cutoff_dispatch():
    # Cutoff below the corrected confidence -> dispatches, labels/conf must match.
    source = _distilled_source(
        temperature=1.0, dispatch_cutoff=0.5,
        policy={"target_accuracy": 0.8},
    )
    baked = _run_compiled_decision(source, CANNED)
    mine = apply_decision(CANNED, LABELS, _calibration_from_source(source))
    assert baked["abstained"] is False and mine.abstained is False
    assert mine.label == baked["label"]
    assert mine.confidence == pytest.approx(baked["confidence"])


def test_decision_math_matches_dispatch_cutoff_abstain():
    # Cutoff above the corrected confidence -> both abstain at the same confidence.
    source = _distilled_source(
        temperature=1.0, dispatch_cutoff=0.9,
        policy={"target_accuracy": 0.99},
    )
    baked = _run_compiled_decision(source, CANNED)
    mine = apply_decision(CANNED, LABELS, _calibration_from_source(source))
    assert baked["abstained"] is True and mine.abstained is True
    assert mine.confidence == pytest.approx(baked["confidence"])


def test_decision_math_matches_conformal_dispatch():
    # A wide quantile keeps the set a singleton -> dispatches.
    source = _distilled_source(
        temperature=1.0, conformal_q=0.5,
        calibration={"alpha": 0.1, "temperature": 1.0, "conformal_quantile": 0.5},
    )
    baked = _run_compiled_decision(source, CANNED)
    mine = apply_decision(CANNED, LABELS, _calibration_from_source(source))
    assert baked["abstained"] is False and mine.abstained is False
    assert mine.label == baked["label"]
    assert mine.confidence == pytest.approx(baked["confidence"])


def test_decision_math_matches_conformal_abstain():
    # A tight quantile forces a multi-label set -> both abstain.
    near = {"billing charge": 0.52, "app crash": 0.48}
    # bar = 1 - q = 0.4; both labels clear it -> set size 2 -> abstain.
    source = _distilled_source(
        temperature=1.0, conformal_q=0.6,
        calibration={"alpha": 0.1, "temperature": 1.0, "conformal_quantile": 0.6},
    )
    baked = _run_compiled_decision(source, near)
    mine = apply_decision(near, LABELS, _calibration_from_source(source))
    assert baked["abstained"] is True and mine.abstained is True
    assert mine.confidence == pytest.approx(baked["confidence"])


def test_class_bias_correction_matches():
    # Per-class bias flips the winner; both the asset and apply_decision agree.
    source = _distilled_source(
        temperature=1.0, dispatch_cutoff=0.1,
        class_bias=[-2.0, 2.0],  # heavily favor class 1
        policy={"target_accuracy": 0.5},
    )
    baked = _run_compiled_decision(source, CANNED)
    mine = apply_decision(CANNED, LABELS, _calibration_from_source(source))
    assert mine.label == baked["label"]  # bias flipped it to "app crash"
    assert mine.label == "app crash"
    assert mine.confidence == pytest.approx(baked["confidence"])


# --------------------------------------------------------------------------- #
# Data-only: battery.json round-trips labels + calibration (ALWAYS RUNS)       #
# --------------------------------------------------------------------------- #


def test_metadata_roundtrips_labels_and_calibration():
    # _collect_bundle is torch-free: it reads the asset's literals and builds the
    # exact battery.json payload. Assert the labels + calibration survive.
    source = _distilled_source(
        temperature=1.3, dispatch_cutoff=0.42, class_bias=[0.1, -0.1],
        policy={"target_accuracy": 0.8},
        ood={"centroids": [[0.0, 0.0], [1.0, 1.0]], "threshold": 0.7, "percentile": 0.95},
    )
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    metadata, tensors, _ = bat._collect_bundle(asset, name="triage", question="which queue?")

    assert metadata["format_version"] == BUNDLE_FORMAT_VERSION
    assert metadata["runtime_kind"] == "distilled"
    assert metadata["name"] == "triage"
    assert metadata["question"] == "which queue?"
    # Labels round-trip exactly (order == registry order == option index).
    assert metadata["labels"] == {"billing": "billing charge", "defect": "app crash"}
    # Calibration round-trips faithfully.
    cal = metadata["calibration"]
    assert cal["temperature"] == 1.3
    assert cal["dispatch_cutoff"] == 0.42
    assert cal["class_bias"] == [0.1, -0.1]
    assert cal["policy"]["target_accuracy"] == 0.8
    # OOD guard survives.
    assert metadata["ood"]["threshold"] == 0.7
    # The head weights are collected as tensors, not in battery.json.
    assert tensors["head_weights"] == [[0.1, 0.2], [0.3, 0.4]]
    assert tensors["head_bias"] == [0.0, 0.0]
    assert "head_weights" not in metadata


def test_metadata_reconstructs_calibration_identically():
    # Calibration.from_dict(to_dict) is a faithful round-trip -> a reloaded
    # Battery applies the exact same decision as the captured one.
    source = _distilled_source(
        temperature=1.1, conformal_q=0.3,
        calibration={"alpha": 0.1, "temperature": 1.1, "conformal_quantile": 0.3},
    )
    captured = _calibration_from_source(source)
    reloaded = Calibration.from_dict(captured.to_dict())
    a = apply_decision(CANNED, LABELS, captured)
    b = apply_decision(CANNED, LABELS, reloaded)
    assert a.to_dict() == b.to_dict()


def test_hosted_runtime_rejected_as_data_only():
    compiler = StrotCompiler(
        nlu_model="openai/gpt-4o-mini", llm_client=lambda p: VALID_PATHS
    )
    asset = compiler.compile(SAMPLE, "triage", REGISTRY)
    with pytest.raises(Exception):
        bat._collect_bundle(asset)


# --------------------------------------------------------------------------- #
# Torch-backed: full export -> from_bundle -> decide round-trip (SKIPS)        #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _torch_distilled_env(monkeypatch, tmp_path):
    """Fake the frozen encoder so the distilled runtime is deterministic.

    Same technique as test_distill.test_encoder_and_distilled_vectors: a tiny
    encoder whose mean-pooled embedding is a fixed vector, wired through the real
    transformers/safetensors APIs. Requires the extra (hence the importorskip in
    each test that uses it).
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    pytest.importorskip("safetensors.numpy")

    class Encoder:
        def load_state_dict(self, *a, **k):
            pass

        def eval(self):
            pass

        def __call__(self, attention_mask):
            hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
            return types.SimpleNamespace(
                last_hidden_state=hidden.expand(attention_mask.shape[0], -1, -1)
            )

    encoder = Encoder()
    tokenizer = lambda texts, **k: {"attention_mask": torch.ones(len(texts), 2)}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _: None)
    monkeypatch.setattr(transformers.AutoModel, "from_config", lambda _: encoder)
    monkeypatch.setattr(safetensors_torch, "load_file", lambda _: {"encoder.x": torch.zeros(1)})
    monkeypatch.setenv("STROT_NLU_CHECKPOINT", str(tmp_path))
    return tmp_path


def test_distilled_bundle_roundtrip_matches_compiled(_torch_distilled_env, tmp_path):
    # The fixed embedding is [2.0, 0.0]; a known head makes the distribution
    # deterministic so both the compiled asset and the Battery agree exactly.
    source = _distilled_source(
        temperature=1.0, dispatch_cutoff=0.3,
        policy={"target_accuracy": 0.7},
    )
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    bundle_dir = tmp_path / "triage_battery"
    export_bundle(asset, bundle_dir, name="triage", question="which queue?")

    # The bundle is a directory with the two canonical files.
    assert (bundle_dir / "battery.json").exists()
    assert (bundle_dir / "weights.safetensors").exists()

    battery = load(bundle_dir)
    assert isinstance(battery, Battery)

    context = "double charged"

    # End-to-end distribution parity: the Battery's distilled runtime must emit
    # the SAME per-label distribution as the baked asset's own _RuntimeRouter on
    # the same context (real torch under the fixture), not merely agree once a
    # distribution is handed to it.
    asset_module = types.ModuleType("asset_live")
    exec(compile(source, "asset_live", "exec"), asset_module.__dict__)
    asset_router = asset_module._RuntimeRouter()
    asset_dist = asset_router.select(
        context, "which queue?", asset_module._ROUTING_OPTIONS
    ).distribution
    battery_dist = battery.distribution(context)
    for label in LABELS:
        assert battery_dist[label] == pytest.approx(asset_dist[label])

    # And the full gated verdict matches the baked CompiledPipeline decision.
    baked = _run_compiled_decision(source, battery_dist)
    verdict = battery.decide(context)
    assert verdict["abstained"] == baked["abstained"]
    if not baked["abstained"]:
        assert verdict["label"] == baked["label"]
    assert verdict["confidence"] == pytest.approx(baked["confidence"])
    assert set(verdict["probabilities"]) == set(LABELS)
    assert sum(verdict["probabilities"].values()) == pytest.approx(1.0)


def test_catalog_helpers_roundtrip(_torch_distilled_env, tmp_path):
    source = _distilled_source()
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    catalog = tmp_path / "catalog"
    export_bundle(asset, catalog / "triage", name="triage")
    export_bundle(asset, catalog / "triage2", name="triage2")

    infos = list_batteries(catalog)
    assert {i["name"] for i in infos} == {"triage", "triage2"}
    assert all("path" in i for i in infos)

    info = battery_info(catalog / "triage")
    assert info["name"] == "triage"
    assert info["runtime_kind"] == "distilled"
    # battery_info is torch-free (reads json only).
    assert "labels" in info
