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


# --------------------------------------------------------------------------- #
# Hub resolution: a bare name -> files cached from the HF hub (MOCKED)         #
# --------------------------------------------------------------------------- #
#
# huggingface_hub.hf_hub_download is monkeypatched to serve files out of a tmp
# bundle written by Stage 1 helpers, so NO real network is touched. The catalog
# / metadata / resolve *plumbing* is torch-free and always runs; only the final
# Battery.decide parity gate is gated behind the torch importorskip.


def _write_battery_json(directory, metadata):
    """Write just a bundle's battery.json (torch-free, no weights file).

    Lets the manifest / info / resolve plumbing be asserted without safetensors
    or numpy installed; the weights-bearing round-trip uses export_bundle under
    the torch fixture instead.
    """
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "battery.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory / "battery.json"


def _metadata_for(name):
    """A torch-free battery.json payload captured from a compiled asset."""
    source = _distilled_source(
        temperature=1.0, dispatch_cutoff=0.3, policy={"target_accuracy": 0.7},
    )
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    metadata, _tensors, _ = bat._collect_bundle(asset, name=name, question="which queue?")
    return metadata


class _HubStub:
    """Stand-in for huggingface_hub.hf_hub_download backed by an on-disk tree.

    ``root`` mirrors the dataset repo layout (``manifest.json`` at the top,
    ``<name>/battery.json`` + ``<name>/weights.safetensors`` per battery). Each
    call records its kwargs and either returns the matching local path or raises
    a 404-shaped error for an unknown file, exactly as the real downloader does.
    """

    class EntryNotFoundError(Exception):
        pass

    def __init__(self, root):
        self.root = root
        self.calls = []

    def __call__(self, *, repo_id, repo_type, filename, revision=None):
        self.calls.append(
            {"repo_id": repo_id, "repo_type": repo_type,
             "filename": filename, "revision": revision}
        )
        target = self.root / filename
        if not target.exists():
            raise self.EntryNotFoundError(f"404: {filename} not in {repo_id}")
        return str(target)


@pytest.fixture
def _hub(monkeypatch, tmp_path):
    """Install a mocked huggingface_hub.hf_hub_download over a tmp repo tree."""
    hub_mod = pytest.importorskip("huggingface_hub")
    repo_root = tmp_path / "repo"
    stub = _HubStub(repo_root)
    monkeypatch.setattr(hub_mod, "hf_hub_download", stub)
    return types.SimpleNamespace(root=repo_root, stub=stub)


def test_list_batteries_returns_hub_manifest(_hub):
    import json

    manifest = {
        "format_version": 1,
        "batteries": [
            {"name": "triage", "task": "triage tickets", "nlu_model": "x",
             "runtime_kind": "distilled", "strot_version": "0.1.3"},
            {"name": "sentiment", "task": "classify sentiment", "nlu_model": "x",
             "runtime_kind": "fine_tuned", "strot_version": "0.1.3"},
        ],
    }
    _hub.root.mkdir(parents=True, exist_ok=True)
    (_hub.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # No source argument -> resolves the hub manifest's batteries list.
    out = list_batteries()
    assert [b["name"] for b in out] == ["triage", "sentiment"]
    # It downloaded exactly the root manifest from the pinned dataset repo.
    assert _hub.stub.calls == [
        {"repo_id": "strot-ai/batteries", "repo_type": "dataset",
         "filename": "manifest.json", "revision": None}
    ]


def test_battery_info_resolves_from_hub(_hub):
    meta = _metadata_for("triage")
    _write_battery_json(_hub.root / "triage", meta)

    info = battery_info("triage")
    assert info["name"] == "triage"
    assert info["runtime_kind"] == "distilled"
    assert "labels" in info  # torch-free, metadata only
    assert _hub.stub.calls[0]["filename"] == "triage/battery.json"
    assert _hub.stub.calls[0]["repo_type"] == "dataset"


def test_repo_override_arg_and_env(_hub, monkeypatch):
    meta = _metadata_for("triage")
    _write_battery_json(_hub.root / "triage", meta)

    # Explicit repo= arg flows through to the download.
    battery_info("triage", repo="acme/batteries")
    assert _hub.stub.calls[-1]["repo_id"] == "acme/batteries"

    # STROT_BATTERY_REPO overrides the pinned default when no arg is given.
    monkeypatch.setenv("STROT_BATTERY_REPO", "env/batteries")
    battery_info("triage")
    assert _hub.stub.calls[-1]["repo_id"] == "env/batteries"

    # The explicit arg still wins over the env var.
    battery_info("triage", repo="acme/batteries")
    assert _hub.stub.calls[-1]["repo_id"] == "acme/batteries"


def test_revision_is_forwarded(_hub):
    meta = _metadata_for("triage")
    _write_battery_json(_hub.root / "triage", meta)
    battery_info("triage", revision="v2")
    assert _hub.stub.calls[-1]["revision"] == "v2"


def test_unknown_battery_name_errors(_hub):
    # Nothing published under that name -> the actionable "not found" guidance.
    with pytest.raises(Exception) as exc:
        battery_info("does-not-exist")
    msg = str(exc.value)
    assert "not found" in msg
    assert "list_batteries" in msg


def test_local_dir_never_calls_hub(_hub, tmp_path):
    # An existing local directory input stays on the Stage 1 local path; the hub
    # mock must not be touched.
    source = _distilled_source()
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    meta, _tensors, _ = bat._collect_bundle(asset, name="triage")
    bundle = tmp_path / "local_triage"
    _write_battery_json(bundle, meta)

    info = battery_info(bundle)
    assert info["name"] == "triage"
    assert _hub.stub.calls == []  # no hub download made for a local dir


def test_list_batteries_local_dir_never_calls_hub(_hub, tmp_path):
    source = _distilled_source()
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    meta, _tensors, _ = bat._collect_bundle(asset, name="triage")
    catalog = tmp_path / "catalog"
    _write_battery_json(catalog / "triage", meta)

    infos = list_batteries(catalog)
    assert {i["name"] for i in infos} == {"triage"}
    assert all("path" in i for i in infos)
    assert _hub.stub.calls == []  # local listing makes no hub call


def test_missing_huggingface_hub_errors(monkeypatch):
    # Simulate huggingface_hub being absent: the lazy import inside the hub path
    # must raise the actionable install guidance, not a bare ImportError.
    import builtins

    # Drop any already-imported huggingface_hub so the lazy import re-runs and
    # hits the faked __import__ below instead of the module cache.
    for mod in list(sys.modules):
        if mod == "huggingface_hub" or mod.startswith("huggingface_hub."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    real_import = builtins.__import__

    def _no_hub(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_hub)

    with pytest.raises(Exception) as exc:
        battery_info("triage")
    msg = str(exc.value)
    assert "huggingface_hub" in msg
    # Points at the actionable remedy (the extra or the direct install).
    assert "strot-ai[local]" in msg or "pip install huggingface_hub" in msg


def test_name_resolves_and_matches_local_load(_hub, _torch_distilled_env, tmp_path):
    # The torch-backed gate: a name resolved from the (mocked) hub builds a
    # Battery whose decision matches the SAME bundle loaded locally.
    source = _distilled_source(
        temperature=1.0, dispatch_cutoff=0.3, policy={"target_accuracy": 0.7},
    )
    asset = types.SimpleNamespace(
        source=source, runtime_kind="distilled", goal="triage tickets",
        nlu_model="SupersonicLabs/Julia-1",
    )
    # Publish the full bundle (battery.json + weights.safetensors) under the
    # repo tree the hub stub serves from.
    export_bundle(asset, _hub.root / "triage", name="triage", question="which queue?")

    context = "double charged"

    # Local load of the very same files, for a reference decision.
    local_battery = load(_hub.root / "triage")
    local_verdict = local_battery.decide(context)

    # Name -> hub resolve: downloads battery.json + weights, builds a Battery.
    hub_battery = load("triage")
    assert isinstance(hub_battery, Battery)
    hub_verdict = hub_battery.decide(context)

    assert hub_verdict == local_verdict

    # It fetched exactly the two per-battery files from the dataset repo.
    fetched = {c["filename"] for c in _hub.stub.calls}
    assert fetched == {"triage/battery.json", "triage/weights.safetensors"}
    assert all(c["repo_type"] == "dataset" for c in _hub.stub.calls)
