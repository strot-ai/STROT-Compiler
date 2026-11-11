"""Unit tests for the fine-tune tier (strot.finetune).

The training path needs torch + transformers + a model download, so it is an
integration concern exercised in the finetune CI job. Here we cover the parts
that must hold with or without the extra: the validated-recipe defaults, the
missing-extra guard, and the small-data guard.
"""

import pytest

from strot import finetune as ft
from strot.exceptions import RouterConfigurationError

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_config_defaults_are_the_validated_recipe():
    c = ft.FineTuneConfig()
    assert c.base_model == "answerdotai/ModernBERT-base"
    assert c.top_layers == 4  # the measured knee (benchmarks/cyber_cached_top.py)
    assert c.lora_rank == 16
    assert c.lora_alpha == 32
    assert ft.DEFAULT_FINETUNE_BASE == c.base_model


def test_module_imports_without_torch():
    # top-level import must not require the finetune extra
    assert hasattr(ft, "finetune_adapter")
    assert ft.MIN_FINETUNE_RECORDS >= 24


@pytest.mark.skipif(HAS_TORCH, reason="torch present: extra guard not exercised")
def test_missing_extra_raises_configuration_error():
    with pytest.raises(RouterConfigurationError):
        ft.finetune_adapter(["x"] * 60, [0, 1] * 30, 2, [0], [1], list(range(2, 60)))


@pytest.mark.skipif(not HAS_TORCH, reason="small-data guard runs only with torch")
def test_small_data_guard_before_any_download():
    # too few records must raise before any model is fetched
    with pytest.raises(ValueError):
        ft.finetune_adapter(["x"] * 4, [0, 1, 0, 1], 2, [0], [1], [2, 3])


def test_result_dataclass_shape():
    r = ft.FineTuneResult(
        measure_accuracy=0.7,
        measure_distributions=[[0.6, 0.4]],
        calibrate_distributions=[[0.5, 0.5]],
        adapter={"__head_bias__": [0.0, 0.0]},
    )
    assert r.measure_accuracy == 0.7
    assert r.config == {}
    assert "__head_bias__" in r.adapter


# --------------------------------------------------------------------- #
# Ensemble (piece A): pure-Python averaging + list-baking, no torch       #
# --------------------------------------------------------------------- #


def test_default_ensemble_size_is_three():
    assert ft.DEFAULT_ENSEMBLE_SIZE == 3
    assert ft.FineTuneConfig().ensemble_size == 3


def test_average_distributions_is_rowwise_mean():
    a = [[1.0, 0.0], [0.0, 1.0]]
    b = [[0.0, 1.0], [0.0, 1.0]]
    assert ft.average_distributions([a, b]) == [[0.5, 0.5], [0.0, 1.0]]
    # single member passes through unchanged
    assert ft.average_distributions([a]) == a
    # empty folds average to empty
    assert ft.average_distributions([[], []]) == []


def test_average_distributions_guards():
    with pytest.raises(ValueError):
        ft.average_distributions([])
    with pytest.raises(ValueError):
        ft.average_distributions([[[0.1, 0.9]], [[0.1, 0.9], [0.2, 0.8]]])


def _fake_member_trainer(per_seed_measure):
    """Return a fake finetune_adapter keyed on config.seed, so each ensemble
    member yields distinct (diverse) measure distributions."""
    def _trainer(texts, labels, n_classes, measure, calibrate, train, config=None):
        seed = config.seed if config else 0
        return ft.FineTuneResult(
            measure_accuracy=0.0,  # ignored: the ensemble recomputes it
            measure_distributions=per_seed_measure[seed],
            calibrate_distributions=[[0.6, 0.4]],
            adapter={"__seed__": seed, "__head_bias__": [0.0, 0.0]},
            config={"base_model": "x", "cut": 0, "n_layers": 1, "lora_rank": 2,
                    "lora_alpha": 4, "max_length": 8, "n_classes": n_classes,
                    "device": "cpu"},
        )
    return _trainer


def test_ensemble_averages_members_and_bakes_the_list():
    labels = [0, 1]
    measure = [0, 1]
    # Seeds 11, 12, 13 (base seed 11 + member index) give diverse per-member
    # rows; averaged, both rows argmax to the true label -> accuracy 1.0.
    per_seed = {
        11: [[0.9, 0.1], [0.2, 0.8]],
        12: [[0.6, 0.4], [0.1, 0.9]],
        13: [[0.7, 0.3], [0.4, 0.6]],
    }
    result = ft.finetune_ensemble(
        texts=["a", "b"], labels=labels, n_classes=2,
        measure=measure, calibrate=[0], train=[0, 1],
        _trainer=_fake_member_trainer(per_seed),
    )
    # The list of per-member adapters is baked (serving averages them).
    assert len(result.adapters) == 3
    assert [a["__seed__"] for a in result.adapters] == [11, 12, 13]
    # Member seeds vary (diversity): seed, seed+1, seed+2.
    assert result.config["ensemble_size"] == 3
    # Judged on the member-averaged rows, not any single member.
    expected = ft.average_distributions([per_seed[s] for s in (11, 12, 13)])
    assert result.measure_distributions == expected
    assert result.measure_accuracy == 1.0
    assert len(result.member_accuracies) == 3


def test_ensemble_size_one_is_single_adapter():
    cfg = ft.FineTuneConfig(ensemble_size=1, seed=5)
    per_seed = {5: [[0.9, 0.1]]}
    result = ft.finetune_ensemble(
        texts=["a"], labels=[0], n_classes=2,
        measure=[0], calibrate=[0], train=[0], config=cfg,
        _trainer=_fake_member_trainer(per_seed),
    )
    assert len(result.adapters) == 1
    assert result.measure_distributions == [[0.9, 0.1]]
    assert result.measure_accuracy == 1.0


def test_ensemble_size_zero_rejected():
    with pytest.raises(ValueError):
        ft.finetune_ensemble(
            ["a"], [0], 2, [0], [0], [0],
            config=ft.FineTuneConfig(ensemble_size=0),
            _trainer=_fake_member_trainer({}),
        )


# --------------------------------------------------------------------- #
# validate_adapter (find/reuse): pure-Python via a fake loader, no torch  #
# --------------------------------------------------------------------- #


def _fake_loader(per_text):
    """A fake router loader: ``forward(texts)`` looks each text up in
    ``per_text`` so the adapter's scoring is deterministic and torch-free."""
    def _loader(adapter, config):
        def forward(texts):
            return [per_text[t] for t in texts]
        return ("tok", forward)
    return _loader


def test_validate_adapter_scores_measure_and_calibrate_no_training(monkeypatch):
    # No torch needed: stub the extra guard and inject a fake loader.
    monkeypatch.setattr(ft, "_import_torch", lambda: None)
    per_text = {
        "a": [0.9, 0.1],  # measure row 0 -> argmax 0 == label 0 (hit)
        "b": [0.2, 0.8],  # measure row 1 -> argmax 1 == label 1 (hit)
        "c": [0.6, 0.4],  # calibrate row
    }
    result = ft.validate_adapter(
        texts=["a", "b", "c"], labels=[0, 1, 0], n_classes=2,
        measure=[0, 1], calibrate=[2],
        adapters=[{"__head_bias__": [0.0, 0.0]}],
        config={"n_classes": 2, "base_model": "x"},
        _loader=_fake_loader(per_text),
    )
    assert result.measure_accuracy == 1.0
    assert result.measure_distributions == [[0.9, 0.1], [0.2, 0.8]]
    assert result.calibrate_distributions == [[0.6, 0.4]]
    assert result.member_accuracies == []  # no per-member training occurred


def test_validate_adapter_averages_multiple_adapters(monkeypatch):
    monkeypatch.setattr(ft, "_import_torch", lambda: None)
    # Two adapters that both see the same texts; averaging makes row 0 argmax 0.
    def _two_loader(adapter, config):
        table = adapter["__rows__"]
        return ("tok", lambda texts: [table[t] for t in texts])

    result = ft.validate_adapter(
        texts=["a"], labels=[0], n_classes=2,
        measure=[0], calibrate=[0],
        adapters=[{"__rows__": {"a": [0.8, 0.2]}}, {"__rows__": {"a": [0.6, 0.4]}}],
        config={"n_classes": 2},
        _loader=_two_loader,
    )
    assert result.measure_distributions == [[0.7, pytest.approx(0.3)]]  # row-wise mean
    assert result.measure_accuracy == 1.0
    assert len(result.adapters) == 2


def test_validate_adapter_n_classes_mismatch_raises(monkeypatch):
    monkeypatch.setattr(ft, "_import_torch", lambda: None)
    with pytest.raises(ValueError):
        ft.validate_adapter(
            texts=["a"], labels=[0], n_classes=3,
            measure=[0], calibrate=[0],
            adapters=[{"__head_bias__": [0.0, 0.0]}],
            config={"n_classes": 2},  # adapter predicts a different label space
            _loader=_fake_loader({"a": [1.0, 0.0]}),
        )


def test_validate_adapter_empty_battery_rejected(monkeypatch):
    monkeypatch.setattr(ft, "_import_torch", lambda: None)
    with pytest.raises(ValueError):
        ft.validate_adapter(
            texts=["a"], labels=[0], n_classes=2,
            measure=[0], calibrate=[0], adapters=[],
            config={"n_classes": 2}, _loader=_fake_loader({}),
        )
