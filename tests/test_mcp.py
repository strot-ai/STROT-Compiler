"""Tests for strot.mcp.tools — the SDK-free, torch-free capability layer.

The discovery tools read a compiled model's baked literals with ast, so they are
exercised here without installing the mcp SDK or torch.
"""

import json

import pytest

from strot.mcp import tools

# A minimal stand-in for a compiled model artifact: just the baked literals the
# discovery tools read (no runtime, so no torch needed).
HEAD_ARTIFACT = '''NLU_MODEL = "Alibaba-NLP/gte-modernbert-base"

QUESTIONS_CONFIG: Dict[str, Any] = {"is_bug": {"type": "predicate"}, "team": {"type": "choice"}}

OOD: Dict[str, Any] = {'centroids': [[1.0, 0.0]], 'threshold': 0.5}

class CompiledDecider:
    pass
'''

DEEP_ARTIFACT = '''NLU_MODEL = "Alibaba-NLP/gte-modernbert-base"

QUESTIONS_CONFIG: Dict[str, Any] = {"q": {"type": "predicate"}}

ADAPTER_CONFIG: Dict[str, Any] = {"base_model": "answerdotai/ModernBERT-base"}
'''


def test_read_model_meta_is_torch_free(tmp_path):
    art = tmp_path / "ticket-triage.strot"
    art.write_text(HEAD_ARTIFACT)
    meta = tools.read_model_meta(art)
    assert meta["name"] == "ticket-triage"
    assert meta["questions"] == ["is_bug", "team"]
    assert meta["encoder"] == "Alibaba-NLP/gte-modernbert-base"
    assert meta["has_drift_gate"] is True
    assert meta["has_adapter"] is False


def test_read_model_meta_detects_deep_adapter(tmp_path):
    art = tmp_path / "deep.strot"
    art.write_text(DEEP_ARTIFACT)
    meta = tools.read_model_meta(art)
    assert meta["has_adapter"] is True
    assert meta["has_drift_gate"] is False


def test_read_model_meta_prefers_json_sidecar(tmp_path):
    art = tmp_path / "m.strot"
    art.write_text(HEAD_ARTIFACT)
    art.with_suffix(".json").write_text(json.dumps({"mode": "head", "nlu": "x-encoder"}))
    meta = tools.read_model_meta(art)
    assert meta["mode"] == "head"  # from the sidecar


def test_list_models_scans_dir(tmp_path):
    (tmp_path / "a.strot").write_text(HEAD_ARTIFACT)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "battery_b_abc.py").write_text(DEEP_ARTIFACT)
    names = {m["name"] for m in tools.list_models(tmp_path)}
    assert names == {"a", "battery_b_abc"}


def test_list_models_missing_root_is_empty(tmp_path):
    assert tools.list_models(tmp_path / "nope") == []


def test_as_records_accepts_inline_and_file(tmp_path):
    records = [{"x": 1}, {"x": 2}]
    assert tools._as_records(records) == records
    p = tmp_path / "data.json"
    p.write_text(json.dumps(records))
    assert tools._as_records(str(p)) == records


def test_tools_import_without_mcp_sdk():
    # Importing the capability layer must not require the mcp SDK (so the core
    # install stays tiny). If this module imported, that is already proven.
    assert hasattr(tools, "list_models") and hasattr(tools, "compile_model")
