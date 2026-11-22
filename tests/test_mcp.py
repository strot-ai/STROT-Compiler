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
    assert hasattr(tools, "list_models") and hasattr(tools, "compile_start")


def _wait(job_id, timeout=3.0):
    import time

    for _ in range(int(timeout / 0.02)):
        s = tools.compile_status(job_id)
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.02)
    return tools.compile_status(job_id)


def test_compile_start_runs_in_background_and_reports_done(monkeypatch):
    monkeypatch.setattr(tools, "_compiles_started", 0)
    monkeypatch.delenv("STROT_MCP_COMPILE_BUDGET", raising=False)
    monkeypatch.setattr(
        tools, "compile_model",
        lambda *a, **k: {"model_path": "/x/triage.strot", "mode": "head"},
    )
    started = tools.compile_start([{"x": 1}], {"q": {"type": "predicate"}})
    assert started["status"] == "running" and "job_id" in started
    done = _wait(started["job_id"])
    assert done["status"] == "done"
    assert done["result"]["model_path"] == "/x/triage.strot"


def test_compile_status_unknown_job():
    assert tools.compile_status("nope")["status"] == "unknown"


def test_compile_error_is_surfaced_not_raised(monkeypatch):
    monkeypatch.setattr(tools, "_compiles_started", 0)
    monkeypatch.delenv("STROT_MCP_COMPILE_BUDGET", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("no torch")

    monkeypatch.setattr(tools, "compile_model", _boom)
    done = _wait(tools.compile_start([{"x": 1}], {"q": {}})["job_id"])
    assert done["status"] == "error" and "no torch" in done["error"]


def test_compile_budget_refuses_over_cap(monkeypatch):
    monkeypatch.setattr(tools, "_compiles_started", 0)
    monkeypatch.setenv("STROT_MCP_COMPILE_BUDGET", "1")
    monkeypatch.setattr(tools, "compile_model", lambda *a, **k: {"model_path": "m"})
    assert tools.compile_start([{"x": 1}], {"q": {}})["status"] == "running"
    refused = tools.compile_start([{"x": 1}], {"q": {}})
    assert refused["status"] == "refused" and "budget" in refused["reason"]
    assert tools.budget_status() == {"compiles_started": 1, "cap": 1, "remaining": 0}
