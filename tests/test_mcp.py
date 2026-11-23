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


def test_compile_model_calls_the_llm_free_door(monkeypatch):
    # compile_model must route to strot.compile (no llm arg).
    import strot

    seen = {}

    class _Result:
        questions = {"q": {}}
        cache_hit = False
        artifact_path = "/x/triage.strot"

    def _fake_compile(records, questions, labels, **kw):
        seen["labels"] = labels
        seen["kw"] = kw
        return _Result()

    monkeypatch.setattr(strot, "compile", _fake_compile)
    out = tools.compile_model(
        [{"body": "a"}], {"q": {"type": "predicate"}}, [{"q": True}],
    )
    assert out["model_path"] == "/x/triage.strot"
    assert seen["labels"] == [{"q": True}]
    assert "llm" not in seen["kw"]  # the whole point: no LLM on this path


def test_compile_start_with_labels_runs_in_background(monkeypatch):
    monkeypatch.setattr(tools, "_compiles_started", 0)
    monkeypatch.delenv("STROT_MCP_COMPILE_BUDGET", raising=False)
    monkeypatch.setattr(
        tools, "compile_model",
        lambda *a, **k: {"model_path": "/x/triage.strot", "mode": "head"},
    )
    started = tools.compile_start([{"x": 1}], {"q": {"type": "predicate"}}, [{"q": True}])
    assert started["status"] == "running" and "job_id" in started
    done = _wait(started["job_id"])
    assert done["status"] == "done"
    assert done["result"]["model_path"] == "/x/triage.strot"


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


def test_set_and_read_model_status(tmp_path):
    m = str(tmp_path / "triage.strot")
    assert tools.model_status(m)["status"] == "active"  # default when never set
    tools.set_status(m, "paused", reason="drift")
    st = tools.model_status(m)
    assert st["status"] == "paused" and st["reason"] == "drift"
    tools.set_status(m, "active")
    assert tools.model_status(m)["status"] == "active"


def test_set_status_rejects_bad_value(tmp_path):
    assert "error" in tools.set_status(str(tmp_path / "m.strot"), "bogus")


def test_decide_record_honors_paused_circuit_breaker(tmp_path):
    m = str(tmp_path / "m.strot")
    tools.set_status(m, "paused", reason="broken")
    # paused short-circuits before any model load, so no torch is needed
    assert tools.decide_record(m, {"x": 1}) == {"paused": True, "reason": "broken"}


def test_recompile_reuses_question_config(tmp_path, monkeypatch):
    m = tmp_path / "triage.strot"
    m.write_text('QUESTIONS_CONFIG: Dict[str, Any] = {"is_bug": {"type": "predicate"}}\n')
    monkeypatch.setattr(tools, "_compiles_started", 0)
    monkeypatch.delenv("STROT_MCP_COMPILE_BUDGET", raising=False)
    captured = {}

    def _fake(data, questions, labels=None, **k):
        captured["questions"] = questions
        captured["name"] = k.get("name")
        return {"model_path": "x"}

    monkeypatch.setattr(tools, "compile_model", _fake)
    started = tools.recompile(str(m), [{"x": 1}])
    _wait(started["job_id"])
    assert captured["questions"] == {"is_bug": {"type": "predicate"}}
    assert captured["name"].endswith("triage-recompiled")  # same dir as the original


def test_recompile_missing_config_errors(tmp_path):
    m = tmp_path / "empty.strot"
    m.write_text("# no QUESTIONS_CONFIG here\n")
    assert tools.recompile(str(m), [{"x": 1}])["status"] == "error"


def test_playbook_covers_the_loop_and_principles():
    from strot.mcp.playbook import PLAYBOOK

    for term in (
        "inspect_data", "pii_scan", "compile_start",
        "evaluate_model", "drift_check", "cluster_rejections", "recompile", "set_status",
        "Labels from any source", "Recompile, don't increment", "Evaluate-safe, act-gated",
    ):
        assert term in PLAYBOOK, term
