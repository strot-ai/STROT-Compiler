"""Tests for strot.inspect_data / strot.pii_scan - the agent's torch-free view of
raw input (and the MCP wrappers over them)."""

import strot
from strot.mcp import tools

SAMPLE = [
    {"ticket_id": 1, "body": "double billing charge; email jane@acme.io", "tier": "pro"},
    {"ticket_id": 2, "body": "app crash defect; ssn 123-45-6789", "tier": "free"},
]


def test_inspect_data_profiles_fields():
    out = strot.inspect_data(SAMPLE)
    assert out["n_records"] == 2
    names = {f["name"] for f in out["fields"]}
    assert names == {"ticket_id", "body", "tier"}
    body = next(f for f in out["fields"] if f["name"] == "body")
    assert "cardinality" in body and "sample_values" in body and "semantic_type" in body


def test_pii_scan_detects_email_and_ssn():
    out = strot.pii_scan(SAMPLE)
    assert out["n_scanned"] == 2
    assert out["detected"].get("EMAIL") == 1 and out["detected"].get("SSN") == 1
    assert out["masker"] in ("regex", "regex+gliner")


def test_pii_scan_clean_data_detects_nothing():
    assert strot.pii_scan([{"body": "a normal request with nothing sensitive"}])["detected"] == {}


def test_mcp_inspect_and_pii_tools_torch_free():
    assert tools.inspect_data([{"a": 1, "b": "x"}])["n_records"] == 1
    assert tools.pii_scan([{"body": "reach me at a@b.io"}])["detected"].get("EMAIL") == 1
