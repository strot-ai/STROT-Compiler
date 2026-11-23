"""The STROT MCP server: registers :mod:`strot.mcp.tools` with the MCP SDK.

Run with ``python -m strot.mcp`` (needs ``pip install "strot-ai[mcp]"``). Each
tool is a thin pass-through to a function in ``tools`` so the capability logic
stays SDK-free and testable. Targets the MCP Python SDK 2.x (``MCPServer``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        'The STROT MCP server needs the mcp extra: pip install "strot-ai[mcp]" '
        "(requires the MCP Python SDK >= 2)."
    ) from exc

from . import tools

server = MCPServer("strot")


@server.tool()
def list_models(root: str = ".") -> List[Dict[str, Any]]:
    """List compiled STROT decision models under a directory (torch-free)."""
    return tools.list_models(root)


@server.tool()
def model_info(path: str) -> Dict[str, Any]:
    """Read a compiled model's metadata (questions, encoder, tier) without loading it."""
    return tools.model_info(path)


@server.tool()
def inspect_data(data: Union[str, list], sample_size: int = 25) -> Dict[str, Any]:
    """Profile raw input - fields, types, cardinality, sample values (torch-free). Plan from this."""
    return tools.inspect_data(data, sample_size=sample_size)


@server.tool()
def pii_scan(data: Union[str, list], sample_size: int = 25) -> Dict[str, Any]:
    """Detect what PII the teacher-boundary masker would redact over a sample (torch-free)."""
    return tools.pii_scan(data, sample_size=sample_size)


@server.tool()
def compile_start(
    data: Union[str, list],
    questions: Union[str, dict],
    labels: Optional[List[Dict[str, Any]]] = None,
    mode: str = "head",
    projection: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile a decision model from labels YOU bring (no LLM) -> returns a job_id
    to poll. The agent-labels door: design the questions and decide each record
    yourself, pass {question: answer} per record. mode='lite' needs no labels.
    Runs minutes; needs [local]."""
    return tools.compile_start(data, questions, labels, mode=mode, projection=projection, name=name)


@server.tool()
def compile_status(job_id: str) -> Dict[str, Any]:
    """Poll a compile job: running | done (with result) | error."""
    return tools.compile_status(job_id)


@server.tool()
def budget_status() -> Dict[str, Any]:
    """Compiles run this session and the cap (STROT_MCP_COMPILE_BUDGET), if set."""
    return tools.budget_status()


@server.tool()
def decide(model_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Run a compiled model on one record -> typed decision JSON. Needs [local]."""
    return tools.decide_record(model_path, record)


@server.tool()
def evaluate_model(
    model_path: str, labeled: List[Dict[str, Any]], confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Run a compiled model over labeled records -> accuracy/coverage/ECE. Needs [local]."""
    return tools.evaluate_model(model_path, labeled, confidence_threshold=confidence_threshold)


@server.tool()
def score(
    decisions: List[Dict[str, Any]], expected: List[Dict[str, Any]], confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Score already-produced decisions against expected answers (torch-free)."""
    return tools.score(decisions, expected, confidence_threshold=confidence_threshold)


@server.tool()
def read_traces(path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Summarize a model's decision traces: volume, refusal rate, confidence, value mix (torch-free)."""
    return tools.read_traces(path, limit=limit)


@server.tool()
def cluster_rejections(traces_path: str, top_terms: int = 10, max_samples: int = 20) -> Dict[str, Any]:
    """Cluster the refused records to surface the gap to close - the diagnostic signal (torch-free)."""
    return tools.cluster_rejections(traces_path, top_terms=top_terms, max_samples=max_samples)


@server.tool()
def drift_check(traces_path: str, refusal_threshold: float = 0.2, confidence_floor: float = 0.5) -> Dict[str, Any]:
    """Drift signal from traces + a recompile recommendation (torch-free)."""
    return tools.drift_check(traces_path, refusal_threshold=refusal_threshold, confidence_floor=confidence_floor)


@server.tool()
def recompile(
    model_path: str, data: Union[str, list],
    labels: Optional[List[Dict[str, Any]]] = None,
    mode: str = "head", name: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompile a model on fresh data (reusing its questions), async -> job_id. You
    re-label the new data (no LLM); mode='lite' needs none. Needs [local]."""
    return tools.recompile(model_path, data, labels, mode=mode, name=name)


@server.tool()
def set_status(model_path: str, status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Set a model's serving status: active | paused (the circuit breaker / promote)."""
    return tools.set_status(model_path, status, reason=reason)


@server.tool()
def model_status(model_path: str) -> Dict[str, Any]:
    """Read a model's serving status (active | paused)."""
    return tools.model_status(model_path)


@server.prompt(
    name="strot_compiler",
    description="The STROT compiler & gatekeeper playbook - load it to drive the tools "
    "as a specialized decision-model compiler (plan -> compile -> monitor -> recompile).",
)
def strot_compiler_playbook() -> str:
    from .playbook import PLAYBOOK

    return PLAYBOOK


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
