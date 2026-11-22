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
def design_questions(data: Union[str, list], intent: str, llm: str = "openai/gpt-4o") -> Dict[str, Any]:
    """Design a typed question config from a one-line intent and a data sample."""
    return tools.design_questions(data, intent, llm=llm)


@server.tool()
def compile_model(
    data: Union[str, list],
    questions: Union[str, dict],
    mode: str = "head",
    llm: str = "openai/gpt-4o",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile a decision model from data + questions (or a bare intent). Needs [local]."""
    return tools.compile_model(data, questions, mode=mode, llm=llm, name=name)


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


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
