"""STROT as MCP tools — expose the compiler for an agent to drive.

    pip install "strot-ai[mcp]"
    python -m strot.mcp

The capability logic lives in :mod:`strot.mcp.tools` (import-light: the discovery
tools are torch-free and need no LLM, so they stay cheap and testable without the
MCP SDK). :mod:`strot.mcp.server` is the thin wrapper that registers them with the
MCP SDK. Keeping the two apart means ``import strot`` never pulls the SDK, and the
core install stays tiny — the ``[mcp]`` extra only matters when you run the server.
"""

from . import tools

__all__ = ["tools"]
