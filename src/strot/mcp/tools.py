"""Capability functions behind the STROT MCP server.

Deliberately import-light: the discovery tools (:func:`list_models`,
:func:`model_info`) read a compiled model's baked literals with ``ast`` - no
torch, no LLM, no model load - so an agent can survey the whole roster for free.
The compile/inference tools import ``strot`` lazily and need the ``[local]``
extra; schema design needs an LLM but not torch.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

Records = List[Dict[str, Any]]
DataArg = Union[str, Records]


def _as_records(data: DataArg) -> Records:
    """Accept inline records or a path to a JSON file of records."""
    if isinstance(data, str):
        return json.loads(Path(data).read_text(encoding="utf-8"))
    return list(data)


def _module_literal(source: str, name: str) -> Any:
    """Literal-eval a top-level ``name = ...`` / ``name: T = ...`` assignment."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign):
            target = next((getattr(t, "id", None) for t in node.targets), None)
        if target == name:
            try:
                return ast.literal_eval(node.value)
            except Exception:
                return None
    return None


def read_model_meta(path: Union[str, Path]) -> Dict[str, Any]:
    """Torch-free metadata for a compiled model (``.strot`` or ``battery_*.py``).

    Reads the baked ``QUESTIONS_CONFIG`` / ``NLU_MODEL`` literals (and the
    optional ``.json`` sidecar) without loading the encoder.
    """
    p = Path(path)
    meta: Dict[str, Any] = {}
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    source = p.read_text(encoding="utf-8", errors="ignore")
    questions = meta.get("questions") or _module_literal(source, "QUESTIONS_CONFIG")
    return {
        "path": str(p),
        "name": p.stem,
        "questions": list(questions) if isinstance(questions, dict) else None,
        "encoder": _module_literal(source, "NLU_MODEL") or meta.get("nlu"),
        "mode": meta.get("mode"),
        "has_adapter": "ADAPTER_CONFIG" in source,  # a deep (LoRA) model
        "has_drift_gate": "OOD" in source and "centroids" in source,
    }


def list_models(root: Union[str, Path] = ".") -> List[Dict[str, Any]]:
    """List compiled decision models under ``root`` (torch-free inventory)."""
    base = Path(root)
    if not base.exists():
        return []
    files = sorted(set(base.rglob("*.strot")) | set(base.rglob("battery_*.py")))
    return [read_model_meta(f) for f in files]


def model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """One compiled model's metadata, without loading it."""
    return read_model_meta(path)


def design_questions(
    data: DataArg, intent: str, llm: str = "openai/gpt-4o", sample_size: int = 25
) -> Dict[str, Any]:
    """Design a typed question config from a one-line intent + a data sample."""
    import strot

    return strot.questions(_as_records(data), intent, llm=llm, sample_size=sample_size)


def compile_model(
    data: DataArg,
    questions: Union[str, Dict[str, Any]],
    mode: str = "head",
    llm: str = "openai/gpt-4o",
    state: str = ".strot",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile a decision model (needs the ``[local]`` extra). Returns its path.

    ``questions`` may be a config dict or a bare intent string. When ``name`` is
    given, the model is saved as ``<name>.strot``.
    """
    import strot

    result = strot.decide(
        _as_records(data), questions, llm=llm, mode=mode, state=state,
        execute=False, name=name,
    )
    path = result.save(name) if name else result.artifact_path
    return {
        "model_path": str(path),
        "mode": mode,
        "questions": list(result.questions or {}),
        "cache_hit": result.cache_hit,
    }


def decide_record(model_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Run a compiled model on one record -> typed decision JSON (needs ``[local]``)."""
    import strot

    return strot.load_model(model_path)(record)
