"""The agent's view of RAW input - before it plans a compilation.

:func:`inspect_data` profiles the data (fields, types, cardinality, sample
values); :func:`pii_scan` reports what the teacher-boundary masker would redact
(the PII-enablement "detect" step). Both are torch-free, so the agent can look
and plan cheaply - which questions, which fields, which tier, and whether PII
handling is needed - before paying for a compile.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .schema import introspect_schema

_TAG_RE = re.compile(r"\[([A-Z][A-Z_]*)\]")


def _records(data: Any, sample_size: int):
    from .jit import as_records

    return as_records(data)[:sample_size]


def inspect_data(data: Any, sample_size: int = 25) -> Dict[str, Any]:
    """Profile raw input: per-field name, semantic/python type, cardinality,
    null rate, and sample values, plus the record count. Torch-free."""
    records = _records(data, sample_size)
    out = introspect_schema(records).to_dict()
    out["n_records"] = len(records)
    return out


def pii_scan(data: Any, sample_size: int = 25) -> Dict[str, Any]:
    """Detect what PII the teacher-boundary masker would redact over a sample -
    placeholder types + counts. Torch-free (regex; also GLiNER if ``[pii]`` is
    installed). This is the 'detect' step of PII enablement: an agent can scan
    before any data reaches the teacher.
    """
    from . import pii

    records = _records(data, sample_size)
    maskers = pii.default_maskers()
    masked = pii.mask_records(records, maskers=maskers)
    counts: Dict[str, int] = {}
    for record in masked:
        for value in record.values():
            for tag in _TAG_RE.findall(str(value)):
                counts[tag] = counts.get(tag, 0) + 1
    return {
        "n_scanned": len(records),
        "detected": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "masker": "regex+gliner" if len(maskers) > len(pii.DEFAULT_MASKERS) else "regex",
    }
