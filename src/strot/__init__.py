"""strot-ai: a decision compiler.

STROT (Structured Task Reasoning and Output Transformation, arXiv:2505.01636)
compiles a specialized decision model per task. STROT runs no LLM and holds no
key: the agent (or a human) brings the labels, a calibrated head is distilled
onto a frozen encoder, and the result is a self-contained model that decides
locally on CPU and emits typed decision JSON.

    # the agent designs the questions and labels a small sample (see `strot playbook`)
    model = strot.compile(data, questions, labels)   # distill a model, no LLM
    model(record)                                     # decide, locally
    model.dispatch("triage"); strot.load_model("triage")   # ship + reuse
"""

from .compiler import CompiledAsset, StrotCompiler
from .constants import CONFIDENCE_THRESHOLD, DEFAULT_NLU_MODEL
from .exceptions import (
    CompilationError,
    FallbackException,
    RouterConfigurationError,
    SchemaIntrospectionError,
    StrotError,
)
from .schema import DatasetSchema, FieldProfile, SchemaProfiler, introspect_schema

__version__ = "2.0.3"

# `compile` is the build verb (re.compile-style: compiled once, decides forever).
# The agent brings the labels; STROT distills the model with no LLM.
from .jit import DecisionModel, RunResult, load_model  # noqa: E402  (needs __version__ for cache keys)
from .jit import train as compile  # noqa: E402,A001  (the one build door; shadows the builtin at the package surface)
from .evaluate import evaluate, score_decisions  # noqa: E402
from .trace import cluster_rejections, drift_check, read_traces, summarize_traces  # noqa: E402
from .inspect import inspect_data, pii_scan  # noqa: E402
from .watch import infer_questions, ingest_traces  # noqa: E402
from .prefetch import prefetch_nlu  # noqa: E402

__all__ = [
    "compile",
    "evaluate",
    "score_decisions",
    "read_traces",
    "summarize_traces",
    "cluster_rejections",
    "drift_check",
    "inspect_data",
    "pii_scan",
    "ingest_traces",
    "infer_questions",
    "load_model",
    "DecisionModel",
    "RunResult",
    "prefetch_nlu",
    "CompiledAsset",
    "StrotCompiler",
    "StrotError",
    "CompilationError",
    "FallbackException",
    "RouterConfigurationError",
    "SchemaIntrospectionError",
    "DEFAULT_NLU_MODEL",
    "DatasetSchema",
    "FieldProfile",
    "SchemaProfiler",
    "introspect_schema",
    "__version__",
]
