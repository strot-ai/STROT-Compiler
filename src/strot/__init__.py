"""strot-ai: a decision compiler.

STROT (Structured Task Reasoning and Output Transformation, arXiv:2505.01636)
compiles a specialized decision model per task: an LLM teacher labels your real
data, a calibrated head is distilled onto a frozen encoder, and the result is a
self-contained model that runs locally on CPU and emits typed decision JSON.

    questions = strot.questions(data, "triage support tickets")  # design the schema
    model = strot.decide(data, questions)                        # compile a model
    model(record)                                                # decide, locally
    model.save("triage"); strot.load_model("triage")             # ship + reuse
"""

from .compiler import (
    CompiledAsset,
    StrotCompiler,
    anthropic_llm_client,
    claude_cli_llm_client,
    codex_cli_llm_client,
    openai_llm_client,
)
from .constants import CONFIDENCE_THRESHOLD, DEFAULT_NLU_MODEL
from .exceptions import (
    CompilationError,
    FallbackException,
    RouterConfigurationError,
    SchemaIntrospectionError,
    StrotError,
)
from .schema import DatasetSchema, FieldProfile, SchemaProfiler, introspect_schema

__version__ = "0.12.0"

from .jit import DecisionModel, RunResult, decide, load_model, questions  # noqa: E402  (needs __version__ for cache keys)
from .prefetch import prefetch_nlu  # noqa: E402

__all__ = [
    "decide",
    "questions",
    "load_model",
    "DecisionModel",
    "RunResult",
    "prefetch_nlu",
    "CompiledAsset",
    "StrotCompiler",
    "openai_llm_client",
    "anthropic_llm_client",
    "claude_cli_llm_client",
    "codex_cli_llm_client",
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
