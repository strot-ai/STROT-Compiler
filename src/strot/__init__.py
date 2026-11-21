"""strot-ai: compile-time LLM code synthesis with deterministic,
self-routing NLU runtimes.

Implements the STROT framework (Structured Task Reasoning and Output
Transformation, arXiv:2505.01636) with an NLU strategy-pattern runtime:
a local CPU transformer (default ``SupersonicLabs/Julia-1``) or a hosted
cloud API, baked into a standalone compiled production asset.
"""

from .compiler import (
    CompiledAsset,
    StrotCompiler,
    anthropic_llm_client,
    claude_cli_llm_client,
    codex_cli_llm_client,
    openai_llm_client,
)
from .exceptions import (
    CompilationError,
    FallbackException,
    RouterConfigurationError,
    SchemaIntrospectionError,
    StrotError,
)
from .router import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_NLU_MODEL,
    DEFAULT_STROT_MODEL,
    MAX_FLAT_OPTIONS,
    STROT_MODELS,
    FineTunedAdapterStrategy,
    HostedApiStrategy,
    LocalEncoderStrategy,
    RouteDecision,
    Router,
    RuntimeStrategy,
    create_router,
    create_strategy,
    resolve_model,
)
from .schema import DatasetSchema, FieldProfile, SchemaProfiler, introspect_schema

__version__ = "0.12.0"

from .jit import DecisionModel, RunResult, decide, load_model, questions, run  # noqa: E402  (needs __version__ for cache keys)
from .prefetch import prefetch_nlu  # noqa: E402

# Battery distribution: a compiled catalog classifier as data (bundle on disk)
# served by a data-parameterized runtime in the package. Imported after
# __version__ so export can stamp the bundle. Local filesystem only for now.
from .battery import (  # noqa: E402
    Battery,
    battery_info,
    export_bundle,
    list_batteries,
    load,
)

__all__ = [
    "run",
    "decide",
    "questions",
    "load_model",
    "DecisionModel",
    "RunResult",
    "prefetch_nlu",
    "Battery",
    "load",
    "list_batteries",
    "battery_info",
    "export_bundle",
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
    "CONFIDENCE_THRESHOLD",
    "DEFAULT_NLU_MODEL",
    "DEFAULT_STROT_MODEL",
    "STROT_MODELS",
    "resolve_model",
    "MAX_FLAT_OPTIONS",
    "RuntimeStrategy",
    "LocalEncoderStrategy",
    "HostedApiStrategy",
    "FineTunedAdapterStrategy",
    "RouteDecision",
    "Router",
    "create_router",
    "create_strategy",
    "DatasetSchema",
    "FieldProfile",
    "SchemaProfiler",
    "introspect_schema",
    "__version__",
]
