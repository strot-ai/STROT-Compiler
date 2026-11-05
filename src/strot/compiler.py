"""Compile time: the STROT Synthesizer Engine.

:class:`StrotCompiler` implements the paper's three phases at compile time:

1. **Schema-aware context construction** — introspect the dataset sample
   into a verified structural schema (:mod:`strot.schema`).
2. **Prompt scaffolding** — build a strict, sectioned prompt for a large
   generative LLM containing the schema, the functional goal, and the
   requested ``nlu_model``.
3. **Feedback-driven synthesis** — ask the LLM for the execution-path code,
   validate its syntax and output contract, and loop error traces back for
   self-correction within a bounded retry budget.

The validated paths are then woven together with a deterministic runtime
router template (:mod:`strot.templates`) matched to the ``nlu_model``,
producing a completely standalone production asset.
"""

from __future__ import annotations

import ast
import json
import re
import types as _types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from . import templates
from .exceptions import CompilationError, RouterConfigurationError
from .router import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_NLU_MODEL,
    HOSTED_PROVIDER_PREFIXES,
)
from .schema import DatasetSchema, introspect_schema

#: A generative LLM client is any callable mapping a prompt to raw model text.
LLMClient = Callable[[str], str]
LLMHook = Callable[[Any], None]

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def openai_llm_client(
    model: str = "gpt-6-luna", temperature: float = 0.0, *,
    reasoning_effort: Optional[str] = None,
    on_result: Optional[LLMHook] = None, on_event: Optional[LLMHook] = None,
    on_turn: Optional[LLMHook] = None,
) -> LLMClient:
    """Convenience factory: a synthesizer client backed by the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RouterConfigurationError(
            "openai_llm_client requires the 'hosted' extra: "
            "pip install strot-ai[hosted]"
        ) from exc
    client = OpenAI()
    effort = reasoning_effort if reasoning_effort is not None else (
        "none" if model == "gpt-6-luna" else None)
    options: Dict[str, Any] = {"temperature": temperature} if effort in (None, "none") else {}
    if effort is not None:
        options["extra_body"] = {"reasoning_effort": effort}

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
        if on_result is not None:
            on_result(response)
        return response.choices[0].message.content or ""

    return _call


def anthropic_llm_client(
    model: str = "claude-haiku-5-5",
    max_tokens: int = 16000,
    *, reasoning_effort: Optional[str] = None,
    on_result: Optional[LLMHook] = None, on_event: Optional[LLMHook] = None,
    on_turn: Optional[LLMHook] = None,
) -> LLMClient:
    """Convenience factory: a synthesizer client backed by the Anthropic API.

    Requires the ``anthropic`` package (``pip install strot-ai[anthropic]``)
    and ``ANTHROPIC_API_KEY`` in the environment.
    """
    if reasoning_effort is not None:
        raise RouterConfigurationError("This Anthropic client does not support reasoning_effort.")
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RouterConfigurationError(
            "anthropic_llm_client requires the 'anthropic' extra: "
            "pip install strot-ai[anthropic]"
        ) from exc
    client = Anthropic()

    def _call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if on_result is not None:
            on_result(response)
        if response.stop_reason == "refusal":
            raise CompilationError(
                "Anthropic synthesizer refused the request.", attempts=0
            )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    return _call


def claude_cli_llm_client(
    model: Optional[str] = None,
    cli: str = "claude",
    timeout: float = 600.0,
    *, reasoning_effort: Optional[str] = None,
    on_result: Optional[LLMHook] = None, on_event: Optional[LLMHook] = None,
    on_turn: Optional[LLMHook] = None,
) -> LLMClient:
    """Use Claude Code for synthesis.

    Requires ``--safe-mode`` support. This skips user customizations and
    keeps CLI authentication. Built-in tools and session persistence are
    disabled. No SDK is required.
    """
    if reasoning_effort is not None:
        raise RouterConfigurationError("This Claude CLI client does not support reasoning_effort.")
    return _cli_llm_client("claude", cli, model, timeout)


def codex_cli_llm_client(
    model: Optional[str] = None,
    cli: str = "codex",
    timeout: float = 600.0,
    *, reasoning_effort: Optional[str] = None,
    on_result: Optional[LLMHook] = None, on_event: Optional[LLMHook] = None,
    on_turn: Optional[LLMHook] = None,
) -> LLMClient:
    """Use ``codex exec`` for synthesis.

    Run ``codex login`` for ChatGPT subscription access. The CLI must support
    ``--ignore-user-config`` and ``--ephemeral``. No SDK is required.
    Shell commands, web search, and project instructions are disabled.
    """
    return _cli_llm_client("codex", cli, model, timeout, reasoning_effort)


def codex_sdk_llm_client(
    model: Optional[str] = None, cli: Optional[str] = None, timeout: float = 600.0, *,
    reasoning_effort: Optional[str] = None,
    on_result: Optional[LLMHook] = None, on_event: Optional[LLMHook] = None,
    on_turn: Optional[LLMHook] = None,
) -> LLMClient:
    """Use the Codex SDK with an installed CLI, or its bundled runtime.

    Automatic selection checks publisher identity and version. Set ``cli`` to
    trust a specific executable name or path instead.
    ``timeout`` bounds response waiting after turn startup. Expiry closes the
    runtime. Startup and user callbacks must return on their own.
    """
    import tempfile
    from threading import TIMEOUT_MAX, Event, Timer
    from ._codex import collect_turn, find_cli

    if not 0 < timeout <= TIMEOUT_MAX:
        raise ValueError("timeout must be a positive finite number of seconds within the platform's timer limit")

    try:
        from openai_codex import ApprovalMode, Codex, CodexConfig, CodexError, Sandbox
    except ImportError as exc:
        raise RouterConfigurationError(
            "codex_sdk_llm_client requires the 'codex-sdk' extra: "
            "pip install strot-ai[codex-sdk]"
        ) from exc

    cli_path = find_cli(cli) if cli is not None else None
    effort = reasoning_effort if reasoning_effort is not None else (
        "none" if model == "gpt-6-luna" else None)

    def _call(prompt: str) -> str:
        hook_error = None
        expired = Event()

        def notify(hook: LLMHook, value: Any) -> None:
            nonlocal hook_error
            try:
                hook(value)
            except Exception as exc:
                hook_error = exc
                raise

        try:
            with tempfile.TemporaryDirectory(prefix="strot-synthesis-") as cwd:
                config = CodexConfig(
                    codex_bin=cli_path or find_cli(), cwd=cwd, experimental_api=False,
                    config_overrides=("project_doc_max_bytes=0", "features.shell_tool=false",
                                      'web_search="disabled"'),
                )
                with Codex(config) as client:
                    turn = client.thread_start(
                        model=model, model_provider="openai", cwd=cwd, ephemeral=True,
                        developer_instructions="",
                        sandbox=Sandbox.read_only, approval_mode=ApprovalMode.deny_all,
                    ).turn(prompt, **({"effort": effort} if effort is not None else {}))
                    if on_turn is not None:
                        notify(on_turn, turn)

                    def expire() -> None:
                        expired.set()
                        client.close()

                    timer = Timer(timeout, expire)
                    timer.daemon = True
                    timer.start()
                    try:
                        if on_event is None:
                            result = turn.run()
                        else:
                            result = collect_turn(turn, lambda event: notify(on_event, event))
                    finally:
                        timer.cancel()
                        timer.join()
                    if expired.is_set():
                        raise TimeoutError(f"response wait exceeded {timeout:g} seconds")
        except Exception as exc:
            if exc is hook_error:
                raise
            if expired.is_set():
                raise CompilationError(
                    f"Codex SDK synthesizer timed out after {timeout:g} seconds of response waiting.",
                    attempts=0,
                ) from exc
            if isinstance(exc, CodexError):
                raise
            if isinstance(exc, (OSError, RuntimeError, UnicodeError)):
                raise CompilationError(f"Codex SDK synthesizer failed: {exc}", attempts=0) from exc
            raise
        if on_result is not None:
            on_result(result)
        if result.status.value != "completed":
            raise CompilationError(
                f"Codex SDK synthesizer stopped with {result.status.value}.", attempts=0,
            )
        if not result.final_response or not result.final_response.strip():
            raise CompilationError("Codex SDK synthesizer returned no text.", attempts=0)
        return result.final_response

    return _call


def _cli_llm_client(
    backend: str, cli: str, model: Optional[str], timeout: float,
    reasoning_effort: Optional[str] = None,
) -> LLMClient:
    import shutil
    import subprocess
    import tempfile

    cli_path = shutil.which(cli)
    if cli_path is None:
        raise RouterConfigurationError(
            f"Cannot find the {cli!r} CLI on PATH. Install {backend} and sign in."
        )
    cli_path = str(Path(cli_path).absolute())
    effort = reasoning_effort if reasoning_effort is not None else (
        "none" if model == "gpt-6-luna" else None)

    def _call(prompt: str) -> str:
        # Use stdin for context and skip project instructions.
        with tempfile.TemporaryDirectory(prefix="strot-synthesis-") as cwd:
            output = Path(cwd) / "response.txt"
            if backend == "codex":
                command = [
                    cli_path, "exec", "--ignore-user-config", "--ephemeral",
                    "-c", "project_doc_max_bytes=0", "-c", "features.shell_tool=false",
                    "-c", 'web_search="disabled"',
                    "--sandbox", "read-only", "--skip-git-repo-check",
                    "--color", "never", "--output-last-message", str(output),
                ]
            else:
                command = [
                    cli_path, "--safe-mode", "-p", "--output-format", "text", "--tools", "",
                    "--no-session-persistence",
                ]
            if backend == "codex" and effort is not None:
                command += ["-c", "model_reasoning_effort=" + json.dumps(effort)]
            if model:
                command += ["--model", model]
            try:
                proc = subprocess.run(
                    command, input=prompt, capture_output=True, text=True,
                    encoding="utf-8", timeout=timeout, cwd=cwd,
                )
                if proc.returncode != 0:
                    detail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()[:500]
                    raise CompilationError(
                        f"{backend} CLI synthesizer exited {proc.returncode}: "
                        f"{detail}", attempts=0,
                    )
                response = output.read_text(encoding="utf-8") if backend == "codex" else proc.stdout
            except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
                raise CompilationError(
                    f"{backend} CLI synthesizer failed: {exc}", attempts=0,
                ) from exc
            if not response.strip():
                raise CompilationError(f"{backend} CLI synthesizer returned no text.", attempts=0)
            return response

    return _call


def _sample_records(sample: Any) -> List[Dict[str, Any]]:
    """Normalize any supported sample shape into a list of dict records."""
    from .schema import SchemaProfiler

    columns = SchemaProfiler()._normalize(sample)
    length = max(len(values) for values in columns.values())
    return [
        {name: (values[i] if i < len(values) else None) for name, values in columns.items()}
        for i in range(length)
    ]


def handler_name(action_key: str) -> str:
    """Map an action-registry key to its compiled function name."""
    sanitized = re.sub(r"\W+", "_", action_key).strip("_").lower()
    if not sanitized:
        raise CompilationError(f"Action key {action_key!r} cannot form a function name.", attempts=0)
    return f"execute_{sanitized}"


@dataclass(frozen=True)
class CompiledAsset:
    """A self-contained, self-routing production artifact."""

    source: str
    nlu_model: str
    runtime_kind: str  # "local" | "hosted"
    goal: str
    schema: DatasetSchema
    attempts: int
    calibration: Optional[Dict[str, Any]] = None  #: fitted stats (None = uncalibrated)
    label_search: Optional[Dict[str, Any]] = None  #: measured-labeling scores (None = not measured)
    model_search: Optional[Dict[str, Any]] = None  #: zero-shot vs distilled scores (None = no distill)
    policy: Optional[Dict[str, Any]] = None  #: fitted target-accuracy dispatch policy

    def save(self, path: Union[str, Path] = "compiled_pipeline.py") -> Path:
        """Persist the asset as an independent Python file."""
        target = Path(path)
        target.write_text(self.source, encoding="utf-8")
        return target


class StrotCompiler:
    """Orchestrates schema introspection, LLM code generation, and assembly
    of the standalone runtime asset."""

    def __init__(
        self,
        nlu_model: str = DEFAULT_NLU_MODEL,
        llm_client: Optional[LLMClient] = None,
        max_retries: int = 3,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        self.nlu_model = nlu_model
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

    @property
    def runtime_kind(self) -> str:
        if self.nlu_model.lower().startswith(HOSTED_PROVIDER_PREFIXES):
            return "hosted"
        return "local"

    # ------------------------------------------------------------------ #
    # Phase 2: prompt scaffolding                                         #
    # ------------------------------------------------------------------ #

    def build_synthesis_prompt(
        self,
        schema: DatasetSchema,
        goal: str,
        action_registry: Mapping[str, str],
    ) -> str:
        """Construct the strict, sectioned synthesizer prompt."""
        contract_lines = [
            f"- `def {handler_name(key)}(record: dict) -> dict:` handles action "
            f"{key!r} ({desc})"
            for key, desc in action_registry.items()
        ]
        contract_lines += [
            "- `ROUTING_LABELS: Dict[str, str]` module-level literal dict mapping "
            f"EVERY action key ({', '.join(map(repr, action_registry))}) to a "
            "short routing label: 1-4 lowercase words, mutually distinct, using "
            "vocabulary that appears in the records themselves. The runtime "
            "decision model sees ONLY these labels as its options - short, "
            "discriminative labels dramatically outperform long descriptions.",
            "- `def route_context(record: dict) -> str:` projects ONLY the "
            "fields relevant to choosing an action into a compact string for "
            "the decision model; drop irrelevant or noisy fields.",
        ]
        return "\n".join(
            [
                "You are the STROT synthesizer. Generate production Python code.",
                "",
                "## VERIFIED DATASET SCHEMA",
                schema.to_json(),
                "",
                "## FUNCTIONAL GOAL",
                goal,
                "",
                "## RUNTIME NLU MODEL",
                f"The routing decision at runtime is made by {self.nlu_model!r} "
                f"({self.runtime_kind} runtime). Do NOT generate routing code; "
                "it is compiled in separately.",
                "",
                "## OUTPUT CONTRACT",
                "Emit exactly one ```python code block containing ONLY these "
                "top-level functions, fully implemented against the schema above:",
                *contract_lines,
                "",
                "Rules: pure standard library only; comprehensive exception "
                "handling inside each function; full type hints; no I/O, no "
                "network calls, no imports of torch/transformers/openai/julia; "
                "return a JSON-serializable dict.",
            ]
        )

    def _synthesize_paths(
        self,
        prompt: str,
        expected_functions: List[str],
        label_keys: Optional[List[str]] = None,
    ) -> "tuple[str, int]":
        return self._synthesize_with(
            prompt,
            lambda code: self._validate_candidate(code, expected_functions, label_keys),
        )

    def _synthesize_with(
        self, prompt: str, validate: Callable[[str], Optional[str]]
    ) -> "tuple[str, int]":
        if self.llm_client is None:
            raise CompilationError(
                "No llm_client configured. Pass a callable (e.g. "
                "strot.compiler.openai_llm_client()) to StrotCompiler.",
                attempts=0,
            )
        current_prompt = prompt
        last_error: Optional[str] = None
        candidate = ""
        for attempt in range(1, self.max_retries + 1):
            raw = self.llm_client(current_prompt)
            candidate = self._extract_code(raw)
            error = validate(candidate)
            if error is None:
                return candidate, attempt
            last_error = error
            current_prompt = self._build_repair_prompt(prompt, candidate, error)
        raise CompilationError(
            f"Synthesizer failed after {self.max_retries} attempts; "
            f"last validation error: {last_error}",
            attempts=self.max_retries,
            last_error=last_error,
        )

    @staticmethod
    def _extract_code(raw: str) -> str:
        match = _CODE_FENCE_RE.search(raw)
        return (match.group(1) if match else raw).strip()

    @staticmethod
    def _validate_candidate(
        code: str,
        expected_functions: List[str],
        label_keys: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return a human-readable validation error, or None if the code is valid."""
        if not code:
            return "The response contained no code."
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError at line {exc.lineno}: {exc.msg}"
        defined = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        missing = [name for name in expected_functions if name not in defined]
        if missing:
            return f"Missing required function definitions: {', '.join(missing)}"
        if label_keys is not None:
            error = StrotCompiler._validate_routing_labels(tree, label_keys)
            if error:
                return error
        # Security lint: generated code must be pure computation. Block ML
        # packages AND stdlib escape hatches to I/O, network, process control,
        # and dynamic execution. NOTE: this is validation, not a sandbox -
        # compiling fully untrusted intent requires external isolation.
        forbidden = {
            "torch", "transformers", "openai", "julia", "huggingface_hub",
            "os", "sys", "subprocess", "socket", "urllib", "http", "ftplib",
            "requests", "shutil", "pathlib", "importlib", "ctypes", "pickle",
            "multiprocessing", "threading", "tempfile", "io",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            bad = names & forbidden
            if bad:
                return f"Forbidden imports in execution paths: {', '.join(sorted(bad))}"
        banned_calls = {"eval", "exec", "open", "__import__", "compile", "input"}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in banned_calls
            ):
                return (
                    f"Forbidden call {node.func.id!r} in execution paths: "
                    "generated code must be pure computation (no I/O, no "
                    "dynamic execution)."
                )
        return None

    @staticmethod
    def _validate_routing_labels(tree: ast.Module, label_keys: List[str]) -> Optional[str]:
        """Check the module-level ROUTING_LABELS literal against the registry."""
        value_node = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ROUTING_LABELS" for t in node.targets
            ):
                value_node = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "ROUTING_LABELS"
            ):
                value_node = node.value
        if value_node is None:
            return "Missing required module-level constant ROUTING_LABELS."
        try:
            labels = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return "ROUTING_LABELS must be a literal dict of str -> str."
        if not isinstance(labels, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v.strip()
            for k, v in labels.items()
        ):
            return "ROUTING_LABELS must be a literal dict of str -> non-empty str."
        if set(labels) != set(label_keys):
            return (
                "ROUTING_LABELS keys must exactly match the action keys: "
                f"{', '.join(label_keys)}"
            )
        return None

    @staticmethod
    def _build_repair_prompt(original_prompt: str, failed_code: str, error: str) -> str:
        return "\n".join(
            [
                original_prompt,
                "",
                "## PREVIOUS ATTEMPT (REJECTED)",
                "```python",
                failed_code,
                "```",
                "",
                "## VALIDATION ERROR",
                error,
                "",
                "Fix the error and emit the corrected code block. Output only code.",
            ]
        )

    # ------------------------------------------------------------------ #
    # Assembly of the standalone asset                                    #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Golden-record promotion gate                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_candidate_module(source: str) -> Any:
        """Exec a candidate asset in-memory for the golden gate.

        A load failure here is an environment problem (missing decision
        runtime on the compile box), never the synthesizer's fault - raise
        instead of burning repair-loop retries.
        """
        import sys

        module = _types.ModuleType("strot_golden_gate")
        # Registered so stdlib machinery (e.g. dataclasses) can resolve the
        # module during exec; dropped again by the gate methods when done.
        sys.modules["strot_golden_gate"] = module
        try:
            exec(compile(source, "<strot-golden-gate>", "exec"), module.__dict__)
        except Exception as exc:
            sys.modules.pop("strot_golden_gate", None)
            raise RouterConfigurationError(
                "Golden-record gate could not load the candidate asset; the "
                f"decision runtime must be installed on the compile box: {exc}"
            ) from exc
        return module

    @staticmethod
    def validate_golden_run(golden: Any, action_registry: Mapping[str, str]) -> None:
        if not isinstance(golden, list) or not golden:
            raise RouterConfigurationError("golden must be a non-empty list.")
        for i, case in enumerate(golden):
            if (
                not isinstance(case, Mapping)
                or not isinstance(case.get("record"), Mapping)
                or case.get("expect") not in action_registry
            ):
                raise RouterConfigurationError(
                    f"golden[{i}] must be {{'record': {{...}}, 'expect': <action key>}} "
                    f"with expect one of {list(action_registry)}."
                )

    def _test_golden_pipeline(self, source: str, golden: List[Mapping[str, Any]]) -> Optional[str]:
        import sys

        module = self._load_candidate_module(source)
        try:
            pipeline = module.CompiledPipeline()
        except Exception as exc:
            raise RouterConfigurationError(
                f"Golden-record gate: decision runtime failed to initialize: {exc}"
            ) from exc
        failures = []
        for i, case in enumerate(golden):
            expect = case["expect"]
            try:
                outcome = pipeline.process(dict(case["record"]))
            except module.FallbackException as exc:
                failures.append(
                    f"golden[{i}] fell to fallback (confidence {exc.confidence:.2f}) "
                    f"but expected action {expect!r}"
                )
                continue
            if outcome["action"] != expect:
                failures.append(
                    f"golden[{i}] decided {outcome['action']!r} (confidence "
                    f"{outcome['confidence']:.2f}) but expected {expect!r}"
                )
        sys.modules.pop("strot_golden_gate", None)
        if failures:
            return (
                "Golden-record promotion gate failed - adjust ROUTING_LABELS and "
                "route_context so these known answers hold:\n" + "\n".join(failures)
            )
        return None

    # ------------------------------------------------------------------ #
    # Compile-time calibration (teacher labels -> fitted temperature + q)  #
    # ------------------------------------------------------------------ #

    def build_teacher_prompt(
        self,
        goal: str,
        action_registry: Mapping[str, str],
        records: List[Dict[str, Any]],
    ) -> str:
        """Ask the synthesizer LLM to act as a labeling teacher, once."""
        lines = [
            "You are the STROT teacher. Label each record with the single "
            "best action for the goal.",
            "",
            "## GOAL",
            goal,
            "",
            "## ACTIONS",
        ]
        lines += [f"- {key}: {desc}" for key, desc in action_registry.items()]
        lines += ["", "## RECORDS (one JSON object per line, in order)"]
        lines += [
            f"{i}: {json.dumps(record, default=str, sort_keys=True)}"
            for i, record in enumerate(records)
        ]
        lines += [
            "",
            "## OUTPUT CONTRACT",
            "Emit exactly one ```python code block containing ONLY one "
            "module-level literal assignment:",
            f"- `LABELS: List[str]` with exactly {len(records)} entries; the "
            "i-th entry is the action key for record i, drawn only from: "
            + ", ".join(map(repr, action_registry)) + ".",
            "",
            "No other statements, functions, or imports.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _validate_teacher_labels(
        code: str, n_records: int, action_registry: Mapping[str, str]
    ) -> Optional[str]:
        if not code:
            return "The response contained no code."
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError at line {exc.lineno}: {exc.msg}"
        value_node = None
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(x, ast.Name) and x.id == "LABELS" for x in targets):
                    value_node = node.value
                    continue
            return "Emit ONLY the LABELS literal assignment."
        if value_node is None:
            return "Missing required module-level literal LABELS."
        try:
            labels = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return "LABELS must be a literal list of action-key strings."
        if not isinstance(labels, list) or len(labels) != n_records:
            return f"LABELS must contain exactly {n_records} entries."
        bad = [x for x in labels if x not in action_registry]
        if bad:
            return f"LABELS contains unknown action keys: {bad[:5]}"
        return None

    def _collect_distributions(
        self,
        source: str,
        records: List[Dict[str, Any]],
        action_registry: Mapping[str, str],
        question: Optional[str],
        goal: str,
    ) -> List[List[float]]:
        import sys

        module = self._load_candidate_module(source)
        try:
            router = module._RuntimeRouter()
            routing_question = question or (
                f"Which execution path best achieves the goal: {goal}?"
            )
            distributions = []
            for record in records:
                decision = router.select(
                    module.route_context(dict(record)),
                    routing_question,
                    module._ROUTING_OPTIONS,
                )
                distributions.append(
                    [
                        max(decision.distribution.get(opt, 0.0), 1e-12)
                        for opt in module._ROUTING_OPTIONS
                    ]
                )
        finally:
            sys.modules.pop("strot_golden_gate", None)
        return distributions

    def _fit_calibration(
        self,
        source: str,
        records: List[Dict[str, Any]],
        labels: List[str],
        action_registry: Mapping[str, str],
        question: Optional[str],
        goal: str,
        target_risk: float,
    ) -> Dict[str, Any]:
        """Run the promoted artifact over teacher-labeled records; fit T and q."""
        import sys

        from . import calibrate

        module = self._load_candidate_module(source)
        try:
            router = module._RuntimeRouter()
            action_keys = list(action_registry)
            routing_question = question or (
                f"Which execution path best achieves the goal: {goal}?"
            )
            distributions: List[List[float]] = []
            true_indices: List[int] = []
            for record, label in zip(records, labels):
                context = module.route_context(dict(record))
                decision = router.select(
                    context, routing_question, module._ROUTING_OPTIONS
                )
                distributions.append(
                    [
                        max(decision.distribution.get(opt, 0.0), 1e-12)
                        for opt in module._ROUTING_OPTIONS
                    ]
                )
                true_indices.append(action_keys.index(label))
        finally:
            sys.modules.pop("strot_golden_gate", None)
        return calibrate.fit_split_calibration(
            distributions, true_indices, alpha=target_risk
        )

    def _resolve_compile_checkpoint(self) -> str:
        import os

        override = os.environ.get("STROT_NLU_CHECKPOINT")
        if override:
            return override
        if Path(self.nlu_model).is_dir():
            return self.nlu_model
        from huggingface_hub import snapshot_download

        return snapshot_download(self.nlu_model)

    def _measure_labelings(
        self,
        source: str,
        records: List[Dict[str, Any]],
        true_indices: List[int],
        action_registry: Mapping[str, str],
        question: Optional[str],
        goal: str,
    ) -> "tuple[str, Dict[str, str], Dict[str, float]]":
        """Score candidate labelings on the teacher-labeled sample; keep the best.

        Candidates: the synthesized ROUTING_LABELS and the bare registry keys.
        Measurement replaces guessing - benchmarks showed synthesis wins when
        integration is the bottleneck and loses when the keys are already
        optimal; scoring both captures both cases. Ties keep the synthesized
        labeling (listed first).
        """
        import sys

        module = self._load_candidate_module(source)
        try:
            router = module._RuntimeRouter()
            routing_question = question or (
                f"Which execution path best achieves the goal: {goal}?"
            )
            candidates: Dict[str, Dict[str, str]] = {
                "synthesized": {k: module.ROUTING_LABELS[k] for k in action_registry},
                "registry_keys": {
                    k: k.replace("_", " ").strip() for k in action_registry
                },
            }
            if candidates["registry_keys"] == candidates["synthesized"]:
                return "synthesized", candidates["synthesized"], {"synthesized": -1.0}
            scores: Dict[str, float] = {}
            for name, labels in candidates.items():
                options = [labels[key] for key in action_registry]
                correct = 0
                for record, true_index in zip(records, true_indices):
                    decision = router.select(
                        module.route_context(dict(record)), routing_question, options
                    )
                    if decision.index == true_index:
                        correct += 1
                scores[name] = round(correct / len(records), 4)
        finally:
            sys.modules.pop("strot_golden_gate", None)
        winner = max(scores, key=lambda k: (scores[k], k == "synthesized"))
        return winner, candidates[winner], scores

    def compile(
        self,
        sample: Any,
        goal: str,
        action_registry: Mapping[str, str],
        question: Optional[str] = None,
        golden: Optional[List[Mapping[str, Any]]] = None,
        target_risk: Optional[float] = None,
        measure_labels: bool = False,
        distill: bool = False,
        target_accuracy: Optional[float] = None,
    ) -> CompiledAsset:
        """Run the full compile-time pipeline and return the production asset.

        Args:
            sample: A representative dataset sample (records, columns, or a
                DataFrame-like object) used for schema introspection.
            goal: The user's functional goal in natural language.
            action_registry: Ordered mapping of action key -> description;
                list position defines the routing option index at runtime.
            question: Optional override for the dynamic selection question
                baked into the asset.
            golden: Optional known-answer records
                (``[{"record": {...}, "expect": "<action key>"}, ...]``). A
                candidate asset is executed against them inside the repair
                loop and only promoted if every expectation holds. Requires
                the decision runtime on the compile box.
            target_risk: Optional miscoverage level ``alpha`` (e.g. 0.05).
                When set, the LLM teacher-labels the sample once, a
                temperature and split-conformal quantile are fitted on the
                promoted artifact's own distributions, and the artifact
                dispatches only on singleton prediction sets. Guarantee is
                MARGINAL: the rate of wrong dispatches over all records is
                <= alpha under exchangeability (the error rate conditional
                on dispatching is bounded by alpha / dispatch_rate, reported
                in the calibration metadata). Requires the decision runtime
                on the compile box.
            measure_labels: Score candidate labelings (the synthesized labels
                vs the bare registry keys) on the teacher-labeled sample and
                bake the winner - measurement instead of guessing. Needs the
                teacher labels, so it shares the one teacher call with
                target_risk (free when calibrating; otherwise triggers it).
        """
        if target_risk is not None and not 0 < target_risk < 1:
            raise RouterConfigurationError("target_risk must be in (0, 1).")
        if target_accuracy is not None and target_risk is not None:
            raise RouterConfigurationError(
                "Set either target_risk (conformal guarantee) or "
                "target_accuracy (empirical dispatch policy), not both."
            )
        if target_accuracy is not None and not 0 < target_accuracy <= 1:
            raise RouterConfigurationError("target_accuracy must be in (0, 1].")
        if distill and self.runtime_kind != "local":
            raise RouterConfigurationError(
                "distill requires a local decision model (the encoder supplies "
                f"the embeddings); got {self.nlu_model!r}."
            )
        if not action_registry:
            raise CompilationError("action_registry must contain at least one action.", attempts=0)
        if golden is not None:
            self.validate_golden_run(golden, action_registry)
        schema = introspect_schema(sample)
        expected = [handler_name(key) for key in action_registry]
        if len(set(expected)) != len(expected):
            raise CompilationError(
                "Action keys collide after sanitization; use distinct keys.", attempts=0
            )
        expected = expected + ["route_context"]
        prompt = self.build_synthesis_prompt(schema, goal, action_registry)
        promoted: Dict[str, str] = {}

        def validate(code: str) -> Optional[str]:
            error = self._validate_candidate(code, expected, list(action_registry))
            if error:
                return error
            source = self._assemble(schema, goal, dict(action_registry), code, question)
            try:
                ast.parse(source)
            except SyntaxError as exc:  # template regression guard
                return f"Assembled asset failed syntax validation: {exc}"
            if golden:
                error = self._test_golden_pipeline(source, golden)
                if error:
                    return error
            promoted["source"] = source
            promoted["code"] = code
            return None

        _, attempts = self._synthesize_with(prompt, validate)
        source = promoted["source"]
        final_code = promoted["code"]

        calibration: Optional[Dict[str, Any]] = None
        label_search: Optional[Dict[str, Any]] = None
        model_search: Optional[Dict[str, Any]] = None
        runtime_override: Optional[str] = None
        if target_risk is not None or measure_labels or distill or target_accuracy is not None:
            records = _sample_records(sample)
            if distill:
                from . import distill as distill_mod

                if len(records) < distill_mod.MIN_DISTILL_RECORDS:
                    raise RouterConfigurationError(
                        f"distill needs at least {distill_mod.MIN_DISTILL_RECORDS} "
                        f"sample records (got {len(records)}); raise sample_size."
                    )
            # Chunked teacher labeling: one call per batch keeps the exact-
            # count contract reliable at any sample size.
            labels: List[str] = []
            _CHUNK = 50
            for _start in range(0, len(records), _CHUNK):
                _chunk = records[_start:_start + _CHUNK]
                teacher_prompt = self.build_teacher_prompt(
                    goal, action_registry, _chunk
                )
                labels_code, _ = self._synthesize_with(
                    teacher_prompt,
                    lambda code, _n=len(_chunk): self._validate_teacher_labels(
                        code, _n, action_registry
                    ),
                )
                labels += ast.literal_eval(
                    next(
                        node.value
                        for node in ast.parse(labels_code).body
                        if isinstance(node, (ast.Assign, ast.AnnAssign))
                    )
                )
            action_keys = list(action_registry)
            true_indices = [action_keys.index(label) for label in labels]

            # Fold hygiene: with distill, teacher-labeled records are split so
            # training, judging (model/label search), and calibration never
            # share data. Without distill, judging and calibration use all
            # records (back-compatible).
            if distill:
                from . import distill as distill_mod

                train_ix, measure_ix, calib_ix = distill_mod.three_way_split(
                    range(len(records))
                )
            else:
                train_ix, measure_ix, calib_ix = [], list(range(len(records))), list(
                    range(len(records))
                )
            records_measure = [records[i] for i in measure_ix]
            true_measure = [true_indices[i] for i in measure_ix]
            records_calib = [records[i] for i in calib_ix]
            labels_calib = [labels[i] for i in calib_ix]

            if measure_labels:
                winner, winner_labels, scores = self._measure_labelings(
                    source, records_measure, true_measure, action_registry, question, goal
                )
                label_search = {"scores": scores, "winner": winner}
                if winner != "synthesized":
                    override = (
                        "\n\n# Measured label search winner "
                        "(overrides synthesized labels)\n"
                        f"ROUTING_LABELS = {winner_labels!r}"
                    )
                    candidate_code = final_code + override
                    candidate_source = self._assemble(
                        schema, goal, dict(action_registry), candidate_code, question
                    )
                    golden_error = (
                        self._test_golden_pipeline(candidate_source, golden)
                        if golden
                        else None
                    )
                    if golden_error:
                        # Goldens are a regression gate: a measured winner that
                        # breaks them is rejected, keeping the synthesized labels.
                        label_search["winner_rejected_by_golden"] = True
                        label_search["winner"] = "synthesized"
                    else:
                        final_code = candidate_code
                        source = candidate_source

            if distill:
                from . import distill as distill_mod

                checkpoint = self._resolve_compile_checkpoint()
                embed = distill_mod.make_encoder_embedder(checkpoint)
                module = self._load_candidate_module(source)
                try:
                    contexts = [
                        module.route_context(dict(record)) for record in records
                    ]
                    routing_question = question or (
                        f"Which execution path best achieves the goal: {goal}?"
                    )
                    router = module._RuntimeRouter()
                    zero_shot_correct = 0
                    for i in measure_ix:
                        decision = router.select(
                            contexts[i], routing_question, module._ROUTING_OPTIONS
                        )
                        if decision.index == true_indices[i]:
                            zero_shot_correct += 1
                finally:
                    import sys as _sys

                    _sys.modules.pop("strot_golden_gate", None)
                zero_shot_acc = round(zero_shot_correct / len(measure_ix), 4)

                vectors = embed([contexts[i] for i in train_ix + measure_ix])
                train_vectors = vectors[: len(train_ix)]
                measure_vectors = vectors[len(train_ix):]
                head_w, head_b = distill_mod.train_head(
                    train_vectors,
                    [true_indices[i] for i in train_ix],
                    n_classes=len(action_keys),
                )
                distilled_acc = distill_mod.head_accuracy(
                    measure_vectors, true_measure, head_w, head_b
                )
                model_search = {
                    "zero_shot": zero_shot_acc,
                    "distilled": distilled_acc,
                    "n_train": len(train_ix),
                    "n_measure": len(measure_ix),
                    "winner": "distilled" if distilled_acc > zero_shot_acc else "zero_shot",
                }
                if model_search["winner"] == "distilled":
                    head_code = (
                        "\n\n# Compile-time-distilled head (train fold only)\n"
                        f"HEAD_WEIGHTS = {head_w!r}\n"
                        f"HEAD_BIAS = {head_b!r}"
                    )
                    candidate_code = final_code + head_code
                    candidate_source = self._assemble(
                        schema, goal, dict(action_registry), candidate_code,
                        question, runtime="distilled",
                    )
                    golden_error = (
                        self._test_golden_pipeline(candidate_source, golden)
                        if golden
                        else None
                    )
                    if golden_error:
                        model_search["winner_rejected_by_golden"] = True
                        model_search["winner"] = "zero_shot"
                    else:
                        final_code = candidate_code
                        source = candidate_source
                        runtime_override = "distilled"

            if target_risk is not None:
                calibration = self._fit_calibration(
                    source, records_calib, labels_calib, action_registry,
                    question, goal, target_risk,
                )
                source = self._assemble(
                    schema,
                    goal,
                    dict(action_registry),
                    final_code,
                    question,
                    temperature=float(calibration["temperature"]),
                    conformal_q=float(calibration["conformal_quantile"]),
                    calibration=calibration,
                    runtime=runtime_override,
                )
                if golden:
                    error = self._test_golden_pipeline(source, golden)
                    if error:
                        raise CompilationError(
                            "Calibrated artifact no longer satisfies the golden "
                            f"records at target_risk={target_risk} (the conformal "
                            "gate abstains on them). Relax target_risk or revisit "
                            f"the goldens:\n{error}",
                            attempts=attempts,
                            last_error=error,
                        )

            if target_accuracy is not None:
                from . import calibrate as calibrate_mod

                true_calib = [
                    action_keys.index(label) for label in labels_calib
                ]
                dists = self._collect_distributions(
                    source, records_calib, action_registry, question, goal
                )
                # Fold hygiene inside the policy fit: corrections on half A,
                # the dispatch cutoff on disjoint half B.
                a_d, a_t = dists[0::2], true_calib[0::2]
                b_d, b_t = dists[1::2], true_calib[1::2]
                if not b_d:
                    a_d, a_t, b_d, b_t = dists, true_calib, dists, true_calib
                temperature = (
                    calibrate_mod.fit_temperature(a_d, a_t) if len(a_d) >= 4 else 1.0
                )
                class_bias = (
                    calibrate_mod.fit_class_bias(a_d, a_t, temperature)
                    if len(a_d) >= 4
                    else None
                )
                corrected = [
                    calibrate_mod.apply_correction(d, temperature, class_bias)
                    for d in b_d
                ]
                confidences = [max(p) for p in corrected]
                correct_flags = [
                    max(range(len(p)), key=p.__getitem__) == t
                    for p, t in zip(corrected, b_t)
                ]
                cutoff = calibrate_mod.selective_cutoff(
                    confidences, correct_flags, target_accuracy
                )
                kept = [
                    ok for conf, ok in zip(confidences, correct_flags)
                    if cutoff is not None and conf >= cutoff
                ]
                policy = {
                    "target_accuracy": target_accuracy,
                    "temperature": temperature,
                    "class_bias": class_bias,
                    "dispatch_cutoff": cutoff,
                    "n_fit": len(a_d),
                    "n_cutoff_fold": len(b_d),
                    "tier1_coverage_est": (
                        round(len(kept) / len(b_d), 4) if b_d else 0.0
                    ),
                    "tier1_accuracy_est": (
                        round(sum(kept) / len(kept), 4) if kept else None
                    ),
                    "note": (
                        "empirical policy fit on held-out labeled records; "
                        "NOT a finite-sample guarantee"
                    ),
                }
                source = self._assemble(
                    schema,
                    goal,
                    dict(action_registry),
                    final_code,
                    question,
                    temperature=temperature,
                    class_bias=class_bias,
                    dispatch_cutoff=cutoff if cutoff is not None else 1.1,
                    policy=policy,
                    runtime=runtime_override,
                )
                if golden:
                    error = self._test_golden_pipeline(source, golden)
                    if error:
                        raise CompilationError(
                            "Policy-fitted artifact no longer satisfies the "
                            f"golden records at target_accuracy={target_accuracy}:"
                            f"\n{error}",
                            attempts=attempts,
                            last_error=error,
                        )
            else:
                policy = None

        if target_risk is None and target_accuracy is None:
            policy = None

        return CompiledAsset(
            source=source,
            nlu_model=self.nlu_model,
            runtime_kind=runtime_override or self.runtime_kind,
            goal=goal,
            schema=schema,
            attempts=attempts,
            calibration=calibration,
            label_search=label_search,
            model_search=model_search,
            policy=policy,
        )

    @staticmethod
    def validate_questions_config(questions: Mapping[str, Any]) -> None:
        """Validate a typed-question config before any synthesis."""
        if not isinstance(questions, Mapping) or not questions:
            raise RouterConfigurationError("questions must be a non-empty mapping.")
        for name, spec in questions.items():
            if not isinstance(spec, Mapping) or spec.get("type") not in (
                "boolean", "score", "choice",
            ):
                raise RouterConfigurationError(
                    f"question {name!r}: type must be 'boolean', 'score', or 'choice'."
                )
            if spec["type"] == "score":
                minimum, maximum = spec.get("min"), spec.get("max")
                if (
                    type(minimum) is not int
                    or type(maximum) is not int
                    or not 2 <= (maximum - minimum + 1) <= 20
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: score needs int min < max spanning "
                        "2-20 levels (the decision model's option range)."
                    )
                if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: score requires instructions anchoring the scale."
                    )
            elif spec["type"] == "boolean":
                if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: boolean requires instructions."
                    )
            else:  # choice
                options = spec.get("options")
                if (
                    not isinstance(options, list)
                    or not 2 <= len(options) <= 20
                    or len(set(options)) != len(options)
                    or not all(isinstance(o, str) and o for o in options)
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: choice needs 2-20 unique non-empty options."
                    )

    def build_decide_prompt(self, schema: DatasetSchema, questions: Mapping[str, Any]) -> str:
        """Construct the decide-mode prompt: expand a typed-question config for the model."""
        return "\n".join(
            [
                "You are the STROT synthesizer. Expand a typed-question config "
                "for a small local decision model - you configure, you do not "
                "write programs.",
                "",
                "## VERIFIED DATASET SCHEMA",
                schema.to_json(),
                "",
                "## DECISION CONFIG (typed questions)",
                json.dumps(questions, indent=2),
                "",
                "## OUTPUT CONTRACT",
                "Emit exactly one ```python code block containing EXACTLY two "
                "top-level definitions and nothing else:",
                "- `QUESTIONS: Dict[str, Any]` literal with the SAME question "
                "names. Per question: keep \"type\" and \"instructions\" (write "
                "crisp instructions if the config omits them). For 'choice' add "
                "\"criteria\": a dict mapping EVERY option (exact strings) to a "
                "short discriminative description using vocabulary from the "
                "records. For 'score' add \"min\" (copy the config int) and "
                "\"rubric\": a list of exactly (max - min + 1) short level "
                "labels, in order from min to max, grounded in the "
                "instructions' anchors. For 'boolean' add \"criteria\": "
                "{\"false\": <short negative-case description>, \"true\": "
                "<short positive-case description>}.",
                "- `def route_context(record: dict) -> str:` projecting ONLY "
                "the fields relevant to answering the questions into a compact "
                "string.",
                "",
                "Rules: short, mutually distinct labels dramatically outperform "
                "long ones; no imports, no other statements.",
            ]
        )

    def _validate_decide_candidate(
        self, code: str, questions: Mapping[str, Any]
    ) -> Optional[str]:
        error = self._validate_candidate(code, ["route_context"])
        if error:
            return error
        tree = ast.parse(code)
        value_node = None
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "QUESTIONS" for t in targets):
                    value_node = node.value
        if value_node is None:
            return "Missing required module-level literal QUESTIONS."
        try:
            expanded = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return "QUESTIONS must be a literal dict."
        if set(expanded) != set(questions):
            return f"QUESTIONS keys must exactly match the config: {sorted(questions)}"
        for name, spec in questions.items():
            item = expanded[name]
            if not isinstance(item, dict) or item.get("type") != spec["type"]:
                return f"question {name!r}: type must stay {spec['type']!r}"
            if not isinstance(item.get("instructions"), str) or not item["instructions"].strip():
                return f"question {name!r}: instructions must be a non-empty string"
            if spec["type"] == "choice":
                criteria = item.get("criteria")
                if not isinstance(criteria, dict) or set(criteria) != set(spec["options"]):
                    return (
                        f"question {name!r}: criteria keys must exactly match "
                        f"the options {spec['options']}"
                    )
                if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
                    return f"question {name!r}: criteria descriptions must be non-empty strings"
            elif spec["type"] == "score":
                expected_len = spec["max"] - spec["min"] + 1
                rubric = item.get("rubric")
                if item.get("min") != spec["min"]:
                    return f"question {name!r}: min must equal {spec['min']}"
                if (
                    not isinstance(rubric, list)
                    or len(rubric) != expected_len
                    or not all(isinstance(v, str) and v.strip() for v in rubric)
                ):
                    return (
                        f"question {name!r}: rubric must be a list of exactly "
                        f"{expected_len} non-empty level labels"
                    )
            else:  # boolean
                criteria = item.get("criteria")
                if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
                    return (
                        f"question {name!r}: criteria must map exactly "
                        "'false' and 'true' to descriptions"
                    )
        return None

    @staticmethod
    def validate_golden_decide(golden: Any, questions: Mapping[str, Any]) -> None:
        if not isinstance(golden, list) or not golden:
            raise RouterConfigurationError("golden must be a non-empty list.")
        for i, case in enumerate(golden):
            if (
                not isinstance(case, Mapping)
                or not isinstance(case.get("record"), Mapping)
                or not isinstance(case.get("expect"), Mapping)
                or not case["expect"]
            ):
                raise RouterConfigurationError(
                    f"golden[{i}] must be {{'record': {{...}}, 'expect': "
                    "{<question>: <expected>}}."
                )
            for qname, expected in case["expect"].items():
                spec = questions.get(qname)
                if spec is None:
                    raise RouterConfigurationError(
                        f"golden[{i}]: unknown question {qname!r}."
                    )
                if spec["type"] == "choice" and expected not in spec["options"]:
                    raise RouterConfigurationError(
                        f"golden[{i}].{qname}: expected must be one of {spec['options']}."
                    )
                if spec["type"] == "boolean" and not isinstance(expected, bool):
                    raise RouterConfigurationError(
                        f"golden[{i}].{qname}: boolean expectation must be true/false."
                    )
                if spec["type"] == "score":
                    is_range = (
                        isinstance(expected, (list, tuple))
                        and len(expected) == 2
                        and all(isinstance(v, (int, float)) for v in expected)
                    )
                    if not is_range and not isinstance(expected, (int, float)):
                        raise RouterConfigurationError(
                            f"golden[{i}].{qname}: score expectation must be a "
                            "number or [low, high] range."
                        )

    def _test_golden_decider(
        self,
        source: str,
        golden: List[Mapping[str, Any]],
        questions: Mapping[str, Any],
    ) -> Optional[str]:
        import sys

        module = self._load_candidate_module(source)
        try:
            decider = module.CompiledDecider()
        except Exception as exc:
            raise RouterConfigurationError(
                f"Golden-record gate: decision runtime failed to initialize: {exc}"
            ) from exc
        failures = []
        for i, case in enumerate(golden):
            decision = decider.process(dict(case["record"]))
            for qname, expected in case["expect"].items():
                qtype = questions[qname]["type"]
                if qtype == "choice":
                    got = decision["choices"][qname]["selected"]
                    if got != expected:
                        failures.append(
                            f"golden[{i}].{qname}: selected {got!r}, expected {expected!r}"
                        )
                elif qtype == "boolean":
                    p = decision["booleans"][qname]["probability"]
                    if (p >= 0.5) != expected:
                        failures.append(
                            f"golden[{i}].{qname}: probability={p:.2f}, expected "
                            f"{'true' if expected else 'false'}"
                        )
                else:  # score
                    value = decision["scores"][qname]["value"]
                    low, high = (
                        expected
                        if isinstance(expected, (list, tuple))
                        else (expected - 0.5, expected + 0.5)
                    )
                    if not low <= value <= high:
                        failures.append(
                            f"golden[{i}].{qname}: value={value}, expected "
                            f"within [{low}, {high}]"
                        )
        sys.modules.pop("strot_golden_gate", None)
        if failures:
            return (
                "Golden-record promotion gate failed - adjust the criteria, "
                "rubrics, and route_context so these known answers hold:\n"
                + "\n".join(failures)
            )
        return None

    def compile_decide(
        self,
        sample: Any,
        questions: Mapping[str, Any],
        golden: Optional[List[Mapping[str, Any]]] = None,
    ) -> CompiledAsset:
        """Decide mode: compile a typed-question decision config.

        The synthesis agent expands the config (criteria descriptions, score
        rubrics, boolean case descriptions) against the verified schema and
        writes the state projection; the local decision model answers all
        questions per record at runtime in one batched pass. With ``golden``,
        candidates are executed against known-answer records inside the
        repair loop and only promoted when every expectation holds.
        """
        if self.runtime_kind != "local":
            raise RouterConfigurationError(
                "decide mode currently requires a local decision model "
                f"(got {self.nlu_model!r})."
            )
        self.validate_questions_config(questions)
        if golden is not None:
            self.validate_golden_decide(golden, questions)
        schema = introspect_schema(sample)
        prompt = self.build_decide_prompt(schema, questions)
        header = templates.DECIDE_HEADER_TEMPLATE.format(
            nlu_model=self.nlu_model,
            schema_literal=repr(schema.to_dict()),
            config_literal=repr(dict(questions)),
        )
        promoted: Dict[str, str] = {}

        def validate(code: str) -> Optional[str]:
            error = self._validate_decide_candidate(code, questions)
            if error:
                return error
            source = (
                "\n\n".join(
                    [header, templates.DECIDE_FOOTER_TEMPLATE.format(generated_paths=code)]
                )
                + "\n"
            )
            try:
                ast.parse(source)
            except SyntaxError as exc:  # template regression guard
                return f"Assembled decision asset failed syntax validation: {exc}"
            if golden:
                error = self._test_golden_decider(source, golden, questions)
                if error:
                    return error
            promoted["source"] = source
            return None

        _, attempts = self._synthesize_with(prompt, validate)
        source = promoted["source"]
        return CompiledAsset(
            source=source,
            nlu_model=self.nlu_model,
            runtime_kind="decide",
            goal=json.dumps(questions, sort_keys=True),
            schema=schema,
            attempts=attempts,
        )

    def _assemble(
        self,
        schema: DatasetSchema,
        goal: str,
        action_registry: Dict[str, str],
        generated_paths: str,
        question: Optional[str],
        temperature: float = 1.0,
        conformal_q: Optional[float] = None,
        calibration: Optional[Dict[str, Any]] = None,
        runtime: Optional[str] = None,
        class_bias: Optional[List[float]] = None,
        dispatch_cutoff: Optional[float] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> str:
        header = templates.ASSET_HEADER_TEMPLATE.format(
            nlu_model=self.nlu_model,
            runtime_kind=runtime or self.runtime_kind,
            goal=goal,
            threshold=self.confidence_threshold,
            temperature=repr(temperature),
            conformal_q=repr(conformal_q),
            calibration_literal=repr(calibration or {}),
            class_bias=repr(class_bias),
            dispatch_cutoff=repr(dispatch_cutoff),
            policy_literal=repr(policy or {}),
            schema_literal=repr(schema.to_dict()),
        )
        if runtime == "distilled":
            runtime_template = templates.DISTILLED_RUNTIME_TEMPLATE
        elif self.runtime_kind == "hosted":
            runtime_template = templates.HOSTED_RUNTIME_TEMPLATE
        else:
            runtime_template = templates.LOCAL_RUNTIME_TEMPLATE
        runtime = runtime_template.format(nlu_model=self.nlu_model)
        handlers_literal = (
            "{"
            + ", ".join(f"{key!r}: {handler_name(key)}" for key in action_registry)
            + "}"
        )
        footer = templates.ASSET_FOOTER_TEMPLATE.format(
            generated_paths=generated_paths,
            registry_literal=repr(action_registry),
            handlers_literal=handlers_literal,
            default_question=question
            or f"Which execution path best achieves the goal: {goal}?",
        )
        return "\n\n".join([header, runtime, footer]) + "\n"
