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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from . import templates
from .exceptions import CompilationError, RouterConfigurationError
from .constants import CONFIDENCE_THRESHOLD, DEFAULT_NLU_MODEL
from .schema import DatasetSchema, introspect_schema

#: A generative LLM client is any callable mapping a prompt to raw model text.
LLMClient = Callable[[str], str]

#: Below this many labeled records for a class, its head under-learns; warn.
MIN_CLASS_RECORDS = 3

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def openai_llm_client(model: str = "gpt-4o", temperature: float = 0.0) -> LLMClient:
    """Convenience factory: a synthesizer client backed by the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RouterConfigurationError(
            "openai_llm_client requires the 'hosted' extra: "
            "pip install strot-ai[hosted]"
        ) from exc
    client = OpenAI()

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return _call


def anthropic_llm_client(
    model: str = "claude-opus-4-8",
    max_tokens: int = 16000,
) -> LLMClient:
    """Convenience factory: a synthesizer client backed by the Anthropic API.

    Requires the ``anthropic`` package (``pip install strot-ai[anthropic]``)
    and ``ANTHROPIC_API_KEY`` in the environment.
    """
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
) -> LLMClient:
    """Use Claude Code for synthesis.

    Requires ``--safe-mode`` support. This skips user customizations and
    keeps CLI authentication. Built-in tools and session persistence are
    disabled. No SDK is required.
    """
    return _cli_llm_client("claude", cli, model, timeout)


def codex_cli_llm_client(
    model: Optional[str] = None,
    cli: str = "codex",
    timeout: float = 600.0,
) -> LLMClient:
    """Use ``codex exec`` for synthesis.

    Run ``codex login`` for ChatGPT subscription access. The CLI must support
    ``--ignore-user-config`` and ``--ephemeral``. No SDK is required.
    Shell commands, web search, and project instructions are disabled.
    """
    return _cli_llm_client("codex", cli, model, timeout)


def _cli_llm_client(backend: str, cli: str, model: Optional[str], timeout: float) -> LLMClient:
    import shutil
    import subprocess
    import tempfile

    cli_path = shutil.which(cli)
    if cli_path is None:
        raise RouterConfigurationError(
            f"Cannot find the {cli!r} CLI on PATH. Install {backend} and sign in."
        )
    cli_path = str(Path(cli_path).absolute())

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
        # Decide compiles only local encoders now (hosted runtime removed).
        return "local"

    # ------------------------------------------------------------------ #
    # Phase 2: prompt scaffolding                                         #
    # ------------------------------------------------------------------ #

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

    def _resolve_compile_checkpoint(self) -> str:
        import os

        override = os.environ.get("STROT_NLU_CHECKPOINT")
        if override:
            return override
        if Path(self.nlu_model).is_dir():
            return self.nlu_model
        from huggingface_hub import snapshot_download

        return snapshot_download(self.nlu_model)

    @staticmethod
    def validate_questions_config(questions: Mapping[str, Any]) -> None:
        """Validate a typed-question config before any synthesis."""
        if not isinstance(questions, Mapping) or not questions:
            raise RouterConfigurationError("questions must be a non-empty mapping.")
        for name, spec in questions.items():
            if not isinstance(spec, Mapping) or spec.get("type") not in (
                "predicate", "boolean", "score", "choice",
            ):
                raise RouterConfigurationError(
                    f"question {name!r}: type must be 'predicate', 'score', or 'choice' "
                    "('boolean' is a deprecated alias for 'predicate')."
                )
            if spec["type"] == "score":
                StrotCompiler._validate_score_spec(name, spec)
                if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: score requires instructions anchoring the scale."
                    )
            elif spec["type"] in ("predicate", "boolean"):
                if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: predicate requires instructions."
                    )
            else:  # choice
                StrotCompiler._validate_choice_spec(name, spec)

    @staticmethod
    def _score_levels(spec: Mapping[str, Any]) -> List["tuple[str, Optional[str]]"]:
        """Normalize a score spec into ordered ``(label, description)`` levels.

        The ``levels`` key carries ``[{"label": ..., "description": ...}, ...]``
        (description optional), lowest to highest. The deprecated ``min``/``max``
        alias synthesizes one level per integer in ``min..max`` (the integers
        themselves as labels, no descriptions).
        """
        levels = spec.get("levels")
        if levels is not None:
            return [(str(level["label"]), level.get("description")) for level in levels]
        minimum, maximum = spec["min"], spec["max"]
        return [(str(value), None) for value in range(minimum, maximum + 1)]

    @staticmethod
    def _validate_score_spec(name: str, spec: Mapping[str, Any]) -> None:
        """Validate a score question: ``levels`` (rich) or ``min``/``max`` (alias)."""
        levels = spec.get("levels")
        has_range = spec.get("min") is not None or spec.get("max") is not None
        if levels is not None and has_range:
            raise RouterConfigurationError(
                f"question {name!r}: set either 'levels' or the deprecated "
                "'min'/'max' alias, not both."
            )
        if levels is not None:
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise RouterConfigurationError(
                    f"question {name!r}: score needs 2-10 ordered levels."
                )
            seen: List[str] = []
            for level in levels:
                if not isinstance(level, Mapping) or "label" not in level:
                    raise RouterConfigurationError(
                        f"question {name!r}: each level must be a mapping with a "
                        "'label' (and an optional 'description')."
                    )
                label = level["label"]
                if not isinstance(label, str) or not label.strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: level labels must be non-empty strings."
                    )
                description = level.get("description")
                if description is not None and (
                    not isinstance(description, str) or not description.strip()
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: level 'description' must be a "
                        "non-empty string when present."
                    )
                seen.append(label)
            if len(set(seen)) != len(seen):
                raise RouterConfigurationError(
                    f"question {name!r}: level labels must be unique."
                )
        else:
            minimum, maximum = spec.get("min"), spec.get("max")
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or not 2 <= (maximum - minimum + 1) <= 10
            ):
                raise RouterConfigurationError(
                    f"question {name!r}: score needs 'levels' (2-10 ordered "
                    "labeled levels) or the deprecated int min < max alias "
                    "spanning 2-10 levels."
                )

    @staticmethod
    def _choice_pairs(spec: Mapping[str, Any]) -> List["tuple[Any, Optional[str]]"]:
        """Normalize a choice spec into ordered ``(value, description)`` pairs.

        The ``choices`` key carries ``[{"value": ..., "description": ...}, ...]``
        (description optional). The deprecated ``options`` alias is a bare list
        of values with no descriptions.
        """
        choices = spec.get("choices")
        if choices is not None:
            return [(c["value"], c.get("description")) for c in choices]
        return [(value, None) for value in spec.get("options", [])]

    @staticmethod
    def _validate_choice_spec(name: str, spec: Mapping[str, Any]) -> None:
        """Validate a choice question: ``choices`` (rich) or ``options`` (alias)."""
        choices, options = spec.get("choices"), spec.get("options")
        if choices is not None and options is not None:
            raise RouterConfigurationError(
                f"question {name!r}: set either 'choices' or the deprecated "
                "'options' alias, not both."
            )
        if choices is not None:
            if not isinstance(choices, list) or not 2 <= len(choices) <= 255:
                raise RouterConfigurationError(
                    f"question {name!r}: choice needs 2-255 choices."
                )
            values: List[Any] = []
            for choice in choices:
                if not isinstance(choice, Mapping) or "value" not in choice:
                    raise RouterConfigurationError(
                        f"question {name!r}: each choice must be a mapping with a "
                        "'value' (and an optional 'description')."
                    )
                value = choice["value"]
                if not isinstance(value, (str, bool)) or (
                    isinstance(value, str) and not value
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: choice values must be non-empty "
                        "strings or booleans."
                    )
                description = choice.get("description")
                if description is not None and (
                    not isinstance(description, str) or not description.strip()
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: choice 'description' must be a "
                        "non-empty string when present."
                    )
                values.append(value)
        elif options is not None:
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 255
                or not all(isinstance(o, (str, bool)) and o != "" for o in options)
            ):
                raise RouterConfigurationError(
                    f"question {name!r}: choice needs 2-255 unique non-empty "
                    "values (deprecated 'options' alias)."
                )
            values = list(options)
        else:
            raise RouterConfigurationError(
                f"question {name!r}: choice needs 'choices' "
                "(or the deprecated 'options' alias)."
            )
        keyed = [(type(v).__name__, v) for v in values]
        if len(set(keyed)) != len(keyed):
            raise RouterConfigurationError(
                f"question {name!r}: choice values must be unique."
            )

    @staticmethod
    def _prompt_config(questions: Mapping[str, Any]) -> Dict[str, Any]:
        """Config as the synthesizer should see it: score levels always explicit.

        The deprecated score ``min``/``max`` alias is expanded into an ordered
        ``levels`` list so the synthesizer only ever reasons about labeled
        levels, never a bare integer range.
        """
        shown: Dict[str, Any] = {}
        for name, spec in questions.items():
            if spec.get("type") == "score":
                shown[name] = {
                    key: value
                    for key, value in spec.items()
                    if key not in ("min", "max", "levels")
                }
                shown[name]["levels"] = [
                    {"label": label, **({"description": desc} if desc else {})}
                    for label, desc in StrotCompiler._score_levels(spec)
                ]
            else:
                shown[name] = dict(spec)
        return shown

    def build_propose_prompt(self, schema: DatasetSchema, intent: str) -> str:
        """Intake prompt: propose a typed question set from a one-line intent."""
        return "\n".join([
            "You are the STROT intake designer. From a one-line decision intent "
            "and the dataset schema, propose the SMALL set of typed questions a "
            "decision model should answer per record. You design the schema - you "
            "do not write programs and you do not invent data.",
            "",
            "## DECISION INTENT",
            intent,
            "",
            "## VERIFIED DATASET SCHEMA",
            schema.to_json(),
            "",
            "## OUTPUT CONTRACT",
            "Emit exactly one ```python code block containing a single top-level "
            "`QUESTIONS: Dict[str, Any]` literal and nothing else. Each entry maps a "
            "short snake_case name to a spec:",
            '- predicate: {"type": "predicate", "instructions": <one line>}',
            '- choice:    {"type": "choice", "instructions": <one line>, '
            '"choices": [{"value": <str|bool>, "description": <short>}...]}',
            '- score:     {"type": "score", "instructions": <one line>, '
            '"levels": [{"label": <str>, "description": <short>}...]} '
            "(2-10 levels, ordered lowest to highest)",
            "",
            "Rules: only questions the intent actually needs (prefer few); ground "
            "options/levels in the schema's fields and vocabulary; no other "
            "statements.",
        ])

    def propose_questions(self, sample: Any, intent: str) -> Dict[str, Any]:
        """Intake: synthesize a typed question set from a one-line intent.

        Returns a validated ``questions`` config for the author to review and
        edit before :meth:`compile_decide`. The LLM designs the decision schema
        here; it does not train or invent data.
        """
        schema = introspect_schema(sample)
        prompt = self.build_propose_prompt(schema, intent)
        promoted: Dict[str, Any] = {}

        def validate(code: str) -> Optional[str]:
            try:
                questions = self._literal_assignment(code, "QUESTIONS")
            except Exception as exc:
                return f"QUESTIONS must be a single top-level literal dict: {exc}"
            if not isinstance(questions, dict) or not questions:
                return "QUESTIONS must be a non-empty dict."
            try:
                self.validate_questions_config(questions)
            except RouterConfigurationError as exc:
                return str(exc)
            promoted["questions"] = questions
            return None

        self._synthesize_with(prompt, validate)
        return promoted["questions"]

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
                json.dumps(self._prompt_config(questions), indent=2),
                "",
                "## OUTPUT CONTRACT",
                "Emit exactly one ```python code block containing EXACTLY two "
                "top-level definitions and nothing else:",
                "- `QUESTIONS: Dict[str, Any]` literal with the SAME question "
                "names. Per question: keep \"type\" and \"instructions\" (write "
                "crisp instructions if the config omits them). For 'choice' add "
                "\"criteria\": a dict mapping EVERY choice value (exact, as given "
                "in the config - keep booleans as booleans) to a short "
                "discriminative description using vocabulary from the records; "
                "sharpen each config 'description' where present. For 'score' add "
                "\"rubric\": a list of {\"label\": <config level label, exact>, "
                "\"description\": <short discriminative description grounded in "
                "the instructions' anchors and the records' vocabulary>} entries, "
                "one per config level, in the SAME order the config lists them "
                "(lowest to highest). For 'predicate' (or the deprecated "
                "'boolean' alias) add \"criteria\": {\"false\": <short "
                "negative-case description>, \"true\": <short positive-case "
                "description>}.",
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
                values = [value for value, _ in self._choice_pairs(spec)]
                criteria = item.get("criteria")
                if not isinstance(criteria, dict) or set(criteria) != set(values):
                    return (
                        f"question {name!r}: criteria keys must exactly match "
                        f"the choice values {values}"
                    )
                if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
                    return f"question {name!r}: criteria descriptions must be non-empty strings"
            elif spec["type"] == "score":
                labels = [label for label, _ in self._score_levels(spec)]
                rubric = item.get("rubric")
                if not isinstance(rubric, list) or len(rubric) != len(labels):
                    return (
                        f"question {name!r}: rubric must be a list of exactly "
                        f"{len(labels)} ordered levels (one per config level)"
                    )
                for level, label in zip(rubric, labels):
                    if (
                        not isinstance(level, dict)
                        or str(level.get("label")) != label
                    ):
                        return (
                            f"question {name!r}: each rubric level must be a "
                            "mapping whose 'label' matches the config levels, in "
                            f"order: {labels}"
                        )
                    description = level.get("description")
                    if not isinstance(description, str) or not description.strip():
                        return (
                            f"question {name!r}: each rubric level needs a "
                            "non-empty 'description'"
                        )
            else:  # predicate (or the deprecated boolean alias)
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
                if spec["type"] == "choice":
                    values = [value for value, _ in StrotCompiler._choice_pairs(spec)]
                    if expected not in values:
                        raise RouterConfigurationError(
                            f"golden[{i}].{qname}: expected must be one of {values}."
                        )
                if spec["type"] in ("predicate", "boolean") and not isinstance(expected, bool):
                    raise RouterConfigurationError(
                        f"golden[{i}].{qname}: predicate expectation must be true/false."
                    )
                if spec["type"] == "score":
                    labels = [label for label, _ in StrotCompiler._score_levels(spec)]
                    if expected not in labels:
                        raise RouterConfigurationError(
                            f"golden[{i}].{qname}: score expectation must be one "
                            f"of the level labels {labels}."
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
                    got = decision["choices"][qname]["choice"]
                    if got != expected:
                        failures.append(
                            f"golden[{i}].{qname}: chose {got!r}, expected {expected!r}"
                        )
                elif qtype in ("predicate", "boolean"):
                    p = decision["predicates"][qname]["probability"]
                    if (p >= 0.5) != expected:
                        failures.append(
                            f"golden[{i}].{qname}: probability={p:.2f}, expected "
                            f"{'true' if expected else 'false'}"
                        )
                else:  # score
                    got = decision["scores"][qname]["score"]
                    if got != expected:
                        failures.append(
                            f"golden[{i}].{qname}: scored {got!r}, expected {expected!r}"
                        )
        sys.modules.pop("strot_golden_gate", None)
        if failures:
            return (
                "Golden-record promotion gate failed - adjust the criteria, "
                "rubrics, and route_context so these known answers hold:\n"
                + "\n".join(failures)
            )
        return None

    @staticmethod
    def _question_runtime(spec: Mapping[str, Any]) -> "tuple[str, List[Any]]":
        """``(base_type, ordered classes)`` for a question's distilled head.

        predicate -> ``[False, True]`` (the head reports P(true)); choice -> the
        choice values in order; score -> the ordered level labels.
        """
        qtype = spec["type"]
        if qtype in ("predicate", "boolean"):
            return "predicate", [False, True]
        if qtype == "choice":
            return "choice", [value for value, _ in StrotCompiler._choice_pairs(spec)]
        return "score", [label for label, _ in StrotCompiler._score_levels(spec)]

    @staticmethod
    def _literal_assignment(code: str, name: str) -> Any:
        """Literal-eval the value of a top-level ``name = ...`` assignment."""
        for node in ast.parse(code).body:
            target = None
            if isinstance(node, ast.Assign) and node.targets:
                target = getattr(node.targets[0], "id", None)
            elif isinstance(node, ast.AnnAssign):
                target = getattr(node.target, "id", None)
            if target == name:
                return ast.literal_eval(node.value)
        raise ValueError(f"no top-level {name!r} assignment")

    def build_decide_teacher_prompt(
        self, questions: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
    ) -> str:
        """Ask the teacher LLM to answer every question for every record - the
        training labels each per-question head is distilled from."""
        spec_lines = []
        for name, spec in questions.items():
            base, classes = self._question_runtime(spec)
            allowed = ("true or false" if base == "predicate"
                       else "exactly one of: " + ", ".join(repr(c) for c in classes))
            spec_lines.append(
                f"- {name!r} ({base}): {spec.get('instructions', '')} "
                f"Answer with {allowed}."
            )
        return "\n".join([
            "You are the STROT teacher. Answer every question for every record "
            "below - these answers train a small local model, so be accurate and "
            "internally consistent.",
            "",
            "## QUESTIONS",
            "\n".join(spec_lines),
            "",
            "## RECORDS",
            json.dumps([dict(r) for r in records], indent=2, default=str),
            "",
            "## OUTPUT CONTRACT",
            "Emit exactly one ```python code block with a single top-level "
            "definition and nothing else:",
            f"- `ANSWERS: list` of EXACTLY {len(records)} dicts, one per record "
            "in the SAME order. Each dict maps EVERY question name to that "
            "record's answer: predicate -> a Python bool (True/False); choice -> "
            "exactly one allowed value (keep booleans as booleans); score -> "
            "exactly one allowed level label (a string).",
        ])

    def _validate_decide_labels(
        self, code: str, n: int, questions: Mapping[str, Any]
    ) -> Optional[str]:
        try:
            answers = self._literal_assignment(code, "ANSWERS")
        except Exception as exc:
            return f"ANSWERS must be a single literal list assignment: {exc}"
        if not isinstance(answers, list) or len(answers) != n:
            got = len(answers) if isinstance(answers, list) else "a non-list"
            return f"ANSWERS must have exactly {n} entries (got {got})."
        for i, row in enumerate(answers):
            if not isinstance(row, dict):
                return f"ANSWERS[{i}] must be a dict."
            for name, spec in questions.items():
                if name not in row:
                    return f"ANSWERS[{i}] is missing question {name!r}."
                base, classes = self._question_runtime(spec)
                value = row[name]
                if base == "predicate" and not isinstance(value, bool):
                    return f"ANSWERS[{i}][{name!r}] must be a bool."
                if base != "predicate" and value not in classes:
                    return (
                        f"ANSWERS[{i}][{name!r}]={value!r} is not an allowed "
                        f"{base} answer."
                    )
        return None

    @staticmethod
    def _l2norm(vec: Sequence[float]) -> List[float]:
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def _lite_prototype_text(label: Any, description: Optional[str]) -> str:
        """The text STROT embeds as a class's zero-shot prototype."""
        if description:
            return f"{label}: {description}"
        return str(label)

    def _lite_heads(self, questions: Mapping[str, Any]) -> Dict[str, Any]:
        """Zero-shot ``lite`` heads: a prototype embedding per class, no training.

        Each class becomes an L2-normalized prototype embedding of its
        description; at runtime the record's embedding is scored against them
        (the same linear ``weights @ x`` runtime as a trained head, so cosine
        nearest-prototype). A predicate uses ``[-proto, +proto]`` so the two
        logits are a sigmoid around the statement direction. No teacher labels,
        no fold split - just the frozen encoder and the config's own text.
        """
        from . import distill as distill_mod

        embedder = distill_mod.make_encoder_embedder(self._resolve_compile_checkpoint())
        heads: Dict[str, Any] = {}
        for name, spec in questions.items():
            base, classes = self._question_runtime(spec)
            if base == "predicate":
                proto = self._l2norm(embedder([spec["instructions"]])[0])
                weights = [[-x for x in proto], list(proto)]  # classes [False, True]
                bias = [0.0, 0.0]
            else:
                pairs = (
                    self._choice_pairs(spec) if base == "choice"
                    else self._score_levels(spec)
                )
                texts = [self._lite_prototype_text(label, desc) for label, desc in pairs]
                weights = [self._l2norm(v) for v in embedder(texts)]
                bias = [0.0] * len(classes)
            heads[name] = {
                "type": base, "classes": classes,
                "weights": weights, "bias": bias, "temperature": 1.0,
            }
        return heads

    def compile_decide(
        self,
        sample: Any,
        questions: Mapping[str, Any],
        golden: Optional[List[Mapping[str, Any]]] = None,
        mode: str = "head",
    ) -> CompiledAsset:
        """Decide mode: compile one decision head per typed question.

        ``mode`` picks the tier:

        * ``"lite"`` - zero-shot prototype heads: each class is an embedding of
          its description, scored by cosine at runtime. No teacher labels, no
          training; the synthesizer still writes the state projection. The local
          $0 floor (instant, bring-no-data).
        * ``"head"`` (default) - the teacher LLM answers every question for every
          sampled record; a calibrated logistic head per question is trained on
          the frozen encoder's embeddings and baked in.
        * ``"deep"`` - a shared top-N LoRA adapter is fine-tuned over all
          questions, then the heads are refit on the adapted embedding.

        Invariant - **recompile, don't increment**: every compile trains a FRESH
        head (and, for ``deep``, a fresh adapter) from the frozen base on the
        current data; it NEVER warm-starts from previously trained weights. A
        compiled model's weights are a pure function of its own
        ``(data, questions, mode)`` - independent of what was compiled before.
        This is what keeps continuous optimization immune to catastrophic
        forgetting: retraining on accumulated data is a clean re-fit, not a
        sequential update that could erode earlier quality. Do not add a
        warm-start-from-prior-weights path without revisiting that guarantee.

        With ``golden``, the baked model is run against known-answer records and
        rejected if any expectation fails.
        """
        if mode not in ("lite", "head", "deep"):
            raise RouterConfigurationError(
                f"decide mode must be 'lite', 'head', or 'deep' (got {mode!r})."
            )
        if self.runtime_kind != "local":
            raise RouterConfigurationError(
                "decide mode requires a local encoder to embed records "
                f"(got {self.nlu_model!r})."
            )
        self.validate_questions_config(questions)
        if golden is not None:
            self.validate_golden_decide(golden, questions)
        schema = introspect_schema(sample)
        records = list(sample)
        from . import distill as distill_mod

        if mode == "head" and len(records) < distill_mod.MIN_DISTILL_RECORDS:
            raise RouterConfigurationError(
                f"decide needs at least {distill_mod.MIN_DISTILL_RECORDS} sample "
                f"records to distill its heads (got {len(records)}); raise "
                "sample_size."
            )
        if mode == "deep":
            from . import finetune as finetune_mod

            if len(records) < finetune_mod.MIN_FINETUNE_RECORDS:
                raise RouterConfigurationError(
                    f"decide mode='deep' needs at least "
                    f"{finetune_mod.MIN_FINETUNE_RECORDS} records to fine-tune the "
                    f"shared adapter (got {len(records)}); raise sample_size or use "
                    "mode='head'."
                )

        # 1. Synthesize the state projection (route_context). The decide prompt
        # also expands criteria/rubrics; the heads learn from labels rather than
        # criteria, so only route_context is used - the block is kept in the
        # baked model as a record of the synthesis.
        prompt = self.build_decide_prompt(schema, questions)
        promoted: Dict[str, str] = {}

        def validate(code: str) -> Optional[str]:
            error = self._validate_decide_candidate(code, questions)
            if error:
                return error
            promoted["code"] = code
            return None

        _, attempts = self._synthesize_with(prompt, validate)
        code = promoted["code"]

        if mode == "lite":
            # Zero-shot tier: prototype heads from the config's own text, no
            # teacher pass and no training. The projection above is still baked.
            question_heads = self._lite_heads(questions)
            return self._bake_decider(
                question_heads, code, schema, questions, golden, attempts,
                winner="lite",
            )

        ns: Dict[str, Any] = {}
        exec(compile(code, "<decide-projection>", "exec"), ns)
        contexts = [ns["route_context"](dict(record)) for record in records]

        # 2. Teacher labels: every question answered for every record.
        from . import pii

        masked = pii.mask_records(records, maskers=pii.default_maskers())
        labels_code, _ = self._synthesize_with(
            self.build_decide_teacher_prompt(questions, masked),
            lambda c: self._validate_decide_labels(c, len(records), questions),
        )
        answers = self._literal_assignment(labels_code, "ANSWERS")

        # Coverage guard: a head can only learn an outcome it has seen. Warn on
        # thin classes so the author brings records covering each outcome
        # (bring-your-own-data; a logistic head needs only a few per class).
        import warnings

        for name, spec in questions.items():
            _, classes = self._question_runtime(spec)
            counts: Dict[Any, int] = {}
            for row in answers:
                counts[row[name]] = counts.get(row[name], 0) + 1
            thin = [c for c in classes if counts.get(c, 0) < MIN_CLASS_RECORDS]
            if thin:
                warnings.warn(
                    f"decide: question {name!r} has thin coverage for {thin} "
                    f"(<{MIN_CLASS_RECORDS} labeled records each); the head will "
                    "under-learn these outcomes. Bring more records covering them.",
                    stacklevel=2,
                )

        # Fold hygiene: each head trains on one fold; its temperature is fit on a
        # disjoint fold so the baked confidence is honest, not overfit. (For deep,
        # the shared adapter also trains on the train fold only.)
        from . import calibrate as calibrate_mod

        train_ix, measure_ix, calib_ix = distill_mod.three_way_split(
            list(range(len(records)))
        )
        fit_ix = train_ix + measure_ix

        # 3. Per-record embeddings shared by every head. head/lite use the frozen
        # encoder; deep adapts a shared top-N LoRA to the domain first and refits
        # the heads on the adapted embedding (one forward pass per record).
        adapter: Optional[Dict[str, Any]] = None
        adapter_config: Optional[Dict[str, Any]] = None
        if mode == "deep":
            from . import finetune as finetune_mod

            q_labels: Dict[str, List[int]] = {}
            q_classes: Dict[str, int] = {}
            for name, spec in questions.items():
                _, classes = self._question_runtime(spec)
                q_labels[name] = [
                    classes.index(answers[i][name]) for i in range(len(records))
                ]
                q_classes[name] = len(classes)
            adapt = finetune_mod.finetune_decide(
                contexts, q_labels, q_classes, train=train_ix,
            )
            vectors = adapt.vectors
            adapter, adapter_config = adapt.adapter, adapt.config
        else:
            vectors = distill_mod.make_encoder_embedder(
                self._resolve_compile_checkpoint()
            )(contexts)

        # 4. One calibrated logistic head per question.
        question_heads: Dict[str, Any] = {}
        for name, spec in questions.items():
            base, classes = self._question_runtime(spec)
            y = [classes.index(answers[i][name]) for i in range(len(records))]
            head_w, head_b = distill_mod.train_head(
                [vectors[i] for i in fit_ix], [y[i] for i in fit_ix],
                n_classes=len(classes),
            )
            calib_logits = [
                [sum(w * x for w, x in zip(row, vectors[i])) + b
                 for row, b in zip(head_w, head_b)]
                for i in calib_ix
            ]
            temperature = (
                calibrate_mod.fit_temperature_logits(
                    calib_logits, [y[i] for i in calib_ix]
                )
                if len(calib_ix) >= 4 else 1.0
            )
            question_heads[name] = {
                "type": base, "classes": classes,
                "weights": head_w, "bias": head_b, "temperature": temperature,
            }

        # 4b. Drift / OOD guard (head + deep): per-class centroids + a distance
        # threshold on the training embeddings, so the runtime refuses records
        # unlike anything seen rather than answering them. Centroids use the
        # finest-grained question's labels; the gate is always on (conservative
        # 0.99 percentile). lite has no record embeddings, so it has no gate.
        ood: Optional[Dict[str, Any]] = None
        if vectors:
            ood_q = max(
                questions, key=lambda q: len(self._question_runtime(questions[q])[1])
            )
            _, ood_classes = self._question_runtime(questions[ood_q])
            ood_labels = [ood_classes.index(answers[i][ood_q]) for i in range(len(records))]
            centroids = distill_mod.class_centroids(
                vectors, ood_labels, n_classes=len(ood_classes)
            )
            ood = {
                "centroids": centroids,
                "threshold": distill_mod.ood_threshold(vectors, centroids, percentile=0.99),
                "percentile": 0.99, "basis": ood_q,
            }

        # 5. Bake the decision model.
        return self._bake_decider(
            question_heads, code, schema, questions, golden, attempts,
            winner=("fine_tuned" if mode == "deep" else "distilled"),
            n_records=len(records), adapter=adapter, adapter_config=adapter_config,
            ood=ood,
        )

    def _bake_decider(
        self,
        question_heads: Dict[str, Any],
        code: str,
        schema: DatasetSchema,
        questions: Mapping[str, Any],
        golden: Optional[List[Mapping[str, Any]]],
        attempts: int,
        winner: str,
        n_records: Optional[int] = None,
        adapter: Optional[Dict[str, Any]] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        ood: Optional[Dict[str, Any]] = None,
    ) -> CompiledAsset:
        """Bake baked heads + projection into a self-contained decider asset.

        Shared by all tiers - they differ only in how ``question_heads`` was
        fitted. ``lite``/``head`` embed with the frozen encoder; ``deep`` passes
        an ``adapter`` and bakes the LoRA-adapted runtime instead. ``ood`` (when
        fitted) bakes the drift gate; an empty literal is a runtime no-op.
        """
        header = templates.DECIDE_DISTILLED_HEADER_TEMPLATE.format(
            nlu_model=self.nlu_model,
            schema_literal=repr(schema.to_dict()),
            config_literal=repr(dict(questions)),
        )
        heads_literal = "QUESTION_HEADS: Dict[str, Any] = " + repr(question_heads)
        ood_literal = "OOD: Dict[str, Any] = " + repr(ood or {})
        parts = [header, code, heads_literal, ood_literal]
        if adapter is not None:
            parts.append("ADAPTER: Dict[str, Any] = " + repr(adapter))
            parts.append("ADAPTER_CONFIG: Dict[str, Any] = " + repr(adapter_config))
            parts.append(templates.DECIDE_FINETUNED_RUNTIME_TEMPLATE.format())
        else:
            parts.append(templates.DECIDE_DISTILLED_RUNTIME_TEMPLATE.format())
        source = "\n\n".join(parts) + "\n"
        if golden:
            error = self._test_golden_decider(source, golden, questions)
            if error:
                raise CompilationError(
                    "Decision model does not satisfy the golden "
                    f"records:\n{error}",
                    attempts=attempts, last_error=error,
                )
        return CompiledAsset(
            source=source,
            nlu_model=self.nlu_model,
            runtime_kind="decide",
            goal=json.dumps(questions, sort_keys=True),
            schema=schema,
            attempts=attempts,
            model_search={"winner": winner, "n_records": n_records},
        )

