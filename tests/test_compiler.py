import ast
import json
import sys
import types
from unittest.mock import Mock

import pytest

from strot.compiler import StrotCompiler, handler_name, openai_llm_client, anthropic_llm_client, claude_cli_llm_client
from strot.exceptions import CompilationError, FallbackException

SAMPLE = [
    {"ticket_id": 1, "body": "double charged", "tier": "pro"},
    {"ticket_id": 2, "body": "app crashes", "tier": "free"},
]


@pytest.mark.parametrize("provider,factory,name,options,expected", [
    ("openai", openai_llm_client, "OpenAI", {},
     {"model": "gpt-6-luna", "temperature": 0.0, "extra_body": {"reasoning_effort": "none"}}),
    ("openai", openai_llm_client, "OpenAI", {"reasoning_effort": "low"},
     {"model": "gpt-6-luna", "extra_body": {"reasoning_effort": "low"}}),
    ("openai", openai_llm_client, "OpenAI", {"reasoning_effort": "medium"},
     {"model": "gpt-6-luna", "extra_body": {"reasoning_effort": "medium"}}),
    ("openai", openai_llm_client, "OpenAI", {"reasoning_effort": "none", "temperature": 0.2},
     {"model": "gpt-6-luna", "temperature": 0.2, "extra_body": {"reasoning_effort": "none"}}),
    ("openai", openai_llm_client, "OpenAI", {"model": "custom-model", "temperature": 0.2},
     {"model": "custom-model", "temperature": 0.2}),
    ("anthropic", anthropic_llm_client, "Anthropic", {},
     {"model": "claude-haiku-5-5", "max_tokens": 16000}),
    ("anthropic", anthropic_llm_client, "Anthropic", {"model": "custom-model"},
     {"model": "custom-model", "max_tokens": 16000}),
])
def test_native_api_result_hooks(monkeypatch, provider, factory, name, options, expected):
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="code"))],
        content=[types.SimpleNamespace(type="thinking", thinking="plan"),
                 types.SimpleNamespace(type="text", text="code")], stop_reason="end_turn")
    module, client = Mock(), Mock()
    getattr(module, name).return_value = client
    client.chat.completions.create.return_value = client.messages.create.return_value = response
    monkeypatch.setitem(sys.modules, provider, module)
    seen = []
    unsupported = lambda _: pytest.fail("Unsupported hook called")
    assert factory(**options, on_result=seen.append, on_event=unsupported, on_turn=unsupported)("prompt") == "code"
    assert seen == [response]
    create = client.chat.completions.create if provider == "openai" else client.messages.create
    create.assert_called_once_with(**expected, messages=[{"role": "user", "content": "prompt"}])
    error = OSError("observer failed")
    with pytest.raises(OSError) as caught:
        factory(**options, on_result=Mock(side_effect=error))("prompt")
    assert caught.value is error


@pytest.mark.parametrize("factory", [anthropic_llm_client, claude_cli_llm_client])
def test_unsupported_effort_is_explicit(factory):
    from strot.exceptions import RouterConfigurationError

    with pytest.raises(RouterConfigurationError, match="reasoning_effort"):
        factory(reasoning_effort="low")

REGISTRY = {
    "billing": "Billing disputes and refunds",
    "defect": "Crashes and bugs",
}

VALID_PATHS = '''```python
ROUTING_LABELS = {"billing": "billing charge", "defect": "app crash"}


def route_context(record: dict) -> str:
    return str(record.get("body", ""))


def execute_billing(record: dict) -> dict:
    try:
        return {"queue": "billing", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "billing", "error": str(exc)}


def execute_defect(record: dict) -> dict:
    try:
        return {"queue": "engineering", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "engineering", "error": str(exc)}
```'''


class ScriptedLLM:
    """Replays a scripted sequence of responses and records every prompt."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def make_compiler(llm, **kwargs):
    return StrotCompiler(llm_client=llm, **kwargs)


class TestPromptScaffolding:
    def test_prompt_contains_schema_goal_and_contract(self):
        llm = ScriptedLLM([VALID_PATHS])
        make_compiler(llm).compile(SAMPLE, "triage tickets", REGISTRY)
        prompt = llm.prompts[0]
        assert "## VERIFIED DATASET SCHEMA" in prompt
        assert '"ticket_id"' in prompt
        assert "triage tickets" in prompt
        assert "SupersonicLabs/Julia-1" in prompt
        assert "execute_billing" in prompt and "execute_defect" in prompt
        assert "ROUTING_LABELS" in prompt and "route_context" in prompt

    def test_missing_routing_labels_feeds_back_and_retries(self):
        no_labels = VALID_PATHS.replace(
            'ROUTING_LABELS = {"billing": "billing charge", "defect": "app crash"}', ""
        )
        llm = ScriptedLLM([no_labels, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        assert "ROUTING_LABELS" in llm.prompts[1]

    def test_mismatched_label_keys_rejected(self):
        wrong_keys = VALID_PATHS.replace('"defect": "app crash"', '"other": "app crash"')
        llm = ScriptedLLM([wrong_keys, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        assert "exactly match the action keys" in llm.prompts[1]


class TestSelfCorrectionLoop:
    def test_syntax_error_feeds_trace_back_and_retries(self):
        broken = "```python\ndef execute_billing(record):\n    return {\n```"
        llm = ScriptedLLM([broken, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        repair_prompt = llm.prompts[1]
        assert "## PREVIOUS ATTEMPT (REJECTED)" in repair_prompt
        assert "SyntaxError" in repair_prompt

    def test_missing_contract_function_is_rejected(self):
        only_one = "```python\ndef execute_billing(record: dict) -> dict:\n    return {}\n```"
        llm = ScriptedLLM([only_one, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        assert "execute_defect" in llm.prompts[1]

    def test_forbidden_imports_are_rejected(self):
        heavy = (
            "```python\nimport torch\n"
            'ROUTING_LABELS = {"billing": "billing", "defect": "defect"}\n'
            "def route_context(record: dict) -> str:\n    return str(record)\n"
            "def execute_billing(record: dict) -> dict:\n    return {}\n"
            "def execute_defect(record: dict) -> dict:\n    return {}\n```"
        )
        llm = ScriptedLLM([heavy, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        assert "Forbidden imports" in llm.prompts[1]

    def test_stdlib_escape_hatches_are_rejected(self):
        sneaky = VALID_PATHS.replace(
            "def route_context(record: dict) -> str:",
            "import subprocess\n\n\ndef route_context(record: dict) -> str:",
        )
        llm = ScriptedLLM([sneaky, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        assert "Forbidden imports" in llm.prompts[1]

    def test_dynamic_execution_calls_are_rejected(self):
        sneaky = VALID_PATHS.replace(
            'return str(record.get("body", ""))',
            'return str(eval("record"))',
        )
        llm = ScriptedLLM([sneaky, VALID_PATHS])
        asset = make_compiler(llm).compile(SAMPLE, "triage", REGISTRY)
        assert asset.attempts == 2
        assert "Forbidden call" in llm.prompts[1]

    def test_exhausted_budget_raises_compilation_error(self):
        llm = ScriptedLLM(["nonsense"] * 3)
        with pytest.raises(CompilationError) as excinfo:
            make_compiler(llm, max_retries=3).compile(SAMPLE, "triage", REGISTRY)
        assert excinfo.value.attempts == 3

    def test_missing_llm_client_is_explicit(self):
        with pytest.raises(CompilationError, match="llm_client"):
            StrotCompiler().compile(SAMPLE, "triage", REGISTRY)


class TestAssetAssembly:
    def test_local_asset_is_valid_python_and_targets_cpu(self):
        asset = make_compiler(ScriptedLLM([VALID_PATHS])).compile(SAMPLE, "triage", REGISTRY)
        ast.parse(asset.source)
        assert asset.runtime_kind == "local"
        assert "from julia import load_model" in asset.source
        assert 'device="cpu"' in asset.source
        assert 'NLU_MODEL = \'SupersonicLabs/Julia-1\'' in asset.source

    def test_hosted_asset_is_valid_python_and_torch_free(self):
        compiler = StrotCompiler(
            nlu_model="openai/gpt-6-luna", llm_client=ScriptedLLM([VALID_PATHS])
        )
        asset = compiler.compile(SAMPLE, "triage", REGISTRY)
        tree = ast.parse(asset.source)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "torch" not in imported and "transformers" not in imported
        assert "openai" in imported
        assert '"selected_index": {"type": "integer"}' in asset.source

    def test_asset_saves_as_standalone_file(self, tmp_path):
        asset = make_compiler(ScriptedLLM([VALID_PATHS])).compile(SAMPLE, "triage", REGISTRY)
        path = asset.save(tmp_path / "compiled_pipeline.py")
        assert path.read_text(encoding="utf-8") == asset.source
        assert "from strot" not in asset.source  # complete isolation

    def test_handler_name_sanitization(self):
        assert handler_name("billing-dispute v2") == "execute_billing_dispute_v2"
        with pytest.raises(CompilationError):
            StrotCompiler(llm_client=lambda p: VALID_PATHS).compile(
                SAMPLE, "triage", {"a b": "x", "a-b": "y"}
            )


class TestCompiledAssetExecution:
    """Execute a hosted-runtime asset end-to-end with a faked openai module."""

    def _fake_openai(self, payload, effort="low"):
        module = types.ModuleType("openai")

        class _Completions:
            @staticmethod
            def create(**kwargs):
                assert kwargs["model"] == "gpt-6-luna"
                assert kwargs["extra_body"] == {"reasoning_effort": effort}
                assert ("temperature" in kwargs) == (effort == "none")
                msg = types.SimpleNamespace(content=json.dumps(payload))
                choice = types.SimpleNamespace(message=msg)
                return types.SimpleNamespace(choices=[choice])

        class OpenAI:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=_Completions())

        module.OpenAI = OpenAI
        return module

    def _run_asset(self, monkeypatch, payload, effort=None):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        if effort is None:
            monkeypatch.delenv("STROT_NLU_REASONING_EFFORT", raising=False)
        else:
            monkeypatch.setenv("STROT_NLU_REASONING_EFFORT", effort)
        monkeypatch.setitem(sys.modules, "openai", self._fake_openai(payload, effort or "low"))
        compiler = StrotCompiler(
            nlu_model="openai/gpt-6-luna", llm_client=ScriptedLLM([VALID_PATHS])
        )
        asset = compiler.compile(SAMPLE, "triage", REGISTRY)
        module = types.ModuleType("compiled_pipeline")
        monkeypatch.setitem(sys.modules, "compiled_pipeline", module)
        exec(compile(asset.source, "compiled_pipeline.py", "exec"), module.__dict__)
        return module.__dict__

    @pytest.mark.parametrize("effort", [None, "none", "medium"])
    def test_confident_route_dispatches_generated_path(self, monkeypatch, effort):
        ns = self._run_asset(monkeypatch, {"selected_index": 1, "confidence": 0.95}, effort)
        outcome = ns["CompiledPipeline"]().process({"ticket_id": 7, "body": "crash"})
        assert outcome["action"] == "defect"
        assert outcome["result"] == {"queue": "engineering", "ticket_id": 7}
        assert outcome["confidence"] == pytest.approx(0.95)

    def test_low_confidence_triggers_asset_fallback(self, monkeypatch):
        ns = self._run_asset(monkeypatch, {"selected_index": 0, "confidence": 0.2})
        with pytest.raises(ns["FallbackException"]) as error:
            ns["CompiledPipeline"]().process({"ticket_id": 7, "body": "???"})
        assert error.value.confidence == pytest.approx(0.2)

    def test_default_route_skips_calibration_math(self, monkeypatch):
        ns = self._run_asset(monkeypatch, {"selected_index": 1, "confidence": 0.95})
        ns["math"] = None
        assert ns["CompiledPipeline"]().process({"ticket_id": 7})["action"] == "defect"
