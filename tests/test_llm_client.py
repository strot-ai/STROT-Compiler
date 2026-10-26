from unittest.mock import Mock

import pytest

import strot


@pytest.mark.parametrize("spec,name,model", [
    ("openai/gpt-6-luna", "openai", "gpt-6-luna"), ("gpt-6-luna", "openai", "gpt-6-luna"),
    ("openai", "openai", None), ("anthropic", "anthropic", None),
    ("anthropic/sonnet", "anthropic", "sonnet"),
    ("codex-sdk", "codex_sdk", None), ("codex-sdk/model", "codex_sdk", "model"),
    ("codex-cli", "codex_cli", None), ("claude-cli/model", "claude_cli", "model"),
    ("claude-code", "claude_cli", None),
])
def test_shared_factory_is_lazy_and_forwards_hooks(monkeypatch, spec, name, model):
    backend = Mock(return_value="compiled text")
    factory = Mock(return_value=backend)
    monkeypatch.setattr("strot.jit." + name + "_llm_client", factory)
    hooks = dict(on_result=Mock(), on_event=Mock(), on_turn=Mock())
    client = strot.llm_client(spec, **hooks)
    factory.assert_not_called()
    assert client("first") == client("second") == "compiled text"
    factory.assert_called_once()
    args, kwargs = factory.call_args
    assert (args[0] if args else kwargs.pop("model", None)) == model
    assert kwargs == hooks
    assert [call.args[0] for call in backend.call_args_list] == ["first", "second"]


def test_shared_factory_preserves_custom_callables():
    assert strot.llm_client(lambda prompt: prompt.upper(), on_event=Mock())("custom") == "CUSTOM"


@pytest.mark.parametrize("spec,name", [("openai", "openai"), ("gpt-6-luna", "openai"),
                                       ("codex-sdk", "codex_sdk"), ("codex-cli", "codex_cli")])
def test_shared_factory_forwards_effort(monkeypatch, spec, name):
    factory = Mock(return_value=lambda prompt: prompt)
    monkeypatch.setattr("strot.jit." + name + "_llm_client", factory)
    assert strot.llm_client(spec, reasoning_effort="low")("prompt") == "prompt"
    assert factory.call_args.kwargs["reasoning_effort"] == "low"
