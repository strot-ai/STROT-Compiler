import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import strot
from strot._codex import find_cli
from strot.exceptions import CompilationError, RouterConfigurationError
from test_compiler import REGISTRY, SAMPLE, VALID_PATHS


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setattr("strot._codex.find_cli", lambda cli=None: cli)
    class CodexError(Exception):
        pass

    result = SimpleNamespace(status=SimpleNamespace(value="completed"), final_response=VALID_PATHS)
    client = MagicMock()
    client.__enter__.return_value = client
    client.thread_start.return_value.turn.return_value.run.return_value = result
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        Codex=factory, CodexConfig=SimpleNamespace, CodexError=CodexError,
        Sandbox=SimpleNamespace(read_only="read-only"), ApprovalMode=SimpleNamespace(deny_all="deny-all"),
    ))
    return client, factory, result, CodexError


@pytest.mark.parametrize("model", [None, "gpt-6-luna"])
def test_sdk_repairs_and_reuses_cached_code(sdk, monkeypatch, tmp_path, model):
    client, factory, result, _ = sdk
    client.thread_start.return_value.turn.return_value.run.side_effect = [
        SimpleNamespace(status=result.status, final_response="not valid code"), result,
    ]
    spec = "codex-sdk" + ("/" + model if model else "")
    first = strot.run(SAMPLE, "triage", REGISTRY, llm=spec, state=tmp_path, execute=False)
    assert first.attempts == 2 and not first.cache_hit
    assert factory.call_count == client.__exit__.call_count == 2
    for call in factory.call_args_list:
        config = call.args[0]
        assert config.codex_bin is None
        assert config.experimental_api is False
        assert config.config_overrides == ("project_doc_max_bytes=0", "features.shell_tool=false",
                                           'web_search="disabled"')
        assert not Path(config.cwd).exists() and Path(config.cwd) != Path.cwd()
    for call in client.thread_start.call_args_list:
        assert call.kwargs == dict(model=model, model_provider="openai", cwd=call.kwargs["cwd"],
                                   ephemeral=True, developer_instructions="",
                                   sandbox="read-only", approval_mode="deny-all")
    assert "PREVIOUS ATTEMPT" in client.thread_start.return_value.turn.call_args.args[0]
    assert client.thread_start.return_value.turn.call_args.kwargs == ({"effort": "none"} if model else {})
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    assert strot.run(SAMPLE, "triage", REGISTRY, llm=spec, state=tmp_path, execute=False).cache_hit


def test_missing_sdk_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    with pytest.raises(RouterConfigurationError, match=r"strot-ai\[codex-sdk\]"):
        strot.codex_sdk_llm_client()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 1e100])
def test_sdk_rejects_invalid_timeout(sdk, timeout):
    with pytest.raises(ValueError, match="positive finite"):
        strot.codex_sdk_llm_client(timeout=timeout)
    sdk[1].assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_sdk_timeout_closes_runtime_without_result_callback(sdk, monkeypatch, streaming, failure):
    client, _, result, error = sdk
    closed, results = Event(), []
    client.close.side_effect = closed.set

    def wait(*args):
        assert closed.wait(2), "runtime did not close"
        if failure:
            raise error("transport closed")
        return result

    client.thread_start.return_value.turn.return_value.run.side_effect = wait
    monkeypatch.setattr("strot._codex.collect_turn", wait)
    llm = strot.codex_sdk_llm_client(
        timeout=0.01, on_event=(lambda event: None) if streaming else None, on_result=results.append)
    with pytest.raises(CompilationError, match="timed out") as caught:
        llm("compile")
    assert caught.value.attempts == 0 and not results
    client.close.assert_called_once()
    client.__exit__.assert_called_once()


def test_timeout_keeps_callback_errors(sdk, monkeypatch):
    client = sdk[0]
    closed = Event()
    client.close.side_effect = closed.set
    error = RuntimeError("observer failed")
    def callback(event):
        assert closed.wait(2)
        raise error
    monkeypatch.setattr("strot._codex.collect_turn", lambda turn, hook: hook(None))
    with pytest.raises(RuntimeError) as caught:
        strot.codex_sdk_llm_client(timeout=0.01, on_event=callback)("compile")
    assert caught.value is error


@pytest.mark.parametrize("effort", [None, "none", "low", "medium"])
def test_sdk_effort_reaches_the_turn(sdk, effort):
    assert strot.codex_sdk_llm_client("gpt-6-luna", reasoning_effort=effort)("prompt") == VALID_PATHS
    sdk[0].thread_start.return_value.turn.assert_called_once_with("prompt", effort=effort or "none")


def test_result_hook_preserves_metadata_and_callback_errors(sdk):
    client, _, result, _ = sdk
    result.usage, result.duration_ms = {"tokens": 42}, 125
    seen = []
    assert strot.codex_sdk_llm_client(on_result=seen.append)("compile") == VALID_PATHS
    assert seen == [result] and seen[0].duration_ms == 125
    failure = RuntimeError("observer failed")
    hook = Mock(side_effect=failure)
    with pytest.raises(RuntimeError) as caught:
        strot.codex_sdk_llm_client(on_result=hook)("compile")
    assert caught.value is failure
    assert client.__exit__.call_count == 2


@pytest.mark.parametrize("entry", ["run", "decide"])
def test_turn_hook_can_cancel_without_compiler_retry(sdk, tmp_path, entry):
    client, factory, result, _ = sdk
    turn = client.thread_start.return_value.turn.return_value
    turn.run.return_value = result
    turn.interrupt.side_effect = lambda: setattr(result.status, "value", "interrupted")
    llm = strot.codex_sdk_llm_client(on_turn=lambda handle: handle.interrupt())
    args = (SAMPLE, "triage", REGISTRY) if entry == "run" else (
        SAMPLE, {"is_bug": {"type": "boolean", "instructions": "Bug report?"}})
    with pytest.raises(CompilationError, match="interrupted"):
        getattr(strot, entry)(*args, llm=llm, state=tmp_path, execute=False)
    turn.interrupt.assert_called_once()
    factory.assert_called_once()
    client.__exit__.assert_called_once()


def test_turn_hook_error_is_not_wrapped(sdk):
    error = OSError("observer failed")
    with pytest.raises(OSError) as caught:
        strot.codex_sdk_llm_client(on_turn=Mock(side_effect=error))("compile")
    assert caught.value is error
    sdk[0].__exit__.assert_called_once()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("expires", [False, True])
def test_runtime_accepts_sdk_configuration(monkeypatch, tmp_path, installed, expires):
    sdk = pytest.importorskip("openai_codex")
    bundled_codex_path = pytest.importorskip("codex_cli_bin").bundled_codex_path

    codex_bin = str(bundled_codex_path()) if installed else None
    monkeypatch.setattr("strot._codex.find_cli", lambda cli=None: codex_bin)
    config_type = sdk.CodexConfig

    def isolated_config(**kwargs):
        return config_type(**kwargs, env={
            "CODEX_HOME": str(tmp_path), "OPENAI_API_KEY": "offline-sdk-key",
            "CODEX_API_KEY": "", "CODEX_ACCESS_TOKEN": "",
            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        })

    monkeypatch.setattr(sdk, "CodexConfig", isolated_config)
    monkeypatch.setattr("openai_codex.client.CodexClient._start_turn",
                        lambda *args, **kwargs: pytest.fail("Model calls are forbidden"))
    processes = []
    codex_type = sdk.Codex
    def tracked_client(config):
        client = codex_type(config)
        processes.append(client._client._proc)
        return client
    monkeypatch.setattr(sdk, "Codex", tracked_client)
    # Start the real runtime and thread, but replace generation before any model call.
    def turn(_, prompt, **kwargs):
        def run():
            if expires:
                processes[-1].wait(timeout=5)
            return SimpleNamespace(status=SimpleNamespace(value="completed"), final_response=prompt)
        return SimpleNamespace(run=run)
    monkeypatch.setattr(sdk.Thread, "turn", turn)
    llm = strot.codex_sdk_llm_client("gpt-6-luna", timeout=0.05 if expires else 600)
    if expires:
        with pytest.raises(CompilationError, match="timed out"):
            llm("compiled code")
    else:
        assert llm("compiled code") == "compiled code"
    assert processes[0].poll() is not None


def test_explicit_relative_cli_survives_directory_change(sdk, monkeypatch, tmp_path):
    monkeypatch.setattr("strot._codex.find_cli", find_cli)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "./codex" if name == "./codex" else None)
    client = strot.codex_sdk_llm_client(cli="./codex")
    monkeypatch.chdir(tmp_path.parent)
    assert client("compile") == VALID_PATHS
    assert sdk[1].call_args.args[0].codex_bin == str(tmp_path / "codex")


@pytest.mark.parametrize("failure", ["startup", "thread", "run", "encoding", "interrupted", "none", "empty"])
def test_sdk_failures_close_resources(sdk, failure):
    client, factory, result, error = sdk
    if failure == "startup":
        factory.side_effect = OSError("runtime missing")
    elif failure == "thread":
        client.thread_start.side_effect = error("login required")
    elif failure == "run":
        client.thread_start.return_value.turn.return_value.run.side_effect = RuntimeError("turn failed")
    elif failure == "encoding":
        client.thread_start.return_value.turn.return_value.run.side_effect = UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid byte")
    elif failure == "interrupted":
        result.status.value = "interrupted"
    else:
        result.final_response = None if failure == "none" else " \n"
    with pytest.raises(error if failure == "thread" else CompilationError):
        strot.codex_sdk_llm_client()("compile this")
    if failure != "startup":
        client.__exit__.assert_called_once()
    for call in factory.call_args_list:
        assert not Path(call.args[0].cwd).exists()


@pytest.mark.parametrize("entry", ["run", "decide"])
@pytest.mark.parametrize("stage", ["startup", "thread", "turn", "run"])
@pytest.mark.parametrize("error_type,args", [
    ("ServerBusyError", (-32001, "server busy", {"retry_after": 1})),
    ("TransportClosedError", ("connection closed",)),
])
def test_sdk_typed_errors_reach_callers_unchanged(monkeypatch, tmp_path, entry, stage, error_type, args):
    sdk = pytest.importorskip("openai_codex")
    error = getattr(sdk, error_type)(*args)
    client = MagicMock()
    client.__enter__.return_value = client
    factory = Mock(return_value=client)
    target = {"startup": factory, "thread": client.thread_start,
              "turn": client.thread_start.return_value.turn,
              "run": client.thread_start.return_value.turn.return_value.run}[stage]
    target.side_effect = error
    monkeypatch.setattr(sdk, "Codex", factory)
    llm = strot.codex_sdk_llm_client(cli=sys.executable)
    args = (SAMPLE, "triage", REGISTRY) if entry == "run" else (
        SAMPLE, {"is_bug": {"type": "boolean", "instructions": "Does this mention crashes?"}})
    with pytest.raises(type(error)) as caught:
        getattr(strot, entry)(*args, llm=llm, state=tmp_path, execute=False)
    assert caught.value is error
    factory.assert_called_once()
    assert not Path(factory.call_args.args[0].cwd).exists()
    if stage != "startup":
        client.__exit__.assert_called_once()


@pytest.mark.parametrize("base", [OSError, RuntimeError, UnicodeError])
def test_sdk_error_subclasses_are_never_wrapped(sdk, base):
    class Error(sdk[3], base):
        pass
    error = Error("SDK failure")
    sdk[1].side_effect = error
    with pytest.raises(Error) as caught:
        strot.codex_sdk_llm_client()("compile")
    assert caught.value is error
