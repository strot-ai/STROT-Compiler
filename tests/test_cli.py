import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import strot
from strot.exceptions import CompilationError, RouterConfigurationError
from strot.jit import _resolve_llm

FIXTURES = Path(__file__).parent / "fixtures"


def test_codex_process_repairs_and_reuses_cache(monkeypatch, tmp_path):
    cases = json.loads((FIXTURES / "support_tickets.json").read_text(encoding="utf-8"))
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/end_to_end.py"))
    cli = tmp_path / "codex"
    cli.write_text(f"#!{sys.executable}\n" + (FIXTURES / "codex_stub.py").read_text(), encoding="utf-8")
    cli.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr("shutil.which", lambda _: str(cli))
    monkeypatch.setenv("STROT_TEST_CLI_LOG", str(log))
    monkeypatch.setenv("STROT_TEST_CLI_RESPONSE", str(FIXTURES / "support_paths.md"))
    goal = example["GOAL"] + " Handle Zoë Pérez’s request."
    args = ([case["record"] for case in cases], goal, example["ACTION_REGISTRY"])
    first = strot.run(*args, llm="codex-cli/gpt-6-luna", state=tmp_path, execute=False)
    assert first.attempts == 2 and not first.cache_hit
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2 and "SyntaxError" in calls[1]["prompt"]
    assert "Zoë Pérez" in calls[0]["prompt"]
    assert calls[0]["args"][-2:] == ["--model", "gpt-6-luna"]
    assert all(Path(call["cwd"]) != Path.cwd() and not Path(call["cwd"]).exists() for call in calls)
    monkeypatch.setattr("shutil.which", lambda _: pytest.fail("CLI resolved on cache hit"))
    assert strot.run(*args, llm="codex-cli/gpt-6-luna", state=tmp_path, execute=False).cache_hit
    assert len(log.read_text().splitlines()) == 2


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_relative_cli_path(backend, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli = tmp_path / backend
    cli.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "if '--output-last-message' in sys.argv:\n"
        "    Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text('ready')\n"
        "else:\n"
        "    print('ready', end='')\n"
    )
    cli.chmod(0o755)
    unsupported = lambda _: pytest.fail("Unsupported hook called")
    client = getattr(strot, backend + "_cli_llm_client")(
        cli="./" + backend, on_result=unsupported, on_event=unsupported, on_turn=unsupported)
    assert client("prompt") == "ready"


@pytest.mark.parametrize("spec", ["codex-cli", "codex-cli/gpt-6-luna", "claude-cli", "claude-cli/claude-haiku-5-5"])
def test_cli_synthesis_and_cache(spec, monkeypatch, tmp_path):
    from test_jit import SAMPLE, QUERY, REGISTRY, VALID_PATHS

    monkeypatch.setattr("shutil.which", lambda cli: "/bin/" + cli)
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        assert QUERY in kwargs["input"]
        assert kwargs["timeout"] == 600
        assert kwargs["encoding"] == "utf-8"
        assert Path(kwargs["cwd"]).is_dir()
        if "--output-last-message" in command:
            assert "--sandbox" in command and "read-only" in command
            expected = [
                "project_doc_max_bytes=0", "features.shell_tool=false", 'web_search="disabled"',
            ]
            if spec.endswith("/gpt-6-luna"):
                expected.append('model_reasoning_effort="none"')
            assert [command[i + 1] for i, value in enumerate(command) if value == "-c"] == expected
            Path(command[command.index("--output-last-message") + 1]).write_text(VALID_PATHS)
        else:
            assert "--safe-mode" in command
            assert command[command.index("--tools") + 1] == ""
            assert "--no-session-persistence" in command
        if "/" in spec:
            assert command[-2:] == ["--model", spec.split("/", 1)[1]]
        else:
            assert "--model" not in command
        return SimpleNamespace(returncode=0,
                               stdout="CLI progress" if spec.startswith("codex") else VALID_PATHS,
                               stderr="progress")

    monkeypatch.setattr(subprocess, "run", invoke)
    first = strot.run(SAMPLE, QUERY, REGISTRY, llm=spec, state=tmp_path, execute=False)
    assert not first.cache_hit and first.artifact_path.exists()
    monkeypatch.setattr("shutil.which", lambda cli: pytest.fail("CLI resolved on cache hit"))
    assert strot.run(SAMPLE, QUERY, REGISTRY, llm=spec, state=tmp_path, execute=False).cache_hit
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_missing_cli(backend, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cli: None)
    with pytest.raises(RouterConfigurationError, match="CLI on PATH"):
        _resolve_llm(backend + "-cli")


def test_codex_cli_effort_override(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cli: "/bin/" + cli)

    def invoke(command, **kwargs):
        assert 'model_reasoning_effort="low"' in command
        Path(command[command.index("--output-last-message") + 1]).write_text("code")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", invoke)
    assert strot.codex_cli_llm_client("gpt-6-luna", reasoning_effort="low")("prompt") == "code"


@pytest.mark.parametrize("backend", ["codex", "claude"])
@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror", "empty", "missing-output", "encoding"])
def test_cli_failures(backend, failure, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cli: "/bin/" + cli)

    def invoke(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "oserror":
            raise OSError("cannot start")
        if failure == "encoding":
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        if backend == "codex" and failure != "missing-output":
            Path(command[command.index("--output-last-message") + 1]).write_text("")
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stdout="login required" if failure == "exit" else "",
                               stderr="startup warning" if failure == "exit" else "")

    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(CompilationError) as error:
        _resolve_llm(backend + "-cli")("prompt")
    if failure == "encoding":
        assert isinstance(error.value.__cause__, UnicodeDecodeError)
    if failure == "exit":
        assert "login required" in str(error.value)


def test_example_defaults_to_offline_stub(monkeypatch, tmp_path, capsys):
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/end_to_end.py"))
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setattr("strot.jit.openai_llm_client", lambda **_: pytest.fail("OpenAI selected"))
    monkeypatch.setattr(sys, "argv", ["end_to_end.py", "--compile-only", "--state", str(tmp_path)])
    example["main"]()
    summary = json.loads(capsys.readouterr().out.splitlines()[0])
    assert Path(summary["artifact"]).is_file()
    assert not summary["cache_hit"] and summary["processed"] == 0


@pytest.mark.parametrize("spec,override,expected", [
    ("gpt-6-luna", [], None), ("openai/gpt-6-luna", [], None),
    ("codex-sdk/gpt-6-luna", [], None), ("codex-cli/gpt-6-luna", [], None),
    ("codex-sdk", [], None), ("claude-cli", [], None),
    ("codex-sdk/gpt-6-luna", ["--reasoning-effort", "medium"], "medium"),
])
def test_example_forwards_effort_override(monkeypatch, tmp_path, spec, override, expected):
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/end_to_end.py"))
    factory = Mock(return_value=example["stub_synthesizer"])
    monkeypatch.setattr(strot, "llm_client", factory)
    monkeypatch.setattr(sys, "argv", ["end_to_end.py", "--compile-only", "--llm", spec,
                                     "--state", str(tmp_path), *override])
    example["main"]()
    factory.assert_called_once_with(spec, reasoning_effort=expected)
