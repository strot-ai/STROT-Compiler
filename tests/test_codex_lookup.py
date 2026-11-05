import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strot._codex import find_cli
from strot.exceptions import RouterConfigurationError


@pytest.fixture
def lookup(monkeypatch, tmp_path):
    installed, calls = {}, []
    monkeypatch.setattr("strot._codex.newest_verified_cli", lambda paths: next(paths, None))
    monkeypatch.setenv("PATH", "")

    def which(name):
        calls.append(name)
        return installed.get(name)

    monkeypatch.setattr("shutil.which", which)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv("HOMEBREW_PREFIX", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr("strot._codex.powershell_path", lambda: str(
        tmp_path / "Windows/System32/WindowsPowerShell/v1.0/powershell.exe"))
    return installed, calls


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_path_is_first_candidate(lookup, monkeypatch, platform):
    installed, calls = lookup
    monkeypatch.setattr(sys, "platform", platform)
    installed["codex"] = "/path/codex"
    assert find_cli() == "/path/codex"
    assert calls == ["codex"]


@pytest.mark.parametrize("platform,location", [
    ("darwin", "/opt/homebrew/bin/codex"), ("darwin", "/usr/local/bin/codex"),
    ("darwin", "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"),
    ("darwin", "/Applications/Codex.app/Contents/Resources/codex"),
    ("darwin", "home/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"),
    ("linux", "/home/linuxbrew/.linuxbrew/bin/codex"),
    ("linux", "home/.linuxbrew/bin/codex"), ("linux", "/opt/chatgpt/resources/codex-cli/bin/codex"),
    ("darwin", "brew/bin/codex"), ("linux", "brew/bin/codex"),
    ("win32", "local/Programs/OpenAI/Codex/bin/codex.exe"),
])
def test_standard_install_locations(lookup, monkeypatch, tmp_path, platform, location):
    installed, _ = lookup
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOMEBREW_PREFIX", str(tmp_path / "brew"))
    path = str(Path(location) if location.startswith("/") else tmp_path / location)
    installed[path] = path
    assert find_cli() == path
    del installed[path]
    assert find_cli() is None


def test_linux_desktop_launcher(lookup, monkeypatch, tmp_path):
    installed, _ = lookup
    monkeypatch.setattr(sys, "platform", "linux")
    launcher = tmp_path / "chatgpt"
    launcher.symlink_to(tmp_path / "desktop/chatgpt")
    installed["chatgpt"] = str(launcher)
    path = str(tmp_path / "desktop/resources/codex-cli/bin/codex")
    installed[path] = path
    assert find_cli() == path


@pytest.mark.parametrize("failure", [None, "timeout", "oserror", "exit"])
def test_windows_store_discovery_is_bounded(lookup, monkeypatch, tmp_path, failure):
    installed, _ = lookup
    monkeypatch.setattr(sys, "platform", "win32")
    powershell = str(tmp_path / "Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    installed[powershell] = "/powershell.exe"
    root = tmp_path / "WindowsApps/OpenAI.Codex_é"
    path = str(root / "app/resources/codex-cli/bin/codex.exe")
    Path(path).parent.mkdir(parents=True)
    Path(path).touch()
    installed[path] = path

    def query(command, **kwargs):
        assert "Get-AppxPackage -Name OpenAI.Codex" in command[-1]
        assert kwargs["timeout"] == 3 and kwargs["encoding"] == "utf-8" and kwargs["check"]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 3)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command)
        if failure:
            raise OSError("not available")
        return SimpleNamespace(stdout="\ufeff" + str(root) + "\nrelative warning\n")

    monkeypatch.setattr(subprocess, "run", query)
    assert find_cli() == (None if failure else path)


def test_explicit_cli_is_absolute_and_never_silently_replaced(lookup, monkeypatch, tmp_path):
    installed, calls = lookup
    monkeypatch.chdir(tmp_path)
    installed["./tools/codex"] = "./tools/codex"
    assert find_cli("./tools/codex") == str(tmp_path / "tools/codex")
    assert calls == ["./tools/codex"]
    with pytest.raises(RouterConfigurationError, match="Cannot find the Codex CLI"):
        find_cli("missing-codex")


@pytest.mark.parametrize("error", [OSError("unreadable"), RuntimeError("no home directory")])
def test_discovery_errors_use_the_bundle(lookup, monkeypatch, error):
    def unavailable():
        raise error
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", unavailable)
    assert find_cli() is None


@pytest.mark.parametrize("prefix", ["", ".", "relative"])
@pytest.mark.parametrize("variable,suffix", [
    ("HOMEBREW_PREFIX", "bin/codex"),
    ("LOCALAPPDATA", "Programs/OpenAI/Codex/bin/codex.exe"),
])
def test_relative_install_roots_do_not_search_the_working_directory(lookup, monkeypatch, prefix, variable, suffix):
    installed, calls = lookup
    monkeypatch.setattr(sys, "platform", "linux" if variable == "HOMEBREW_PREFIX" else "win32")
    monkeypatch.setenv(variable, prefix)
    path = str(Path(prefix) / suffix)
    installed[path] = path
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected process"))
    assert find_cli() is None
    assert path not in calls
