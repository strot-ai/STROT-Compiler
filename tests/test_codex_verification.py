import base64
import hashlib
from http.client import IncompleteRead
from importlib.metadata import PackageNotFoundError
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strot import _codex
from strot._codex import find_cli
from strot.exceptions import RouterConfigurationError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected process"))
    monkeypatch.setattr(_codex, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network"))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.163.0")


@pytest.mark.parametrize("platform,magic", [("darwin", b"\xcf\xfa\xed\xfe"),
                                           ("win32", b"MZ\0\0"), ("linux", b"\x7fELF")])
@pytest.mark.parametrize("prerelease", ["alpha.2.10", "beta.1", "rc.1"])
def test_only_verified_versions_are_ranked(monkeypatch, tmp_path, platform, magic, prerelease):
    monkeypatch.setattr(sys, "platform", platform)
    versions = ["0.162.0", "0.163.0", "0.164.0-alpha.2.9", f"0.164.0-{prerelease}", "0.164.0", "9.0.0"]
    paths = []
    for index, version in enumerate(versions):
        path = tmp_path / str(index)
        path.write_bytes(magic)
        paths.append(str(path))
    verified = lambda path: path.name != "5"
    monkeypatch.setattr(_codex, "_signed", verified)
    monkeypatch.setattr(_codex, "_linux_digest", verified)
    probes = []
    def run(command):
        probes.append(command[0])
        return "codex-cli " + versions[int(Path(command[0]).name)]
    monkeypatch.setattr(_codex, "_run", run)
    assert _codex.newest_verified_cli(paths + paths) == paths[4]
    assert probes == paths[:5]
    assert _codex.newest_verified_cli(reversed(paths)) == paths[4]
    assert _codex.newest_verified_cli(paths[:4]) == paths[3]
    assert _codex.newest_verified_cli(paths[:1] + paths[5:]) is None
    versions[0] = f"0.163.0-{prerelease}"
    assert _codex.newest_verified_cli(paths[:1]) is None


@pytest.mark.parametrize("platform,magic", [("darwin", b"\xcf\xfa\xed\xfe"),
                                           ("win32", b"MZ\0\0"), ("linux", b"\x7fELF")])
def test_lookup_without_bundle_uses_sdk_runtime_pin(monkeypatch, tmp_path, platform, magic):
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr("importlib.metadata.version", missing)
    monkeypatch.setattr("importlib.metadata.requires", lambda name: [
        "pydantic>=2.12", "openai-codex-cli-bin==0.164.0"])
    monkeypatch.setattr(sys, "platform", platform)
    path = tmp_path / "codex"
    path.write_bytes(magic)
    monkeypatch.setattr(_codex, "_signed", lambda path: True)
    monkeypatch.setattr(_codex, "_linux_digest", lambda path: True)
    monkeypatch.setattr(_codex, "_run", lambda command: "codex-cli 0.164.0")
    assert _codex.newest_verified_cli([path]) == str(path)
    monkeypatch.setattr(_codex, "_run", lambda command: "codex-cli 0.163.0")
    assert _codex.newest_verified_cli([path]) is None


@pytest.mark.parametrize("requirements", [[], ["openai-codex-cli-bin>=0.164.0"]])
def test_bundle_free_lookup_needs_a_known_baseline(monkeypatch, requirements):
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr("importlib.metadata.version", missing)
    monkeypatch.setattr("importlib.metadata.requires", lambda name: requirements)
    with pytest.raises(RouterConfigurationError, match="Set cli="):
        _codex.newest_verified_cli([])


@pytest.mark.parametrize("native", [False, True])
def test_only_desktop_wrappers_use_the_nested_payload(monkeypatch, tmp_path, native):
    monkeypatch.setattr(sys, "platform", "darwin")
    wrapper = tmp_path / "bin/codex"
    wrapper.parent.mkdir()
    wrapper.write_bytes(b"\xcf\xfa\xed\xfe" if native else b"#!/bin/sh\nexit 1")
    payload = tmp_path / "CodexCLI.app/Contents/MacOS/codex"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"\xcf\xfa\xed\xfe")
    monkeypatch.setattr(_codex, "_signed", lambda path: path == payload or native)
    monkeypatch.setattr(_codex, "_run", lambda cmd: "codex-cli 0.163.0")
    assert _codex.newest_verified_cli([str(wrapper)]) == str(wrapper if native else payload)
    payload.unlink()
    assert _codex.newest_verified_cli([str(wrapper)]) == (str(wrapper) if native else None)


@pytest.mark.parametrize("failure", ["signature", "timeout", "version", "invalid-version", "missing"])
def test_failed_verification_uses_bundle(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(sys, "platform", "darwin")
    path = tmp_path / "codex"
    if failure != "missing":
        path.write_bytes(b"\xcf\xfa\xed\xfe")
    def run(command):
        if failure == "signature":
            assert command[0] == "/usr/bin/codesign"
            raise subprocess.CalledProcessError(1, command)
        if command[0] == "/usr/bin/codesign":
            assert '2DC432GLL2' in command[-2] and 'identifier "codex"' in command[-2]
            return ""
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        return "codex-cli invalid" if failure == "invalid-version" else "unexpected version"
    monkeypatch.setattr(_codex, "_run", run)
    assert _codex.newest_verified_cli([str(path)]) is None


@pytest.mark.parametrize("output,expected", [("verified", True), ("", False), ("Valid", False)])
def test_windows_checks_publisher_with_literal_path(monkeypatch, tmp_path, output, expected):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_codex, "powershell_path", lambda: str(
        tmp_path / "Windows/System32/WindowsPowerShell/v1.0/powershell.exe"))
    def run(command):
        assert command[0].endswith("System32/WindowsPowerShell/v1.0/powershell.exe")
        assert "-LiteralPath 'a''b.exe'" in command[-1]
        assert "'Valid'" in command[-1] and "'OpenAI OpCo, LLC'" in command[-1]
        return output
    monkeypatch.setattr(_codex, "_run", run)
    assert _codex._signed(Path("a'b.exe")) is expected


@pytest.mark.parametrize("failure", [None, "mismatch", "network", "truncated", "invalid", "traversal", "redirect", "missing"])
def test_linux_hashes_binary_without_executing_it(monkeypatch, tmp_path, failure):
    path = tmp_path / "bin/codex"
    path.parent.mkdir()
    path.write_bytes(b"\x7fELFofficial binary")
    metadata = dict(version="../../elsewhere" if failure == "traversal" else "0.163.0",
                    target="x86_64-unknown-linux-musl")
    (tmp_path / "codex-package.json").write_text(json.dumps(metadata))
    if failure == "missing":
        (tmp_path / "codex-package.json").unlink()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if failure is None else "0" * 64
    record = dict(spec=dict(data=dict(hash=dict(algorithm="sha256", value=digest))))
    bundle = dict(rekorBundle=dict(Payload=dict(body=base64.b64encode(json.dumps(record).encode()).decode())))
    def fetch(url, timeout):
        assert url == "https://releases.openai.com/codex/releases/0.163.0/codex-x86_64-unknown-linux-musl.sigstore"
        assert timeout == 3
        if failure == "network":
            raise OSError("offline")
        if failure == "truncated":
            raise IncompleteRead(b"partial")
        response = io.BytesIO(b"invalid" if failure == "invalid" else json.dumps(bundle).encode())
        response.geturl = lambda: "https://example.org/untrusted" if failure == "redirect" else url
        return response
    monkeypatch.setattr(_codex, "urlopen", fetch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_codex, "_run", lambda cmd: "codex-cli 0.163.0" if failure is None else
                        pytest.fail("Unverified binary executed"))
    assert _codex.newest_verified_cli([str(path)]) == (str(path) if failure is None else None)


def test_windows_uses_os_directory_not_environment(monkeypatch, tmp_path):
    import ctypes

    system = str(tmp_path / "real-system")
    def directory(buffer, size):
        buffer.value = system
        return len(system)
    monkeypatch.setenv("SystemRoot", "untrusted")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        GetSystemDirectoryW=directory)), raising=False)
    assert _codex.powershell_path() == str(Path(system) / "WindowsPowerShell/v1.0/powershell.exe")
    monkeypatch.setattr(ctypes.windll.kernel32, "GetSystemDirectoryW", lambda *a: 0)
    with pytest.raises(OSError):
        _codex.powershell_path()


@pytest.mark.parametrize("failure", ["changed", "changed-after-probe", "pipe", "symlink"])
def test_unstable_or_special_files_are_not_executed(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(sys, "platform", "darwin")
    path = tmp_path / "codex"
    if failure == "pipe":
        if not hasattr(os, "mkfifo"):
            pytest.skip("No named pipes")
        os.mkfifo(path)
    elif failure == "symlink":
        path.symlink_to(path)
    else:
        path.write_bytes(b"\xcf\xfa\xed\xfe")
        def changed(path):
            path.write_bytes(b"replaced")
            return True
        if failure == "changed":
            monkeypatch.setattr(_codex, "_signed", changed)
        else:
            monkeypatch.setattr(_codex, "_signed", lambda path: True)
            monkeypatch.setattr(_codex, "_run", lambda cmd: changed(path) and "codex-cli 0.163.0")
    assert _codex.newest_verified_cli([str(path)]) is None


def test_verification_commands_are_bounded(monkeypatch):
    def run(command, **kwargs):
        assert kwargs == dict(check=True, capture_output=True, encoding="utf-8", timeout=5)
        return SimpleNamespace(stdout=" result\n")
    monkeypatch.setattr(subprocess, "run", run)
    assert _codex._run(["trusted-system-tool"]) == "result"


def test_reading_a_binary_does_not_count_as_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    path = tmp_path / "codex"
    path.write_bytes(b"\xcf\xfa\xed\xfe")
    original, calls = Path.stat, []
    def accessed(path, **kwargs):
        info = list(original(path, **kwargs))
        calls.append(path)
        info[7] += len(calls)
        return os.stat_result(info)
    monkeypatch.setattr(Path, "stat", accessed)
    monkeypatch.setattr(_codex, "_signed", lambda path: True)
    monkeypatch.setattr(_codex, "_run", lambda cmd: "codex-cli 0.163.0")
    assert _codex.newest_verified_cli([str(path)]) == str(path)


def test_selected_binary_is_checked_after_other_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"\xcf\xfa\xed\xfe")
    second.write_bytes(b"\xcf\xfa\xed\xfe")
    def signed(path):
        if path == first:
            return True
        first.write_bytes(b"replaced")
        return False
    monkeypatch.setattr(_codex, "_signed", signed)
    monkeypatch.setattr(_codex, "_run", lambda cmd: "codex-cli 0.163.0")
    assert _codex.newest_verified_cli([str(first), str(second)]) is None


def test_lookup_ranks_all_locations_but_explicit_cli_skips_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PATH", "/one:/two")
    installed = {"codex": "/one/codex", "/two/codex": "/two/codex",
                 "/opt/homebrew/bin/codex": "/brew/codex"}
    monkeypatch.setattr("shutil.which", installed.get)
    def select(paths):
        assert list(paths) == ["/one/codex", "/two/codex", "/brew/codex"]
        return "/brew/codex"
    monkeypatch.setattr(_codex, "newest_verified_cli", select)
    assert find_cli() == "/brew/codex"
    assert find_cli("codex") == "/one/codex"
