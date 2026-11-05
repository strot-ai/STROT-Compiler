"""Codex runtime verification and typed progress collection."""

import base64
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from urllib.request import urlopen

from .exceptions import RouterConfigurationError


def collect_turn(turn, on_event):
    from contextlib import closing

    from openai_codex import TurnResult
    from openai_codex.generated.v2_all import AgentMessageThreadItem, MessagePhase
    from openai_codex.models import (
        ItemCompletedNotification, ThreadTokenUsageUpdatedNotification, TurnCompletedNotification,
    )

    completed, usage, items = None, None, []
    with closing(turn.stream()) as events:
        for event in events:
            payload = event.payload
            if isinstance(payload, ItemCompletedNotification) and payload.turn_id == turn.id:
                items.append(payload.item)
            elif isinstance(payload, ThreadTokenUsageUpdatedNotification) and payload.turn_id == turn.id:
                usage = payload.token_usage
            elif isinstance(payload, TurnCompletedNotification) and payload.turn.id == turn.id:
                completed = payload.turn
            on_event(event)
    if completed is None:
        raise RuntimeError("turn completed event not received")
    if completed.status.value == "failed":
        detail = completed.error.message if completed.error else None
        raise RuntimeError(detail or f"turn failed with status {completed.status.value}")
    final, fallback = None, None
    for item in reversed(items):
        message = getattr(item, "root", item)
        if isinstance(message, AgentMessageThreadItem):
            if message.phase == MessagePhase.final_answer:
                final = message.text
                break
            if message.phase is None and fallback is None:
                fallback = message.text
    return TurnResult(
        id=completed.id, status=completed.status, error=completed.error,
        started_at=completed.started_at, completed_at=completed.completed_at,
        duration_ms=completed.duration_ms, final_response=final if final is not None else fallback,
        items=items, usage=usage,
    )


def _run(command):
    return subprocess.run(command, check=True, capture_output=True,
                          encoding="utf-8", timeout=5).stdout.strip()


def _identity(path):
    info = path.stat()
    return (info.st_mode, info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def powershell_path():
    import ctypes

    directory = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(directory, len(directory))
    if not 0 < length < len(directory):
        raise OSError("Cannot find the Windows system directory")
    return str(Path(directory.value) / "WindowsPowerShell/v1.0/powershell.exe")


def _signed(path):
    if sys.platform == "darwin":
        _run(["/usr/bin/codesign", "--verify", "--strict", "-R",
              '=anchor apple generic and certificate leaf[subject.OU] = "2DC432GLL2" '
              'and identifier "codex"', str(path)])
        return True
    if sys.platform == "win32":
        literal = str(path).replace("'", "''")
        return _run([powershell_path(), "-NoProfile", "-NonInteractive", "-Command",
                     "$ErrorActionPreference='Stop'; "
                     f"$s=Get-AuthenticodeSignature -LiteralPath '{literal}'; "
                     "if ($s.Status -eq 'Valid' -and "
                     "$s.SignerCertificate.GetNameInfo('SimpleName',$false) -ceq "
                     "'OpenAI OpCo, LLC') { 'verified' }"]) == "verified"
    return False


def _linux_digest(path):
    # The manifest is only a hint. Trust comes from the digest fetched over HTTPS.
    manifest = path.parent.parent / "codex-package.json"
    if not manifest.is_file() or manifest.stat().st_size > 65536:
        return False
    with manifest.open("rb") as source:
        metadata = json.loads(source.read(65536))
    version, target = metadata["version"], metadata["target"]
    if (not isinstance(version, str) or
            not re.fullmatch(r"\d+\.\d+\.\d+(?:-alpha\.\d+(?:\.\d+)?)?", version) or
            target not in ("aarch64-unknown-linux-musl", "x86_64-unknown-linux-musl")):
        return False
    url = f"https://releases.openai.com/codex/releases/{version}/codex-{target}.sigstore"
    with urlopen(url, timeout=3) as response:
        if not response.geturl().startswith("https://releases.openai.com/"):
            return False
        bundle = json.loads(response.read(262145))
    record = json.loads(base64.b64decode(bundle["rekorBundle"]["Payload"]["body"], validate=True))
    digest = record["spec"]["data"]["hash"]
    if (digest["algorithm"] != "sha256" or not isinstance(digest["value"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", digest["value"])):
        return False
    actual = hashlib.sha256()
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            actual.update(chunk)
    return actual.hexdigest() == digest["value"]


def newest_verified_cli(paths):
    """Return the newest verified runtime at least as new as the SDK's runtime.

    Unknown wrappers, missing signatures, network errors and old versions use
    the SDK bundle. Explicit overrides do not pass through this function.
    """
    from importlib.metadata import PackageNotFoundError, requires, version

    from packaging.requirements import Requirement
    from packaging.version import Version

    def parsed(value):
        return Version(re.sub(r"-alpha\.(\d+)\.(\d+)$", r"a\1.post\2", value))

    try:
        minimum = parsed(version("openai-codex-cli-bin"))
    except PackageNotFoundError:
        pins = [pin.version for value in requires("openai-codex") or []
                if (requirement := Requirement(value)).name == "openai-codex-cli-bin"
                for pin in requirement.specifier if pin.operator == "==" and "*" not in pin.version]
        if len(pins) != 1:
            raise RouterConfigurationError(
                "Cannot determine the SDK's required Codex CLI version. Set cli= explicitly.")
        minimum = parsed(pins[0])
    selected, selected_identity, newest, seen = None, None, minimum, set()
    for candidate in paths:
        try:
            path = Path(candidate).resolve(strict=True)
            # Use the nested payload only for desktop shell wrappers.
            nested = path.parent.parent / "CodexCLI.app/Contents/MacOS/codex"
            if sys.platform == "darwin" and nested.is_file() and path.is_file():
                with path.open("rb") as binary:
                    if binary.read(2) == b"#!":
                        path = nested.resolve(strict=True)
            if path in seen:
                continue
            seen.add(path)
            identity = _identity(path)
            if not stat.S_ISREG(identity[0]):
                continue
            with path.open("rb") as binary:
                magic = binary.read(4)
            if sys.platform == "linux":
                trusted = magic == b"\x7fELF" and _linux_digest(path)
            else:
                trusted = (magic[:2] == b"MZ" if sys.platform == "win32" else
                           magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
                                     b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")) and _signed(path)
            if not trusted or _identity(path) != identity:
                continue
            output = _run([str(path), "--version"])
            match = re.fullmatch(r"codex-cli (\S+)", output)
            if match and _identity(path) == identity:
                current = parsed(match[1])
                if current >= minimum and (selected is None or current > newest):
                    selected, selected_identity, newest = str(path), identity, current
        except (OSError, HTTPException, RuntimeError, subprocess.SubprocessError, ValueError,
                KeyError, TypeError):
            continue
    try:
        return selected if selected and _identity(Path(selected)) == selected_identity else None
    except OSError:
        return None


def find_cli(cli=None):
    if cli is not None:
        found = shutil.which(os.path.expanduser(cli))
        if found is None:
            raise RouterConfigurationError(f"Cannot find the Codex CLI {cli!r}.")
        return str(Path(found).absolute())

    def candidates():
        yield "codex"
        for directory in os.get_exec_path():
            if Path(directory).is_absolute():
                yield Path(directory) / ("codex.exe" if sys.platform == "win32" else "codex")
        if sys.platform in ("darwin", "linux"):
            prefixes = (["/opt/homebrew", "/usr/local"] if sys.platform == "darwin" else
                        ["/home/linuxbrew/.linuxbrew", str(Path.home() / ".linuxbrew")])
            for prefix in [os.environ.get("HOMEBREW_PREFIX"), *prefixes]:
                if prefix and Path(prefix).is_absolute():
                    yield Path(prefix) / "bin/codex"
        if sys.platform == "darwin":
            for root in (Path("/Applications"), Path.home() / "Applications"):
                for app in ("ChatGPT.app", "Codex.app"):
                    for binary in ("codex-cli/bin/codex", "codex"):
                        yield root / app / "Contents/Resources" / binary
        elif sys.platform == "linux":
            launcher = shutil.which("chatgpt") or shutil.which("/usr/bin/chatgpt")
            if launcher:
                try:
                    yield Path(launcher).resolve().parent / "resources/codex-cli/bin/codex"
                except (OSError, RuntimeError):
                    pass
            for root in ("/opt/ChatGPT", "/opt/chatgpt", "/usr/lib/chatgpt"):
                yield Path(root) / "resources/codex-cli/bin/codex"
        elif sys.platform == "win32":
            local = os.environ.get("LOCALAPPDATA")
            if local and Path(local).is_absolute():
                yield Path(local) / "Programs/OpenAI/Codex/bin/codex.exe"
            try:
                powershell = powershell_path()
                if shutil.which(powershell):
                    result = subprocess.run([
                        powershell, "-NoProfile", "-NonInteractive", "-Command",
                        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
                        "Get-AppxPackage -Name OpenAI.Codex | Select-Object -ExpandProperty InstallLocation",
                    ], capture_output=True, encoding="utf-8", check=True, timeout=3)
                    for line in result.stdout.splitlines():
                        root = Path(line.strip().lstrip("\ufeff"))
                        if root.is_absolute():
                            yield from root.rglob("codex.exe")
            except (OSError, subprocess.SubprocessError, UnicodeError):
                pass

    paths = (str(Path(found).absolute()) for candidate in candidates()
             if (found := shutil.which(str(candidate))))
    try:
        return newest_verified_cli(paths)
    except (OSError, RuntimeError):
        return None
