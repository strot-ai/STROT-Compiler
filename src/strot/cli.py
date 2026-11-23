"""The ``strot`` CLI: wire the compiler playbook into whatever agent you use.

STROT ships the methodology as agent instructions (see ``strot/skill``). This CLI
drops the right file into your project so Claude / Codex / any agent can pick it
up, and prints the playbook for anything else.

    strot init [--agent claude|codex|generic] [--dir .]
    strot playbook
"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path

# Where each agent looks for its instructions. Same methodology, different
# envelope — all rendered from the one PLAYBOOK.md.
_TARGETS = {
    "claude": Path(".claude/skills/strot/SKILL.md"),
    "codex": Path("AGENTS.md"),
    "generic": Path("STROT_PLAYBOOK.md"),
}
_SOURCE = {"claude": "SKILL.md", "codex": "AGENTS.md", "generic": "PLAYBOOK.md"}


def _skill_dir():
    return resources.files("strot") / "skill"


def _read(name: str) -> str:
    return (_skill_dir() / name).read_text(encoding="utf-8")


def _init(agent: str, dest_root: Path) -> int:
    target = dest_root / _TARGETS[agent]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_read(_SOURCE[agent]), encoding="utf-8")
    written = [target]

    # Claude/Codex load PLAYBOOK.md + WATCHER.md + the compile script as bundled
    # resources; copy them next to the instruction file so the agent can use them.
    if agent in ("claude", "codex"):
        for doc in ("PLAYBOOK.md", "WATCHER.md"):
            (target.parent / doc).write_text(_read(doc), encoding="utf-8")
            written.append(target.parent / doc)
        scripts_dst = target.parent / "scripts"
        scripts_dst.mkdir(exist_ok=True)
        with resources.as_file(_skill_dir() / "scripts" / "compile_model.py") as src:
            shutil.copy(src, scripts_dst / "compile_model.py")
        written.append(scripts_dst / "compile_model.py")

    for p in written:
        print(f"wrote {p}")
    print()
    print("Next: install the compile extra (torch) if you haven't —")
    print('  pip install "strot-ai[local]"')
    print(f"Then point {agent} at it; it is now the STROT compiler.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="strot", description="STROT decision compiler.")
    sub = ap.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="install the compiler playbook for your agent")
    p_init.add_argument("--agent", choices=sorted(_TARGETS), default="claude")
    p_init.add_argument("--dir", default=".", help="project root to write into")

    sub.add_parser("playbook", help="print the compiler playbook to stdout")

    args = ap.parse_args(argv)
    if args.cmd == "init":
        return _init(args.agent, Path(args.dir))
    if args.cmd == "playbook":
        print(_read("PLAYBOOK.md"))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
