"""Tests for the packaged agent playbook + the `strot` CLI (torch-free).

STROT ships its methodology as agent instructions (strot/skill) and a tiny CLI
that wires them into whatever agent you use. These verify the files are present
and self-consistent, and that `strot init` / `strot playbook` work.
"""

from importlib import resources

import strot
from strot import cli


def _skill(name):
    return (resources.files("strot") / "skill" / name).read_text(encoding="utf-8")


def test_playbook_describes_the_llm_free_flow():
    pb = _skill("PLAYBOOK.md")
    # The whole point: the agent is the teacher, STROT holds no key.
    assert "You are the teacher" in pb
    assert "no API key" in pb or "no key" in pb
    assert "strot.compile(" in pb
    # The loop steps are present.
    for step in ("INSPECT", "DESIGN", "LABEL", "COMPILE", "VERIFY", "RECOMPILE"):
        assert step in pb, step


def test_skill_and_agents_files_point_at_the_playbook_and_script():
    for fname in ("SKILL.md", "AGENTS.md"):
        text = _skill(fname)
        assert "PLAYBOOK.md" in text
        assert "compile_model.py" in text
    # SKILL.md carries the Claude frontmatter.
    assert _skill("SKILL.md").startswith("---\nname: strot-compiler")


def test_compile_is_the_public_build_verb():
    assert "compile" in strot.__all__


def test_cli_playbook_prints_the_methodology(capsys):
    rc = cli.main(["playbook"])
    out = capsys.readouterr().out
    assert rc == 0 and "You are the teacher" in out


def test_cli_init_writes_the_claude_skill(tmp_path, capsys):
    rc = cli.main(["init", "--agent", "claude", "--dir", str(tmp_path)])
    assert rc == 0
    skill = tmp_path / ".claude" / "skills" / "strot"
    assert (skill / "SKILL.md").exists()
    assert (skill / "PLAYBOOK.md").exists()
    assert (skill / "scripts" / "compile_model.py").exists()


def test_cli_init_codex_writes_agents_md(tmp_path):
    cli.main(["init", "--agent", "codex", "--dir", str(tmp_path)])
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "PLAYBOOK.md").exists()


def test_cli_init_generic_is_a_single_playbook(tmp_path):
    cli.main(["init", "--agent", "generic", "--dir", str(tmp_path)])
    assert (tmp_path / "STROT_PLAYBOOK.md").exists()
