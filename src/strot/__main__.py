"""``python -m strot`` -> the CLI (strot init / strot playbook)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
