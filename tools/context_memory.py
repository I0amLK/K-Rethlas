#!/usr/bin/env python3
"""Fixture-only context CLI; use the installed shared package and isolated Python."""
from pathlib import Path

from rethlas_core.context_views_cli import main


if __name__ == "__main__":
    raise SystemExit(main(repository_root=Path(__file__).absolute().parents[1]))
