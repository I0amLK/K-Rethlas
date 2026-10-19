#!/usr/bin/env python3
"""Fixture-only validity CLI; use the installed package and isolated Python."""
from pathlib import Path

from rethlas_core.validity_cli import main


if __name__ == "__main__":
    raise SystemExit(main(repository_root=Path(__file__).absolute().parents[1]))
