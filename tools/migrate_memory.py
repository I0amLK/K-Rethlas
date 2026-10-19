#!/usr/bin/env python3
"""Run the installed controller-only migration CLI without import-path injection."""
from pathlib import Path

from rethlas_core.legacy_migration_cli import main


if __name__ == "__main__":
    raise SystemExit(main(repository_root=Path(__file__).absolute().parent.parent))
