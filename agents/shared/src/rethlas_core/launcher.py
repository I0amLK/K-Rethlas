"""Read-only inspection of legacy runs; no model launch or file-based promotion."""
from __future__ import annotations

from pathlib import Path
import re


def production_completed() -> bool:
    # M1 has no production review driver. Fixture receipts and legacy filenames
    # cannot satisfy completion, even if their own JSON says otherwise.
    return False


def inspect_run(generation_root: Path, problem: str) -> dict:
    root = generation_root.absolute()
    path = Path(problem)
    if (path.is_absolute() or ".." in path.parts or path.suffix != ".md"
            or not path.parts or path.parts[0] != "data"):
        raise ValueError("problem must be a relative data/*.md path")
    source = root / path
    if any(parent.is_symlink() for parent in (source, *source.parents)) or not source.is_file():
        raise ValueError("problem must be a real file within generation data")
    problem_id = path.relative_to("data").with_suffix("").as_posix()
    results = root / "results" / problem_id
    log_dir = root / "logs" / problem_id / "iter"
    if any(parent.is_symlink() for target in (results, log_dir) for parent in (target, *target.parents)):
        raise ValueError("run paths cannot contain symlinks")
    log_paths = sorted(log_dir.glob(path.stem + "_iter_*.md")) if log_dir.is_dir() else []
    sessions, iterations = set(), []
    for log in log_paths:
        if log.is_symlink() or not log.is_file() or log.stat().st_size > 8388608:
            raise ValueError("invalid legacy log")
        match = re.search(r"_iter_([0-9]+)\.md$", log.name)
        if match:
            iterations.append(int(match.group(1)))
        for line in log.read_text(encoding="utf-8").splitlines():
            if "session id: " in line:
                sessions.add(line.split("session id: ", 1)[1].strip())
                break
    if len(sessions) > 1 or (iterations and not sessions):
        raise ValueError("ambiguous or missing legacy session identity")
    return {
        "problem_id": problem_id, "next_iteration": max(iterations, default=-1) + 1,
        "resume_available": bool(sessions), "pause_present": (results / "PAUSE_AFTER_ITERATION").exists(),
        "legacy_verified_file_present": (results / "blueprint_verified.md").is_file(),
        "status": "unreviewed", "production_completed": production_completed(),
        "live_available": False,
    }
