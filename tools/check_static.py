"""Parse changed Python/JSON/TOML without importing or installing project code."""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from jsonschema import Draft202012Validator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes, failures, counts = {}, [], {"python": 0, "json": 0, "toml": 0, "schemas": 0, "test_methods_written": 0}
    directories = (root / "agents/shared", root / "tests", root / "tools", root / "docs/examples")
    paths = {path for directory in directories for path in directory.rglob("*")
             if path.is_file() and not any(part.endswith(".egg-info") or part == "__pycache__" for part in path.parts)}
    paths.update(root / relative for relative in (
        "agents/verification/api/server.py", "agents/verification/mcp/server.py",
        "agents/verification/scripts/test_verify_endpoint.py", "agents/generation/mcp/__init__.py",
        "agents/generation/mcp/server.py", "agents/verification/.codex/config.toml",
        "agents/generation/.codex/config.toml", "agents/generation/.codex/agents/subgoal-prover.toml"))
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            if path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
                compile(tree, relative, "exec")  # Compile only; never exec/import it.
                counts["python"] += 1
                if path.parent == root / "tests":
                    counts["test_methods_written"] += sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_") for node in ast.walk(tree))
            elif path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                counts["json"] += 1
                if path.name.endswith(".schema.json"):
                    Draft202012Validator.check_schema(value)
                    counts["schemas"] += 1
            elif path.suffix == ".toml":
                tomllib.loads(path.read_text(encoding="utf-8"))
                counts["toml"] += 1
        except (ValueError, SyntaxError) as exc:
            failures.append({"path": relative, "error": type(exc).__name__})
    for name, schema_name in (("policy", "policy"), ("math-memory", "math_memory"), ("report", "verification_output")):
        schema = json.loads((root / f"agents/shared/src/rethlas_core/schemas/{schema_name}.schema.json").read_text())
        example = json.loads((root / f"docs/examples/{name}.json").read_text())
        for error in Draft202012Validator(schema).iter_errors(example):
            failures.append({"example": name, "error": error.message})
    shell_checks = []
    for relative in ("agents/generation/tests/run_example.sh", "agents/generation/site/serve.sh"):
        result = subprocess.run(["bash", "-n", str(root / relative)], capture_output=True, text=True)
        shell_checks.append({"path": relative, "exit_code": result.returncode, "stderr": result.stderr})
        if result.returncode:
            failures.append({"shell": relative})
    diff = subprocess.run(["git", "-C", str(root), "diff", "--check"], capture_output=True, text=True)
    if diff.returncode:
        failures.append({"diff_check": diff.stdout + diff.stderr})
    packages = sorted(f"{distribution.metadata['Name']}=={distribution.version}" for distribution in metadata.distributions())
    summary = {"scope": "static_parse_not_runtime_validation", "python": sys.version, "counts": counts,
        "failures": failures, "shell_checks": shell_checks, "git_diff_check_exit": diff.returncode,
        "project_imported": False,
        "package_installed": any(item.lower().startswith("rethlas-core==") for item in packages),
        "runtime_tests_executed": False,
        "file_hashes": hashes}
    (args.output / "static-checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "dependencies.txt").write_text("\n".join(packages) + "\n")
    print(json.dumps({"counts": counts, "failures": failures, "runtime_tests_executed": False}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
