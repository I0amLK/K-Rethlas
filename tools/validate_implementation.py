"""Static execution-ledger validation; never imports the runtime package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEDGER_PATHS = {
    "M1": ("docs/implementation-status.graph.json", "docs/implementation-status.md"),
    "M2a": ("docs/milestones/m2a/status.graph.json", "docs/milestones/m2a/status.md"),
    "M2b": ("docs/milestones/m2b/status.graph.json", "docs/milestones/m2b/status.md"),
    "M2c": ("docs/milestones/m2c/status.graph.json", "docs/milestones/m2c/status.md"),
    "M2d": ("docs/milestones/m2d/status.graph.json", "docs/milestones/m2d/status.md"),
}
EXPECTED_TASK_COUNT = 15
ALLOWED_STATES = {"not_started", "implemented_unverified", "partial", "blocked", "deferred", "passed", "pending"}


def load(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("invalid JSON constant")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def validate(root: Path, milestone: str = "M1") -> list[str]:
    if milestone not in LEDGER_PATHS:
        raise ValueError("unknown milestone")
    graph_path, markdown_path = LEDGER_PATHS[milestone]
    data = load(root / graph_path)
    markdown = (root / markdown_path).read_text(encoding="utf-8")
    errors = []
    if data["schema_version"] != "1.0" or data["milestone"] != milestone:
        errors.append("unknown ledger contract")
    roadmap = load(root.parent / "analysis/upgrade-todo.graph.json")
    roadmap_nodes = roadmap["nodes"]
    if len({node["id"] for node in roadmap_nodes}) != len(roadmap_nodes):
        errors.append("duplicate roadmap node ID")
    tasks = {node["id"]: node for node in roadmap_nodes if node["kind"] == "task"}
    if len(tasks) != EXPECTED_TASK_COUNT:
        errors.append("expected 15 roadmap tasks")
    nodes = data["nodes"]
    if len({node["id"] for node in nodes}) != len(nodes):
        errors.append("duplicate node ID")
    task_nodes = [node for node in nodes if node["kind"] == "task_status"]
    if {node["roadmap_task"] for node in task_nodes} != set(tasks) or len(task_nodes) != len(tasks):
        errors.append("roadmap coverage mismatch")
    for node in task_nodes:
        original = tasks.get(node["roadmap_task"])
        if original is None:
            errors.append("unknown roadmap task: " + node["id"])
        else:
            acceptance_ids = node["acceptance_ids"]
            original_ids = [item["id"] for item in original["acceptance"]]
            if (set(acceptance_ids) != set(original_ids)
                    or len(acceptance_ids) != len(set(acceptance_ids))
                    or len(original_ids) != len(set(original_ids))):
                errors.append("acceptance IDs changed: " + node["id"])
        if any(value not in ALLOWED_STATES for value in node["states"].values()):
            errors.append("unknown state: " + node["id"])
        marker = "<!-- task:" + node["roadmap_task"] + " -->"
        if marker not in markdown:
            errors.append("missing Markdown task: " + node["id"])
        if node.get("overall") == "done":
            errors.append(milestone + " cannot claim full roadmap acceptance")
    ids = {node["id"] for node in nodes}
    for edge in data["edges"]:
        if edge["source"] not in ids or edge["target"] not in ids:
            errors.append("dangling execution edge")
    for evidence in (node for node in nodes if node["kind"] == "evidence"):
        relative = Path(evidence["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append("unsafe evidence path")
            continue
        target = root / relative
        try:
            regular = (not target.is_symlink() and root.resolve() in target.resolve().parents and target.is_file())
        except (OSError, RuntimeError):
            regular = False
        if not regular:
            errors.append("missing/nonregular evidence")
            continue
        if hashlib.sha256(target.read_bytes()).hexdigest() != evidence["sha256"]:
            errors.append("evidence digest mismatch: " + str(relative))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--milestone", choices=tuple(LEDGER_PATHS), default="M1")
    args = parser.parse_args()
    try:
        errors = validate(args.root.absolute(), milestone=args.milestone)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Execution ledger rejected: " + type(exc).__name__)
        return 1
    for error in errors:
        print(error)
    if errors:
        return 1
    print("Validated execution ledger against the unchanged static roadmap; no runtime pass inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
