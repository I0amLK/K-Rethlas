#!/usr/bin/env python3
"""Migrate, inspect, back up or clear the disposable vectors of one memory scope."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))
from memory_graph import MemoryGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Generation or verification memory root")
    parser.add_argument("--scope", required=True, help="Problem id or verification run id")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="Initialize/import this scope without an embedding request")
    commands.add_parser("clear-index", help="Delete only rebuildable embedding rows")
    export = commands.add_parser("export", help="Export authoritative graph as a JSON snapshot")
    export.add_argument("destination", type=Path)
    backup = commands.add_parser("backup", help="Consistent SQLite backup, including any live WAL")
    backup.add_argument("destination", type=Path)
    args = parser.parse_args()
    memory = MemoryGraph(args.root)
    if args.command == "migrate":
        snapshot = memory.export_snapshot(args.scope)
        result = {"database_path": str(memory.database_path(args.scope)),
                  "node_count": len(snapshot["nodes"]), "edge_count": len(snapshot["edges"])}
    elif args.command == "clear-index":
        result = memory.clear_index(args.scope)
    elif args.command == "backup":
        result = memory.backup(args.scope, args.destination)
    else:
        snapshot = memory.export_snapshot(args.scope)
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        with args.destination.open("x", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        result = {"export_path": str(args.destination.resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
