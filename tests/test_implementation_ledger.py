"""Isolated synthetic ledger checks; never assert acceptance of a repository ledger."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "implementation_ledger_validator", Path(__file__).resolve().parents[1] / "tools/validate_implementation.py"
)
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


class ImplementationLedgerTests(unittest.TestCase):
    PATHS = {
        "M1": ("docs/implementation-status.graph.json", "docs/implementation-status.md"),
        "M2a": ("docs/milestones/m2a/status.graph.json", "docs/milestones/m2a/status.md"),
        "M2b": ("docs/milestones/m2b/status.graph.json", "docs/milestones/m2b/status.md"),
        "M2c": ("docs/milestones/m2c/status.graph.json", "docs/milestones/m2c/status.md"),
        "M2d": ("docs/milestones/m2d/status.graph.json", "docs/milestones/m2d/status.md"),
    }

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "Rethlas"
        self.root.mkdir()
        (self.workspace / "analysis").mkdir()
        self.roadmap = {"nodes": [
            {"id": f"task-{index:02d}", "kind": "task", "acceptance": [
                {"id": f"task-{index:02d}-A1"}, {"id": f"task-{index:02d}-A2"},
            ]}
            for index in range(15)
        ]}
        self.write_roadmap()
        self.evidence = self.root / "evidence/synthetic.txt"
        self.evidence.parent.mkdir()
        self.evidence.write_bytes(b"Synthetic static fixture; not runtime execution evidence.\n")
        task_nodes = [
            {"id": "status-" + task["id"], "kind": "task_status", "roadmap_task": task["id"],
             "acceptance_ids": [item["id"] for item in task["acceptance"]],
             "states": {"implementation": "implemented_unverified", "tests": "pending", "acceptance": "blocked"},
             "overall": "partial"}
            for task in self.roadmap["nodes"]
        ]
        nodes = task_nodes + [{
            "id": "evidence-synthetic", "kind": "evidence", "path": "evidence/synthetic.txt",
            "sha256": hashlib.sha256(self.evidence.read_bytes()).hexdigest(),
        }]
        edges = [{"source": node["id"], "target": "evidence-synthetic"} for node in task_nodes]
        self.graphs = {}
        for milestone, (graph_path, markdown_path) in self.PATHS.items():
            (self.root / graph_path).parent.mkdir(parents=True, exist_ok=True)
            self.graphs[milestone] = {
                "schema_version": "1.0", "milestone": milestone,
                "nodes": deepcopy(nodes), "edges": deepcopy(edges),
            }
            self.write_graph(milestone)
            (self.root / markdown_path).write_text(
                "\n".join("<!-- task:" + task["id"] + " -->" for task in self.roadmap["nodes"]),
                encoding="utf-8",
            )

    def write_roadmap(self):
        (self.workspace / "analysis/upgrade-todo.graph.json").write_text(
            json.dumps(self.roadmap), encoding="utf-8"
        )

    def write_graph(self, milestone):
        (self.root / self.PATHS[milestone][0]).write_text(json.dumps(self.graphs[milestone]), encoding="utf-8")

    def test_default_and_explicit_m1_are_compatible(self):
        self.assertEqual(ledger.validate(self.root), [])
        self.assertEqual(ledger.validate(self.root, "M1"), [])
        self.assertEqual(ledger.validate(self.root, milestone="M1"), [])
        self.graphs["M2a"]["milestone"] = "M1"
        self.write_graph("M2a")
        self.assertEqual(ledger.validate(self.root), [])
        self.assertIn("unknown ledger contract", ledger.validate(self.root, "M2a"))

    def test_m2a_uses_only_its_selected_paths(self):
        for path in self.PATHS["M1"]:
            (self.root / path).unlink()
        self.assertEqual(ledger.validate(self.root, milestone="M2a"), [])
        with self.assertRaises(FileNotFoundError):
            ledger.validate(self.root)

    def test_missing_selected_m2a_files_do_not_fall_back_to_m1(self):
        for path in self.PATHS["M2a"]:
            with self.subTest(path=path):
                target = self.root / path
                original = target.read_bytes()
                target.unlink()
                with self.assertRaises(FileNotFoundError):
                    ledger.validate(self.root, "M2a")
                target.write_bytes(original)
        self.assertEqual(ledger.validate(self.root), [])

    def test_m2b_uses_only_its_selected_paths(self):
        for milestone in ("M1", "M2a"):
            for path in self.PATHS[milestone]:
                (self.root / path).unlink()
        self.assertEqual(ledger.validate(self.root, milestone="M2b"), [])
        for milestone in ("M1", "M2a"):
            with self.subTest(milestone=milestone), self.assertRaises(FileNotFoundError):
                ledger.validate(self.root, milestone)

    def test_missing_selected_m2b_files_do_not_fall_back(self):
        for path in self.PATHS["M2b"]:
            with self.subTest(path=path):
                target = self.root / path
                original = target.read_bytes()
                target.unlink()
                with self.assertRaises(FileNotFoundError):
                    ledger.validate(self.root, "M2b")
                self.assertEqual(ledger.validate(self.root, "M1"), [])
                self.assertEqual(ledger.validate(self.root, "M2a"), [])
                target.write_bytes(original)

    def test_m2c_uses_only_selected_files_without_fallback(self):
        for milestone in ("M1", "M2a", "M2b"):
            for path in self.PATHS[milestone]:
                (self.root / path).unlink()
        self.assertEqual(ledger.validate(self.root, "M2c"), [])
        (self.root / self.PATHS["M2c"][0]).unlink()
        with self.assertRaises(FileNotFoundError):
            ledger.validate(self.root, "M2c")

    def test_cli_selects_m2c_and_rejects_missing_selected_file(self):
        for milestone in ("M1", "M2a", "M2b"):
            (self.root / self.PATHS[milestone][0]).unlink()
        argv = ["validate_implementation", "--root", str(self.root), "--milestone", "M2c"]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            self.assertEqual(ledger.main(), 0)
        (self.root / self.PATHS["M2c"][0]).unlink()
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            self.assertEqual(ledger.main(), 1)

    def test_m2d_uses_only_selected_files_without_fallback(self):
        for milestone in ("M1", "M2a", "M2b", "M2c"):
            for path in self.PATHS[milestone]:
                (self.root / path).unlink()
        self.assertEqual(ledger.validate(self.root, "M2d"), [])
        (self.root / self.PATHS["M2d"][0]).unlink()
        with self.assertRaises(FileNotFoundError):
            ledger.validate(self.root, "M2d")

    def test_cli_selects_m2d_and_rejects_missing_selected_file(self):
        for milestone in ("M1", "M2a", "M2b", "M2c"):
            (self.root / self.PATHS[milestone][0]).unlink()
        argv = ["validate_implementation", "--root", str(self.root), "--milestone", "M2d"]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            self.assertEqual(ledger.main(), 0)
        (self.root / self.PATHS["M2d"][0]).unlink()
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            self.assertEqual(ledger.main(), 1)

    def test_unknown_milestone_is_rejected(self):
        with self.assertRaises(ValueError):
            ledger.validate(self.root, "M2e")

    def test_milestone_and_schema_must_match_selected_contract(self):
        for milestone in self.PATHS:
            for field, value in (("milestone", "M2a" if milestone == "M1" else "M1"),
                                 ("schema_version", "2.0")):
                with self.subTest(milestone=milestone, field=field):
                    original = self.graphs[milestone][field]
                    self.graphs[milestone][field] = value
                    self.write_graph(milestone)
                    self.assertIn("unknown ledger contract", ledger.validate(self.root, milestone))
                    self.graphs[milestone][field] = original
                    self.write_graph(milestone)

    def test_missing_task_is_rejected_for_all_milestones(self):
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                del self.graphs[milestone]["nodes"][0]
                self.write_graph(milestone)
                self.assertIn("roadmap coverage mismatch", ledger.validate(self.root, milestone))

    def test_duplicate_task_reference_is_not_complete_coverage(self):
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                nodes = self.graphs[milestone]["nodes"]
                nodes[1]["roadmap_task"] = nodes[0]["roadmap_task"]
                self.write_graph(milestone)
                self.assertIn("roadmap coverage mismatch", ledger.validate(self.root, milestone))

    def test_unknown_task_is_rejected_without_skipping_other_checks(self):
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                node = self.graphs[milestone]["nodes"][0]
                node["roadmap_task"] = "unknown-task"
                node["states"]["tests"] = "invented-state"
                self.write_graph(milestone)
                errors = ledger.validate(self.root, milestone)
                self.assertIn("roadmap coverage mismatch", errors)
                self.assertIn("unknown roadmap task: " + node["id"], errors)
                self.assertIn("unknown state: " + node["id"], errors)

    def test_roadmap_must_still_contain_fifteen_tasks(self):
        self.roadmap["nodes"].pop()
        self.write_roadmap()
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                # Even matching a shortened roadmap does not satisfy the contract.
                del self.graphs[milestone]["nodes"][14]
                del self.graphs[milestone]["edges"][14]
                self.write_graph(milestone)
                self.assertIn("expected 15 roadmap tasks", ledger.validate(self.root, milestone))

    def test_acceptance_ids_cannot_be_missing_changed_or_duplicated(self):
        for milestone in self.PATHS:
            node = self.graphs[milestone]["nodes"][0]
            original = node["acceptance_ids"][:]
            for name, ids in (("missing", original[:-1]), ("changed", [original[0], "foreign-ID"]),
                              ("duplicate", original + [original[0]])):
                with self.subTest(milestone=milestone, mutation=name):
                    node["acceptance_ids"] = ids
                    self.write_graph(milestone)
                    self.assertIn("acceptance IDs changed: " + node["id"], ledger.validate(self.root, milestone))

    def test_duplicate_node_ids_are_rejected(self):
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                self.graphs[milestone]["nodes"].append(deepcopy(self.graphs[milestone]["nodes"][-1]))
                self.write_graph(milestone)
                self.assertIn("duplicate node ID", ledger.validate(self.root, milestone))

    def test_duplicate_roadmap_node_and_acceptance_ids_are_rejected(self):
        self.roadmap["nodes"].append(deepcopy(self.roadmap["nodes"][0]))
        self.roadmap["nodes"][1]["acceptance"].append(deepcopy(self.roadmap["nodes"][1]["acceptance"][0]))
        self.write_roadmap()
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                errors = ledger.validate(self.root, milestone)
                self.assertIn("duplicate roadmap node ID", errors)
                self.assertIn("acceptance IDs changed: status-task-01", errors)

    def test_existing_state_vocabulary_remains_accepted(self):
        states = ("not_started", "implemented_unverified", "partial", "blocked", "deferred", "passed", "pending")
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                self.graphs[milestone]["nodes"][0]["states"] = {"axis-" + str(i): state for i, state in enumerate(states)}
                self.write_graph(milestone)
                self.assertEqual(ledger.validate(self.root, milestone), [])

    def test_unknown_state_and_overall_done_are_rejected(self):
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                node = self.graphs[milestone]["nodes"][0]
                node["states"]["tests"] = "invented-state"
                node["overall"] = "done"
                self.write_graph(milestone)
                errors = ledger.validate(self.root, milestone)
                self.assertIn("unknown state: " + node["id"], errors)
                self.assertIn(milestone + " cannot claim full roadmap acceptance", errors)

    def test_missing_markdown_task_marker_is_rejected(self):
        for milestone, (_, markdown_path) in self.PATHS.items():
            with self.subTest(milestone=milestone):
                (self.root / markdown_path).write_text("Synthetic text without task markers.", encoding="utf-8")
                self.assertIn("missing Markdown task: status-task-00", ledger.validate(self.root, milestone))

    def test_dangling_source_and_target_are_rejected(self):
        for milestone in self.PATHS:
            edge = self.graphs[milestone]["edges"][0]
            for endpoint in ("source", "target"):
                with self.subTest(milestone=milestone, endpoint=endpoint):
                    original = edge[endpoint]
                    edge[endpoint] = "missing-node"
                    self.write_graph(milestone)
                    self.assertIn("dangling execution edge", ledger.validate(self.root, milestone))
                    edge[endpoint] = original

    def test_bad_evidence_hash_is_rejected(self):
        for milestone in self.PATHS:
            for bad_hash in ("0" * 64, "not-a-sha256"):
                with self.subTest(milestone=milestone, bad_hash=bad_hash):
                    self.graphs[milestone]["nodes"][-1]["sha256"] = bad_hash
                    self.write_graph(milestone)
                    self.assertIn("evidence digest mismatch: evidence/synthetic.txt", ledger.validate(self.root, milestone))

    def test_changed_evidence_bytes_are_rejected(self):
        self.evidence.write_bytes(b"Changed synthetic evidence.\n")
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                self.assertIn("evidence digest mismatch: evidence/synthetic.txt", ledger.validate(self.root, milestone))

    def test_unsafe_evidence_paths_are_rejected(self):
        for milestone in self.PATHS:
            for unsafe in ("../outside.txt", str(self.evidence)):
                with self.subTest(milestone=milestone, path=unsafe):
                    self.graphs[milestone]["nodes"][-1]["path"] = unsafe
                    self.write_graph(milestone)
                    self.assertIn("unsafe evidence path", ledger.validate(self.root, milestone))

    def test_missing_directory_and_symlink_evidence_are_rejected(self):
        symlink = self.root / "evidence/link.txt"
        symlink.symlink_to(self.evidence)
        for milestone in self.PATHS:
            for relative in ("evidence/missing.txt", "evidence", "evidence/link.txt"):
                with self.subTest(milestone=milestone, path=relative):
                    self.graphs[milestone]["nodes"][-1]["path"] = relative
                    self.write_graph(milestone)
                    self.assertIn("missing/nonregular evidence", ledger.validate(self.root, milestone))

    def test_parent_symlink_cannot_escape_evidence_root(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        (outside / "synthetic.txt").write_bytes(self.evidence.read_bytes())
        (self.root / "external-evidence").symlink_to(outside, target_is_directory=True)
        for milestone in self.PATHS:
            with self.subTest(milestone=milestone):
                self.graphs[milestone]["nodes"][-1]["path"] = "external-evidence/synthetic.txt"
                self.write_graph(milestone)
                self.assertIn("missing/nonregular evidence", ledger.validate(self.root, milestone))

    def test_duplicate_json_keys_and_nonfinite_constants_stay_rejected(self):
        for milestone, (graph_path, _) in self.PATHS.items():
            for invalid in ('{"schema_version":"1.0","schema_version":"1.0"}', '{"value":NaN}'):
                with self.subTest(milestone=milestone, data=invalid):
                    (self.root / graph_path).write_text(invalid, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        ledger.validate(self.root, milestone)

    def test_cli_default_remains_m1(self):
        (self.root / self.PATHS["M2a"][0]).unlink()
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root)]), redirect_stdout(StringIO()) as output:
            self.assertEqual(ledger.main(), 0)
        self.assertIn("no runtime pass inferred", output.getvalue())

    def test_cli_selects_m2a(self):
        (self.root / self.PATHS["M1"][0]).unlink()
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root), "--milestone", "M2a"]), redirect_stdout(StringIO()):
            self.assertEqual(ledger.main(), 0)

    def test_cli_missing_m2a_is_rejected_not_reported_as_validated(self):
        (self.root / self.PATHS["M2a"][0]).unlink()
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root), "--milestone", "M2a"]), redirect_stdout(StringIO()) as output:
            self.assertEqual(ledger.main(), 1)
        self.assertIn("Execution ledger rejected", output.getvalue())
        self.assertNotIn("Validated execution ledger", output.getvalue())

    def test_cli_selects_m2b_without_other_ledgers(self):
        for milestone in ("M1", "M2a"):
            (self.root / self.PATHS[milestone][0]).unlink()
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root), "--milestone", "M2b"]), redirect_stdout(StringIO()) as output:
            self.assertEqual(ledger.main(), 0)
        self.assertIn("no runtime pass inferred", output.getvalue())

    def test_cli_missing_m2b_is_rejected_not_reported_as_validated(self):
        (self.root / self.PATHS["M2b"][0]).unlink()
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root), "--milestone", "M2b"]), redirect_stdout(StringIO()) as output:
            self.assertEqual(ledger.main(), 1)
        self.assertIn("Execution ledger rejected", output.getvalue())
        self.assertNotIn("Validated execution ledger", output.getvalue())

    def test_cli_default_remains_m1_when_m2b_is_absent(self):
        (self.root / self.PATHS["M2b"][0]).unlink()
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root)]), redirect_stdout(StringIO()):
            self.assertEqual(ledger.main(), 0)

    def test_cli_rejects_unknown_milestone(self):
        with patch.object(sys, "argv", ["validate_implementation", "--root", str(self.root), "--milestone", "M2e"]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                ledger.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
