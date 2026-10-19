"""Real managed research CLI definitions; execution requires local authorization.

Only public/synthetic text is used. Provider credential files are absent and no
external requests are sent. COMMAND_RECORDS contain hashes, not credentials or
private invocation paths.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from rethlas_core.context_views import exact_ref
from rethlas_core.private_fs import canonical_json
from rethlas_core.research_records import candidate, relation


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/research.py"
COMMAND_RECORDS = []


class ResearchCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rethlas-research-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / "project"
        self.credentials = self.parent / "credentials"
        self.credentials.mkdir(mode=0o700)
        self.tokens = []
        self.counter = 0
        initialized = self.call("init", "--root", self.root, "--owner", "cli-test")
        self.assertEqual(initialized["status"], "initialized")
        self.config = {"version": "research-client-1", "root": str(self.root),
            "reader": None, "writer": None, "indexer": None, "runner": None,
            "embedding": {"mode": "lexical"}, "text_provider": None}
        for field, role in (("reader", "project_reader"), ("writer", "project_writer"),
                            ("runner", "research_runner")):
            path = self.credentials / (field + ".token")
            issued = self.call("issue", "--root", self.root, "--role", role, "--token-file", path)
            self.assertEqual(issued["status"], "issued")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.stat().st_nlink, 1)
            token = path.read_text(encoding="ascii")
            self.assertTrue(token and not any(char.isspace() for char in token))
            self.tokens.append(token)
            self.config[field] = {"run_id": issued["run_id"], "token_file": str(path)}
        self.config["indexer"] = deepcopy(self.config["writer"])
        self.config_path = self.parent / "client.json"
        self.write_config()
        premise, premise_r = candidate("cli.premise", "cli.premise.1", "Assumption", "Synthetic premise", "Assume x is positive. 正数前提。")
        claim, claim_r = candidate("cli.claim", "cli.claim.1", "Claim", "Synthetic conclusion", "Then x + 1 is positive. 正数加一仍为正数。")
        self.premise, self.claim = exact_ref(premise_r), exact_ref(claim_r)
        manifest = json.loads((self.root / "memory/manifest.json").read_bytes())
        self.call("commit", "--config", self.config_path, "--request", self.input({
            "version": 1, "request_id": "seed", "expected_head": manifest["head"],
            "entities": [premise, claim], "revisions": [premise_r, claim_r],
            "relations": [relation("depends_on", claim_r, premise_r)], "evidence": []}))
        self.call("index", "--config", self.config_path)

    def write_config(self):
        self.config_path.write_bytes(canonical_json(self.config))
        self.config_path.chmod(0o600)

    def input(self, value=None, *, raw=None):
        self.counter += 1
        path = self.parent / ("request-%d.json" % self.counter)
        path.write_bytes(canonical_json(value) if raw is None else raw)
        path.chmod(0o600)
        return path

    def call(self, command, *arguments, expected=0, module=False):
        entry = ["-m", "rethlas_core.research_cli"] if module else [str(TOOL)]
        process = subprocess.run([sys.executable, "-I", "-B", *entry, command, *map(str, arguments)],
            cwd=ROOT.parent, stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        COMMAND_RECORDS.append({"test": self.id(), "command": command, "module": module,
            "exit_code": process.returncode, "stdout_sha256": hashlib.sha256(process.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr).hexdigest(), "stderr_bytes": len(process.stderr),
            "provider_mode": "lexical"})
        for token in self.tokens:
            self.assertNotIn(token.encode("ascii"), process.stdout + process.stderr)
        self.assertEqual(process.returncode, expected, process.stdout.decode() + process.stderr.decode())
        self.assertEqual(process.stderr, b"", "CLI failures must not echo argv or paths")
        value = json.loads(process.stdout)
        if expected:
            self.assertEqual(set(value), {"status", "error"})
        return value

    def test_actual_tool_and_module_search_exact_body_and_context(self):
        value = self.call("search", "--config", self.config_path, "--query", "正数加一", "--top-k", "5", module=True)
        self.assertEqual(value["retrieval_mode"], "lexical")
        self.assertIn(self.claim, [item["ref"] for item in value["results"]])
        context = self.call("get", "--config", self.config_path, "--request", self.input(self.claim))
        self.assertEqual({row["revision_id"] for row in context["dto"]["revisions"]},
                         {self.claim["revision_id"], self.premise["revision_id"]})
        self.assertTrue(context["overlay"]["input_eligible"])
        self.assertTrue(all(row["assertion_state"] == "candidate" and row["validity"] == "needs_review"
                            for row in context["dto"]["revisions"]))
        navigation = self.call("expand", "--config", self.config_path, "--request", self.input({"refs": [self.claim], "depth": 1}))
        self.assertEqual(navigation["status"], "navigation_not_complete_mathematical_context")
        unknown = {**self.claim, "revision_id": "unknown"}
        self.assertEqual(self.call("get", "--config", self.config_path, "--request", self.input(unknown), expected=2)["status"], "rejected")

    def test_actual_unicode_append_is_idempotent_and_index_rebuild_is_explicit(self):
        request = {"problem_id": "中文/一致连续", "channel": "events", "request_id": "unicode-record",
                   "record": {"source": "https://example.invalid/public-synthetic", "text": "公开合成来源"}}
        first = self.call("append", "--config", self.config_path, "--request", self.input(request))
        self.assertEqual(first["problem_id"], request["problem_id"])
        self.assertEqual(first["status"], "committed_candidate")
        second = self.call("append", "--config", self.config_path, "--request", self.input(request))
        self.assertEqual(first["ref"], second["ref"])
        self.assertTrue(second["commit"]["replayed"])
        stale = self.call("search", "--config", self.config_path, "--query", "公开合成来源", expected=3)
        self.assertEqual(stale["status"], "conflict")
        self.call("index", "--config", self.config_path)
        found = self.call("search", "--config", self.config_path, "--query", "公开合成来源")
        source = next(item for item in found["results"] if item["ref"] == first["ref"])
        self.assertEqual(source["type"], "Reference")
        self.assertEqual(source["metadata"], {"problem_id": request["problem_id"], "channel": "events"})
        (self.root / "index/current.json").unlink()
        self.assertEqual(self.call("search", "--config", self.config_path, "--query", "公开合成来源", expected=2)["error"], "FileNotFoundError")
        self.call("index", "--config", self.config_path)
        self.assertTrue(self.call("search", "--config", self.config_path, "--query", "公开合成来源")["results"])
        self.assertFalse(list(self.parent.rglob("*.jsonl")))

    def test_actual_withdrawal_changes_current_but_never_erases_bodies(self):
        bodies = {path.name: path.read_bytes() for path in (self.root / "memory/bodies").iterdir() if path.is_file()}
        value = self.call("withdraw", "--config", self.config_path, "--request", self.input({
            "version": "project-1", "request_id": "withdraw-premise", "targets": [self.premise],
            "reason": "Synthetic current-use regression"}))
        self.assertEqual(value["status"], "withdrawn")
        self.assertEqual(self.call("search", "--config", self.config_path, "--query", "正数")["count"], 0)
        self.call("get", "--config", self.config_path, "--request", self.input(self.claim), expected=2)
        status = self.call("status", "--config", self.config_path)
        self.assertTrue(status["current"][self.claim["revision_id"]]["math_blocked"])
        self.assertEqual({path.name: path.read_bytes() for path in (self.root / "memory/bodies").iterdir() if path.is_file()}, bodies)

    def test_actual_readonly_and_bad_json_never_create_legacy_files(self):
        self.config["writer"] = self.config["indexer"] = self.config["runner"] = None
        self.write_config()
        self.call("append", "--config", self.config_path, "--request", self.input({
            "problem_id": "p", "channel": "events", "record": {"text": "candidate"}}), expected=2)
        self.call("append", "--config", self.config_path, "--request", self.input(raw=b'{"x":1,"x":2}'), expected=2)
        self.call("search", "--config", self.config_path, "--query", "正数", "--top-k", "0", expected=2)
        self.assertFalse(list(self.parent.rglob("*.jsonl")))

    def test_actual_parser_errors_and_existing_token_output_are_redacted(self):
        for arguments in (("--owner", "private-invalid-role"), ("--root", self.root, "--owner", "owner", "--secret-option", "private-value")):
            value = self.call("init", *arguments, expected=2)
            self.assertEqual(value, {"status": "rejected", "error": "ValueError"})
        path = Path(self.config["reader"]["token_file"])
        before = path.read_bytes()
        self.call("issue", "--root", self.root, "--role", "project_reader", "--token-file", path, expected=2)
        self.assertEqual(path.read_bytes(), before)

    def test_actual_text_approval_is_operator_only_and_no_request_is_sent(self):
        self.config["embedding"] = {"mode": "bge_live", "endpoint": "https://example.invalid/v1/embeddings",
                                    "key_file": str(self.parent / "never-created-secret")}
        self.write_config()
        self.call("approve-online", "--config", self.config_path, "--request", self.input({
            "allowed_sha256": [hashlib.sha256("public synthetic".encode()).hexdigest()],
            "materials": "public_or_synthetic"}))
        messages = [{"role": "system", "content": "Review only public synthetic mathematics."},
                    {"role": "user", "content": "Assume x > 0. Then x + 1 > 0."}]
        self.call("approve-text", "--config", self.config_path, "--request", self.input({
            "messages": messages, "purpose": "review", "materials": "public_or_synthetic"}))
        used = self.call("online-usage", "--config", self.config_path)["used"]
        self.assertEqual(used.get("text_requests", 0), 0)
        self.assertEqual(used.get("embedding_requests", 0), 0)
        self.assertFalse((self.parent / "never-created-secret").exists())


class ResearchCliRoutingTests(unittest.TestCase):
    def invoke(self, arguments):
        from rethlas_core.research_cli import main
        output = StringIO()
        with redirect_stdout(output):
            code = main(arguments)
        return code, json.loads(output.getvalue())

    def test_status_has_no_provider_dependency(self):
        from types import SimpleNamespace
        from rethlas_core import research_cli
        config, service, retrieval = {}, object(), object()
        memory, selected = SimpleNamespace(reader={}), {"run_id": "fixed", "token": "not-printed"}
        expected = {"request_status": "bound", "historical": {"stage": "awaiting_review_approval"}}
        with patch.object(research_cli, "connect", return_value=(config, service, retrieval, memory)) as connect, \
             patch.object(research_cli, "grant", return_value=selected), \
             patch.object(research_cli, "configured_text", side_effect=AssertionError("must stay offline")), \
             patch("rethlas_core.research.ResearchPipeline") as pipeline:
            pipeline.return_value.status.return_value = expected
            code, value = self.invoke(["research-status", "--config", "/operator/client.json", "--request-id", "research-one"])
        self.assertEqual((code, value), (0, expected))
        connect.assert_called_once_with(Path("/operator/client.json"), providers=False)
        pipeline.assert_called_once_with(service, retrieval, None)
        pipeline.return_value.status.assert_called_once_with(selected, request_id="research-one")

    def test_explicit_revise_review_and_bound_approval_are_returned_unchanged(self):
        from types import SimpleNamespace
        from rethlas_core import research_cli
        selected = {"run_id": "fixed", "token": "not-printed"}
        provider = object()
        cases = {
            "run": {"request_id": "new", "question": "public synthetic"},
            "revise": {"request_id": "new", "parent_request_id": "parent", "instructions": "address the gap"},
            "review": {"request_id": "new", "proof_ref": {"entity_id": "p", "revision_id": "p.1", "body_sha256": "0" * 64}, "statement": "public synthetic"},
        }
        for command, request in cases.items():
            expected = {"request_status": "bound", "historical": {"stage": "awaiting_generation_approval" if command != "review" else "awaiting_review_approval"},
                        "approval_required": {"purpose": "generation" if command != "review" else "review", "messages": [], "input_sha256": "0" * 64}}
            with self.subTest(command=command), patch.object(research_cli, "connect", return_value=({"version": "research-client-1"}, "service", "retrieval", SimpleNamespace(reader={}))) as connect, \
                 patch.object(research_cli, "grant", return_value=selected), \
                 patch.object(research_cli, "configured_text", return_value=provider), \
                 patch.object(research_cli, "_request", return_value=request), \
                 patch("rethlas_core.research.ResearchPipeline") as pipeline:
                getattr(pipeline.return_value, command).return_value = expected
                code, value = self.invoke([command, "--config", "/operator/client.json", "--request", "/operator/request.json"])
            self.assertEqual((code, value), (0, expected))
            connect.assert_called_once_with(Path("/operator/client.json"), providers=command in {"run", "revise"})
            getattr(pipeline.return_value, command).assert_called_once_with(selected, **request)

    def test_uncertain_provider_outcome_is_not_rejected_as_no_effect(self):
        from rethlas_core import research_cli
        from rethlas_core.provider import ProviderTimeout
        with patch.object(research_cli, "connect", side_effect=ProviderTimeout("never echo private path or credential")):
            self.assertEqual(self.invoke(["status", "--config", "/operator/client.json"]),
                             (4, {"status": "indeterminate", "error": "ProviderTimeout"}))


if __name__ == "__main__":
    unittest.main()
