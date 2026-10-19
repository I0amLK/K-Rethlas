"""Actual subprocess CLI interactions; synthetic fixture bytes only."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from rethlas_core.authorization import ROLE_ACTIONS
from rethlas_core.private_fs import digest

from test_context_views import ContextFixtureMixin, VERSION, active, manifest_for
from test_memory_store import encoded, write_private


REPOSITORY = Path(__file__).absolute().parents[1]
TOOL = REPOSITORY / "tools/context_memory.py"
COMMAND_RECORDS = []


def archive_output(value):
    """Keep tests/denials/counts, never graph/DTO bodies or authority credentials."""
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if key in {"graph", "bundle", "revisions", "relations"} and type(item) in (dict, list):
                result[key] = {"sha256": digest(item), "count": len(item)}
            elif key == "token":
                result[key] = "[REDACTED]"
            else:
                result[key] = archive_output(item)
        return result
    if type(value) is list:
        return [archive_output(v) for v in value]
    return value


class ContextCliTests(ContextFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.credentials = self.base / "credentials"
        self.credentials.mkdir(mode=0o700)
        self.paths = {}
        for name, grant in self.grants.items():
            path = self.credentials / name
            write_private(path, grant["token"].encode("ascii"))
            self.paths[name] = path
        self.source_path = self.credentials / "source"
        write_private(self.source_path, self.source_token.encode("ascii"))
        self.counter = 0

    def input(self, value):
        self.counter += 1
        path = self.base / ("request-%d.json" % self.counter)
        write_private(path, encoded(value))
        return path

    def call(self, command, *args, expected=0, root=None):
        argv = [sys.executable, "-I", "-B", str(TOOL), command, "--workspace", str(root or self.workspace.root),
                *map(str, args)]
        result = subprocess.run(argv, cwd=REPOSITORY.parent, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        text = result.stdout.decode("utf-8")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = {"unparsed_stdout": text}
        COMMAND_RECORDS.append({"test": self.id(), "argv": argv, "exit_code": result.returncode,
                                "stdout": archive_output(value), "stderr": result.stderr.decode("utf-8")})
        for grant in self.grants.values():
            self.assertNotIn(grant["token"], text + result.stderr.decode("utf-8"))
        self.assertEqual(result.returncode, expected, result.stderr.decode("utf-8") + text)
        return value

    def auth(self, name="left"):
        return ["--token-file", self.paths[name], "--run-id", self.grants[name]["run_id"]]

    def seal_cli(self, name):
        before = self.call("branch-read", *self.auth(name))
        request = {"version": VERSION, "request_id": "seal-" + name, "expected_head": before["head"], "active": active(before["graph"])}
        return self.call("seal", *self.auth(name), "--request", self.input(request))

    def test_actual_init_issue_and_public_branch_read(self):
        path = self.base / "cli-workspace"
        result = self.call("init", "--source-root", self.source.root, "--source-token-file", self.source_path,
            "--source-run-id", "source-run", "--spec", self.input(self.spec), root=path)
        self.assertEqual(result["status"], "initialized")
        credential = self.credentials / "cli"
        grant = self.call("issue", "--kind", "branch", "--branch-id", "left", "--action", "context.branch.read",
                          "--token-file", credential, root=path)
        exported = self.call("branch-read", "--token-file", credential, "--run-id", grant["run_id"], root=path)
        self.assertEqual(exported["graph"]["revisions"], self.read_branch()["graph"]["revisions"])
        self.call("issue", "--kind", "branch", "--branch-id", "left", "--action", "context.branch.read",
                  "--token-file", credential, root=path, expected=2)

    def test_actual_branch_commit_replay_cas_and_cross_branch_denial(self):
        right = self.call("branch-read", *self.auth("right"))
        request = self.branch_request()
        path = self.input(request)
        committed = self.call("branch-commit", *self.auth(), "--request", path)
        replay = self.call("branch-commit", *self.auth(), "--request", path)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], committed["head"])
        modified = deepcopy(request)
        modified["request_id"] = "different"
        self.call("branch-commit", *self.auth(), "--request", self.input(modified), expected=3)
        self.call("branch-read", "--token-file", self.paths["left"], "--run-id", self.grants["right"]["run_id"], expected=2)
        self.assertEqual(self.call("branch-read", *self.auth("right")), right)

    def test_actual_seal_barrier_and_closed_replay(self):
        request = self.branch_request()
        self.call("branch-commit", *self.auth(), "--request", self.input(request))
        self.seal_cli("left")
        self.call("branch-read", *self.auth(), expected=2)
        self.call("branch-commit", *self.auth(), "--request", self.input(request), expected=2)
        self.call("issue", "--kind", "join", "--action", "context.join.read", "--token-file", self.credentials / "join", expected=2)
        self.seal_cli("right")
        result = self.call("recover", "--kind", "branch", "--branch-id", "left")
        self.assertEqual(result["status"], "sealed")

    def test_actual_conflict_resolution_join_and_view_isolated_probe(self):
        for name in ("left", "right"):
            self.call("branch-commit", *self.auth(name), "--request", self.input(self.branch_request(name)))
            self.seal_cli(name)
        join_token = self.credentials / "join"
        grant = self.call("issue", "--kind", "join", "--action", "context.join.prepare", "--action", "context.join.read",
                          "--token-file", join_token)
        auth = ["--token-file", join_token, "--run-id", grant["run_id"]]
        request = {"version": VERSION, "request_id": "join", "snapshot_sha256": self.workspace.config["snapshot_sha256"],
                   "selection": {"choices": [], "historical": []}}
        self.call("join-prepare", *auth, "--request", self.input(request), expected=3)
        choices = active(self.graph)
        for choice in choices:
            if choice["entity_id"] in ("claim", "proof"):
                choice["revision_id"] = choice["entity_id"] + ":left"
        request["selection"]["choices"] = choices
        result = self.call("join-prepare", *auth, "--request", self.input(request))
        self.assertTrue(self.call("join-prepare", *auth, "--request", self.input(request))["replayed"])
        joined = self.call("join-read", *auth)["bundle"]
        manifest = manifest_for(joined["graph"], workspace_id=self.workspace.config["workspace_id"],
            snapshot_sha256=self.workspace.config["snapshot_sha256"], join_sha256=result["join_sha256"],
            claim="claim:left", proof="proof:left")
        bad = deepcopy(manifest)
        bad["dependencies"] = []
        self.call("build-view", "--request", self.input(bad), expected=2)
        self.call("build-view", "--request", self.input(manifest))
        verifier_token = self.credentials / "verifier"
        verifier = self.call("issue", "--kind", "view", "--action", "context.view.read", "--action", "context.view.probe",
                             "--token-file", verifier_token)
        auth = ["--token-file", verifier_token, "--run-id", verifier["run_id"]]
        dto = self.call("view-read", *auth)
        self.assertNotIn("PRIVATE_", json.dumps(dto))
        self.assertIn("assumption:0", {r["revision_id"] for r in dto["revisions"]})
        self.call("branch-read", *auth, expected=2)
        probe = self.call("probe", *auth)
        self.assertEqual(probe["status"], "synthetic_context_observed")
        self.assertEqual(probe["observation"]["input_sha256"], digest(dto))
        self.assertFalse(probe["mathematical_review"])
        again = self.call("probe", *auth)
        self.assertTrue(again["observation"]["empty_work"])

    def test_actual_published_view_rebinding_and_history_access_rejected(self):
        verifier, manifest, _ = self.freeze_view()
        credential = self.credentials / "view-bound"
        write_private(credential, verifier["token"].encode("ascii"))
        auth = ["--token-file", credential, "--run-id", verifier["run_id"]]
        before = (self.workspace.root / "view.json").read_bytes()
        self.assertTrue(self.call("build-view", "--request", self.input(manifest))["replayed"])
        changed = deepcopy(manifest)
        changed["request_id"] = "changed-view"
        self.call("build-view", "--request", self.input(changed), expected=3)
        self.assertEqual((self.workspace.root / "view.json").read_bytes(), before)
        self.call("join-read", *auth, expected=2)
        self.call("view-read", *auth, "--history", "private", expected=2)
        dto = self.call("view-read", *auth)
        self.assertEqual(dto["manifest_sha256"], digest(manifest))

    def test_actual_private_roots_symlinks_and_source_overlap_reject(self):
        alias = self.base / "workspace-alias"
        alias.symlink_to(self.workspace.root, target_is_directory=True)
        self.call("branch-read", *self.auth(), root=alias, expected=2)
        self.call("init", "--source-root", self.source.root, "--source-token-file", self.source_path,
            "--source-run-id", "source-run", "--spec", self.input(self.spec), root=self.source.root / "nested", expected=2)
        self.workspace.root.chmod(0o755)
        try:
            self.call("branch-read", *self.auth(), expected=2)
        finally:
            self.workspace.root.chmod(0o700)

    def test_actual_unknown_mode_command_and_extra_history_are_rejected(self):
        self.call("branch-read", *self.auth(), "--history", "generation", expected=2)
        self.call("probe", *self.auth(), "--command", "/usr/bin/true", expected=2)
        self.call("probe", *self.auth(), "--mode", "nonsandbox", expected=2)
        self.call("view-read", *self.auth(), "--head", "f" * 64, expected=2)

    def test_actual_strict_json_duplicate_keys_and_nonfinite_rejected(self):
        for raw in (b'{"version":1,"version":1}', b'{"version":NaN}', b'{"value":"\\xff"}'):
            self.counter += 1
            path = self.base / ("bad-%d.json" % self.counter)
            write_private(path, raw)
            self.call("branch-commit", *self.auth(), "--request", path, expected=2)

    def test_actual_unknown_input_permissions_and_private_error_output(self):
        self.call("branch-read", "--token-file", self.source_path, "--run-id", "source-run", expected=2)
        request = self.branch_request()
        request["history"] = "PRIVATE_REQUEST_MARKER"
        result = self.call("branch-commit", *self.auth(), "--request", self.input(request), expected=2)
        self.assertNotIn("PRIVATE_REQUEST_MARKER", json.dumps(result))
        self.paths["left"].chmod(0o644)
        self.call("branch-read", *self.auth(), expected=2)
        self.paths["left"].chmod(0o600)
        link = self.credentials / "linked"
        link.symlink_to(self.paths["left"])
        self.call("branch-read", "--token-file", link, "--run-id", self.grants["left"]["run_id"], expected=2)


if __name__ == "__main__":
    unittest.main()
