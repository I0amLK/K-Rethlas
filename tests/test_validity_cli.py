"""Actual T09 CLI subprocess interactions with synthetic fixture bytes only.

Execution needs separate authorization. Sandbox success is required, never skipped
or replaced with a fake observation. COMMAND_RECORDS omit bodies and credentials.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rethlas_core.authorization import ROLE_ACTIONS
from rethlas_core.context_views import exact_ref
from rethlas_core.private_fs import digest
from rethlas_core.validity import VERSION

from test_context_views import edge, manifest_for, mathematical_fixture
from test_memory_store import encoded, read_json, request_fixture, write_private


REPOSITORY = Path(__file__).absolute().parents[1]
TOOL = REPOSITORY / "tools/validity_memory.py"
COMMAND_RECORDS = []


def archive_output(value):
    """Keep structured denials and actual observations, not disclosed full text."""
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if key in {"token", "capability", "secret", "api_key"}:
                result[key] = "[REDACTED]"
            elif key in {"graph", "bundle", "dto", "descriptor", "projection", "control"} and type(item) in (dict, list):
                result[key] = {"sha256": digest(item), "count": len(item)}
            elif key == "revisions" and type(item) is list:
                observed = all(type(row) is dict and "body_bytes" in row and "body" not in row for row in item)
                result[key] = [archive_output(row) for row in item] if observed else {"sha256": digest(item), "count": len(item)}
            elif key == "relations" and type(item) is list:
                result[key] = {"sha256": digest(item), "count": len(item)}
            elif key in {"body", "reason", "unparsed_stdout"} and type(item) is str:
                result[key] = {"sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(), "bytes": len(item.encode("utf-8"))}
            else:
                result[key] = archive_output(item)
        return result
    if type(value) is list:
        return [archive_output(item) for item in value]
    return value


class ValidityCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="validity-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "service"
        self.credentials = self.base / "credentials"
        self.credentials.mkdir(mode=0o700)
        self.counter = 0
        self.grants = {}
        self.tokens = []
        self.graph = mathematical_fixture()
        initialized = self.call("init", "--owner", "owner")
        self.assertEqual(initialized["status"], "initialized")
        self.genesis = read_json(self.root / "memory/manifest.json")["head"]
        self.issue("writer", "validity_writer")
        self.seed = {"version": 1, "request_id": "fixture-seed", "expected_head": self.genesis,
                     **{key: self.graph[key] for key in ("entities", "revisions", "relations", "evidence")}}
        committed = self.call("commit", *self.auth("writer"), "--request", self.input(self.seed))
        self.head = committed["head"]
        self.registration = self.register_request()
        self.call("register-use", "--request", self.input(self.registration))
        self.issue("reader", "validity_reader", "use-main")

    def input(self, value=None, *, raw=None):
        self.counter += 1
        path = self.base / ("request-%d.json" % self.counter)
        write_private(path, encoded(value) if raw is None else raw)
        return path

    def call(self, command, *args, expected=0, root=None):
        argv = [sys.executable, "-I", "-B", str(TOOL), command, "--root", str(root or self.root), *map(str, args)]
        completed = subprocess.run(argv, cwd=REPOSITORY.parent,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        text = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = {"unparsed_stdout": text}
        archived_stderr = stderr
        record_argv = list(argv)
        for token in self.tokens:
            archived_stderr = archived_stderr.replace(token, "[REDACTED]")
            record_argv = [part.replace(token, "[REDACTED]") for part in record_argv]
        COMMAND_RECORDS.append({"test": self.id(), "argv": record_argv, "exit_code": completed.returncode,
                                "stdout": archive_output(value), "stderr": archived_stderr})
        for token in self.tokens:
            self.assertNotIn(token, text + stderr, "CLI disclosed a capability")
        self.assertEqual(completed.returncode, expected, stderr + text)
        self.assertEqual(stderr, "", "CLI errors must be structured without argv/path diagnostics")
        self.assertIsInstance(value, dict)
        if expected:
            self.assertEqual(set(value), {"status", "error"})
        return value

    def issue(self, name, role, use_id=None, *, root=None):
        path = self.credentials / name
        args = ["--role", role, "--token-file", path]
        if use_id is not None:
            args += ["--use-id", use_id]
        for action in sorted(ROLE_ACTIONS[role]):
            args += ["--action", action]
        result = self.call("issue", *args, root=root)
        self.assertEqual(result["status"], "issued")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.stat().st_nlink, 1)
        token = path.read_text(encoding="ascii")
        self.assertTrue(token and not any(char.isspace() for char in token))
        self.tokens.append(token)
        self.grants[name] = {"path": path, "run_id": result["run_id"]}
        return result

    def auth(self, name="reader"):
        grant = self.grants[name]
        return ["--token-file", grant["path"], "--run-id", grant["run_id"]]

    def register_request(self, *, graph=None, head=None, use_id="use-main", request_id=None,
                         claim="claim:0", proof="proof:0"):
        manifest = manifest_for(graph or self.graph, claim=claim, proof=proof)
        return {"version": VERSION, "request_id": request_id or "register-" + use_id,
                "use_id": use_id, "expected_head": head or self.head,
                **{key: manifest[key] for key in ("roots", "dependencies", "references")}}

    def exact(self, revision_id):
        return exact_ref(next(row for row in self.graph["revisions"] if row["revision_id"] == revision_id))

    def probe(self, request_id="probe-one"):
        if "verifier" not in self.grants:
            self.issue("verifier", "validity_verifier", "use-main")
        request = {"version": VERSION, "request_id": request_id}
        return self.call("probe", *self.auth("verifier"), "--request", self.input(request)), request

    def publish(self, probe_id, request_id="publish-one", *, expected=0):
        if "publisher" not in self.grants:
            self.issue("publisher", "validity_publisher", "use-main")
        request = {"version": VERSION, "request_id": request_id, "probe_id": probe_id}
        return self.call("publish", *self.auth("publisher"), "--request", self.input(request), expected=expected), request

    def withdraw(self, revision_id, request_id="withdraw-one"):
        if "withdrawer" not in self.grants:
            self.issue("withdrawer", "validity_withdrawer")
        request = {"version": VERSION, "request_id": request_id,
                   "targets": [self.exact(revision_id)], "reason": "合成撤回原因，不是数学审查。"}
        return self.call("withdraw", *self.auth("withdrawer"), "--request", self.input(request)), request

    def history_bytes(self, *, manifest=True):
        memory = self.root / "memory"
        paths = [memory / "store.json"] + ([memory / "manifest.json"] if manifest else [])
        for directory in (memory / "bodies", memory / "graphs", memory / "receipts", self.root / "uses"):
            paths += [path for path in directory.iterdir() if path.is_file()]
        return {str(path.relative_to(self.root)): path.read_bytes() for path in paths}

    def assert_existing_history_unchanged(self, before):
        for name, raw in before.items():
            self.assertEqual((self.root / name).read_bytes(), raw, name)

    def test_actual_success_observation_publication_and_support_withdrawal_replay(self):
        before = self.history_bytes()
        read = self.call("read", *self.auth())
        self.assertEqual(read["status"], "current_fixture_input")
        self.assertTrue(read["current"]["fixture_input_eligible"])
        self.assertFalse(read["current"]["mathematical_review"])
        self.assertFalse(read["current"]["production_completed"])
        self.assertNotIn("PRIVATE_", json.dumps(read["dto"], ensure_ascii=False))
        lookup = {"version": VERSION, "refs": [self.exact("proof:0")], "depth": 1}
        found = self.call("lookup", *self.auth(), "--request", self.input(lookup))
        self.assertEqual(found["retrieval_kind"], "exact_id_not_embedding")
        self.assertIn("assumption:0", {row["revision_id"] for row in found["revisions"]})
        observed, probe_request = self.probe()
        probe = observed["probe"]
        self.assertEqual(observed["status"], "synthetic_input_observed")
        self.assertEqual(probe["status"], "accepted")
        actual = probe["observation"]
        self.assertEqual(actual["input_files"], ["context.json", "driver.py", "overlay.json", "use.json"])
        self.assertEqual(actual["dto_sha256"], digest(read["dto"]))
        self.assertEqual(actual["descriptor_sha256"], digest(read["descriptor"]))
        driver = REPOSITORY / "agents/shared/src/rethlas_core/validity_fixture_driver.py"
        self.assertEqual(actual["driver_sha256"], hashlib.sha256(driver.read_bytes()).hexdigest())
        self.assertEqual(actual["overlay_sha256"], read["current"]["overlay_sha256"])
        self.assertEqual(actual["revisions"], [{**{key: row[key] for key in ("entity_id", "revision_id", "type", "body_sha256")},
                    "body_bytes": len(row["body"].encode("utf-8"))} for row in read["dto"]["revisions"]])
        self.assertTrue(actual["empty_work"] and actual["input_readonly"])
        self.assertFalse(actual["mathematical_review"])
        self.assertEqual(actual["relations"], len(read["dto"]["relations"]))
        published, publish_request = self.publish(probe["probe_id"])
        self.assertEqual(published["status"], "fixture_published")
        self.assertFalse(published["receipt"]["mathematical_review"])
        self.assertFalse(published["receipt"]["production_completed"])
        withdrawn, withdrawal_request = self.withdraw("assumption:0")
        self.assertEqual(withdrawn["status"], "withdrawn")
        replay = self.call("withdraw", *self.auth("withdrawer"), "--request", self.input(withdrawal_request))
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["withdrawal_id"], withdrawn["withdrawal_id"])
        current = self.call("status", *self.auth())["current"]
        self.assertFalse(current["fixture_input_eligible"])
        self.assertIn(self.exact("assumption:0"), current["blockers"])
        self.call("read", *self.auth(), expected=2)
        self.call("lookup", *self.auth(), "--request", self.input(lookup), expected=2)
        self.call("probe", *self.auth("verifier"), "--request", self.input({"version": VERSION, "request_id": "blocked-probe"}), expected=2)
        self.publish(probe["probe_id"], "blocked-publish", expected=2)
        old_probe = self.call("probe", *self.auth("verifier"), "--request", self.input(probe_request))
        self.assertTrue(old_probe["replayed"])
        self.assertEqual(old_probe["probe"], probe)
        self.assertFalse(old_probe["current"]["fixture_input_eligible"])
        old_publish = self.call("publish", *self.auth("publisher"), "--request", self.input(publish_request))
        self.assertTrue(old_publish["replayed"])
        self.assertEqual(old_publish["receipt"], published["receipt"])
        self.assertFalse(old_publish["current"]["fixture_input_eligible"])
        self.assert_existing_history_unchanged(before)

    def test_actual_citation_only_withdrawal_preserves_dto_but_requires_fresh_observation(self):
        before = self.history_bytes()
        read = self.call("read", *self.auth())
        observed, _ = self.probe()
        published, original_publish = self.publish(observed["probe"]["probe_id"])
        self.withdraw("source:0")
        after = self.call("read", *self.auth())
        self.assertEqual(after["dto"], read["dto"])
        self.assertEqual(after["descriptor"], read["descriptor"])
        self.assertTrue(after["current"]["fixture_input_eligible"])
        self.assertEqual(after["current"]["blockers"], [])
        citation = next(row for row in after["current"]["overlay"]["citation_only"] if row["revision_id"] == "source:0")
        self.assertTrue(citation["withdrawn"])
        self.assertEqual(citation["use"], "historical_citation_only")
        self.assertTrue(citation["not_authorized_as_premise"])
        lookup = self.call("lookup", *self.auth(), "--request", self.input({"version": VERSION, "refs": [self.exact("source:0")], "depth": 0}))
        self.assertEqual(lookup["revisions"][0]["use"], "historical_citation_only")
        self.assertTrue(lookup["revisions"][0]["not_authorized_as_premise"])
        self.publish(observed["probe"]["probe_id"], "stale-citation-publish", expected=3)
        replay = self.call("publish", *self.auth("publisher"), "--request", self.input(original_publish))
        self.assertEqual(replay["receipt"], published["receipt"])
        self.assertTrue(replay["current"]["fixture_input_eligible"])
        self.assertNotEqual(replay["receipt"]["overlay_sha256"], replay["current"]["overlay_sha256"])
        refreshed, _ = self.probe("refreshed-probe")
        self.assertEqual(refreshed["probe"]["observation"]["overlay_sha256"], after["current"]["overlay_sha256"])
        self.assertNotEqual(refreshed["probe"]["observation"]["overlay_sha256"], observed["probe"]["observation"]["overlay_sha256"])
        latest, _ = self.publish(refreshed["probe"]["probe_id"], "refreshed-publish")
        self.assertTrue(latest["current"]["fixture_input_eligible"])
        self.assertFalse(latest["receipt"]["mathematical_review"])
        self.assert_existing_history_unchanged(before)

    def test_actual_role_resource_foreign_scope_and_extra_flags_are_denied(self):
        before = self.history_bytes()
        other = self.register_request(use_id="use-other")
        self.call("register-use", "--request", self.input(other))
        self.issue("other-reader", "validity_reader", "use-other")
        self.call("read", "--token-file", self.grants["reader"]["path"], "--run-id", self.grants["other-reader"]["run_id"], expected=2)
        self.call("read", *self.auth("writer"), expected=2)
        self.call("probe", *self.auth(), "--request", self.input({"version": VERSION, "request_id": "reader-cannot-probe"}), expected=2)
        foreign = self.base / "foreign"
        self.call("init", "--owner", "different-owner", root=foreign)
        self.issue("foreign-writer", "validity_writer", root=foreign)
        self.call("commit", *self.auth("foreign-writer"), "--request", self.input(self.seed), expected=2)
        self.call("issue", "--role", "validity_reader", "--use-id", "unknown-use", "--action", "validity.read",
                  "--token-file", self.credentials / "unknown-use", expected=2)
        self.call("issue", "--role", "validity_writer", "--use-id", "use-main", "--action", "validity.commit",
                  "--token-file", self.credentials / "wrong-resource", expected=2)
        self.call("issue", "--role", "validity_reader", "--use-id", "use-main", "--action", "validity.withdraw",
                  "--token-file", self.credentials / "wrong-action", expected=2)
        for option in ("--mode", "--history", "--head", "--command", "--ignore-revocation", "--fallback", "--roo"):
            result = self.call("read", *self.auth(), option, "PRIVATE_FLAG_MARKER", expected=2)
            self.assertNotIn("PRIVATE_FLAG_MARKER", json.dumps(result))
        self.assert_existing_history_unchanged(before)
        self.assertEqual(read_json(self.root / "memory/manifest.json")["head"], self.head)

    def test_actual_strict_json_versions_unknown_refs_and_self_reported_verdicts_reject(self):
        before = self.history_bytes()
        for raw in (b'{"version":1,"version":1}', b'{"version":NaN}', b'{"version":Infinity}',
                    b'{"value":"\xff"}', b'{"value":"\\ud800"}', b'{"value":1e999}'):
            self.call("commit", *self.auth("writer"), "--request", self.input(raw=raw), expected=2)
        for version in (True, 1.0):
            request = request_fixture(self.head, label="bad-version")
            request["version"] = version
            self.call("commit", *self.auth("writer"), "--request", self.input(request), expected=2)
        for depth in (True, 1.0, 9):
            self.call("lookup", *self.auth(), "--request", self.input({"version": VERSION, "refs": [self.exact("claim:0")], "depth": depth}), expected=2)
        unknown = {"entity_id": "uncommitted-branch", "revision_id": "uncommitted-branch:0", "body_sha256": "f" * 64}
        self.call("lookup", *self.auth(), "--request", self.input({"version": VERSION, "refs": [unknown], "depth": 0}), expected=2)
        self.issue("withdrawer", "validity_withdrawer")
        self.call("withdraw", *self.auth("withdrawer"), "--request", self.input({"version": VERSION, "request_id": "unknown-withdrawal", "targets": [unknown], "reason": "PRIVATE_REASON_MARKER"}), expected=2)
        malformed = deepcopy(self.registration)
        malformed["request_id"] = "unknown-registration"
        malformed["use_id"] = "unknown-branch"
        malformed["roots"]["proof"] = unknown
        self.call("register-use", "--request", self.input(malformed), expected=2)
        self.issue("publisher", "validity_publisher", "use-main")
        for extra in ({"correct": True}, {"observation": {"mathematical_review": True}}, {"receipt": "M1-receipt"}):
            request = {"version": VERSION, "request_id": "caller-verdict", "probe_id": "0" * 64, **extra}
            result = self.call("publish", *self.auth("publisher"), "--request", self.input(request), expected=2)
            self.assertNotIn("PRIVATE_", json.dumps(result))
        self.publish("0" * 64, "unknown-observation", expected=3)
        self.assert_existing_history_unchanged(before)
        self.assertEqual(read_json(self.root / "memory/manifest.json")["head"], self.head)

    def test_actual_private_paths_credentials_and_requests_do_not_repair_or_overwrite(self):
        before = self.history_bytes()
        alias = self.base / "service-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.call("read", *self.auth(), root=alias, expected=2)
        self.call("read", *self.auth(), root=self.base / "absent", expected=2)
        self.assertFalse((self.base / "absent").exists())
        self.call("init", "--owner", "owner", "--domain", "production", root=self.base / "production", expected=2)
        self.assertFalse((self.base / "production").exists())
        self.call("init", "--owner", "owner", root=self.root, expected=3)
        self.call("init", "--owner", "owner", root=REPOSITORY / "tests", expected=2)
        credential = self.grants["reader"]["path"]
        original = credential.read_bytes()
        for destination in (credential, self.root / "new-token", REPOSITORY / "new-token"):
            self.call("issue", "--role", "validity_reader", "--use-id", "use-main", "--action", "validity.read",
                      "--token-file", destination, expected=2)
        self.assertEqual(credential.read_bytes(), original)
        linked = self.credentials / "linked"
        linked.symlink_to(credential)
        self.call("read", "--token-file", linked, "--run-id", self.grants["reader"]["run_id"], expected=2)
        linked.unlink()
        os.link(credential, linked)
        self.call("read", *self.auth(), expected=2)
        linked.unlink()
        credential.chmod(0o644)
        try:
            self.call("read", *self.auth(), expected=2)
        finally:
            credential.chmod(0o600)
        self.root.chmod(0o755)
        try:
            self.call("read", *self.auth(), expected=2)
        finally:
            self.root.chmod(0o700)
        request = self.input({"version": VERSION, "refs": [self.exact("claim:0")], "depth": 0})
        request_alias = self.base / "request-alias"
        request_alias.symlink_to(request)
        self.call("lookup", *self.auth(), "--request", request_alias, expected=2)
        request_alias.unlink()
        os.link(request, request_alias)
        self.call("lookup", *self.auth(), "--request", request, expected=2)
        self.assert_existing_history_unchanged(before)

    def test_actual_graph_replay_keeps_new_floor_and_retroactive_premise_edits_reject(self):
        before = self.history_bytes(manifest=False)
        request = request_fixture(self.head, label="new-candidate")
        committed = self.call("commit", *self.auth("writer"), "--request", self.input(request))
        self.assertNotEqual(committed["head"], self.head)
        self.assertEqual(committed["generation"], 2)
        replay = self.call("commit", *self.auth("writer"), "--request", self.input(self.seed))
        self.assertTrue(replay["replayed"])
        self.assertTrue(replay["historical_only"])
        self.assertEqual(replay["head"], self.head)
        self.assertEqual(replay["accepted_graph"]["head"], committed["head"])
        changed = deepcopy(self.seed)
        changed["expected_head"] = committed["head"]
        self.call("commit", *self.auth("writer"), "--request", self.input(changed), expected=3)
        stale = request_fixture(self.head, label="stale-parent")
        self.call("commit", *self.auth("writer"), "--request", self.input(stale), expected=3)
        retroactive = {"version": 1, "request_id": "retroactive-premise", "expected_head": committed["head"],
                       "entities": [], "revisions": [], "relations": [edge("claim:0", "assumption:0", "depends_on", name="retroactive")], "evidence": []}
        self.call("commit", *self.auth("writer"), "--request", self.input(retroactive), expected=2)
        self.assertEqual(read_json(self.root / "memory/manifest.json")["head"], committed["head"])
        self.assertTrue(self.call("status", *self.auth())["current"]["fixture_input_eligible"])
        self.assert_existing_history_unchanged(before)

    def test_actual_committed_legacy_wrapper_is_historical_not_registered_mathematical_input(self):
        legacy = deepcopy(next(row for row in self.graph["revisions"] if row["revision_id"] == "source:0"))
        legacy.update(entity_id="legacy-source", revision_id="legacy:revision:source")
        legacy["provenance"]["source"] = "legacy-jsonl://snapshot/synthetic"
        entity = deepcopy(next(row for row in self.graph["entities"] if row["entity_id"] == "source"))
        entity.update(entity_id="legacy-source")
        entity["provenance"]["source"] = "legacy-jsonl://snapshot/synthetic"
        additions = [legacy]
        for name in ("claim", "proof"):
            item = deepcopy(next(row for row in self.graph["revisions"] if row["revision_id"] == name + ":0"))
            item["revision_id"] = name + ":legacy"
            additions.append(item)
        edges = [edge("proof:legacy", "claim:legacy", "attempts_to_prove"),
                 edge("proof:legacy", "assumption:0", "assumes"),
                 edge("claim:legacy", "definition:0", "in_context"),
                 edge("claim:legacy", "legacy:revision:source", "cites")]
        # The shared edge helper splits a simple fixture ID at ':'. This opaque
        # legacy revision belongs to a different exact entity and is explicit.
        edges[-1]["target"]["entity_id"] = "legacy-source"
        request = {"version": 1, "request_id": "legacy-candidate", "expected_head": self.head,
                   "entities": [entity], "revisions": additions, "relations": edges, "evidence": []}
        committed = self.call("commit", *self.auth("writer"), "--request", self.input(request))
        graph = deepcopy(self.graph)
        graph["entities"].append(entity)
        graph["revisions"].extend(additions)
        graph["relations"].extend(edges)
        registration = self.register_request(graph=graph, head=committed["head"], use_id="legacy-use", claim="claim:legacy", proof="proof:legacy")
        registration["references"] = [exact_ref(legacy)]
        self.call("register-use", "--request", self.input(registration), expected=2)
        audited = self.call("audit")
        stored = next(row for row in audited["graph"]["graph"]["revisions"] if row["revision_id"] == legacy["revision_id"])
        self.assertEqual(stored, legacy)
        self.assertTrue(self.call("status", *self.auth())["current"]["fixture_input_eligible"])

    def test_actual_operator_audit_queue_events_and_explicit_recovery_flags(self):
        before = self.history_bytes()
        self.withdraw("assumption:0")
        audit = self.call("audit")
        self.assertEqual(audit["status"], "operator_historical_audit")
        self.assertEqual(audit["graph"]["head"], self.head)
        self.assertFalse(audit["mathematical_review"])
        self.assertTrue(all(row["assertion_state"] == "candidate" and row["validity"] == "needs_review"
                            for row in audit["graph"]["graph"]["revisions"]))
        queue = self.call("review-queue")
        self.assertLessEqual(len(queue["entries"]), 128)
        self.assertTrue(any(row["kind"] == "direct_withdrawal" and row["revision"]["revision_id"] == "assumption:0" for row in queue["entries"]))
        self.assertTrue(any(row["kind"] == "math_dependency_blocked" for row in queue["entries"]))
        self.assertNotIn("合成撤回原因", json.dumps(queue, ensure_ascii=False))
        events = self.call("events")
        self.assertLessEqual(len(events["entries"]), 128)
        self.assertTrue(any(row["kind"] == "WITHDRAW" for row in events["entries"]))
        self.call("events", "--cursor", "invalid-cursor", expected=2)
        self.call("review-queue", "--cursor", "invalid-cursor", expected=2)
        self.assertEqual(self.call("recover", "--kind", "control")["status"], "stable")
        self.assertEqual(self.call("recover", "--kind", "graph")["status"], "no_pending_graph")
        self.call("recover", "--kind", "control", "--cancel", expected=2)
        self.call("recover", "--kind", "control", "--token-file", self.grants["reader"]["path"], expected=2)
        self.call("recover", "--kind", "control", "--request", self.input({"version": VERSION}), expected=2)
        self.call("reset", expected=2)
        self.assert_existing_history_unchanged(before)


if __name__ == "__main__":
    unittest.main()
