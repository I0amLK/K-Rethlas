"""Synthetic legacy byte-capture and offline-package contract fixtures.

No historical files, providers or model services are used. Imports only define
fixtures; the operator-authorized acceptance entrypoint runs them separately.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import unittest
from unittest import mock

from test_memory_store import (
    ACTIONS, RUN_ID, MemoryFixtureMixin, encoded, request_fixture, sha, write_private,
)
from rethlas_core.authorization import AuthorizationError
from rethlas_core.legacy_migration import (
    LIMITS, PROFILES, MigrationCapacity, MigrationError, MigrationQuarantine,
    apply, capture, convert_snapshot, dry_run, prepare,
)
from rethlas_core.memory_io import read_input
from rethlas_core.memory_store import MemoryCancelled, MemoryConflict, MemoryStore
from rethlas_core.private_fs import PrivateDirectory, PrivateFilesystemError
from rethlas_core.contracts import validate_math_memory


OBSERVED = "2026-10-17T00:00:00Z"


def legacy_line(channel="proof_steps", record=None, *, ending=b"\n", timestamp=OBSERVED):
    value = {"timestamp_utc": timestamp, "channel": channel,
             "record": {"id": "step/一", "status": "solved"} if record is None else record}
    return json.dumps(value, ensure_ascii=False).encode("utf-8") + ending


class MigrationFixtureMixin(MemoryFixtureMixin):
    def setUp(self):
        super().setUp()
        self.sources = self.base / "sources"
        self.sources.mkdir(mode=0o700)
        self.spec_path = self.sources / "spec.json"
        self.bundle = self.base / "bundle"

    def spec(self, inputs=None, *, allow=False, namespace="公开合成/迁移"):
        inputs = inputs if inputs is not None else [("main", "generation-10", "proof_steps", legacy_line(), None)]
        sources = []
        for alias, profile, channel, raw, scope in inputs:
            path = self.sources / (alias + ".jsonl")
            write_private(path, raw)
            scope = scope if scope is not None else (
                {"problem_id": "题/α", "branch_id": "支线\\一", "source_run_id": None}
                if profile == "generation-10" else
                {"problem_id": None, "branch_id": None, "source_run_id": "原run/甲"})
            sources.append({"alias": alias, "profile": profile, "channel": channel,
                            "scope": scope, "path": str(path), "expected_sha256": sha(raw)})
        value = {"version": "legacy-jsonl-1", "source_namespace": namespace,
                 "import_observed_at": OBSERVED, "options": {"allow_quarantine": allow},
                 "sources": sources}
        self.save_spec(value)
        return value

    def save_spec(self, value):
        write_private(self.spec_path, encoded(value))

    def converted(self, spec=None):
        if spec is None:
            self.spec()
        snapshot, raw, _ = capture(self.spec_path)
        return convert_snapshot(snapshot, raw, actor="legacy-jsonl-import/owner/" + RUN_ID)

    def prepared(self, **kwargs):
        if not self.spec_path.exists():
            self.spec()
        return prepare(self.spec_path, self.bundle, self.store, self.token, run_id=RUN_ID, **kwargs)

    def applied(self, prepared=None, **kwargs):
        result = prepared if prepared is not None else self.prepared()
        return apply(self.bundle, result["bundle_sha256"], self.store, self.token, run_id=RUN_ID, **kwargs)

    def tree_bytes(self, root=None):
        root = root or self.base
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def repin(self):
        """Deliberate untrusted self-consistency, not authorized correctness."""
        manifest = json.loads((self.bundle / "manifest.json").read_bytes())
        for member in manifest["files"]:
            raw = (self.bundle / member["name"]).read_bytes()
            member.update(bytes=len(raw), sha256=sha(raw))
        write_private(self.bundle / "manifest.json", encoded(manifest))
        return sha(encoded(manifest))


class LegacyConversionTests(MigrationFixtureMixin, unittest.TestCase):
    def test_two_profiles_all_fifteen_channels_without_importing_legacy(self):
        inputs = [("s" + str(i), profile, channel, legacy_line(channel, {}), None)
                  for i, (profile, channel) in enumerate(
                      (p, c) for p in PROFILES for c in PROFILES[p])]
        self.spec(inputs)
        result = self.converted(True)
        self.assertEqual(result["counts"]["accepted_lines"], 15)
        self.assertEqual(result["counts"]["source_files"], 15)
        self.assertEqual(len(result["graph"]["entities"]), 15)
        self.assertTrue(all(item["type"] == "Reference" for item in result["graph"]["entities"]))

    def test_exact_utf8_crlf_spacing_and_unterminated_final_line(self):
        first = legacy_line(record={"id": "一/\\", "proof": "α\r\nβ"}, ending=b"\r\n")
        second = b' { "timestamp_utc" : "old date", "channel": "proof_steps", "record": {} } '
        raw = first + second
        self.spec([("main", "generation-10", "proof_steps", raw, None)])
        result = self.converted(True)
        revisions = result["graph"]["revisions"]
        self.assertEqual([item["body"].encode("utf-8") for item in revisions], [first, second])
        self.assertEqual([item["body_sha256"] for item in revisions], [sha(first), sha(second)])
        self.assertEqual([row["offset"] for row in result["map"]["lines"]], [0, len(first)])
        self.assertEqual(sum(row["length"] for row in result["map"]["lines"]), len(raw))

    def test_empty_file_zero_lines_and_lf_does_not_add_phantom_line(self):
        self.spec([("a", "generation-10", "proof_steps", b"", None),
                   ("b", "generation-10", "failed_paths", legacy_line("failed_paths", {}), None)])
        result = self.converted(True)
        self.assertEqual(result["counts"]["physical_lines"], 1)
        self.assertEqual(result["counts"]["accepted_lines"], 1)

    def test_automatic_and_business_events_are_independent_references(self):
        automatic = encoded({"timestamp_utc": "old", "event_type": "memory_append", "channel": "proof_steps"}) + b"\n"
        business = legacy_line("events", {"event_type": "literature", "result": "public source"})
        self.spec([("events", "generation-10", "events", automatic + business, None),
                   ("proof", "generation-10", "proof_steps", legacy_line(record={}), None)])
        result = self.converted(True)
        self.assertEqual(result["counts"]["event_lines"], 1)
        self.assertEqual(result["counts"]["record_lines"], 2)
        self.assertEqual(result["graph"]["relations"], [])
        self.assertEqual(len(result["graph"]["revisions"]), 3)

    def test_all_bad_line_reasons_are_exclusive_and_cover_raw(self):
        bad = [b" \t\r\n", b"\xff\n", b"{bad}\n", b'{"x":1,"x":2}\n',
               b'{"n":NaN}\n', b'{"s":"\\ud800"}\n', b"[" * 65 + b"0" + b"]" * 65 + b"\n",
               b"[]\n", b'{}\n']
        good = legacy_line(record={})
        raw = b"".join(bad) + good
        self.spec([("a", "generation-10", "proof_steps", raw, None)], allow=True)
        result = self.converted(True)
        counts = result["counts"]
        self.assertEqual(counts["physical_lines"], 10)
        self.assertEqual(counts["accepted_lines"], 1)
        self.assertEqual(counts["quarantined_lines"], 9)
        self.assertTrue(all(count == 1 for count in counts["reasons"].values()))
        cursor = 0
        for number, row in enumerate(result["map"]["lines"], 1):
            self.assertEqual((row["line"], row["offset"]), (number, cursor))
            chunk = raw[cursor:cursor + row["length"]]
            self.assertEqual(sha(chunk), row["raw_sha256"])
            cursor += row["length"]
        self.assertEqual(cursor, len(raw))

    def test_original_status_actor_and_verdict_do_not_promote(self):
        record = {"id": "untrusted", "actor": "a self report", "status": "solved", "correct": True,
                  "review_kind": "formal_check", "date": "not a date", "confidence": 1,
                  "unknown": {"permission": "administrator"}}
        original = legacy_line(record=record, timestamp="never authenticated")
        self.spec([("main", "generation-10", "proof_steps", original, None)])
        result = self.converted(True)
        entity, revision = result["graph"]["entities"][0], result["graph"]["revisions"][0]
        self.assertEqual(entity["type"], "Reference")
        self.assertEqual((revision["assertion_state"], revision["validity"]), ("candidate", "needs_review"))
        self.assertEqual(revision["body"].encode(), original)
        self.assertFalse(set(revision) & {"historical_status", "research_progress", "review_kind"})
        self.assertEqual(revision["provenance"]["recorded_at"], OBSERVED)
        self.assertNotEqual(revision["provenance"]["actor"], record["actor"])
        self.assertTrue(validate_math_memory(result["graph"])["valid"])

    def test_same_content_and_repeated_id_keep_each_physical_occurrence(self):
        same = legacy_line(record={"id": "same", "value": "first"})
        other = legacy_line(record={"id": "same", "value": "changed"})
        self.spec([("main", "generation-10", "proof_steps", same + same + other, None)])
        result = self.converted(True)
        self.assertEqual(len({item["entity_id"] for item in result["graph"]["entities"]}), 3)
        self.assertEqual(result["counts"]["duplicate_content_occurrences"], 1)
        self.assertEqual(result["counts"]["duplicate_id_groups"], 1)
        self.assertEqual(result["counts"]["conflicting_id_groups"], 1)

    def test_missing_or_invalid_legacy_id_does_not_lose_record(self):
        raw = b"".join(legacy_line(record=record) for record in ({}, {"id": None}, {"id": []}, {"id": ""}))
        self.spec([("main", "generation-10", "proof_steps", raw, None)])
        result = self.converted(True)
        self.assertEqual(len(result["graph"]["entities"]), 4)
        self.assertEqual(result["counts"]["quarantined_lines"], 0)

    def test_profile_and_original_scope_prevent_sanitizer_merging(self):
        inputs = []
        for alias, scope in (("a", {"problem_id": "题/一", "branch_id": None, "source_run_id": None}),
                             ("b", {"problem_id": "题_一", "branch_id": None, "source_run_id": None})):
            inputs.append((alias, "generation-10", "verification_reports", legacy_line("verification_reports"), scope))
        inputs.append(("c", "verification-5", "verification_reports", legacy_line("verification_reports"), None))
        self.spec(inputs)
        result = self.converted(True)
        self.assertEqual(len({row["revision_id"] for row in result["map"]["lines"]}), 3)
        self.assertEqual(result["counts"]["duplicate_id_groups"], 0)

    def test_two_pass_links_resolve_only_unique_exact_channel_scoped_ids(self):
        self.spec([("a", "generation-10", "proof_steps", legacy_line(record={
            "used_examples": ["toy/一", "missing"], "used_results": ["theorem"], "plan_id": "plan"}), None),
                   ("z", "generation-10", "toy_examples", legacy_line("toy_examples", {"id": "toy/一"}), None)])
        result = self.converted(True)
        self.assertEqual(result["counts"]["references"]["resolved"], 1)
        self.assertEqual(result["counts"]["references"]["unresolved"], 3)
        self.assertEqual([item["type"] for item in result["graph"]["relations"]], ["cites"])
        self.assertEqual(result["graph"]["relations"][0]["target"]["revision_id"], result["map"]["lines"][1]["revision_id"])

    def test_later_duplicate_target_makes_earlier_link_ambiguous(self):
        target = legacy_line("toy_examples", {"id": "toy"})
        self.spec([("a", "generation-10", "proof_steps", legacy_line(record={"used_examples": ["toy"]}), None),
                   ("z", "generation-10", "toy_examples", target + target, None)])
        result = self.converted(True)
        self.assertEqual(result["counts"]["references"]["ambiguous"], 1)
        self.assertEqual(result["graph"]["relations"], [])

    def test_link_types_and_cross_scope_targets_never_guessed(self):
        other = {"problem_id": "different", "branch_id": None, "source_run_id": None}
        self.spec([("a", "generation-10", "proof_steps", legacy_line(record={
            "used_examples": ["toy", None, {"id": "toy"}], "adapted_from": ["narrative"]}), None),
                   ("z", "generation-10", "toy_examples", legacy_line("toy_examples", {"id": "toy"}), other)])
        result = self.converted(True)
        self.assertEqual(result["counts"]["references"]["unresolved"], 2)
        self.assertEqual(result["counts"]["references"]["invalid_type"], 2)
        self.assertEqual(result["graph"]["relations"], [])

    def test_input_path_move_does_not_change_logical_conversion(self):
        spec = self.spec()
        before = self.converted(True)
        relocated = self.sources / "moved.jsonl"
        shutil.copyfile(Path(spec["sources"][0]["path"]), relocated)
        spec["sources"][0]["path"] = str(relocated)
        self.save_spec(spec)
        self.assertEqual(self.converted(True), before)
        self.assertNotIn(str(self.base), encoded(before).decode())

    def test_snapshot_changes_are_explicit_new_import_not_incremental_merge(self):
        spec = self.spec()
        before = self.converted(True)
        path = Path(spec["sources"][0]["path"])
        path.write_bytes(path.read_bytes() + legacy_line(record={"id": "new"}))
        with self.assertRaises(MigrationError):
            capture(self.spec_path)
        spec["sources"][0]["expected_sha256"] = sha(path.read_bytes())
        self.save_spec(spec)
        after = self.converted(True)
        self.assertNotEqual(before["snapshot_sha256"], after["snapshot_sha256"])
        self.assertNotEqual(before["map"]["lines"][0]["revision_id"], after["map"]["lines"][0]["revision_id"])

    def test_spec_closed_scope_date_and_profile_validation(self):
        original = self.spec()
        for label, modify in (
            ("outer", lambda v: v.update(extra=True)),
            ("option", lambda v: v["options"].update(rebase=True)),
            ("date", lambda v: v.update(import_observed_at="2026-02-30T00:00:00Z")),
            ("profile", lambda v: v["sources"][0].update(profile="guess")),
            ("channel", lambda v: v["sources"][0].update(channel="statement_checks")),
            ("scope", lambda v: v["sources"][0]["scope"].update(source_run_id="unrelated")),
        ):
            with self.subTest(label=label):
                spec = deepcopy(original)
                modify(spec)
                self.save_spec(spec)
                with self.assertRaises(MigrationError):
                    capture(self.spec_path)

    def test_duplicate_alias_source_identity_and_path_rejected(self):
        original = self.spec()
        for change in ("alias", "identity", "path"):
            spec = deepcopy(original)
            second = deepcopy(spec["sources"][0])
            if change == "alias":
                second["scope"]["problem_id"] = "different"
                alternate = self.sources / "other.jsonl"
                write_private(alternate, legacy_line())
                second["path"] = str(alternate)
            elif change == "identity":
                second["alias"] = "second"
                alternate = self.sources / "other.jsonl"
                write_private(alternate, legacy_line())
                second["path"] = str(alternate)
            else:
                second["alias"] = "second"
                second["scope"]["problem_id"] = "different"
            spec["sources"].append(second)
            self.save_spec(spec)
            with self.subTest(change=change), self.assertRaises(MigrationError):
                capture(self.spec_path)


class LegacyPackageTests(MigrationFixtureMixin, unittest.TestCase):
    def test_dry_run_no_writes_or_target_access(self):
        self.spec()
        before = self.tree_bytes()
        with mock.patch.object(MemoryStore, "__init__", side_effect=AssertionError("must not open store")):
            result = dry_run(self.spec_path)
        self.assertEqual(result["status"], "dry-run")
        self.assertFalse(result["target_checked"])
        self.assertEqual(before, self.tree_bytes())
        self.assertFalse(self.bundle.exists())

    def test_prepare_then_apply_exact_replay_does_not_advance_again(self):
        self.spec()
        before = self.tree_bytes(self.sources)
        prepared = self.prepared()
        self.assert_unchanged_head()
        first = self.applied(prepared)
        second = self.applied(prepared)
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["receipt"], second["receipt"])
        self.assertEqual(self.status()["generation"], 1)
        self.assertEqual(len(list((self.root / "receipts").iterdir())), 1)
        self.assertEqual(before, self.tree_bytes(self.sources))
        self.assertEqual(self.export()["graph"]["entities"][0]["type"], "Reference")

    def test_package_digest_chain_binds_raw_map_and_exact_request(self):
        prepared = self.prepared()
        manifest = json.loads((self.bundle / "manifest.json").read_bytes())
        self.assertEqual(prepared["bundle_sha256"], sha(encoded(manifest)))
        self.assertNotIn("manifest.json", [item["name"] for item in manifest["files"]])
        for item in manifest["files"]:
            data = (self.bundle / item["name"]).read_bytes()
            self.assertEqual((len(data), sha(data)), (item["bytes"], item["sha256"]))
        committed = self.applied(prepared)
        source_manifest = (self.bundle / "source-manifest.json").read_bytes()
        evidence = self.export()["graph"]["evidence"]
        self.assertIn(sha(source_manifest), [item["sha256"] for item in evidence])
        request = json.loads((self.bundle / "request.json").read_bytes())
        self.assertEqual(request["request_id"], committed["receipt"]["binding"]["request_id"])
        for path in self.bundle.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(self.bundle.stat().st_mode & 0o777, 0o700)
        self.assertNotIn(self.token, encoded(manifest).decode())

    def test_move_package_and_delete_original_sources_replays(self):
        prepared = self.prepared()
        first = self.applied(prepared)
        shutil.rmtree(self.sources)
        moved = self.base / "moved"
        self.bundle.rename(moved)
        replay = apply(moved, prepared["bundle_sha256"], self.store, self.token, run_id=RUN_ID)
        self.assertTrue(replay["replayed"])
        self.assertEqual(first["head"], replay["head"])

    def test_bad_lines_default_reject_and_explicit_subset_retains_raw(self):
        raw = legacy_line() + b"bad\n\n"
        spec = self.spec([("main", "generation-10", "proof_steps", raw, None)])
        diagnostic = dry_run(self.spec_path)
        self.assertTrue(diagnostic["requires_allow_quarantine"])
        with self.assertRaises(MigrationQuarantine):
            self.prepared()
        self.assertFalse(self.bundle.exists())
        spec["options"]["allow_quarantine"] = True
        self.save_spec(spec)
        result = self.prepared()
        self.applied(result)
        self.assertEqual((self.bundle / "raw-main.jsonl").read_bytes(), raw)
        self.assertEqual(len(self.export()["graph"]["revisions"]), 1)
        self.assertEqual(result["counts"]["quarantined_lines"], 2)

    def test_zero_accepted_is_noop_without_package_or_receipt(self):
        self.spec([("main", "generation-10", "proof_steps", b"bad\n", None)], allow=True)
        before = self.authoritative_files()
        result = self.prepared()
        self.assertEqual(result["status"], "no-op")
        self.assertFalse(self.bundle.exists())
        self.assertEqual(self.authoritative_files(), before)

    def test_pin_is_required_and_self_reported_wrong_hash_rejected(self):
        prepared = self.prepared()
        for pin in (None, "", "0" * 64, prepared["bundle_sha256"].upper()):
            with self.subTest(pin_type=type(pin).__name__), self.assertRaises(MigrationError):
                apply(self.bundle, pin, self.store, self.token, run_id=RUN_ID)
        self.assert_unchanged_head()

    def test_repin_cannot_authorize_modified_request_or_map(self):
        self.prepared()
        original = (self.bundle / "request.json").read_bytes()
        request = json.loads(original)
        request["revisions"][0]["validity"] = "usable"
        write_private(self.bundle / "request.json", encoded(request))
        with self.assertRaises(MigrationError):
            apply(self.bundle, self.repin(), self.store, self.token, run_id=RUN_ID)
        write_private(self.bundle / "request.json", original)
        mapping = json.loads((self.bundle / "map.json").read_bytes())
        mapping["lines"][0]["offset"] = 1
        write_private(self.bundle / "map.json", encoded(mapping))
        with self.assertRaises(MigrationError):
            apply(self.bundle, self.repin(), self.store, self.token, run_id=RUN_ID)
        self.assert_unchanged_head()

    def test_member_growth_shrink_and_same_length_tamper_are_migration_errors(self):
        prepared = self.prepared()
        before = self.authoritative_files()
        for name in ("raw-main.jsonl", "request.json"):
            path = self.bundle / name
            original = path.read_bytes()
            for label, altered in (("growth", original + b" "), ("shrink", original[:-1]),
                                   ("same_length", b"!" + original[1:])):
                with self.subTest(member=name, alteration=label):
                    write_private(path, altered)
                    with self.assertRaises(MigrationError):
                        self.applied(prepared)
                    self.assertEqual(before, self.authoritative_files())
            write_private(path, original)
        self.assertFalse(self.applied(prepared)["replayed"])

    def test_member_growth_between_inventory_and_read_is_migration_error(self):
        prepared = self.prepared()
        read_bytes = PrivateDirectory.read_bytes
        path = self.bundle / "raw-main.jsonl"
        original = path.read_bytes()
        changed = False

        def read(directory, name, **kwargs):
            nonlocal changed
            if directory.root == self.bundle and name == path.name and not changed:
                write_private(path, original + b" ")
                changed = True
            return read_bytes(directory, name, **kwargs)

        with mock.patch.object(PrivateDirectory, "read_bytes", new=read):
            with self.assertRaises(MigrationError):
                self.applied(prepared)
        self.assertTrue(changed)
        self.assert_unchanged_head()

    def test_unknown_member_and_noncanonical_manifest_rejected(self):
        prepared = self.prepared()
        write_private(self.bundle / "unknown", b"extra")
        with self.assertRaises(MigrationError):
            self.applied(prepared)
        (self.bundle / "unknown").unlink()
        manifest = json.loads((self.bundle / "manifest.json").read_bytes())
        pretty = json.dumps(manifest, indent=2).encode()
        write_private(self.bundle / "manifest.json", pretty)
        with self.assertRaises(MigrationError):
            apply(self.bundle, sha(pretty), self.store, self.token, run_id=RUN_ID)

    def test_empty_or_interrupted_package_is_not_repaired(self):
        self.bundle.mkdir(mode=0o700)
        write_private(self.bundle / "lock", b"")
        write_private(self.bundle / "raw-main.jsonl", legacy_line())
        before = self.tree_bytes(self.bundle)
        with self.assertRaises((MigrationError, OSError)):
            apply(self.bundle, "0" * 64, self.store, self.token, run_id=RUN_ID)
        self.assertEqual(before, self.tree_bytes(self.bundle))
        self.spec()
        with self.assertRaises(MigrationError):
            self.prepared()
        self.assertEqual(before, self.tree_bytes(self.bundle))

    def test_target_run_epoch_and_store_are_frozen(self):
        prepared = self.prepared()
        wrong, token = self.make_store("wrong")
        with self.assertRaises(MemoryConflict):
            apply(self.bundle, prepared["bundle_sha256"], wrong, token, run_id=RUN_ID)
        other = self.store.issue(run_id="other-run", actions=ACTIONS)
        with self.assertRaises(MemoryConflict):
            apply(self.bundle, prepared["bundle_sha256"], self.store, other, run_id="other-run")
        self.store.authority.bump_epoch(RUN_ID)
        renewed = self.store.issue(run_id=RUN_ID, actions=ACTIONS)
        with self.assertRaises(MemoryConflict):
            apply(self.bundle, prepared["bundle_sha256"], self.store, renewed, run_id=RUN_ID)
        self.assert_unchanged_head()

    def test_read_only_prepare_commit_only_apply_and_token_renewal(self):
        self.spec()
        reader = self.store.issue(run_id=RUN_ID, actions=["memory.read"])
        prepared = prepare(self.spec_path, self.bundle, self.store, reader, run_id=RUN_ID)
        with self.assertRaises(AuthorizationError):
            apply(self.bundle, prepared["bundle_sha256"], self.store, reader, run_id=RUN_ID)
        writer = self.store.issue(run_id=RUN_ID, actions=["memory.commit"])
        with mock.patch.object(self.store, "status", side_effect=AssertionError("apply must not read status")):
            first = apply(self.bundle, prepared["bundle_sha256"], self.store, writer, run_id=RUN_ID)
        renewed = self.store.issue(run_id=RUN_ID, actions=["memory.commit"])
        replay = apply(self.bundle, prepared["bundle_sha256"], self.store, renewed, run_id=RUN_ID)
        self.assertEqual(first["head"], replay["head"])
        self.assertTrue(replay["replayed"])

    def test_apply_passes_verified_memory_request_without_reopening_file(self):
        prepared = self.prepared()
        real = self.store.commit

        def commit(token, request, **kwargs):
            (self.bundle / "request.json").unlink()
            return real(token, request, **kwargs)

        with mock.patch.object(self.store, "commit", side_effect=commit):
            self.applied(prepared)
        self.assertEqual(self.status()["generation"], 1)

    def test_existing_research_or_orphan_store_rejected_at_prepare(self):
        self.spec()
        self.commit(request_fixture(self.initial_head))
        with self.assertRaises(MemoryConflict):
            self.prepared()
        empty, token = self.make_store("orphan")
        write_private(empty.root / "bodies" / ("a" * 64 + ".md"), b"orphan")
        with self.assertRaises(MemoryConflict):
            prepare(self.spec_path, self.bundle, empty, token, run_id=RUN_ID)
        self.assertFalse(self.bundle.exists())

    def test_fixture_only_not_production_and_overlap_not_allowed(self):
        self.spec()
        production, token = self.make_store("production", domain="production")
        with self.assertRaises(MigrationError):
            prepare(self.spec_path, self.bundle, production, token, run_id=RUN_ID)
        for target in (self.sources / "package", self.root / "package", self.base):
            with self.subTest(location=target.name), self.assertRaises(MigrationError):
                prepare(self.spec_path, target, self.store, self.token, run_id=RUN_ID)
        forbidden = self.base / "credentials"
        forbidden.mkdir(mode=0o700)
        with self.assertRaises(MigrationError):
            prepare(self.spec_path, forbidden / "package", self.store, self.token,
                    run_id=RUN_ID, forbidden_roots=(forbidden,))

    def test_package_member_permissions_symlink_hardlink_and_fifo_rejected(self):
        prepared = self.prepared()
        path = self.bundle / "raw-main.jsonl"
        original = path.read_bytes()
        path.chmod(0o644)
        with self.assertRaises(PrivateFilesystemError):
            self.applied(prepared)
        path.chmod(0o600)
        linked = self.base / "linked"
        os.link(path, linked)
        with self.assertRaises(PrivateFilesystemError):
            self.applied(prepared)
        linked.unlink()
        path.unlink()
        path.symlink_to(self.spec_path)
        with self.assertRaises(PrivateFilesystemError):
            self.applied(prepared)
        path.unlink()
        os.mkfifo(path, mode=0o600)
        with self.assertRaises(PrivateFilesystemError):
            self.applied(prepared)
        path.unlink()
        write_private(path, original)

    def test_source_unsafe_and_changed_during_read_rejected(self):
        spec = self.spec()
        path = Path(spec["sources"][0]["path"])
        linked = self.sources / "linked.jsonl"
        os.link(path, linked)
        with self.assertRaises(ValueError):
            capture(self.spec_path)
        linked.unlink()
        original = path.read_bytes()
        real_read = os.read
        changed = False

        def read(fd, count):
            nonlocal changed
            result = real_read(fd, count)
            if result == original and not changed:
                path.write_bytes(original + b"\n")
                changed = True
            return result

        with mock.patch("rethlas_core.memory_io.os.read", side_effect=read):
            with self.assertRaises(ValueError):
                capture(self.spec_path)
        self.assertTrue(changed)

    def test_cancelled_before_prepare_does_not_create_package(self):
        self.spec()
        event = threading.Event()
        event.set()
        with self.assertRaises(MemoryCancelled):
            self.prepared(cancel_event=event)
        self.assertFalse(self.bundle.exists())
        self.assert_unchanged_head()

    def test_cancelled_during_package_write_leaves_rejected_partial_package(self):
        self.spec()
        event = threading.Event()
        real_write = PrivateDirectory.write_bytes

        def write(directory, name, data, *args, **kwargs):
            result = real_write(directory, name, data, *args, **kwargs)
            if directory.root == self.bundle and name == "map.json":
                event.set()
            return result

        with mock.patch.object(PrivateDirectory, "write_bytes", new=write):
            with self.assertRaises(MemoryCancelled):
                self.prepared(cancel_event=event)
        self.assertFalse((self.bundle / "manifest.json").exists())
        with self.assertRaises((MigrationError, OSError)):
            apply(self.bundle, "0" * 64, self.store, self.token, run_id=RUN_ID)
        self.assert_unchanged_head()


class LegacyCapacityTests(MigrationFixtureMixin, unittest.TestCase):
    def test_physical_line_exact_boundary_and_one_byte_over(self):
        base = legacy_line(record={}, ending=b"")
        maximum = LIMITS["line_bytes"]
        raw = base + b" " * (maximum - len(base))
        self.spec([("main", "generation-10", "proof_steps", raw, None)])
        self.assertEqual(self.converted(True)["counts"]["accepted_lines"], 1)
        self.spec([("main", "generation-10", "proof_steps", raw + b" ", None)], allow=True)
        with self.assertRaises(MigrationCapacity):
            self.converted(True)

    def test_accepted_limit_never_truncated_or_quarantined(self):
        raw = legacy_line(record={}) * LIMITS["accepted"]
        self.spec([("main", "generation-10", "proof_steps", raw, None)])
        self.assertEqual(self.converted(True)["counts"]["accepted_lines"], LIMITS["accepted"])
        self.spec([("main", "generation-10", "proof_steps", raw + legacy_line(record={}), None)], allow=True)
        with self.assertRaises(MigrationCapacity):
            self.converted(True)

    def test_line_count_boundary_includes_blanks(self):
        self.spec([("main", "generation-10", "proof_steps", b"\n" * LIMITS["lines"], None)], allow=True)
        self.assertEqual(self.converted(True)["counts"]["physical_lines"], LIMITS["lines"])
        self.spec([("main", "generation-10", "proof_steps", b"\n" * (LIMITS["lines"] + 1), None)], allow=True)
        with self.assertRaises(MigrationCapacity):
            self.converted(True)

    def test_file_and_raw_total_limits_are_hard_rejections(self):
        maximum = LIMITS["file_bytes"]
        raw = b"x" * maximum
        self.spec([("main", "generation-10", "proof_steps", raw + b"x", None)], allow=True)
        with self.assertRaises(ValueError):
            capture(self.spec_path)
        inputs = [("s" + str(i), "generation-10", "proof_steps", raw,
                   {"problem_id": str(i), "branch_id": None, "source_run_id": None}) for i in range(4)]
        self.spec(inputs)
        snapshot, frozen, _ = capture(self.spec_path)
        self.assertEqual(sum(map(len, frozen.values())), LIMITS["raw_bytes"])
        inputs.append(("fifth", "verification-5", "events", b"x", None))
        self.spec(inputs)
        with self.assertRaises(MigrationCapacity):
            capture(self.spec_path)

    def test_sixteen_files_and_seventeenth_rejected(self):
        inputs = [("s" + str(i), "generation-10", "proof_steps", b"",
                   {"problem_id": str(i), "branch_id": None, "source_run_id": None}) for i in range(17)]
        self.spec(inputs[:16])
        self.assertEqual(len(capture(self.spec_path)[0]["sources"]), 16)
        self.spec(inputs)
        with self.assertRaises(MigrationError):
            capture(self.spec_path)

    def test_target_smaller_request_body_entries_and_hydration_limits(self):
        self.spec()
        for name, limit in (("request_bytes", 256), ("body_bytes", 1), ("entries", 1),
                            ("graph_bytes", 512), ("exchange_bytes", 512)):
            with self.subTest(name=name):
                store, token = self.make_store("small-" + name, limits={name: limit})
                with self.assertRaises(MigrationCapacity):
                    prepare(self.spec_path, self.base / ("bundle-" + name), store, token, run_id=RUN_ID)
                self.assertEqual(store.status(token, run_id=RUN_ID)["generation"], 0)
                self.assertFalse((self.base / ("bundle-" + name)).exists())

    def test_package_capacity_is_checked_before_creating_any_files(self):
        self.spec()
        with mock.patch.dict(LIMITS, {"bundle_bytes": 1}):
            with self.assertRaises(MigrationCapacity):
                self.prepared()
        self.assertFalse(self.bundle.exists())

    def test_reference_index_limit_is_not_silently_trimmed(self):
        self.spec([("main", "generation-10", "proof_steps", legacy_line(record={
            "used_results": ["narrative"] * (LIMITS["references"] + 1)}), None)], allow=True)
        with self.assertRaises(MigrationCapacity):
            self.converted(True)
        self.assertFalse(self.bundle.exists())

    def test_preparation_checks_remaining_store_bytes_and_files(self):
        self.spec()
        store, token = self.make_store("byte-budget", limits={"store_bytes": 65536})
        with self.assertRaises(MigrationCapacity):
            prepare(self.spec_path, self.bundle, store, token, run_id=RUN_ID)
        self.assertFalse(self.bundle.exists())
        store, token = self.make_store("file-budget", limits={"files": 32})
        raw = b"".join(legacy_line(record={"id": "line" + str(i)}) for i in range(20))
        self.spec([("main", "generation-10", "proof_steps", raw, None)])
        with self.assertRaises(MigrationCapacity):
            prepare(self.spec_path, self.bundle, store, token, run_id=RUN_ID)
        self.assertFalse(self.bundle.exists())

    def test_default_request_capacity_is_not_silently_split(self):
        raw = legacy_line(record={"id": "same", "body": "x" * 4096}) * 128
        self.spec([("main", "generation-10", "proof_steps", raw, None)])
        with self.assertRaises(MigrationCapacity):
            self.prepared()
        self.assertFalse(self.bundle.exists())
        self.assert_unchanged_head()
