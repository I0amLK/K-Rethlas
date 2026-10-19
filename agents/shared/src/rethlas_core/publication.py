"""Trusted, private fixture receipts; model-writable files are never promotion evidence."""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
import secrets

from .private_fs import PrivateDirectory, canonical_json, digest
from .verification import coverage_errors, normalize_findings, require_valid_output


class PublicationConflict(ValueError):
    pass


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identifier(value: str) -> str:
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) or value in {".", ".."}:
        raise ValueError("invalid identifier")
    return value


class FixturePublication:
    """A separate test domain. This class cannot issue production-review receipts.

    Only the controller can access this root. Findings accepted by this object
    remain registered even if a worker deletes its own draft JSONL files.
    """

    def __init__(self, root: Path, *, create: bool = False):
        self.directory = PrivateDirectory(root, create=create)
        with self.directory.lock():
            try:
                self.key = self.directory.read_bytes("receipt-key")
            except FileNotFoundError:
                if not create:
                    raise
                self.key = secrets.token_bytes(32)
                self.directory.write_bytes("receipt-key", self.key, exclusive=True)
            if len(self.key) != 32:
                raise ValueError("invalid receipt key")
            self.runs = self.directory.child("runs", create=create)
            self.objects = self.directory.child("objects", create=create)
            self.receipts = self.directory.child("receipts", create=create)
            try:
                self.directory.read_json("manifest.json")
            except FileNotFoundError:
                if not create:
                    raise
                self.directory.write_json("manifest.json", {"schema_version": "1.0", "runs": {}})

    def _run(self, run_id: str) -> PrivateDirectory:
        return self.runs.child(identifier(run_id))

    def _object(self, text: str) -> str:
        name = text_hash(text) + ".md"
        data = text.encode("utf-8")
        try:
            self.objects.write_bytes(name, data, exclusive=True)
        except FileExistsError:
            if self.objects.read_bytes(name) != data:
                raise ValueError("content-addressed object mismatch")
        return name

    def begin(self, binding: dict, statement: str, proof: str) -> dict | None:
        if binding["domain"] != "fixture":
            raise ValueError("fixture publication requires fixture domain")
        if binding["statement_sha256"] != text_hash(statement) or binding["proof_sha256"] != text_hash(proof):
            raise ValueError("input binding mismatch")
        run_id = identifier(binding["run_id"])
        with self.directory.lock():
            manifest = self.directory.read_json("manifest.json")
            if run_id in manifest["runs"]:
                # A committed run stays sealed even if the final state update
                # failed or its ledger is missing. Never recreate it on replay.
                result = self._status_locked(run_id)
                if result["envelope"]["binding"] != binding:
                    raise PublicationConflict("run is already bound to another request or revision")
                return result
            run = self.runs.child(run_id, create=True)
            try:
                state = run.read_json("state.json")
            except FileNotFoundError:
                state = None
            if state is not None:
                if state["binding"] != binding:
                    raise PublicationConflict("run is already bound to another request or revision")
                raise PublicationConflict("run already started; no implicit retry or replacement")
            self._object(statement)
            self._object(proof)
            run.write_json("state.json", {
                "binding": binding, "state": "started", "findings": [], "events": [],
            })
        return None

    def append_findings(self, run_id: str, records: list[dict]) -> None:
        findings = normalize_findings(records)
        run_id = identifier(run_id)
        with self.directory.lock():
            if run_id in self.directory.read_json("manifest.json")["runs"]:
                raise PublicationConflict("findings are sealed by the committed manifest")
            run = self._run(run_id)
            state = run.read_json("state.json")
            if state["state"] != "started":
                raise PublicationConflict("findings are sealed")
            event = {"sequence": len(state["events"]), "records": records,
                     "normalized": findings, "previous": digest(state["events"])}
            name = f"finding-{event['sequence']:08d}.json"
            run.write_bytes(name, canonical_json(event), exclusive=True)
            state["events"].append({"name": name, "sha256": digest(event)})
            state["findings"].extend(findings)
            run.write_json("state.json", state)

    def _registered(self, run: PrivateDirectory, state: dict) -> list[dict]:
        findings, previous = [], []
        for index, reference in enumerate(state["events"]):
            event = run.read_json(reference["name"])
            if digest(event) != reference["sha256"] or event["sequence"] != index or event["previous"] != digest(previous):
                raise ValueError("finding ledger integrity failure")
            normalized = normalize_findings(event["records"])
            if normalized != event["normalized"]:
                raise ValueError("finding ledger normalization changed")
            findings.extend(normalized)
            previous.append(reference)
        if findings != state["findings"]:
            raise ValueError("finding ledger mismatch")
        return findings

    def complete_fixture(self, run_id: str, output: dict, binding: dict, *, before_commit=None) -> dict:
        # Caller holds authority.locked_check during this commit. Lock order is
        # authority -> publication. No model/provider mode can be passed here.
        require_valid_output(output)
        run_id = identifier(run_id)
        with self.directory.lock():
            manifest = self.directory.read_json("manifest.json")
            if run_id in manifest["runs"]:
                raise PublicationConflict("run is sealed by the committed manifest")
            run = self._run(run_id)
            state = run.read_json("state.json")
            if state["state"] != "started" or state["binding"] != binding:
                raise PublicationConflict("run is sealed or binding changed")
            findings = self._registered(run, state)
            errors = coverage_errors(output, findings)
            if errors:
                raise ValueError("registered finding omitted from final report")
            for key in ("statement_sha256", "proof_sha256"):
                data = self.objects.read_bytes(binding[key] + ".md")
                if hashlib.sha256(data).hexdigest() != binding[key]:
                    raise ValueError("frozen input object changed")
            envelope = {
                "schema_version": "1.0", "binding": binding,
                "verification_kind": "fixture", "production_completed": False,
                "status": "fixture_completed" if output["verdict"] == "correct" else "fixture_rejected",
                "findings_sha256": digest(findings), "output_sha256": digest(output),
                "output": output,
            }
            signature = hmac.new(self.key, canonical_json(envelope), hashlib.sha256).hexdigest()
            receipt = {"envelope": envelope, "signature": signature}
            receipt_id = digest(receipt)
            self.receipts.write_bytes(receipt_id + ".json", canonical_json(receipt), exclusive=True)
            manifest["runs"][run_id] = receipt_id
            # The manifest is the authority for completion, not mutable drafts or
            # an orphan receipt left by an interrupted commit.
            if before_commit is not None:
                before_commit()
            self.directory.write_json("manifest.json", manifest)
            state["state"] = "completed"
            run.write_json("state.json", state)
            return self._status_locked(run_id)

    def status(self, run_id: str, statement: str | None = None, proof: str | None = None) -> dict:
        with self.directory.lock():
            return self._status_locked(identifier(run_id), statement, proof)

    def _status_locked(self, run_id: str, statement: str | None = None, proof: str | None = None) -> dict:
        manifest = self.directory.read_json("manifest.json")
        receipt_id = manifest["runs"].get(run_id)
        if receipt_id is None:
            return {"status": "unreviewed", "production_completed": False}
        receipt = self.receipts.read_json(receipt_id + ".json")
        envelope = receipt["envelope"]
        expected = hmac.new(self.key, canonical_json(envelope), hashlib.sha256).hexdigest()
        if digest(receipt) != receipt_id or not hmac.compare_digest(expected, receipt["signature"]):
            raise ValueError("receipt integrity failure")
        binding = envelope["binding"]
        if binding["run_id"] != run_id or binding["domain"] != "fixture" or envelope["verification_kind"] != "fixture" or envelope["production_completed"] is not False:
            raise ValueError("receipt has invalid trust domain")
        if ((statement is not None and text_hash(statement) != binding["statement_sha256"])
                or (proof is not None and text_hash(proof) != binding["proof_sha256"])):
            return {"status": "stale", "production_completed": False}
        run = self._run(run_id)
        state = run.read_json("state.json")
        if state["binding"] != binding or digest(self._registered(run, state)) != envelope["findings_sha256"]:
            raise ValueError("receipt no longer matches finding ledger")
        for key in ("statement_sha256", "proof_sha256"):
            if hashlib.sha256(self.objects.read_bytes(binding[key] + ".md")).hexdigest() != binding[key]:
                raise ValueError("receipt input object is corrupt")
        output = envelope["output"]
        require_valid_output(output)
        if digest(output) != envelope["output_sha256"] or coverage_errors(output, state["findings"]):
            raise ValueError("receipt output binding is corrupt")
        return {**output, "status": envelope["status"], "production_completed": False,
                "envelope": envelope, "receipt": {"id": receipt_id, "signature": receipt["signature"]}}
