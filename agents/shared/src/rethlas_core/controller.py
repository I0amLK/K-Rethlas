"""Trusted fixed-driver orchestration. No live model/provider broker in M1."""
from __future__ import annotations

from pathlib import Path
import threading

from .authorization import Authority
from .private_fs import digest, strict_json_loads
from .publication import FixturePublication, identifier, text_hash
from .verification import require_valid_output
from .worker import WorkerCancelled, WorkerUnavailable, run_fixture_process


class Controller:
    def __init__(self, root: Path, *, driver: str = "disabled", timeout: float = 5):
        if driver not in {"disabled", "fixture"}:
            raise ValueError("M1 supports only disabled or trusted fixture driver")
        if not 0 < timeout <= 30:
            raise ValueError("invalid driver timeout")
        self.driver = driver
        self.timeout = timeout
        self.authority = Authority(root / "authority")
        self.publications = FixturePublication(root / "fixture-publications")
        self.config = {"driver": driver, "protocol": "1.0", "timeout": timeout,
                       "review_kind": "fixture", "network": "none"}

    @staticmethod
    def validate_request(payload: dict) -> dict:
        required = {"owner", "run_id", "domain", "request_id", "statement", "proof"}
        if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - required - {"dependencies"}:
            raise ValueError("invalid verification request fields")
        for key in ("owner", "run_id", "request_id"):
            identifier(payload[key])
        if payload["domain"] not in {"fixture", "production"}:
            raise ValueError("invalid domain")
        for key in ("statement", "proof"):
            value = payload[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 20000:
                raise ValueError("invalid input text")
        dependencies = payload.get("dependencies", [])
        if not isinstance(dependencies, list) or len(dependencies) > 100:
            raise ValueError("invalid dependency list")
        import re
        seen = set()
        for entry in dependencies:
            if not isinstance(entry, dict) or set(entry) != {"revision_id", "sha256"}:
                raise ValueError("invalid dependency binding")
            identifier(entry["revision_id"])
            if entry["revision_id"] in seen or not isinstance(entry["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("invalid dependency binding")
            seen.add(entry["revision_id"])
        return {**payload, "dependencies": sorted(dependencies, key=lambda item: item["revision_id"])}

    def verify(self, token: str, payload: dict, *, cancel_event: threading.Event | None = None) -> dict:
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise WorkerCancelled("verification request cancelled")

        check_cancelled()
        payload = self.validate_request(payload)
        identity = {key: payload[key] for key in ("owner", "run_id", "domain")}
        claims = self.authority.check(token, action="verify", **identity)
        if self.driver != "fixture" or payload["domain"] != "fixture":
            raise WorkerUnavailable("live Codex/provider broker is unavailable in M1")
        binding = {
            **identity, "request_id": payload["request_id"], "epoch": claims["epoch"],
            "statement_sha256": text_hash(payload["statement"]),
            "proof_sha256": text_hash(payload["proof"]),
            "dependencies_sha256": digest(payload["dependencies"]),
            "config_sha256": digest(self.config),
        }
        with self.authority.locked_check(token, action="verify", **identity):
            check_cancelled()
            prior = self.publications.begin(binding, payload["statement"], payload["proof"])
        if prior is not None:
            return prior
        from .private_fs import canonical_json
        result = strict_json_loads(run_fixture_process(canonical_json({
            "statement": payload["statement"], "proof": payload["proof"],
        }), timeout=self.timeout, cancel_event=cancel_event))
        if not isinstance(result, dict) or set(result) != {"output", "findings"}:
            raise ValueError("invalid fixture driver output")
        require_valid_output(result["output"])
        with self.authority.locked_check(token, action="verify", **identity) as commit_claims:
            check_cancelled()
            self.publications.append_findings(payload["run_id"], result["findings"])

            def before_commit():
                import time
                check_cancelled()
                if not commit_claims["issued_at"] <= time.time() < commit_claims["expires_at"]:
                    from .authorization import AuthorizationError
                    raise AuthorizationError("capability expired before commit")

            return self.publications.complete_fixture(payload["run_id"], result["output"], binding,
                                                        before_commit=before_commit)

    def status(self, token: str, *, statement: str | None = None, proof: str | None = None) -> dict:
        claims = self.authority.authenticate(token, action="status")
        if claims["domain"] != "fixture":
            return {"status": "unreviewed", "production_completed": False}
        with self.authority.locked_check(token, owner=claims["owner"], run_id=claims["run_id"],
                                         domain=claims["domain"], action="status"):
            result = self.publications.status(claims["run_id"], statement, proof)
            if "envelope" in result:
                binding = result["envelope"]["binding"]
                if (any(binding[key] != claims[key] for key in ("owner", "run_id", "domain", "epoch"))
                        or binding["config_sha256"] != digest(self.config)):
                    return {"status": "stale", "production_completed": False}
            return result
