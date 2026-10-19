"""Independent M1 CLI/socket acceptance using only local synthetic state.

Run with the installed dedicated runtime. tools/accept_cli.py additionally writes
redacted command evidence; ordinary unittest discovery does not write evidence.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT / "agents/generation"
VERIFICATION = ROOT / "agents/verification"
COMMAND_RECORDS: list[dict] = []
OBSERVATIONS: dict = {}
_SECRETS: list[str] = []


def redact(value: str) -> str:
    for secret in _SECRETS:
        value = value.replace(secret, "[REDACTED synthetic capability]")
    value = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
    return re.sub(r"[A-Za-z0-9_-]{100,}\.[A-Za-z0-9_-]{40,}", "[REDACTED capability]", value)


class CliAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rethlas-cli-accept-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        (self.work / "home").mkdir()
        self.env = {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "HOME": str(self.work / "home"), "TMPDIR": str(self.work),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def command(self, arguments, *, cwd=None, env=None, expected=0, timeout=20):
        arguments = [str(argument) for argument in arguments]
        started = time.monotonic()
        result = subprocess.run(arguments, cwd=cwd or self.work,
                                env={**self.env, **(env or {})}, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout)
        COMMAND_RECORDS.append({
            "test": self.id(), "command": arguments, "cwd": str(cwd or self.work),
            "environment_overrides": env or {}, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        if expected is not None:
            self.assertEqual(result.returncode, expected,
                             redact(f"command exit: {arguments!r}\n{result.stdout}\n{result.stderr}"))
        return result

    def python(self, code, *arguments, **kwargs):
        return self.command([sys.executable, "-I", "-B", "-c", code, *arguments], **kwargs)

    def cli(self, *arguments, **kwargs):
        return self.command([sys.executable, "-I", "-B", "-m", "rethlas_core.cli", *arguments], **kwargs)

    def json_cli(self, *arguments, **kwargs):
        return json.loads(self.cli(*arguments, **kwargs).stdout)

    def test_core_imports_are_isolated_and_side_effect_free_in_three_cwds(self):
        private = self.work / "must-not-create-private-root"
        code = """
import json, pathlib, sys
import rethlas_core, rethlas_core.contracts, rethlas_core.verification
modules = (rethlas_core, rethlas_core.contracts, rethlas_core.verification)
assert sys.flags.isolated == 1
assert 'rethlas_core.api' not in sys.modules
assert 'rethlas_core.controller' not in sys.modules
assert all(not hasattr(module, name) for module in modules for name in ('APP', 'app'))
assert not pathlib.Path(sys.argv[1]).exists()
print(json.dumps({'isolated': True, 'core': rethlas_core.__file__, 'app_created': False, 'private_root_created': False}))
"""
        for cwd in (GENERATION, VERIFICATION, self.work):
            with self.subTest(cwd=str(cwd)):
                result = self.python(code, private, cwd=cwd,
                                     env={"RETHLAS_PRIVATE_ROOT": str(private)})
                self.assertTrue(json.loads(result.stdout)["isolated"])
        self.assertFalse(private.exists())
        OBSERVATIONS["isolated_import_cwds"] = [str(GENERATION), str(VERIFICATION), "third-party TemporaryDirectory"]

    def test_installed_console_entrypoint_is_really_invoked(self):
        console = Path(sys.executable).parent / "rethlas-control"
        self.assertTrue(console.is_file(), "editable installation must expose rethlas-control")
        result = self.command([console, "--help"])
        for command in ("validate-policy", "fixture-demo", "serve", "site-sync", "launch"):
            self.assertIn(command, result.stdout)

    def test_cli_accepts_all_three_documented_examples(self):
        for command, example in (("validate-policy", "policy.json"),
                                 ("validate-math", "math-memory.json"),
                                 ("validate-report", "report.json")):
            with self.subTest(command=command):
                result = self.json_cli(command, ROOT / "docs/examples" / example)
                self.assertTrue(result["valid"])

    def test_cli_rejects_invalid_contracts_and_duplicate_json_keys(self):
        for command in ("validate-policy", "validate-math", "validate-report"):
            with self.subTest(command=command):
                invalid = self.work / (command + ".json")
                invalid.write_text('{}', encoding="utf-8")
                result = self.json_cli(command, invalid, expected=1)
                self.assertFalse(result["valid"])
                self.assertTrue(result["errors"])
                invalid.write_text('{"schema_version":"1.0","schema_version":"1.0"}', encoding="utf-8")
                rejected = self.cli(command, invalid, expected=2)
                self.assertIn("rejected", rejected.stderr)

    def test_fixture_demo_persists_a_signed_fixture_not_math_success(self):
        private = self.work / "demo-private"
        self.assertFalse(private.exists())
        result = self.json_cli("fixture-demo", "--private-root", private)
        self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
        self.assertEqual(result["status"], "fixture_completed")
        self.assertIs(result["production_completed"], False)
        self.assertEqual(result["envelope"]["verification_kind"], "fixture")
        self.assertIs(result["envelope"]["production_completed"], False)
        self.assertIn("no mathematical review", result["verification_report"]["summary"])
        self.assertRegex(result["receipt"]["signature"], r"^[0-9a-f]{64}$")
        run_id = result["envelope"]["binding"]["run_id"]
        # status() checks the persisted HMAC and object/finding bindings internally;
        # no key is read or printed by the acceptance harness.
        check = self.python("""
import json, pathlib, sys
from rethlas_core.publication import FixturePublication
result = FixturePublication(pathlib.Path(sys.argv[1]) / 'fixture-publications').status(sys.argv[2])
assert result['receipt']['id'] == sys.argv[3]
print(json.dumps({'persisted_signature_valid': True, 'status': result['status'], 'production_completed': result['production_completed']}))
""", private, run_id, result["receipt"]["id"])
        self.assertTrue(json.loads(check.stdout)["persisted_signature_valid"])
        OBSERVATIONS["fixture_demo"] = {"signed_state_validated": True, "status": result["status"], "production_completed": False}

    def test_cli_init_issue_serve_actual_socket_and_status_staleness(self):
        private = self.work / "service-private"
        initialized = self.json_cli("init", "--private-root", private)
        self.assertEqual(initialized["status"], "initialized")
        credentials = self.work / "credentials"
        credentials.mkdir(mode=0o700)
        token_file = credentials / "fixture.capability"
        issued = self.cli("issue", "--private-root", private, "--owner", "cli-owner",
                          "--run-id", "cli-fixture-run", "--domain", "fixture",
                          "--token-file", token_file, "--ttl", "600")
        token = token_file.read_text(encoding="ascii")
        _SECRETS.append(token)
        self.assertEqual(issued.stdout.strip(), '{"status":"credential-written"}')
        self.assertFalse(token in issued.stdout + issued.stderr, "capability must never reach command output")
        self.assertEqual(stat.S_IMODE(token_file.stat().st_mode), 0o600)
        encoded_claims = token.split(".", 1)[0]
        claims = json.loads(base64.urlsafe_b64decode(encoded_claims + "=" * (-len(encoded_claims) % 4)))
        self.assertEqual(claims["actions"], ["status", "verify"])
        self.assertEqual(claims["domain"], "fixture")
        self.assertEqual(claims["role"], "generation")
        self.assertEqual(claims["expires_at"] - claims["issued_at"], 600)
        overwrite = self.cli("issue", "--private-root", private, "--owner", "cli-owner",
                             "--run-id", "cli-fixture-run", "--domain", "fixture",
                             "--token-file", token_file, expected=2)
        self.assertIn("rejected", overwrite.stderr)
        self.assertTrue(token_file.read_text(encoding="ascii") == token, "credential overwrite was refused")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}/verify"
        arguments = [sys.executable, "-I", "-B", "-m", "rethlas_core.cli", "serve",
                     "--private-root", str(private), "--driver", "fixture",
                     "--host", "127.0.0.1", "--port", str(port)]
        started = time.monotonic()
        process = subprocess.Popen(arguments, cwd=self.work, env=self.env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        http_results = []

        def request(method, path, payload=None, authenticated=False):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["Authorization"] = "Bearer " + token
            try:
                connection.request(method, path, body=None if payload is None else json.dumps(payload), headers=headers)
                response = connection.getresponse()
                body = response.read().decode("utf-8")
                http_results.append({"method": method, "path": path, "authenticated": authenticated,
                                     "status": response.status, "body": body})
                return response.status, json.loads(body)
            finally:
                connection.close()

        try:
            deadline = time.monotonic() + 10
            while True:
                self.assertIsNone(process.poll(), "new CLI service exited before readiness")
                try:
                    status, health = request("GET", "/health")
                    if status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                if time.monotonic() >= deadline:
                    self.fail("new CLI fixture service did not become ready within 10 seconds")
                time.sleep(0.05)
            self.assertFalse(health["live_available"])
            self.assertEqual(health["driver"], "fixture")
            self.assertEqual(request("POST", "/verify", {})[0], 401)
            # This existing operator script calls the installed shared client,
            # rather than TestClient/ASGI transport or an internal controller.
            verified = self.command([sys.executable, "-I", "-B",
                                     VERIFICATION / "scripts/test_verify_endpoint.py",
                                     "--token-file", token_file, "--owner", "cli-owner",
                                     "--run-id", "cli-fixture-run", "--endpoint", endpoint])
            result = json.loads(verified.stdout)
            self.assertEqual(result["status"], "fixture_completed")
            self.assertFalse(result["production_completed"])
            statement = self.work / "statement.md"
            proof = self.work / "proof.md"
            statement.write_text("Synthetic endpoint-test statement.", encoding="utf-8")
            proof.write_text("Synthetic body; no mathematical review.", encoding="utf-8")
            current = self.json_cli("status", "--endpoint", endpoint, "--token-file", token_file,
                                    "--statement", statement, "--proof", proof, expected=1)
            self.assertEqual(current["status"], "fixture_completed")
            self.assertEqual(current["receipt"], result["receipt"])
            self.assertFalse(current["production_completed"])
            proof.write_text("Tampered synthetic proof.", encoding="utf-8")
            stale = self.json_cli("status", "--endpoint", endpoint, "--token-file", token_file,
                                  "--statement", statement, "--proof", proof, expected=1)
            self.assertEqual(stale, {"status": "stale", "production_completed": False})
            mismatch = {"owner": "cli-owner", "run_id": "outside-capability", "domain": "fixture",
                        "request_id": "out-of-scope", "statement": "Synthetic.", "proof": "Synthetic."}
            self.assertEqual(request("POST", "/verify", mismatch, authenticated=True)[0], 403)
            OBSERVATIONS["actual_socket"] = {
                "endpoint": endpoint, "health": health, "shared_client_script_exit": verified.returncode,
                "capability_file_mode": "0600", "capability_actions": claims["actions"],
                "capability_domain": claims["domain"], "capability_ttl_seconds": 600,
                "fixture_status_exit": 1, "fixture_status": current["status"],
                "tampered_proof_status": stale["status"], "production_completed": False,
                "out_of_scope_http_status": 403,
            }
        finally:
            # Only this process group, created immediately above, is terminated.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=5)
            COMMAND_RECORDS.append({"test": self.id(), "command": arguments, "cwd": str(self.work),
                                    "exit_code": process.returncode, "stdout": stdout, "stderr": stderr,
                                    "http_interactions": http_results,
                                    "elapsed_seconds": round(time.monotonic() - started, 3),
                                    "termination": "owned process group only"})
            self.assertFalse(token in stdout + stderr, "service logs must not contain capability")

    def test_sandbox_probe_is_real_and_records_available_or_blocked(self):
        completed = self.cli("sandbox-probe", expected=None, timeout=12)
        result = json.loads(completed.stdout)
        self.assertIn(result["status"], ("available", "blocked"))
        self.assertEqual(completed.returncode, 0 if result["status"] == "available" else 2)
        self.assertEqual(result["backend"], "bubblewrap")
        if result["status"] == "blocked":
            self.assertTrue(result["reason"])
        OBSERVATIONS["sandbox_probe"] = {**result, "exit_code": completed.returncode}

    def test_generation_wrapper_has_valid_bash_syntax(self):
        self.command(["/usr/bin/bash", "-n", GENERATION / "tests/run_example.sh"])

    def test_generation_wrapper_dry_run_and_live_fail_closed(self):
        binaries = self.work / "bin"
        binaries.mkdir()
        marker = self.work / "codex-was-executed"
        sentinel = binaries / "codex"
        sentinel.write_text('#!/usr/bin/bash\nprintf executed > "$CODEX_SENTINEL"\nexit 97\n', encoding="utf-8")
        sentinel.chmod(0o700)
        env = {"RETHLAS_PYTHON": sys.executable, "PATH": str(binaries) + ":/usr/bin:/bin",
               "CODEX_SENTINEL": str(marker)}
        dry = self.command(["/usr/bin/bash", GENERATION / "tests/run_example.sh"], env={**env, "DRY_RUN": "1"})
        result = json.loads(dry.stdout)
        self.assertEqual(result["status"], "unreviewed")
        self.assertFalse(result["production_completed"])
        live = self.command(["/usr/bin/bash", GENERATION / "tests/run_example.sh"],
                            env={**env, "DRY_RUN": "0"}, expected=69)
        self.assertIn("No Codex process was started", live.stderr)
        self.assertFalse(marker.exists(), "neither wrapper invocation may execute Codex")
        OBSERVATIONS["generation_wrapper"] = {"dry_run_exit": 0, "non_dry_exit": 69, "codex_executed": False}

    def test_launch_classification_legacy_filename_resume_and_conflict(self):
        generation = self.work / "generation"
        source = generation / "data/algebra/nested/example.md"
        source.parent.mkdir(parents=True)
        source.write_text("Synthetic theorem.", encoding="utf-8")
        results = generation / "results/algebra/nested/example"
        results.mkdir(parents=True)
        (results / "blueprint_verified.md").write_text("Forged legacy verified filename.", encoding="utf-8")
        (results / "PAUSE_AFTER_ITERATION").touch()
        logs = generation / "logs/algebra/nested/example/iter"
        logs.mkdir(parents=True)
        (logs / "example_iter_2.md").write_text("session id: synthetic-session-A\n", encoding="utf-8")
        args = ("launch", "--generation-root", generation, "--problem", "data/algebra/nested/example.md", "--dry-run")
        result = self.json_cli(*args)
        self.assertEqual(result["problem_id"], "algebra/nested/example")
        self.assertEqual(result["next_iteration"], 3)
        for key in ("resume_available", "pause_present", "legacy_verified_file_present"):
            self.assertTrue(result[key])
        self.assertEqual(result["status"], "unreviewed")
        self.assertFalse(result["production_completed"])
        self.assertFalse(result["live_available"])
        source.write_text("Changed synthetic statement.", encoding="utf-8")
        self.assertFalse(self.json_cli(*args)["production_completed"])
        (logs / "example_iter_3.md").write_text("session id: synthetic-session-B\n", encoding="utf-8")
        self.cli(*args, expected=2)
        self.cli("launch", "--generation-root", generation, "--problem", "data/../secret.md", "--dry-run", expected=2)
        OBSERVATIONS["legacy_launch"] = {"classified_problem_id": result["problem_id"], "next_iteration": 3,
                                         "legacy_verified_promotes": False, "conflicting_sessions_exit": 2}

    def test_site_sync_only_temporary_drafts_repeat_and_manual_landing(self):
        results, content = self.work / "results", self.work / "content"
        content.mkdir()
        landing, notes = content / "_index.md", content / "notes.md"
        landing.write_text("Hand-authored landing.\n", encoding="utf-8")
        notes.write_text("Hand-authored notes.\n", encoding="utf-8")
        problem = results / "algebra/example"
        problem.mkdir(parents=True)
        source = problem / "blueprint_verified.md"
        source.write_text("# Synthetic\n<script>not executed</script>\n", encoding="utf-8")
        args = ("site-sync", "--results", results, "--content", content)
        first = self.json_cli(*args)
        page = content / "algebra/example.md"
        first_body = page.read_text(encoding="utf-8")
        second = self.json_cli(*args)
        self.assertEqual(first, second)
        self.assertEqual(first, {"pages": 1, "review_status": "unreviewed", "production_completed": False})
        self.assertEqual(page.read_text(encoding="utf-8"), first_body)
        self.assertIn("UNREVIEWED DRAFT", first_body)
        self.assertIn('review_status = "unreviewed"', first_body)
        self.assertIn("&lt;script&gt;", first_body)
        self.assertNotIn("<script>", first_body)
        self.assertEqual(landing.read_text(encoding="utf-8"), "Hand-authored landing.\n")
        source.unlink()
        self.assertEqual(self.json_cli(*args)["pages"], 0)
        self.assertFalse(page.exists())
        self.assertEqual(landing.read_text(encoding="utf-8"), "Hand-authored landing.\n")
        self.assertEqual(notes.read_text(encoding="utf-8"), "Hand-authored notes.\n")
        OBSERVATIONS["site_sync"] = {"temporary_only": True, "repeat_idempotent": True,
                                     "manual_landing_preserved": True, "only_unreviewed_drafts": True}

    def test_both_mcp_imports_package_aliases_and_unavailable_gate(self):
        private = self.work / "must-not-create-mcp-private"
        code = """
import importlib.util, json, pathlib, sys
from rethlas_core.worker import WorkerUnavailable
root, private = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
observed = []
for name, relative, package in (
    ('accept_generation_mcp_server', 'agents/generation/mcp/server.py', False),
    ('accept_verification_mcp_server', 'agents/verification/mcp/server.py', False),
    ('accept_generation_mcp_package', 'agents/generation/mcp/__init__.py', True),
):
    path = root / relative
    generation = path.parents[1]
    state_paths = (generation / 'memory', generation / 'results', private)
    before = [item.exists() for item in state_paths]
    kwargs = {'submodule_search_locations': [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert not hasattr(module, 'APP') and not hasattr(module, 'app')
    assert 'rethlas_core.api' not in sys.modules
    if package:
        server = sys.modules[name + '.server']
        for alias in module.__all__:
            assert getattr(module, alias) is getattr(server, alias)
    try:
        module.build_mcp_app()
    except WorkerUnavailable:
        gate = 'WorkerUnavailable'
    else:
        raise AssertionError('M1 unmanaged worker gate was bypassed')
    assert [item.exists() for item in state_paths] == before
    assert not private.exists()
    observed.append({'module': name, 'APP_created': False, 'gate': gate, 'package_aliases_checked': package})
print(json.dumps(observed))
"""
        result = self.python(code, ROOT, private,
                             env={"RETHLAS_PRIVATE_ROOT": str(private)}, timeout=30)
        observed = json.loads(result.stdout)
        self.assertEqual(len(observed), 3)
        self.assertTrue(all(item["gate"] == "WorkerUnavailable" for item in observed))
        OBSERVATIONS["mcp_imports"] = observed


if __name__ == "__main__":
    unittest.main()
