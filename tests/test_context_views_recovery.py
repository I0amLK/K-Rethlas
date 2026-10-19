"""Real spawned processes/SIGKILL and synthetic fault recovery for T08."""
from __future__ import annotations

from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError, ROLE_ACTIONS
from rethlas_core.context_views import ContextError, ContextIndeterminate, ContextRecoveryRequired
from rethlas_core.context_workspace import ContextWorkspace, _branch_dir
from rethlas_core.private_fs import PrivateDirectory, canonical_json
from rethlas_core.worker import SandboxWorker, WorkerCancelled, WorkerTimeout, WorkerUnavailable

from test_context_views import ContextFixtureMixin, VERSION, active
from test_memory_store import read_json, write_private, encoded


PROCESS_TIMEOUT = 20


def seal_process(root, grant, request, stage, ready, release):
    workspace = ContextWorkspace(Path(root))
    def checkpoint(actual):
        if actual == stage:
            ready.set()
            if not release.wait(PROCESS_TIMEOUT):
                raise RuntimeError("fixture release deadline")
    workspace._checkpoint = checkpoint
    workspace.seal(grant["token"], request, run_id=grant["run_id"])


def commit_process(root, grant, request, start, output):
    try:
        if not start.wait(PROCESS_TIMEOUT):
            raise RuntimeError("fixture start deadline")
        result = ContextWorkspace(Path(root)).branch_commit(grant["token"], request, run_id=grant["run_id"])
        output.put({"status": result["status"], "head": result["head"], "replayed": result["replayed"]})
    except Exception as exc:
        output.put({"error": type(exc).__name__})


class ContextRecoveryTests(ContextFixtureMixin, unittest.TestCase):
    def test_real_sigkill_each_seal_boundary_requires_matching_evidence(self):
        context = multiprocessing.get_context("spawn")
        # Independent workspaces, since sealing is deliberately irreversible.
        for number, stage in enumerate(("after_seal_intent", "after_seal_close", "after_seal_receipt")):
            with self.subTest(stage=stage):
                workspace = ContextWorkspace.initialize(self.base / ("kill-%d" % number), self.source,
                    self.source_token, run_id="source-run", spec=self.spec)
                grant = workspace.issue(kind="branch", branch_id="left", actions=list(ROLE_ACTIONS["context_branch"]))
                snapshot = workspace.branch_read(grant["token"], run_id=grant["run_id"])
                request = {"version": VERSION, "request_id": "seal", "expected_head": snapshot["head"], "active": active(snapshot["graph"])}
                ready, release = context.Event(), context.Event()
                process = context.Process(target=seal_process, args=(str(workspace.root), grant, request, stage, ready, release))
                process.start()
                try:
                    self.assertTrue(ready.wait(PROCESS_TIMEOUT), "seal did not reach checkpoint")
                    os.kill(process.pid, signal.SIGKILL)
                    process.join(PROCESS_TIMEOUT)
                    self.assertEqual(process.exitcode, -signal.SIGKILL)
                finally:
                    if process.is_alive():
                        process.kill()
                        process.join(PROCESS_TIMEOUT)
                    process.close()
                reopened = ContextWorkspace(workspace.root)
                with self.assertRaises(AuthorizationError):
                    reopened.branch_read(grant["token"], run_id=grant["run_id"])
                result = reopened.recover_branch("left", apply=True)
                self.assertEqual(result["status"], "sealed")
                self.assertEqual(result["head"], snapshot["head"])
                with self.assertRaises(ContextRecoveryRequired):
                    reopened.issue(kind="join", actions=["context.join.read"])

    def test_real_competing_same_request_commits_once(self):
        context = multiprocessing.get_context("spawn")
        start, output = context.Event(), context.Queue()
        request = self.branch_request()
        processes = [context.Process(target=commit_process, args=(str(self.workspace.root), self.branch(), request, start, output))
                     for _ in range(2)]
        for process in processes:
            process.start()
        start.set()
        try:
            results = [output.get(timeout=PROCESS_TIMEOUT) for _ in processes]
            for process in processes:
                process.join(PROCESS_TIMEOUT)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual({r["status"] for r in results}, {"committed"})
            self.assertEqual(len({r["head"] for r in results}), 1)
            self.assertEqual(sorted(r["replayed"] for r in results), [False, True])
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join(PROCESS_TIMEOUT)
                process.close()
            output.close()
            output.join_thread()

    def test_real_commit_waiting_on_seal_is_denied_after_closure(self):
        context = multiprocessing.get_context("spawn")
        request = self.branch_request()
        snapshot = self.read_branch()
        seal_request = {"version": VERSION, "request_id": "race-seal", "expected_head": snapshot["head"],
                        "active": active(snapshot["graph"])}
        ready, release, start, output = context.Event(), context.Event(), context.Event(), context.Queue()
        sealer = context.Process(target=seal_process, args=(str(self.workspace.root), self.branch(), seal_request,
                                                          "after_seal_intent", ready, release))
        writer = context.Process(target=commit_process, args=(str(self.workspace.root), self.branch(), request, start, output))
        sealer.start()
        try:
            self.assertTrue(ready.wait(PROCESS_TIMEOUT))
            writer.start()
            start.set()
            release.set()
            sealer.join(PROCESS_TIMEOUT)
            writer.join(PROCESS_TIMEOUT)
            self.assertEqual(sealer.exitcode, 0)
            self.assertEqual(writer.exitcode, 0)
            self.assertEqual(output.get(timeout=PROCESS_TIMEOUT), {"error": "AuthorizationError"})
            self.assertEqual(self.workspace.recover_branch("left")["head"], snapshot["head"])
        finally:
            release.set()
            for process in (sealer, writer):
                if process.pid is not None:
                    if process.is_alive():
                        process.kill()
                        process.join(PROCESS_TIMEOUT)
                    process.close()
            output.close()
            output.join_thread()

    def test_seal_fsync_failure_is_indeterminate_not_success(self):
        original = PrivateDirectory.write_bytes
        def fail_after_write(directory, name, data, **kwargs):
            result = original(directory, name, data, **kwargs)
            if name.endswith(".intent.json") and directory.root.name == "seals":
                raise OSError("synthetic directory fsync failure")
            return result
        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_after_write), self.assertRaises(ContextIndeterminate):
            self.seal_branch("left")
        with self.assertRaises(AuthorizationError):
            self.read_branch()
        self.workspace = ContextWorkspace(self.workspace.root)
        self.assertEqual(self.workspace.recover_branch("left", apply=True)["status"], "sealed")

    def test_retained_atomic_temporary_does_not_hide_recoverable_authority(self):
        def checkpoint(stage):
            if stage == "after_seal_intent":
                raise RuntimeError("synthetic stop")
        with mock.patch.object(self.workspace, "_checkpoint", side_effect=checkpoint), self.assertRaises(RuntimeError):
            self.seal_branch("left")
        # SIGKILL may leave the temporary before registry.json replacement.
        for directory in (self.workspace.root, self.workspace.root / "authority"):
            write_private(directory / (".tmp-" + "a" * 32), b"synthetic partial atomic write")
        self.workspace = ContextWorkspace(self.workspace.root)
        self.assertEqual(self.workspace.recover_branch("left", apply=True)["status"], "sealed")
        self.assertTrue((self.workspace.root / "authority" / (".tmp-" + "a" * 32)).exists())

    def test_registry_replace_failure_during_seal_never_implies_success(self):
        original = PrivateDirectory.write_bytes
        def fail_after_registry(directory, name, data, **kwargs):
            result = original(directory, name, data, **kwargs)
            if directory.root == self.workspace.root / "authority" and name == "registry.json":
                raise OSError("synthetic registry directory fsync failure")
            return result
        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_after_registry), self.assertRaises(ContextIndeterminate):
            self.seal_branch("left")
        self.workspace = ContextWorkspace(self.workspace.root)
        with self.assertRaises(AuthorizationError):
            self.read_branch()
        self.assertEqual(self.workspace.recover_branch("left", apply=True)["status"], "sealed")

    def test_pre_replace_admission_failure_does_not_start_store_read(self):
        original = PrivateDirectory.write_bytes
        def reject_state(directory, name, data, **kwargs):
            if directory.root == self.workspace.root and name == "state.json":
                raise OSError("synthetic write failure before replace")
            return original(directory, name, data, **kwargs)
        with mock.patch.object(PrivateDirectory, "write_bytes", new=reject_state), \
                mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), \
                self.assertRaises(ContextIndeterminate):
            self.read_branch()

    def test_closed_without_intent_is_not_completed_branch(self):
        self.workspace.authority.close_run(self.branch()["run_id"])
        self.assertEqual(self.workspace.recover_branch("left")["status"], "unsealed")
        with self.assertRaises(ContextRecoveryRequired):
            self.workspace.issue(kind="join", actions=["context.join.read"])

    def test_sealed_receipt_without_closed_run_blocks_join(self):
        self.seal_branch("left")
        self.seal_branch("right")
        path = self.workspace.root / "authority/registry.json"
        registry = read_json(path)
        registry["runs"][self.branch()["run_id"]]["state"] = "open"
        write_private(path, encoded(registry))
        with self.assertRaises(ContextRecoveryRequired):
            self.workspace.issue(kind="join", actions=["context.join.read"])

    def test_join_intent_cannot_auto_publish_or_rebind(self):
        def checkpoint(stage):
            if stage == "after_join_intent":
                raise RuntimeError("synthetic stop")
        with mock.patch.object(self.workspace, "_checkpoint", side_effect=checkpoint), self.assertRaises(RuntimeError):
            self.join(conflict=True)
        self.workspace = ContextWorkspace(self.workspace.root)
        intent = read_json(self.workspace.root / "join.intent.json")
        request = intent["request"]
        grant = self.workspace.issue(kind="join", actions=list(ROLE_ACTIONS["context_join"]))
        with self.assertRaises(ContextRecoveryRequired):
            self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])
        self.assertFalse((self.workspace.root / "join.json").exists())
        result = self.workspace.recover_publication(kind="join", request=request, apply=True)
        self.assertEqual(result["status"], "published")
        self.assertTrue(self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])["replayed"])

    def test_view_intent_only_is_unreadable_until_operator_recovery(self):
        def checkpoint(stage):
            if stage == "after_view_intent":
                raise RuntimeError("synthetic stop")
        with mock.patch.object(self.workspace, "_checkpoint", side_effect=checkpoint), self.assertRaises(RuntimeError):
            self.freeze_view()
        manifest = read_json(self.workspace.root / "view.intent.json")["manifest"]
        self.workspace = ContextWorkspace(self.workspace.root)
        with self.assertRaises(ContextError):
            self.workspace.issue(kind="view", actions=["context.view.read"])
        self.workspace.recover_publication(kind="view", request=manifest, apply=True)
        verifier = self.workspace.issue(kind="view", actions=["context.view.read"])
        self.assertTrue(self.workspace.view_read(verifier["token"], run_id=verifier["run_id"])["revisions"])

    def test_crash_after_read_admission_never_refunds(self):
        def checkpoint(stage):
            if stage == "after_admission":
                raise RuntimeError("synthetic stop")
        with mock.patch.object(self.workspace, "_checkpoint", side_effect=checkpoint), self.assertRaises(RuntimeError):
            self.read_branch()
        state = read_json(self.workspace.root / "state.json")
        self.assertEqual(state["counts"]["read." + _branch_dir("left")], 1)
        self.workspace = ContextWorkspace(self.workspace.root)
        self.read_branch()
        state = read_json(self.workspace.root / "state.json")
        self.assertEqual(state["counts"]["read." + _branch_dir("left")], 2)


class ContextSandboxTests(ContextFixtureMixin, unittest.TestCase):
    def test_actual_probe_has_fresh_work_and_exact_observation(self):
        verifier, _, _ = self.freeze_view()
        result = self.workspace.probe(verifier["token"], run_id=verifier["run_id"])
        self.assertTrue(result["observation"]["empty_work"])
        self.assertTrue(result["observation"]["input_readonly"])
        self.assertFalse(result["mathematical_review"])
        self.assertNotIn("PRIVATE_", canonical_json(result).decode())
        again = self.workspace.probe(verifier["token"], run_id=verifier["run_id"])
        self.assertEqual(result, again)

    def test_real_sandbox_denies_workspace_source_history_network_and_cache(self):
        root = PrivateDirectory(self.base / "sandbox", create=True)
        input_dir = root.child("input", create=True)
        work_dir = root.child("work", create=True)
        target = str(self.workspace.root / "snapshot.json")
        source = str(self.source.root / "authority/key")
        marker = self.base / "host-history"
        marker.write_text("PRIVATE_HOST_HISTORY", encoding="utf-8")
        (input_dir.root / "escape").symlink_to(marker)
        code = "\n".join([
            "from pathlib import Path", "import socket",
            "for name in " + repr([target, source, str(marker), "/input/escape", "/work/previous-conversation"]) + ":",
            "    try: Path(name).read_bytes()", "    except OSError: pass", "    else: raise AssertionError('host/history visible')",
            "for address in [('127.0.0.1', 9), ('1.1.1.1', 53)]:",
            "    sock = socket.socket(); sock.settimeout(0.2)", "    try: sock.connect(address)",
            "    except OSError: pass", "    else: raise AssertionError('network reachable')", "    finally: sock.close()",
            "print('denied')"])
        result = SandboxWorker().run_bounded(input_dir.root, work_dir.root, ["/usr/bin/python3.12", "-I", "-B", "-c", code])
        self.assertEqual(result.stdout, b"denied\n")

    def test_real_bounded_output_timeout_cancel_and_unavailable_never_fallback(self):
        root = PrivateDirectory(self.base / "sandbox", create=True)
        input_dir = root.child("input", create=True)
        work_dir = root.child("work", create=True)
        worker = SandboxWorker()
        for stream in (1, 2):
            with self.subTest(stream=stream), self.assertRaisesRegex(WorkerUnavailable, "capacity"):
                worker.run_bounded(input_dir.root, work_dir.root,
                    ["/usr/bin/python3.12", "-I", "-B", "-c", "import os; os.write(%d, b'x' * 300000)" % stream], output_bytes=1024)
        with self.assertRaises(WorkerTimeout):
            worker.run_bounded(input_dir.root, work_dir.root,
                ["/usr/bin/python3.12", "-I", "-B", "-c", "import time; time.sleep(10)"], timeout=0.2)
        cancelled = threading.Event()
        timer = threading.Timer(0.2, cancelled.set)
        timer.start()
        try:
            with self.assertRaises(WorkerCancelled):
                worker.run_bounded(input_dir.root, work_dir.root,
                    ["/usr/bin/python3.12", "-I", "-B", "-c", "import time; time.sleep(10)"], cancel_event=cancelled)
        finally:
            timer.cancel()
            timer.join()
        with self.assertRaises(WorkerUnavailable):
            SandboxWorker("/nonexistent/context-bwrap").run_bounded(input_dir.root, work_dir.root, ["/usr/bin/true"])

    def test_real_sandbox_descendant_does_not_outlive_timeout_cleanup(self):
        root = PrivateDirectory(self.base / "descendant-sandbox", create=True)
        input_dir = root.child("input", create=True)
        work_dir = root.child("work", create=True)
        code = "\n".join([
            "import os, time", "from pathlib import Path", "pid = os.fork()", "if pid == 0:",
            "    time.sleep(1)", "    Path('/work/escaped').write_text('must not survive')", "    os._exit(0)",
            "time.sleep(20)"])
        with self.assertRaises(WorkerTimeout):
            SandboxWorker().run_bounded(input_dir.root, work_dir.root,
                ["/usr/bin/python3.12", "-I", "-B", "-c", code], timeout=0.3)
        time.sleep(1.1)
        self.assertFalse((work_dir.root / "escaped").exists())

    def test_probe_reauthorizes_after_dispatch_revoke(self):
        verifier, _, _ = self.freeze_view()
        claims = self.workspace.authority.authenticate(verifier["token"], action="context.view.probe")
        original = SandboxWorker.run_bounded
        def run(worker, *args, **kwargs):
            result = original(worker, *args, **kwargs)
            self.workspace.authority.revoke(claims["token_id"])
            return result
        with mock.patch.object(SandboxWorker, "run_bounded", new=run), self.assertRaises(AuthorizationError):
            self.workspace.probe(verifier["token"], run_id=verifier["run_id"])


if __name__ == "__main__":
    unittest.main()
