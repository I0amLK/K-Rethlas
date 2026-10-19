"""Target-host isolation tests; a skipped backend is not target-host acceptance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from rethlas_core.private_fs import canonical_json
from rethlas_core.worker import SandboxWorker, WorkerCancelled, WorkerTimeout, run_fixture_process


class SandboxTests(unittest.TestCase):
    def test_target_host_files_network_and_descendants(self):
        worker = SandboxWorker()
        probe = worker.probe()
        if probe["status"] != "available":
            self.skipTest("BLOCKED target-host backend: " + probe["reason"])
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            root = Path(temporary)
            inputs, work = root / "inputs", root / "work"
            inputs.mkdir()
            work.mkdir()
            (root / "private-key").write_text("synthetic secret")
            (root / "other-problem.md").write_text("other synthetic problem")
            (root / "publish").mkdir()
            (inputs / "approved.md").write_text("approved input")
            (work / "escape").symlink_to(root / "private-key")
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            code = '''import json, socket, subprocess
from pathlib import Path
root = Path(ROOT)
result = {}
result['approved_read'] = Path('/input/approved.md').read_text() == 'approved input'
Path('/work/output.md').write_text('allowed')
result['allowed_write'] = True
for name, target in [('private',root/'private-key'),('other_problem',root/'other-problem.md'),('symlink',Path('/work/escape'))]:
    try:
        target.read_bytes()
        result[name+'_denied'] = False
    except OSError:
        result[name+'_denied'] = True
for name,target in [('publish',root/'publish/out.md'),('input',Path('/input/unapproved.md')),('code',Path('/usr/rethlas-write-probe'))]:
    try:
        target.write_text('forbidden')
        result[name+'_write_denied'] = False
    except OSError:
        result[name+'_write_denied'] = True
for name,address in [('loopback',('127.0.0.1',PORT)),('external',('192.0.2.1',9))]:
    with socket.socket() as client:
        client.settimeout(0.2)
        try:
            client.connect(address)
            result[name+'_denied'] = False
        except OSError:
            result[name+'_denied'] = True
child = subprocess.run(['/usr/bin/python3.12','-I','-c','from pathlib import Path; assert not Path('+repr(str(root/'private-key'))+').exists()'],capture_output=True)
result['child_confined'] = child.returncode == 0
print(json.dumps(result))
'''.replace("ROOT", repr(str(root))).replace("PORT", str(port))
            result = worker.run(inputs, work, ["/usr/bin/python3.12", "-I", "-c", code])
            observations = json.loads(result.stdout)
            self.assertTrue(all(observations.values()), observations)
            self.assertEqual((work / "output.md").read_text(), "allowed")
            self.assertEqual(list((root / "publish").iterdir()), [])

    def test_cancelled_fixture_never_starts(self):
        cancelled = threading.Event()
        cancelled.set()
        with patch("rethlas_core.worker.subprocess.Popen") as process:
            with self.assertRaises(WorkerCancelled):
                run_fixture_process(b"{}", cancel_event=cancelled)
            process.assert_not_called()

    def test_cancellation_kills_active_process(self):
        cancelled = threading.Event()
        started = []
        real_popen = subprocess.Popen

        def record(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            cancelled.set()
            return process

        with patch("rethlas_core.worker.subprocess.Popen", side_effect=record):
            with self.assertRaises(WorkerCancelled):
                run_fixture_process(canonical_json({"statement": "synthetic", "proof": "[fixture:timeout]"}),
                                    timeout=5, cancel_event=cancelled)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)
        self.assertTrue(started[0].stdin.closed)
        self.assertTrue(started[0].stdout.closed)

    def test_fixture_deadline_before_communicate_closes_pipes(self):
        started = []
        real_popen = subprocess.Popen

        def record(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process

        with patch("rethlas_core.worker.subprocess.Popen", side_effect=record):
            with self.assertRaises(WorkerTimeout):
                run_fixture_process(b"{}", timeout=0)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)
        self.assertTrue(started[0].stdin.closed)
        self.assertTrue(started[0].stdout.closed)

    def test_sandbox_timeout_closes_all_pipes(self):
        worker = SandboxWorker()
        probe = worker.probe()
        if probe["status"] != "available":
            self.skipTest("BLOCKED target-host backend: " + probe["reason"])
        started = []
        real_popen = subprocess.Popen

        def record(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process

        with tempfile.TemporaryDirectory() as temporary:
            inputs, work = Path(temporary) / "inputs", Path(temporary) / "work"
            inputs.mkdir()
            work.mkdir()
            with patch("rethlas_core.worker.subprocess.Popen", side_effect=record):
                with self.assertRaises(WorkerTimeout):
                    worker.run(inputs, work, ["/usr/bin/python3.12", "-I", "-c",
                                             "import time; time.sleep(60)"], timeout=0.1)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)
        for stream in (started[0].stdin, started[0].stdout, started[0].stderr):
            self.assertTrue(stream.closed)

    def test_timeout_kills_driver_and_child_group(self):
        real_popen = subprocess.Popen
        started = []

        def record(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process

        with patch("rethlas_core.worker.subprocess.Popen", side_effect=record):
            with self.assertRaises(WorkerTimeout):
                run_fixture_process(canonical_json({"statement": "synthetic", "proof": "[fixture:timeout]"}), timeout=0.4)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)
        self.assertTrue(started[0].stdin.closed)
        self.assertTrue(started[0].stdout.closed)
        # The group may temporarily contain an orphan zombie; it must have no
        # executing descendants. Check Linux proc state rather than PID reuse.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "stat").read_text()
                fields = status[status.rfind(")") + 2:].split()
                if int(fields[2]) == started[0].pid:
                    self.assertEqual(fields[0], "Z", "fixture descendant still running")
            except (OSError, ValueError):
                continue


if __name__ == "__main__":
    unittest.main()
