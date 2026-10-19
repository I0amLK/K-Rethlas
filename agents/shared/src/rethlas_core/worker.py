"""Offline worker isolation and bounded trusted-fixture process execution."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time


class WorkerUnavailable(RuntimeError):
    pass


class WorkerTimeout(TimeoutError):
    pass


class WorkerCancelled(RuntimeError):
    pass


def terminate_group(process: subprocess.Popen) -> None:
    # The group may outlive its leader; do not skip cleanup after poll().
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    finally:
        # Cancellation can precede communicate(), which normally closes pipes.
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (BrokenPipeError, ConnectionResetError):
                    pass


def run_fixture_process(payload: bytes, *, timeout: float = 5,
                        cancel_event: threading.Event | None = None) -> bytes:
    """Run only our installed, fixed synthetic driver, never a requested command.

    This is a trusted protocol fixture, NOT a fallback for an isolated worker.
    Its environment has no provider credentials, HOME, proxies, or model config.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise WorkerCancelled("fixture request cancelled")
    command = [sys.executable, "-I", "-m", "rethlas_core.fixture_driver"]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, cwd="/",
    )
    try:
        deadline, pending = time.monotonic() + timeout, payload
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise WorkerCancelled("fixture request cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout("fixture driver deadline exceeded")
            try:
                output, _ = process.communicate(pending, timeout=min(remaining, 0.05))
                break
            except subprocess.TimeoutExpired:
                # communicate() preserves its buffered state across bounded waits.
                pending = None
        if cancel_event is not None and cancel_event.is_set():
            raise WorkerCancelled("fixture request cancelled")
        if process.returncode != 0 or len(output) > 262144:
            raise ValueError("fixture driver failed")
        return output
    finally:
        terminate_group(process)


class SandboxWorker:
    """Linux bubblewrap, no network, no host home/repository/private mounts."""

    def __init__(self, executable: str = "/usr/bin/bwrap"):
        self.executable = executable

    def command(self, input_dir: Path, work_dir: Path, command: list[str]) -> list[str]:
        for path in (input_dir, work_dir):
            if (not path.is_absolute() or any(parent.is_symlink() for parent in (path, *path.parents))
                    or not path.is_dir()):
                raise ValueError("sandbox mounts must be real absolute directories without symlink ancestors")
        if input_dir == work_dir or input_dir in work_dir.parents or work_dir in input_dir.parents:
            raise ValueError("sandbox input and work directories must be separate")
        args = [
            self.executable, "--die-with-parent", "--new-session", "--unshare-all",
            "--cap-drop", "ALL", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "HOME", "/nonexistent",
            "--ro-bind", "/usr", "/usr",
        ]
        for name in ("bin", "lib", "lib64"):
            path = Path("/") / name
            if path.is_symlink():
                args.extend(["--symlink", os.readlink(path), str(path)])
            elif path.is_dir():
                args.extend(["--ro-bind", str(path), str(path)])
        args.extend([
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(input_dir), "/input", "--bind", str(work_dir), "/work",
            "--chdir", "/work", "--", *command,
        ])
        return args

    def run(self, input_dir: Path, work_dir: Path, command: list[str], *,
            stdin: bytes = b"", timeout: float = 5) -> subprocess.CompletedProcess:
        try:
            process = subprocess.Popen(
                self.command(input_dir, work_dir, command), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/",
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                start_new_session=True,
            )
        except OSError as exc:
            raise WorkerUnavailable("bubblewrap could not start") from exc
        try:
            stdout, stderr = process.communicate(stdin, timeout=timeout)
            if process.returncode != 0:
                # Do not include potentially sensitive command arguments or outputs.
                raise WorkerUnavailable("bubblewrap worker failed; inspect target-host probe")
            return subprocess.CompletedProcess(command, 0, stdout, stderr)
        except subprocess.TimeoutExpired as exc:
            raise WorkerTimeout("sandbox worker deadline exceeded") from exc
        finally:
            terminate_group(process)

    def run_bounded(self, input_dir: Path, work_dir: Path, command: list[str], *,
                    timeout: float = 5, output_bytes: int = 262144,
                    cancel_event: threading.Event | None = None) -> subprocess.CompletedProcess:
        """Fixed-controller callers only; enforce each output bound while reading.

        No stdin: approved context is an immutable file on the read-only mount.
        This does not accept a command supplied by a worker/request API.
        """
        if type(timeout) not in (int, float) or not 0 < timeout <= 30:
            raise ValueError("sandbox deadline must be finite and at most 30 seconds")
        if type(output_bytes) is not int or not 0 < output_bytes <= 262144:
            raise ValueError("sandbox output limit must be a bounded strict integer")
        if cancel_event is not None and cancel_event.is_set():
            raise WorkerCancelled("sandbox request cancelled")
        try:
            process = subprocess.Popen(
                self.command(input_dir, work_dir, command), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/",
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, start_new_session=True,
            )
        except OSError as exc:
            raise WorkerUnavailable("bubblewrap could not start") from exc
        outputs = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                for name in outputs:
                    stream = getattr(process, name)
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                while selector.get_map() or process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        raise WorkerCancelled("sandbox request cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WorkerTimeout("sandbox worker deadline exceeded")
                    for key, _ in selector.select(min(remaining, 0.05)):
                        chunk = os.read(key.fileobj.fileno(), min(65536, output_bytes - len(outputs[key.data]) + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            outputs[key.data].extend(chunk)
                            if len(outputs[key.data]) > output_bytes:
                                raise WorkerUnavailable("sandbox output capacity reached")
            if cancel_event is not None and cancel_event.is_set():
                raise WorkerCancelled("sandbox request cancelled")
            if process.returncode != 0:
                raise WorkerUnavailable("bubblewrap worker failed; inspect target-host probe")
            return subprocess.CompletedProcess(command, 0, bytes(outputs["stdout"]), bytes(outputs["stderr"]))
        finally:
            terminate_group(process)

    def probe(self) -> dict:
        with tempfile.TemporaryDirectory(prefix="rethlas-probe-") as directory:
            root = Path(directory)
            source, work = root / "input", root / "work"
            source.mkdir()
            work.mkdir()
            (source / "sentinel").write_text("approved", encoding="utf-8")
            try:
                result = self.run(source, work, ["/usr/bin/python3.12", "-I", "-c",
                    "from pathlib import Path; "
                    "assert Path('/input/sentinel').read_text() == 'approved'; "
                    "Path('/work/output').write_text('ok')"], timeout=5)
                if result.returncode or (work / "output").read_text() != "ok":
                    raise WorkerUnavailable("sandbox probe did not produce expected output")
            except (WorkerUnavailable, WorkerTimeout, OSError) as exc:
                return {"status": "blocked", "backend": "bubblewrap", "reason": str(exc)}
        return {"status": "available", "backend": "bubblewrap", "network": "disabled"}
