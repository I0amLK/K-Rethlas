"""Native CLI trust contract and bounded owned-process adapter.

Codex owns authentication/configuration. This is not a pure-text or upstream
HTTP/token isolation contract. Event streams are discarded, not interpreted as
proof that tools did not run. Only the final-message file becomes research data.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

from .codex_binding import OBSERVED_EFFORTS
from .codex_provider import VERIFIED_VERSION, _schema
from .private_fs import PrivateDirectory, canonical_json, digest
from .provider import (ProviderConfigurationError, ProviderIndeterminate,
                       ProviderResponseError, ProviderTimeout, _cancelled, _messages, _model)


LIMITS = {"attempts": 2, "purpose_attempts": 1, "stdin_bytes": 16384,
          "final_bytes": 16384, "stream_bytes": 1048576, "seconds": 300,
          "term_seconds": 2, "kill_seconds": 3, "cleanup_seconds": 5,
          "minimum_ttl_seconds": 315}
# Retained official source: protocol-002/downloads/codex-rs_{exec,core,config}_*.
# Exec itself uses Never; do not guess the top-level -a flag placement.
OVERRIDES = ("features.multi_agent=false", "agents.enabled=false", "features.multi_agent_v2.enabled=false",
             "agents.max_concurrent_threads_per_session=1", "agents.max_depth=0",
             'approvals_reviewer="user"')
TRUST = {"pure_text": False, "configuration": "inherited_unresolved",
         "native_tools_mcp_hooks": "inherited; disable known extra-model extensions before approval",
         "upstream_http_token_cost_cap": False, "extra_sources_excluded": False,
         "input_display": "complete Rethlas stdin, not complete upstream HTTP",
         "remote_cancellation_guaranteed": False,
         "cwd_policy": "fresh empty owned cwd per purpose; user configuration still inherited"}


def native_profile(value):
    if (type(value) is not dict or value.get("type") != "codex_cli_native"
            or not {"type"} <= value.keys() <= {"type", "model", "reasoning_effort"}):
        raise ProviderConfigurationError("invalid native CLI selection")
    if "model" in value:
        if _model(value["model"]).lower() in {"auto", "default", "unknown"}:
            raise ProviderConfigurationError("omit native model to inherit")
    if "reasoning_effort" in value and value["reasoning_effort"] not in OBSERVED_EFFORTS:
        raise ProviderConfigurationError("invalid native effort selection")
    return deepcopy(value)


def _installation():
    entry = shutil.which("codex")
    if entry is None:
        raise ProviderConfigurationError("official Codex CLI is not installed")
    path = Path(entry).resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 536870912:
        raise ProviderConfigurationError("invalid Codex entry artifact")
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1048576):
            sha.update(chunk)
    from .codex_provider import STATIC_CAPABILITY_SOURCE
    if sha.hexdigest() != STATIC_CAPABILITY_SOURCE["binary_sha256"]:
        raise ProviderConfigurationError("Codex artifact differs from the retained compatibility installation")
    return {"entry": str(path), "entry_sha256": sha.hexdigest(),
            "compatibility_version": VERIFIED_VERSION,
            "compatibility": "retained installation hash; native exec compatibility check pending"}


def validate_binding(value):
    if (type(value) is not dict or set(value) != {"schema", "version", "provider_type", "mode", "selection",
            "resolved", "resolution_status", "installation", "trust", "limits", "overrides", "configuration_sha256"}
            or value["schema"] != "rethlas.codex-native-binding" or type(value["version"]) is not int
            or value["version"] != 1 or value["provider_type"] != "codex_cli_native" or value["mode"] != "live"
            or value["resolved"] != {"model": None, "reasoning_effort": None, "provider": None, "destination": None}
            or value["resolution_status"] != "inherited_unresolved" or value["trust"] != TRUST
            or value["limits"] != LIMITS or value["overrides"] != list(OVERRIDES)):
        raise ProviderConfigurationError("invalid native CLI binding")
    if type(value["selection"]) is not dict or not set(value["selection"]) <= {"model", "reasoning_effort"}:
        raise ProviderConfigurationError("invalid native CLI binding selection")
    native_profile({"type": "codex_cli_native", **value["selection"]})
    install = value["installation"]
    if (type(install) is not dict or set(install) != {"entry", "entry_sha256", "compatibility_version", "compatibility"}
            or type(install["entry"]) is not str or not Path(install["entry"]).is_absolute()
            or install["compatibility_version"] != VERIFIED_VERSION
            or type(install["entry_sha256"]) is not str or len(install["entry_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in install["entry_sha256"])):
        raise ProviderConfigurationError("invalid native CLI installation binding")
    if value["configuration_sha256"] != digest({k: v for k, v in value.items() if k != "configuration_sha256"}):
        raise ProviderConfigurationError("native CLI binding digest differs")
    return deepcopy(value)


def _final(path, *, directory=None):
    """Never read a seventeenth KiB, even to detect a growing file."""
    with directory._fd() if directory is not None else nullcontext(None) as parent:
        return _final_at(path, parent)


def _final_at(path, parent):
    fd = os.open(path.name if parent is not None else path,
                 os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid()
                or not 0 < before.st_size <= LIMITS["final_bytes"]):
            raise ProviderResponseError("unsafe or oversized native final file")
        raw = bytearray()
        while len(raw) < before.st_size:
            chunk = os.read(fd, before.st_size - len(raw))
            if not chunk:
                raise ProviderResponseError("short native final file")
            raw.extend(chunk)
        after = os.fstat(fd)
        named = os.stat(path.name if parent is not None else path, dir_fd=parent, follow_symlinks=False)
        facts = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
        if facts(before) != facts(after) or facts(after) != facts(named):
            raise ProviderResponseError("native final file changed")
        try:
            return bytes(raw).decode("utf-8")
        except UnicodeError:
            raise ProviderResponseError("native final file is not UTF-8") from None
    finally:
        os.close(fd)


def _reap(process):
    # A descendant can outlive its leader. Always signal the owned group and
    # allow the full TERM grace before KILL, even if the leader already exited.
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        grace = time.monotonic() + 2
        while time.monotonic() < grace:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(min(.02, max(0, grace - time.monotonic())))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            raise ProviderIndeterminate("native owned process cleanup failed") from None
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _run(argv, stdin, *, cwd, cancel_event=None, deadline=None):
    """Fixed transport. Raw events/errors never leave this function."""
    process = None
    deadline = min(time.monotonic() + LIMITS["seconds"], deadline) if deadline is not None else time.monotonic() + LIMITS["seconds"]
    received, sent = 0, 0
    try:
        _cancelled(cancel_event, request_started=True)
        process = subprocess.Popen(argv, cwd=cwd, env=os.environ.copy(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        with selectors.DefaultSelector() as selector:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_WRITE if stream is process.stdin else selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                _cancelled(cancel_event, request_started=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderTimeout("native stage deadline exceeded; no resend")
                for key, _ in selector.select(min(remaining, .05)):
                    if key.fileobj is process.stdin:
                        try:
                            sent += os.write(key.fd, stdin[sent:])
                        except BrokenPipeError:
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                            continue
                        if sent == len(stdin):
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                        continue
                    budget = LIMITS["stream_bytes"] - received
                    if budget <= 0:
                        raise ProviderResponseError("native stream byte limit reached")
                    chunk = os.read(key.fd, min(65536, budget))
                    received += len(chunk)
                    if received == LIMITS["stream_bytes"]:
                        raise ProviderResponseError("native stream byte limit reached")
                    if not chunk:
                        selector.unregister(key.fileobj)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderTimeout("native stage deadline exceeded; no resend")
            if process.wait(timeout=remaining) != 0 or sent != len(stdin):
                raise ProviderIndeterminate("native process exited unsuccessfully; no resend")
        return {"stream_bytes": received, "exit_code": 0}
    except subprocess.TimeoutExpired:
        raise ProviderTimeout("native stage deadline exceeded; no resend") from None
    except (ProviderIndeterminate, ProviderResponseError):
        raise
    except OSError:
        raise ProviderIndeterminate("native process dispatch failed; no resend") from None
    finally:
        if process is not None:
            _reap(process)


def _cleanup_tree(root):
    """A separate owned helper bounds potentially large native disk cleanup."""
    process = None
    deadline = time.monotonic() + LIMITS["cleanup_seconds"]
    try:
        process = subprocess.Popen([sys.executable, "-I", "-B", "-c",
            "import shutil,sys; shutil.rmtree(sys.argv[1])", str(root)],
            cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        if process.wait(timeout=max(.01, deadline - time.monotonic() - .25)) != 0:
            raise ProviderIndeterminate("native owned cleanup failed")
    except subprocess.TimeoutExpired:
        raise ProviderIndeterminate("native owned cleanup deadline exceeded") from None
    except OSError:
        raise ProviderIndeterminate("native owned cleanup unavailable") from None
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise ProviderIndeterminate("native cleanup helper did not exit") from None


def _argv(binding, directory):
    argv = [binding["installation"]["entry"], "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "-C", str(directory / "work"), "--output-schema",
        str(directory / "response.schema.json"), "--json", "-o", str(directory / "final.json")]
    for setting in OVERRIDES:
        argv.extend(["-c", setting])
    if "model" in binding["selection"]:
        argv.extend(["-m", binding["selection"]["model"]])
    if "reasoning_effort" in binding["selection"]:
        argv.extend(["-c", "model_reasoning_effort=" + json.dumps(binding["selection"]["reasoning_effort"])])
    return [*argv, "-"]


class CodexNativeProvider:
    provider_type, mode, model, endpoint = "codex_cli_native", "live", None, None

    def __init__(self, *, selection):
        self.selection = native_profile(selection)

    def binding(self):
        value = {"schema": "rethlas.codex-native-binding", "version": 1,
            "provider_type": self.provider_type, "mode": "live",
            "selection": {k: v for k, v in self.selection.items() if k != "type"},
            "resolved": {"model": None, "reasoning_effort": None, "provider": None, "destination": None},
            "resolution_status": "inherited_unresolved", "installation": _installation(),
            "trust": deepcopy(TRUST), "limits": deepcopy(LIMITS), "overrides": list(OVERRIDES)}
        value["configuration_sha256"] = digest(value)
        return validate_binding(value)

    def prepare(self, messages, *, purpose, service_id, request_key, run_id):
        if purpose not in ("generation", "review"):
            raise ProviderConfigurationError("unknown native purpose")
        stdin = canonical_json(_messages(messages))
        binding = self.binding()
        directory = Path(tempfile.mkdtemp(prefix="rethlas-native-", dir="/tmp"))
        try:
            owned = PrivateDirectory(directory)
            work = owned.child("work", create=True)
            # Metadata/schema live beside the cwd, not inside its initially empty tree.
            owned.write_bytes("owner.json", canonical_json({"schema": "rethlas.native-owned-directory", "version": 1,
                "service_id": service_id, "request_key": request_key, "run_id": run_id, "purpose": purpose,
                "cwd_device": work._identity[0], "cwd_inode": work._identity[1]}), exclusive=True)
            schema = canonical_json(_schema(purpose))
            owned.write_bytes("response.schema.json", schema, exclusive=True)
        except Exception:
            _cleanup_tree(directory)
            raise ProviderConfigurationError("native stage preparation failed") from None
        argv = _argv(binding, directory)
        return {"schema": "rethlas.codex-native-execution", "version": 1, "service_id": service_id,
            "request_key": request_key, "run_id": run_id, "purpose": purpose, "provider": binding,
            "stdin": stdin.decode(), "stdin_sha256": hashlib.sha256(stdin).hexdigest(),
            "output_schema": _schema(purpose), "schema_sha256": hashlib.sha256(schema).hexdigest(),
            "cwd": str(directory / "work"), "argv": argv, "limits": deepcopy(LIMITS)}

    def validate_execution(self, envelope):
        try:
            return self._validate_execution(envelope)
        except ProviderConfigurationError:
            raise
        except Exception:
            raise ProviderConfigurationError("native owned input or directory changed; reapproval required") from None

    def _validate_execution(self, envelope):
        fields = {"schema", "version", "service_id", "request_key", "run_id", "purpose", "provider", "stdin",
            "stdin_sha256", "output_schema", "schema_sha256", "cwd", "argv", "limits"}
        if (type(envelope) is not dict or set(envelope) != fields
                or envelope["schema"] != "rethlas.codex-native-execution" or type(envelope["version"]) is not int
                or envelope["version"] != 1 or envelope["purpose"] not in ("generation", "review")
                or envelope["limits"] != LIMITS or type(envelope["stdin"]) is not str):
            raise ProviderConfigurationError("invalid native execution envelope")
        binding = validate_binding(envelope["provider"])
        if self.binding() != binding:
            raise ProviderConfigurationError("native installation or selection changed")
        from .private_fs import strict_json_loads
        try:
            stdin = canonical_json(_messages(strict_json_loads(envelope["stdin"])))
        except Exception:
            raise ProviderConfigurationError("invalid native retained stdin") from None
        schema = canonical_json(_schema(envelope["purpose"]))
        if (stdin.decode() != envelope["stdin"] or hashlib.sha256(stdin).hexdigest() != envelope["stdin_sha256"]
                or envelope["output_schema"] != _schema(envelope["purpose"])
                or envelope["schema_sha256"] != hashlib.sha256(schema).hexdigest()):
            raise ProviderConfigurationError("native exact input or schema differs")
        root = Path(envelope["cwd"]).parent
        if root.parent != Path("/tmp") or not root.name.startswith("rethlas-native-") or Path(envelope["cwd"]).name != "work":
            raise ProviderConfigurationError("native cwd ownership differs")
        owned = PrivateDirectory(root)
        work = owned.child("work")
        expected_owner = {"schema": "rethlas.native-owned-directory", "version": 1,
            **{k: envelope[k] for k in ("service_id", "request_key", "run_id", "purpose")},
            "cwd_device": work._identity[0], "cwd_inode": work._identity[1]}
        if (owned.read_bytes("owner.json", max_bytes=4096) != canonical_json(expected_owner)
                or owned.read_bytes("response.schema.json", max_bytes=16384) != schema
                or list(work.entries(max_entries=1)) or envelope["argv"] != _argv(binding, root)
                or (root / "final.json").exists() or (root / "final.json").is_symlink()):
            raise ProviderConfigurationError("native cwd, argv or schema changed; new exact approval required")
        return deepcopy(envelope)

    def execute(self, envelope, *, cancel_event=None):
        # The trusted pipeline supplies an envelope built here and approved in its
        # journal. No caller-configurable executable, transport or mock mode.
        deadline = time.monotonic() + LIMITS["seconds"]
        envelope = self.validate_execution(envelope)
        root = Path(envelope["cwd"]).parent
        owned = PrivateDirectory(root)
        try:
            facts = _run(envelope["argv"], envelope["stdin"].encode(), cwd=envelope["cwd"], cancel_event=cancel_event, deadline=deadline)
            content = _final(root / "final.json", directory=owned)
            if time.monotonic() >= deadline:
                raise ProviderTimeout("native end-to-end stage deadline exceeded; no resend")
        except (ProviderIndeterminate, ProviderResponseError, ProviderConfigurationError):
            raise
        except Exception:
            raise ProviderIndeterminate("native response unavailable; no resend") from None
        return {"content": content, "model": None, "mode": "live", "provider_type": self.provider_type,
            "requested_model": self.selection.get("model"), "requested_reasoning_effort": self.selection.get("reasoning_effort"),
            "resolved_model": None, "resolved_reasoning_effort": None, "reported_model": None,
            "reported_reasoning_effort": None, "observed": {}, "usage": {},
            "response_sha256": hashlib.sha256(content.encode()).hexdigest(), "execution_sha256": digest(envelope), **facts}

    @staticmethod
    def cleanup(envelope):
        root = Path(envelope["cwd"]).parent
        # Only this adapter's unique, retained /tmp tree, never a project/user cwd.
        if root.parent != Path("/tmp") or not root.name.startswith("rethlas-native-"):
            raise ProviderConfigurationError("native cleanup ownership differs")
        if root.exists():
            from .private_fs import strict_json_loads
            owned = PrivateDirectory(root)
            owner = strict_json_loads(owned.read_bytes("owner.json", max_bytes=4096))
            if any(owner.get(k) != envelope[k] for k in ("service_id", "request_key", "run_id", "purpose")):
                raise ProviderConfigurationError("native cleanup marker differs")
            _cleanup_tree(root)
