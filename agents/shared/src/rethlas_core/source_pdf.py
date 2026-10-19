"""Selected-byte PDF extraction with a fixed no-network bubblewrap runtime."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

from .memory_io import read_input
from .private_fs import canonical_json, strict_json_loads
from .worker import SandboxWorker, terminate_group


MAXIMUM_PDF_BYTES = 20971520
MAXIMUM_OUTPUT_BYTES = 1048576
PINNED_PYPDF = "6.19.0"


class SourceError(ValueError):
    """A selected source cannot be imported within its declared bounds."""


class SourceUnavailable(RuntimeError):
    """A required source dependency or isolation boundary is unavailable."""


class SourceCancelled(SourceError):
    pass


def _cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise SourceCancelled("source operation cancelled")


def validate_pages(pages):
    if pages is None:
        return None
    if (type(pages) is not list or not 1 <= len(pages) <= 200
            or any(type(page) is not int or not 1 <= page <= 200 for page in pages)
            or pages != sorted(set(pages))):
        raise SourceError("pages must be increasing unique integers between 1 and 200")
    return list(pages)


def pdf_available():
    """Inspect dependencies only; actual extraction also verifies sandbox startup."""
    try:
        _runtime()
        return {"status": "available", "backend": "bubblewrap", "extractor": "pypdf",
                "version": PINNED_PYPDF, "network": "disabled", "runtime_probe": "not_run"}
    except SourceUnavailable:
        return {"status": "unavailable", "backend": "bubblewrap", "reason": "PDF dependency or isolation unavailable"}


def _runtime():
    if not Path("/usr/bin/bwrap").is_file() or not Path("/usr/bin/python3.12").is_file():
        raise SourceUnavailable("PDF requires bubblewrap and Python 3.12")
    try:
        spec = importlib.util.find_spec("pypdf")
        if spec is None or spec.submodule_search_locations is None:
            raise SourceUnavailable("pinned pypdf is not installed")
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1 or importlib.metadata.version("pypdf") != PINNED_PYPDF:
            raise SourceUnavailable("PDF requires the pinned pypdf version")
        package = Path(locations[0]).resolve(strict=True)
        if not package.is_dir() or package.name != "pypdf":
            raise SourceUnavailable("PDF runtime package is unavailable")
        return package
    except (ImportError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        raise SourceUnavailable("pinned PDF runtime unavailable") from exc


def _run(command, *, cancel_event=None, timeout=30):
    _cancelled(cancel_event)
    try:
        process = subprocess.Popen(command, cwd="/", stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    except OSError as exc:
        raise SourceUnavailable("PDF isolation could not start") from exc
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": MAXIMUM_OUTPUT_BYTES, "stderr": 65536}
    try:
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            for name in buffers:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map() or process.poll() is None:
                _cancelled(cancel_event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SourceError("PDF extraction deadline exceeded")
                for key, _ in selector.select(min(remaining, 0.05)):
                    name = key.data
                    chunk = os.read(key.fileobj.fileno(), min(65536, limits[name] - len(buffers[name]) + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        buffers[name].extend(chunk)
                        if len(buffers[name]) > limits[name]:
                            raise SourceError("PDF worker output limit exceeded")
        _cancelled(cancel_event)
        if process.returncode != 0:
            raise SourceUnavailable("PDF isolation or constrained worker failed")
        return bytes(buffers["stdout"])
    finally:
        terminate_group(process)


def extract_pdf(data, *, pages=None, cancel_event=None):
    """Extract only selected PDF bytes; no fallback to an unsandboxed parser."""
    if type(data) is not bytes or not 0 < len(data) <= MAXIMUM_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise SourceError("selected PDF must be a bounded PDF document")
    pages = validate_pages(pages)
    _cancelled(cancel_event)
    package = _runtime()
    worker_source = read_input(Path(__file__).with_name("source_pdf_worker.py"), 65536,
                               single_link=True, stable=True)
    with tempfile.TemporaryDirectory(prefix="rethlas-source-pdf-") as temporary:
        root = Path(temporary)
        source, work = root / "input", root / "work"
        source.mkdir(mode=0o700)
        work.mkdir(mode=0o700)
        for name, raw in (("source.pdf", data), ("worker.py", worker_source),
                          ("request.json", canonical_json({"pages": pages}))):
            path = source / name
            with path.open("xb") as output:
                path.chmod(0o600)
                output.write(raw)
        command = SandboxWorker().command(source, work,
            ["/usr/bin/python3.12", "-I", "-B", "/input/worker.py"])
        split = command.index("--")
        command[split:split] = ["--dir", "/runtime", "--ro-bind", str(package), "/runtime/pypdf"]
        raw = _run(command, cancel_event=cancel_event)
    try:
        result = strict_json_loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise SourceError("PDF worker returned invalid structured data") from exc
    if type(result) is dict and set(result) == {"error"}:
        errors = {"encrypted_pdf", "page_limit", "invalid_page_range", "extracted_text_limit",
                  "extracted_output_limit", "no_extractable_text", "invalid_pdf"}
        if result["error"] not in errors:
            raise SourceError("PDF worker returned an unknown rejection")
        raise SourceError("PDF extraction rejected: " + result["error"])
    if (type(result) is not dict or set(result) != {"version", "extractor", "total_pages", "pages", "warnings"}
            or result["version"] != "source-pdf-1" or result["extractor"] != {"name": "pypdf", "version": PINNED_PYPDF}
            or type(result["total_pages"]) is not int or not 1 <= result["total_pages"] <= 200
            or type(result["pages"]) is not list or type(result["warnings"]) is not list
            or any(type(item) is not str or not item.startswith("page_without_text:") for item in result["warnings"])):
        raise SourceError("PDF worker returned invalid metadata")
    expected = list(range(1, result["total_pages"] + 1)) if pages is None else pages
    if (len(result["pages"]) != len(expected)
            or any(type(item) is not dict or set(item) != {"page", "text"}
                   or type(item["page"]) is not int or item["page"] != number or type(item["text"]) is not str
                   for item, number in zip(result["pages"], expected))):
        raise SourceError("PDF worker page selection differs")
    return result
