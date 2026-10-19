"""Synthetic/local source imports; no public network or model requests."""
from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from rethlas_core.contracts import validate_math_memory
from rethlas_core.private_fs import canonical_json
from rethlas_core.source_import import (
    SourceCancelled, SourceError, SourceUnavailable, _fetch_url, fetch_url,
    pdf_available, preview_file, preview_url, reference_records, reference_refs,
    safe_display, verify_file,
)
from rethlas_core.source_pdf import _run, extract_pdf
from rethlas_core.source_url import _https_bytes, _public, _verified_addresses, parse_url
from rethlas_core.worker import SandboxWorker


COMMAND_RECORDS = []


def make_pdf(path, pages=("First synthetic page", "Second synthetic page"), *, encrypt=False):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        content = DecodedStreamObject()
        content.set_data(("BT /F1 12 Tf 20 200 Td (" + text + ") Tj ET").encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    if encrypt:
        writer.encrypt("synthetic-password")
    writer.write(path)


class SourceFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rethlas-source-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "参考.md"
        self.path.write_bytes("# 假设\r\nx > 0。\n".encode("utf-8"))

    def test_exact_utf8_path_hidden_and_candidate_reference(self):
        value = preview_file(self.path)
        self.assertEqual(value["segments"][0]["text"], "# 假设\r\nx > 0。\n")
        self.assertNotIn(str(self.root), json.dumps(value, ensure_ascii=False))
        verify_file(self.path, value)
        records = reference_records(value)
        self.assertEqual(records["relations"], [])
        self.assertEqual(records["entities"][0]["type"], "Reference")
        self.assertEqual(records["revisions"][0]["assertion_state"], "candidate")
        self.assertEqual(records["revisions"][0]["validity"], "needs_review")
        self.assertTrue(validate_math_memory({"schema_version": "1.0", **records})["valid"])
        self.assertEqual(reference_refs(value)[0]["body_sha256"], records["revisions"][0]["body_sha256"])

    def test_same_name_different_source_not_merged(self):
        other = self.root / "other"
        other.mkdir()
        path = other / self.path.name
        path.write_bytes(self.path.read_bytes())
        self.assertNotEqual(preview_file(path)["source_id"], preview_file(self.path)["source_id"])

    def test_repreview_stable_identity_and_existing_graph_idempotence(self):
        before = preview_file(self.path)
        first = reference_records(before)
        after = preview_file(self.path)
        self.assertEqual(reference_refs(before), reference_refs(after))
        self.assertEqual(reference_records(after, existing_graph={"schema_version": "1.0", **first}), first)
        changed = deepcopy(first)
        changed["revisions"][0]["body"] += "changed"
        with self.assertRaises(SourceError):
            reference_records(after, existing_graph=changed)

    def test_actual_service_import_retry_and_repreview_do_not_mutate_history(self):
        from rethlas_core.project_service import ProjectService
        service = ProjectService.initialize(self.root / "project", owner="source-test")
        writer = service.issue(role="project_writer")
        before = preview_file(self.path)
        records = reference_records(before)
        request = {"version": 1, "request_id": "import-selected", "expected_head": service.config["genesis"]["head"], **records}
        first = service.commit(writer["token"], request, run_id=writer["run_id"])
        self.assertTrue(service.commit(writer["token"], request, run_id=writer["run_id"])["replayed"])
        with service.snapshot(writer["token"], run_id=writer["run_id"]) as (state, graph, _withdrawals, _claims):
            old_head = state["accepted_graph"]["head"]
            old_records = deepcopy(graph)
            selected = preview_file(self.path)
            self.assertEqual(reference_refs(before), reference_refs(selected))
            self.assertEqual(reference_records(selected, existing_graph=graph), records)
            self.assertTrue(all(ref["revision_id"] in {item["revision_id"] for item in graph["revisions"]}
                                for ref in reference_refs(selected)))
        # Caller sees every exact selected ref already exists and skips a new
        # commit. The source helper does not forge a new expected_head or request.
        with service.snapshot(writer["token"], run_id=writer["run_id"]) as (state, graph, _withdrawals, _claims):
            self.assertEqual(state["accepted_graph"]["head"], old_head)
            self.assertEqual(graph, old_records)
        self.assertEqual(first["head"], old_head)

    def test_changed_original_refuses_reuse(self):
        value = preview_file(self.path)
        self.path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(SourceError, "changed after preview"):
            verify_file(self.path, value)
        after = preview_file(self.path)
        self.assertNotEqual(value["source_id"], after["source_id"])

    def test_no_symlink_ancestor_leaf_hardlink_or_special_file(self):
        alias = self.root / "alias.md"
        alias.symlink_to(self.path)
        with self.assertRaises(SourceError):
            preview_file(alias)
        alias.unlink()
        alias.hardlink_to(self.path)
        with self.assertRaises(SourceError):
            preview_file(alias)
        alias.unlink()
        directory_alias = self.root / "linked"
        target = self.root / "target"
        target.mkdir()
        (target / "text.md").write_text("synthetic")
        directory_alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(SourceError):
            preview_file(directory_alias / "text.md")
        fifo = self.root / "fifo.txt"
        os.mkfifo(fifo)
        with self.assertRaises(SourceError):
            preview_file(fifo)

    def test_invalid_utf8_empty_extension_size_and_page_selection(self):
        for raw in (b"\xff", b"   \n", b"x" * (2097152 + 1)):
            with self.subTest(raw=len(raw)):
                self.path.write_bytes(raw)
                with self.assertRaises(SourceError):
                    preview_file(self.path)
        with self.assertRaises(SourceError):
            preview_file(self.path.with_suffix(".py"))
        with self.assertRaises(SourceError):
            preview_file(self.path, pages=[1])
        for pages in ([True], [0], [201], [], [2, 1], [1, 1], "1"):
            with self.subTest(pages=pages):
                with self.assertRaises(SourceError):
                    preview_file(self.path, pages=pages)

    def test_multibyte_segments_preserve_bytes_and_select_explicit_subset(self):
        text = "定理—" * 1000
        self.path.write_text(text, encoding="utf-8")
        value = preview_file(self.path)
        self.assertEqual("".join(segment["text"] for segment in value["segments"]), text)
        self.assertTrue(all(len(segment["text"].encode("utf-8")) <= 2048 for segment in value["segments"]))
        wanted = [value["segments"][1]["segment_id"]]
        self.assertEqual(len(reference_refs(value, segment_ids=wanted)), 1)
        self.assertEqual(len(reference_records(value, segment_ids=wanted)["revisions"]), 1)
        for selected in ([], ["missing"], [wanted[0], wanted[0]]):
            with self.assertRaises(SourceError):
                reference_refs(value, segment_ids=selected)

    def test_whitespace_segments_are_retained_exactly(self):
        text = "start" + " " * 5000 + "end"
        self.path.write_text(text, encoding="utf-8")
        value = preview_file(self.path)
        self.assertEqual("".join(item["text"] for item in value["segments"]), text)
        self.assertEqual(len(reference_records(value)["revisions"]), len(value["segments"]))

    def test_terminal_controls_escaped_but_original_kept(self):
        raw = "proof\x1b[2J\x07\x9b31m‮end\n\t"
        self.path.write_text(raw, encoding="utf-8")
        value = preview_file(self.path)
        self.assertEqual(value["segments"][0]["text"], raw)
        rendered = safe_display(raw)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)
        self.assertNotIn("\x9b", rendered)
        self.assertNotIn("‮", rendered)
        self.assertIn("\\u001b", rendered)
        self.assertTrue(rendered.endswith("\n\t"))

    def test_cancelled_before_read(self):
        event = threading.Event()
        event.set()
        with patch("rethlas_core.source_import.read_input", side_effect=AssertionError("must not read")):
            with self.assertRaises(SourceCancelled):
                preview_file(self.path, cancel_event=event)

    def test_segment_tampering_refused(self):
        value = preview_file(self.path)
        value["segments"][0]["text"] += "changed"
        with self.assertRaises(SourceError):
            reference_records(value)


class SourcePDFTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rethlas-source-pdf-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "synthetic.pdf"
        make_pdf(self.path)

    def test_actual_sandbox_extracts_pages_and_subset(self):
        value = preview_file(self.path)
        self.assertEqual(value["extractor"], {"name": "pypdf", "version": "6.19.0"})
        self.assertEqual(value["selection"], {"pages": [1, 2], "total_pages": 2})
        self.assertTrue(value["complete"])
        self.assertEqual([item["text"] for item in value["segments"]], ["First synthetic page", "Second synthetic page"])
        subset = preview_file(self.path, pages=[2])
        verify_file(self.path, subset)
        self.assertFalse(subset["complete"])
        self.assertEqual(subset["segments"][0]["text"], "Second synthetic page")
        self.assertTrue(subset["segments"][0]["locator"].startswith("page=2:"))
        self.assertEqual(reference_refs(value, segment_ids=[value["segments"][1]["segment_id"]]), reference_refs(subset))

    def test_encrypted_corrupt_empty_and_page_range_rejected(self):
        make_pdf(self.path, encrypt=True)
        with self.assertRaisesRegex(SourceError, "encrypted_pdf"):
            preview_file(self.path)
        self.path.write_bytes(b"%PDF-1.7\nnot a pdf\n")
        with self.assertRaisesRegex(SourceError, "invalid_pdf"):
            preview_file(self.path)
        make_pdf(self.path, pages=("",))
        with self.assertRaisesRegex(SourceError, "no_extractable_text"):
            preview_file(self.path)
        make_pdf(self.path)
        with self.assertRaisesRegex(SourceError, "invalid_page_range"):
            preview_file(self.path, pages=[3])

    def test_pdf_input_limit_and_total_page_limit(self):
        with self.assertRaises(SourceError):
            extract_pdf(b"%PDF-1.7\n" + b"x" * 20971520)
        make_pdf(self.path, pages=("x",) * 201)
        with self.assertRaisesRegex(SourceError, "page_limit"):
            preview_file(self.path)

    def test_pdf_subset_selection_tamper_rejected(self):
        value = preview_file(self.path, pages=[2])
        for changed in ({"pages": [1], "total_pages": 2}, {"pages": [2], "total_pages": True},
                        {"pages": [2, 2], "total_pages": 2}):
            with self.subTest(changed=changed):
                bad = deepcopy(value)
                bad["selection"] = changed
                with self.assertRaises(SourceError):
                    verify_file(self.path, bad)

    def test_pdf_worker_output_limit_is_enforced_while_running(self):
        source, work = self.root / "inputs", self.root / "work"
        source.mkdir()
        work.mkdir()
        command = SandboxWorker().command(source, work,
            ["/usr/bin/python3.12", "-I", "-c", "import sys; sys.stdout.buffer.write(b'x'*1048577)"])
        with self.assertRaisesRegex(SourceError, "output limit"):
            _run(command)

    def test_missing_isolation_never_falls_back(self):
        with patch("rethlas_core.source_pdf._runtime", side_effect=SourceUnavailable("synthetic unavailable")):
            self.assertEqual(pdf_available()["status"], "unavailable")
            with self.assertRaises(SourceUnavailable):
                preview_file(self.path)

    def test_actual_worker_mounts_only_selected_inputs_and_blocks_network(self):
        outside = self.root / "not-selected.md"
        outside.write_text("synthetic private material")
        source, work = self.root / "inputs", self.root / "work"
        source.mkdir()
        work.mkdir()
        (source / "selected.pdf").write_bytes(self.path.read_bytes())
        code = """import json, socket
from pathlib import Path
result = {'selected': Path('/input/selected.pdf').read_bytes().startswith(b'%PDF-'),
          'other_denied': not Path(OTHER).exists(), 'home_denied': not Path('/home/lk').exists()}
for name, address in [('loopback', ('127.0.0.1', 9)), ('external', ('192.0.2.1', 9))]:
    client = socket.socket()
    client.settimeout(0.2)
    try:
        client.connect(address)
        result[name + '_denied'] = False
    except OSError:
        result[name + '_denied'] = True
    finally:
        client.close()
print(json.dumps(result))
""".replace("OTHER", repr(str(outside)))
        command = SandboxWorker().command(source, work, ["/usr/bin/python3.12", "-I", "-c", code])
        result = json.loads(_run(command))
        self.assertTrue(all(result.values()), result)

    def test_running_pdf_cancel_reaps_owned_group_and_closes_streams(self):
        source, work = self.root / "inputs", self.root / "work"
        source.mkdir()
        work.mkdir()
        command = SandboxWorker().command(source, work,
            ["/usr/bin/python3.12", "-I", "-c", "import time; time.sleep(60)"])
        event, started = threading.Event(), []
        real_popen = subprocess.Popen
        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            event.set()
            return process
        with patch("rethlas_core.source_pdf.subprocess.Popen", side_effect=launch):
            with self.assertRaises(SourceCancelled):
                _run(command, cancel_event=event)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)
        self.assertTrue(started[0].stdout.closed)
        self.assertTrue(started[0].stderr.closed)

    def test_pdf_address_space_limit_is_applied_before_parser(self):
        source, work = self.root / "inputs", self.root / "work"
        source.mkdir()
        work.mkdir()
        code = "import resource; resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912)); " \
               "print(resource.getrlimit(resource.RLIMIT_AS)); bytearray(600*1024*1024)"
        command = SandboxWorker().command(source, work, ["/usr/bin/python3.12", "-I", "-c", code])
        with self.assertRaises(SourceUnavailable):
            _run(command)

    def test_timeout_and_cancel_kill_owned_process_group(self):
        source, work = self.root / "inputs", self.root / "work"
        source.mkdir()
        work.mkdir()
        command = SandboxWorker().command(source, work, ["/usr/bin/python3.12", "-I", "-c", "import time; time.sleep(60)"])
        started = []
        real_popen = subprocess.Popen
        def record(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process
        with patch("rethlas_core.source_pdf.subprocess.Popen", side_effect=record):
            with self.assertRaisesRegex(SourceError, "deadline"):
                _run(command, timeout=0.1)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)
        self.assertTrue(started[0].stdout.closed)
        self.assertTrue(started[0].stderr.closed)
        event = threading.Event()
        event.set()
        with patch("rethlas_core.source_pdf.subprocess.Popen", side_effect=AssertionError("must not start")):
            with self.assertRaises(SourceCancelled):
                _run(command, cancel_event=event)


class SourceURLTests(unittest.TestCase):
    def test_preview_never_dns_fetches_and_fetch_requires_explicit_approval(self):
        with patch("rethlas_core.source_import._https_bytes", side_effect=AssertionError("no fetch")), \
             patch("rethlas_core.source_url._resolve", side_effect=AssertionError("no DNS")):
            value = preview_url("https://papers.example.org/example.pdf")
            self.assertFalse(value["network_performed"])
            for approved in (False, None, 1, "yes"):
                with self.assertRaises(SourceError):
                    fetch_url("https://papers.example.org/example.pdf", approved=approved)

    def test_unsafe_url_rejected_before_dns(self):
        urls = ["http://papers.example.org/a", "file:///etc/passwd", "https://localhost/a",
                "https://127.0.0.1/a", "https://10.1.1.1/a", "https://169.254.169.254/a",
                "https://[::1]/a", "https://[fe80::1]/a", "https://[::ffff:127.0.0.1]/a",
                "https://[64:ff9b::7f00:1]/a", "https://user:pass@papers.example.org/a",
                "https://papers.example.org:444/a", "https://papers.example.org/a#secret",
                "https://papers.example.org\\evil/a", "https://papers.example.org/\r\nX:y"]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(SourceError):
                    preview_url(url)

    def test_dns_mixed_public_private_and_special_addresses_rejected(self):
        good = {"family": socket.AF_INET, "address": "93.184.216.34"}
        self.assertEqual(_verified_addresses([good]), [good])
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "192.0.2.1", "224.0.0.1", "::1", "fc00::1", "fe80::1", "2001:db8::1", "::ffff:93.184.216.34", "2002:5db8:d822::"):
            with self.subTest(address=address):
                family = socket.AF_INET6 if ":" in address else socket.AF_INET
                with self.assertRaises(SourceError):
                    _verified_addresses([good, {"family": family, "address": address}])

    def test_dns_credential_free_owned_worker_and_deadline_cleanup(self):
        from rethlas_core.source_url import _resolve
        processes, captured = [], []
        real_popen = subprocess.Popen
        class FinishedResolver:
            returncode = 0
            stdin = stdout = stderr = None
            pid = -1
            def communicate(self, _pending, timeout):
                return b'[{"family":2,"address":"93.184.216.34"}]', None
            def wait(self, timeout=None): return 0
        def finished(command, **kwargs):
            captured.append((command, kwargs))
            return FinishedResolver()
        with patch("rethlas_core.source_url.subprocess.Popen", side_effect=finished), \
             patch("rethlas_core.source_url.terminate_group"):
            value = _resolve("papers.example.org", 443, deadline=time.monotonic() + 1)
        self.assertEqual(value, [{"family": 2, "address": "93.184.216.34"}])
        self.assertEqual(captured[0][1]["env"], {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        self.assertIn("-I", captured[0][0])
        def stalled(_command, **kwargs):
            process = real_popen(["/usr/bin/python3.12", "-I", "-c", "import time; time.sleep(60)"], **kwargs)
            processes.append(process)
            return process
        with patch("rethlas_core.source_url.subprocess.Popen", side_effect=stalled):
            with self.assertRaisesRegex(SourceError, "DNS deadline"):
                _resolve("papers.example.org", 443, deadline=time.monotonic() + 0.1)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)

    def test_rebinding_uses_verified_ip_without_second_lookup(self):
        parsed = parse_url("https://papers.example.org/paper")
        seen = []
        class DummySocket:
            def settimeout(self, value): pass
            def connect(self, value): seen.append(value); raise OSError("stop before TLS")
            def close(self): pass
        with patch("rethlas_core.source_url._resolve", return_value=[{"family": socket.AF_INET, "address": "93.184.216.34"}]) as resolve, \
             patch("rethlas_core.source_url.socket.socket", return_value=DummySocket()):
            with self.assertRaises(SourceError):
                _https_bytes(parsed)
        self.assertEqual(seen, [("93.184.216.34", 443)])
        self.assertEqual(resolve.call_count, 1)

    def test_static_html_source_is_data_and_drops_active_content(self):
        html = b"<h1>Lemma</h1><script>secret instruction</script><style>secret style</style><template>secret template</template><p>x &gt; 0</p><div hidden>secret hidden</div><span hidden/><p>visible tail</p>"
        value = _fetch_url("https://papers.example.org/paper", transport=lambda *_args, **_kwargs: {"mime": "text/html", "data": html})
        text = "".join(item["text"] for item in value["segments"])
        self.assertIn("Lemma", text)
        self.assertIn("x > 0", text)
        self.assertIn("visible tail", text)
        self.assertNotIn("secret", text)
        self.assertEqual(value["kind"], "html")
        self.assertEqual(reference_records(value)["entities"][0]["type"], "Reference")

    def test_transport_response_limits_and_types(self):
        for result in ({"mime": "image/png", "data": b"x"}, {"mime": "text/plain", "data": b""},
                       {"mime": "text/plain", "data": b"x" * (2097152 + 1)},
                       {"mime": "text/plain", "data": "not bytes"}):
            with self.subTest(mime=result["mime"], size=len(result["data"])):
                with self.assertRaises(SourceError):
                    _fetch_url("https://papers.example.org/paper", transport=lambda *_a, **_k: result)


class SourceHTTPSLoopbackTests(unittest.TestCase):
    """Use only a generated CA and an owned loopback server via trusted patches."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rethlas-source-https-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.certificate, self.key = self.root / "cert.pem", self.root / "key.pem"
        command = ["/usr/bin/openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                   "-keyout", str(self.key), "-out", str(self.certificate), "-subj", "/CN=papers.example.org",
                   "-addext", "subjectAltName=DNS:papers.example.org"]
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.request_paths = []
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.request_paths.append(self.path)
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "https://papers.example.org/other")
                    self.end_headers()
                    return
                if self.path == "/rate":
                    self.send_response(429)
                    self.end_headers()
                    return
                if self.path == "/slow":
                    time.sleep(0.2)
                if self.path == "/large":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", "2097153")
                    self.end_headers()
                    return
                if self.path == "/compressed":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Encoding", "gzip")
                    self.end_headers()
                    return
                if self.path == "/pdf":
                    make_pdf(owner.root / "url.pdf", pages=("HTTPS synthetic PDF page",))
                    payload = (owner.root / "url.pdf").read_bytes()
                    mime = "application/pdf"
                else:
                    payload = b"<p>Synthetic local lemma.</p>"
                    mime = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
                    # A deliberately timed-out client closes its own TLS socket.
                    pass
            def log_message(self, *_args): pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate, self.key)
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup_server)
        self.context = ssl.create_default_context(cafile=str(self.certificate))

    def cleanup_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def get(self, path):
        parsed = parse_url("https://papers.example.org" + path)
        # This is a test-only in-process trusted connection boundary. Production
        # parse_url/_verified_addresses and persisted config have no private flag.
        parsed["port"] = self.server.server_port
        with patch("rethlas_core.source_url._resolve", return_value=[{"family": socket.AF_INET, "address": "127.0.0.1"}]), \
             patch("rethlas_core.source_url._verified_addresses", side_effect=lambda value: value), \
             patch("rethlas_core.source_url.ssl.create_default_context", return_value=self.context):
            return _https_bytes(parsed)

    def test_actual_https_html_and_no_redirect_or_rate_retry(self):
        response = self.get("/paper")
        self.assertEqual(response["mime"], "text/html")
        self.assertIn(b"Synthetic local lemma", response["data"])
        for path in ("/redirect", "/rate"):
            with self.assertRaises(SourceError):
                self.get(path)
        self.assertEqual(self.request_paths, ["/paper", "/redirect", "/rate"])
        COMMAND_RECORDS.append({"kind": "owned_loopback_https", "paths": self.request_paths,
                                "redirect_followed": False, "retry": False, "cleanup": "registered"})

    def test_actual_https_pdf_reaches_isolated_extractor(self):
        response = self.get("/pdf")
        self.assertEqual(response["mime"], "application/pdf")
        value = _fetch_url("https://papers.example.org/pdf", transport=lambda *_a, **_k: response)
        self.assertEqual(value["segments"][0]["text"], "HTTPS synthetic PDF page")
        self.assertEqual(value["locator"], "https://papers.example.org/pdf")
        self.assertTrue(value["segments"][0]["locator"].startswith("page=1:"))

    def test_actual_https_size_encoding_and_slow_deadline_rejected(self):
        for path in ("/large", "/compressed"):
            with self.assertRaises(SourceError):
                self.get(path)
        started = time.monotonic()
        with patch("rethlas_core.source_url.FETCH_SECONDS", 0.05):
            with self.assertRaises(SourceError):
                self.get("/slow")
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(self.request_paths, ["/large", "/compressed", "/slow"])


if __name__ == "__main__":
    unittest.main()
