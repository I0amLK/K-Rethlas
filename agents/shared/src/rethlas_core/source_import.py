"""Bounded previews of explicitly selected reference material.

Previews are data, not authority. Only reference_records builds candidate Reference
records; callers commit them through the existing authenticated project service.
No function scans directories, reads authentication, or sends material to a model.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import unicodedata

from .context_views import exact_ref
from .memory_io import read_input
from .private_fs import canonical_json, digest
from .research_records import candidate
from .source_pdf import (MAXIMUM_PDF_BYTES, SourceCancelled, SourceError,
                         SourceUnavailable, _cancelled, extract_pdf, pdf_available,
                         validate_pages)
from .source_url import PDF_BYTES, TEXT_BYTES, _https_bytes, parse_url


MAXIMUM_TEXT_BYTES = 2097152
SEGMENT_BYTES = 2048
MAXIMUM_SEGMENTS = 2048
_PREVIEW_FIELDS = {"version", "source_id", "title", "kind", "source_sha256", "locator",
                   "extractor", "captured_at", "segments", "warnings", "complete", "selection"}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def safe_display(text):
    """Escape terminal controls only; the caller must also disable rich markup."""
    if type(text) is not str:
        raise SourceError("display text must be a string")
    output = []
    for char in text:
        if char in "\n\t":
            output.append(char)
        elif (unicodedata.category(char) in ("Cc", "Cs")
              or unicodedata.bidirectional(char) in ("LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI")):
            output.append("\\u" + format(ord(char), "04x"))
        else:
            output.append(char)
    return "".join(output)


def _text(data):
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise SourceError("source text must be valid UTF-8") from exc
    if not text.strip():
        raise SourceError("source contains no nonempty text")
    return text


def _chunks(text):
    raw = text.encode("utf-8")
    start = 0
    while start < len(raw):
        end = min(start + SEGMENT_BYTES, len(raw))
        while end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        part = raw[start:end].decode("utf-8")
        yield start, end, part
        start = end


def _segments(source_id, texts):
    result = []
    for locator, text in texts:
        for start, end, part in _chunks(text):
            location = locator + ":bytes=" + str(start) + "-" + str(end)
            sha = _sha(part.encode("utf-8"))
            identity = digest({"source_id": source_id, "locator": location, "text_sha256": sha})
            result.append({"segment_id": "segment." + identity, "locator": location,
                           "text": part, "text_sha256": sha})
            if len(result) > MAXIMUM_SEGMENTS:
                raise SourceError("source segment capacity reached")
    if not result:
        raise SourceError("source contains no selected nonempty text")
    return result


def _preview(data, *, identity, title, kind, locator, pages=None, cancel_event=None):
    _cancelled(cancel_event)
    sha = _sha(data)
    source_id = "source." + digest({"identity": identity, "source_sha256": sha})
    warnings, selection = [], {"pages": None, "total_pages": None}
    if kind == "pdf":
        extracted = extract_pdf(data, pages=pages, cancel_event=cancel_event)
        extractor = extracted["extractor"]
        selection = {"pages": [item["page"] for item in extracted["pages"]],
                     "total_pages": extracted["total_pages"]}
        texts = [("page=" + str(item["page"]), item["text"]) for item in extracted["pages"]]
        warnings = ["PDF text extraction may omit or reorder mathematical notation; compare selected pages with the original.",
                    *extracted["warnings"]]
        complete = len(extracted["pages"]) == extracted["total_pages"]
    else:
        if pages is not None:
            raise SourceError("page selection requires a PDF source")
        if len(data) > MAXIMUM_TEXT_BYTES:
            raise SourceError("source text size limit exceeded")
        text = _text(data)
        if kind == "html":
            parser = _StaticHTML()
            try:
                parser.feed(text)
                parser.close()
            except (ValueError, RecursionError) as exc:
                raise SourceError("static HTML could not be extracted") from exc
            text = "".join(parser.parts)
            if not text.strip():
                raise SourceError("static HTML contains no usable text; download a readable document")
            extractor = {"name": "static-html", "version": "1"}
            warnings = ["Static HTML text only; scripts, styles and active content are not executed. Layout and CSS visibility are not evaluated."]
        else:
            extractor = {"name": "exact-utf8", "version": "1"}
        texts, complete = [("text", text)], True
    _cancelled(cancel_event)
    return {"version": "source-preview-1", "source_id": source_id, "title": title, "kind": kind,
            "source_sha256": sha, "locator": locator, "extractor": extractor,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "segments": _segments(source_id, texts), "warnings": warnings,
            "complete": complete, "selection": selection}


def _file_identity(path):
    # Only this digest is kept in the preview; a private absolute path is never
    # embedded in a graph body, model input, URL, exception or evidence locator.
    return digest({"kind": "local-file", "path": str(path.absolute())})


def preview_file(path, *, pages=None, cancel_event=None):
    """Read one selected MD/TXT/PDF, preserving its exact byte hash."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".md", ".txt", ".pdf"):
        raise SourceError("select a Markdown, TXT or PDF file")
    pages = validate_pages(pages)
    _cancelled(cancel_event)
    try:
        data = read_input(path, MAXIMUM_PDF_BYTES if suffix == ".pdf" else MAXIMUM_TEXT_BYTES,
                          single_link=True, stable=True)
    except (OSError, ValueError) as exc:
        raise SourceError("selected source must be a stable, bounded, single-link regular file without symlinks") from exc
    title = path.name
    if len(title.encode("utf-8")) > 255:
        raise SourceError("selected source name is too long")
    return _preview(data, identity=_file_identity(path), title=title,
                    kind="pdf" if suffix == ".pdf" else "markdown" if suffix == ".md" else "text",
                    locator="local:" + title + "#sha256=" + _sha(data), pages=pages, cancel_event=cancel_event)


def verify_file(path, preview):
    """Recheck selected original bytes and selection binding before use."""
    _validate_preview(preview)
    path = Path(path)
    try:
        data = read_input(path, MAXIMUM_PDF_BYTES if preview["kind"] == "pdf" else MAXIMUM_TEXT_BYTES,
                          single_link=True, stable=True)
    except (OSError, ValueError) as exc:
        raise SourceError("selected source is no longer a stable regular file") from exc
    expected = "source." + digest({"identity": _file_identity(path), "source_sha256": _sha(data)})
    if _sha(data) != preview["source_sha256"] or expected != preview["source_id"]:
        raise SourceError("selected source changed after preview; review it again")
    if preview["kind"] == "pdf":
        pages = validate_pages(preview["selection"]["pages"])
        if pages is None or any(page > preview["selection"]["total_pages"] for page in pages):
            raise SourceError("PDF selected pages differ from the preview")
    return None


def preview_url(url):
    """Syntactic/literal-IP preflight only: no DNS, HTTP or authentication."""
    value = parse_url(url)
    return {"version": "source-url-preview-1", "url": value["url"], "host": value["host"],
            "requires_fetch_approval": True, "network_performed": False,
            "dns_public_check": "at_fetch", "automatic_redirect": False, "automatic_retry": False}


def fetch_url(url, *, approved=False, pages=None, cancel_event=None):
    """Fetch only after a person's explicit URL approval, never on URL entry."""
    if approved is not True:
        raise SourceError("explicit URL fetch approval is required")
    return _fetch_url(url, pages=pages, cancel_event=cancel_event)


def _fetch_url(url, *, pages=None, cancel_event=None, transport=None):
    # This private dependency-injection seam is for trusted local transport
    # tests. Neither persisted settings nor UI/MCP input can supply transport.
    parsed = parse_url(url)
    pages = validate_pages(pages)
    _cancelled(cancel_event)
    result = (transport or _https_bytes)(parsed, cancel_event=cancel_event)
    if (type(result) is not dict or set(result) != {"mime", "data"}
            or result["mime"] not in {"text/plain", "text/markdown", "text/html", "application/pdf"}
            or type(result["data"]) is not bytes):
        raise SourceError("source transport returned invalid structured data")
    kind = {"text/plain": "text", "text/markdown": "markdown", "text/html": "html", "application/pdf": "pdf"}[result["mime"]]
    maximum = PDF_BYTES if kind == "pdf" else TEXT_BYTES
    if not 0 < len(result["data"]) <= maximum:
        raise SourceError("source response size limit exceeded")
    title = parsed["host"] + ":" + urlsplit_path_title(parsed["target"])
    return _preview(result["data"], identity=digest({"kind": "https", "url": url}),
                    title=title[:255], kind=kind, locator=url, pages=pages, cancel_event=cancel_event)


def urlsplit_path_title(target):
    # A small display label only; full approved URL remains the source locator.
    return target.split("?", 1)[0].rsplit("/", 1)[-1] or "/"


class _StaticHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.ignored = [], []

    def handle_starttag(self, tag, attrs):
        if self.ignored:
            if tag == self.ignored[-1]:
                self.ignored.append(tag)
            return
        attributes = dict(attrs)
        hidden = "hidden" in attributes or attributes.get("aria-hidden", "").lower() == "true"
        if tag in ("script", "style", "template", "noscript", "iframe", "object", "svg") or hidden:
            self.ignored.append(tag)
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "pre", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.ignored:
            if tag == self.ignored[-1]:
                self.ignored.pop()
            return
        if tag in ("p", "div", "li", "tr", "h1", "h2", "h3", "pre", "section", "article"):
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def _validate_preview(preview):
    if (type(preview) is not dict or set(preview) != _PREVIEW_FIELDS or preview["version"] != "source-preview-1"
            or type(preview["source_id"]) is not str or not preview["source_id"].startswith("source.")
            or len(preview["source_id"]) != 71
            or type(preview["title"]) is not str or not preview["title"].strip()
            or preview["kind"] not in ("pdf", "markdown", "text", "html")
            or type(preview["source_sha256"]) is not str or len(preview["source_sha256"]) != 64
            or type(preview["locator"]) is not str or not preview["locator"]
            or type(preview["extractor"]) is not dict or set(preview["extractor"]) != {"name", "version"}
            or any(type(item) is not str or not item for item in preview["extractor"].values())
            or type(preview["captured_at"]) is not str or type(preview["complete"]) is not bool
            or type(preview["selection"]) is not dict or set(preview["selection"]) != {"pages", "total_pages"}
            or type(preview["warnings"]) is not list or any(type(item) is not str for item in preview["warnings"])
            or type(preview["segments"]) is not list or not 1 <= len(preview["segments"]) <= MAXIMUM_SEGMENTS):
        raise SourceError("invalid source preview")
    try:
        when = datetime.fromisoformat(preview["captured_at"])
        if (when.tzinfo is None or any(char not in "0123456789abcdef" for char in preview["source_sha256"])
                or any(char not in "0123456789abcdef" for char in preview["source_id"][7:])):
            raise ValueError("invalid binding")
    except (ValueError, TypeError) as exc:
        raise SourceError("invalid source preview metadata") from exc
    selection = preview["selection"]
    if preview["kind"] == "pdf":
        pages = validate_pages(selection["pages"])
        total = selection["total_pages"]
        if (pages is None or type(total) is not int or not 1 <= total <= 200
                or any(page > total for page in pages) or preview["complete"] != (len(pages) == total)):
            raise SourceError("invalid selected PDF page metadata")
    elif selection != {"pages": None, "total_pages": None} or not preview["complete"]:
        raise SourceError("non-PDF source page metadata differs")
    ids, total_bytes = [], 0
    for segment in preview["segments"]:
        if (type(segment) is not dict or set(segment) != {"segment_id", "locator", "text", "text_sha256"}
                or type(segment["locator"]) is not str or type(segment["text"]) is not str
                or not segment["text"] or len(segment["text"].encode("utf-8")) > SEGMENT_BYTES
                or segment["text_sha256"] != _sha(segment["text"].encode("utf-8"))
                or segment["segment_id"] != "segment." + digest({"source_id": preview["source_id"],
                    "locator": segment["locator"], "text_sha256": segment["text_sha256"]})):
            raise SourceError("source preview segment binding differs")
        if preview["kind"] == "pdf":
            prefix = segment["locator"].split(":", 1)[0]
            if prefix not in {"page=" + str(page) for page in pages}:
                raise SourceError("PDF segment is not part of the selected pages")
        total_bytes += len(segment["text"].encode("utf-8"))
        ids.append(segment["segment_id"])
    if total_bytes > (1048576 if preview["kind"] == "pdf" else MAXIMUM_TEXT_BYTES):
        raise SourceError("source preview text capacity reached")
    if len(ids) != len(set(ids)):
        raise SourceError("source preview has duplicate segments")


def _selected(preview, segment_ids):
    _validate_preview(preview)
    if segment_ids is None:
        wanted = [segment["segment_id"] for segment in preview["segments"]]
    else:
        if (type(segment_ids) is not list or not segment_ids or any(type(item) is not str for item in segment_ids)
                or len(segment_ids) != len(set(segment_ids))):
            raise SourceError("selected segment IDs must be a nonempty unique list")
        wanted = segment_ids
    by_id = {segment["segment_id"]: segment for segment in preview["segments"]}
    if not set(wanted) <= set(by_id):
        raise SourceError("selected segment is not in this preview")
    return [by_id[identifier] for identifier in wanted]


def _records(preview, segment):
    # The same exact source/page segment always gets the same identity. Capture
    # time belongs to provenance, not body; an existing graph is reused below.
    identity = segment["segment_id"].split(".", 1)[1]
    metadata = {"version": "source-reference-1", "source_id": preview["source_id"], "title": preview["title"],
                "kind": preview["kind"], "source_sha256": preview["source_sha256"], "locator": preview["locator"],
                "extractor": preview["extractor"], "segment": {key: segment[key] for key in ("segment_id", "locator", "text_sha256")}}
    body = "# Reference source\n\n" + canonical_json(metadata).decode("utf-8") + "\n\n## Selected text\n\n" + segment["text"]
    origin = {"source": "source-import:" + preview["source_id"], "actor": "trusted-host",
              "recorded_at": preview["captured_at"]}
    evidence_id = "source-evidence." + identity
    entity, revision = candidate("reference." + identity, "reference." + identity + ".0", "Reference",
        preview["title"] + " [" + segment["locator"] + "]", body, evidence_ids=[evidence_id], origin=origin)
    evidence = {"evidence_id": evidence_id, "locator": "source:" + preview["source_id"] + ":" + segment["locator"],
                "sha256": preview["source_sha256"], "provenance": origin}
    return entity, revision, evidence


def reference_records(preview, *, segment_ids=None, existing_graph=None):
    """Build candidate citations; no automatic mathematical-premise relations.

    existing_graph is an authenticated snapshot supplied by the controller. Exact
    previously imported records are reused; same identity with changed bytes is
    rejected. Passing the same snapshot before _append makes retries idempotent.
    """
    selected = _selected(preview, segment_ids)
    collections = {key: [] for key in ("entities", "revisions", "relations", "evidence")}
    existing = {}
    if existing_graph is not None:
        if type(existing_graph) is not dict:
            raise SourceError("existing source graph must be an authenticated graph snapshot")
        for collection, field in (("entities", "entity_id"), ("revisions", "revision_id"), ("evidence", "evidence_id")):
            if type(existing_graph.get(collection)) is not list:
                raise SourceError("existing source graph is incomplete")
            existing[collection] = {item[field]: item for item in existing_graph[collection]}
    for segment in selected:
        entity, revision, evidence = _records(preview, segment)
        for collection, field, item in (("entities", "entity_id", entity), ("revisions", "revision_id", revision),
                                         ("evidence", "evidence_id", evidence)):
            previous = existing.get(collection, {}).get(item[field])
            if previous is not None:
                if ({key: value for key, value in previous.items() if key != "provenance"}
                        != {key: value for key, value in item.items() if key != "provenance"}):
                    raise SourceError("an immutable imported source identity differs")
                item = previous
            collections[collection].append(deepcopy(item))
    return collections


def reference_refs(preview, *, segment_ids=None):
    """Exact selected Reference refs; generated from source/segment identities."""
    return [exact_ref(_records(preview, segment)[1]) for segment in _selected(preview, segment_ids)]
