"""Fixed PDF text worker; copied into a no-network, selected-input sandbox."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys


MAXIMUM_TEXT_BYTES = 1048576


def main():
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    # Only the pinned installed package is mounted here, not its site-packages
    # directory, the repository, HOME, or any other selected-source directory.
    sys.path.insert(0, "/runtime")
    from pypdf import PdfReader, __version__

    try:
        request = json.loads(Path("/input/request.json").read_text(encoding="utf-8"))
        reader = PdfReader("/input/source.pdf", strict=True)
        if reader.is_encrypted:
            raise ValueError("encrypted_pdf")
        count = len(reader.pages)
        if not 0 < count <= 200:
            raise ValueError("page_limit")
        wanted = list(range(1, count + 1)) if request["pages"] is None else request["pages"]
        if not wanted or any(page > count for page in wanted):
            raise ValueError("invalid_page_range")
        total, pages, warnings = 0, [], []
        for number in wanted:
            text = reader.pages[number - 1].extract_text() or ""
            total += len(text.encode("utf-8"))
            if total > MAXIMUM_TEXT_BYTES:
                raise ValueError("extracted_text_limit")
            if not text.strip():
                warnings.append("page_without_text:" + str(number))
            pages.append({"page": number, "text": text})
        if not total or not any(item["text"].strip() for item in pages):
            raise ValueError("no_extractable_text")
        result = {"version": "source-pdf-1", "extractor": {"name": "pypdf", "version": __version__},
                  "total_pages": count, "pages": pages, "warnings": warnings}
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAXIMUM_TEXT_BYTES:
            raise ValueError("extracted_output_limit")
    except ValueError as exc:
        known = {"encrypted_pdf", "page_limit", "invalid_page_range", "extracted_text_limit",
                 "extracted_output_limit", "no_extractable_text"}
        error = str(exc) if str(exc) in known else "invalid_pdf"
        encoded = json.dumps({"error": error}, separators=(",", ":")).encode("ascii")
    except Exception:
        encoded = b'{"error":"invalid_pdf"}'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
