"""Fixed stdlib-only sandbox probe; it observes input, never verifies mathematics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main():
    root = Path("/input")
    names = sorted(p.name for p in root.iterdir())
    if names != ["context.json", "driver.py"] or list(Path("/work").iterdir()):
        raise ValueError("unexpected fixture input or nonempty work directory")
    raw = (root / "context.json").read_bytes()
    if len(raw) > 8388608:
        raise ValueError("fixture context is oversized")
    dto = json.loads(raw.decode("utf-8"))
    revisions = []
    for item in dto["revisions"]:
        body = item["body"].encode("utf-8")
        actual = hashlib.sha256(body).hexdigest()
        if actual != item["body_sha256"]:
            raise ValueError("fixture received incorrect body bytes")
        revisions.append({"entity_id": item["entity_id"], "revision_id": item["revision_id"],
                          "type": item["type"], "body_sha256": actual, "body_bytes": len(body)})
    readonly = False
    try:
        (root / "forbidden-write").write_bytes(b"fixture")
    except OSError:
        readonly = True
    if not readonly:
        raise ValueError("fixture input is writable")
    result = {"version": "context-observation-1", "input_files": names,
              "input_sha256": hashlib.sha256(raw).hexdigest(),
              "driver_sha256": hashlib.sha256((root / "driver.py").read_bytes()).hexdigest(),
              "manifest_sha256": dto["manifest_sha256"], "revisions": revisions,
              "relations": len(dto["relations"]), "empty_work": True, "input_readonly": True,
              "mathematical_review": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
