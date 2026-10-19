"""Fixed stdlib-only input observation, never mathematical or current-use review."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat


_FILES = ["context.json", "driver.py", "overlay.json", "use.json"]
_DTO_FIELDS = {"version", "policy_hash", "roots", "revisions", "relations"}
_REVISION_FIELDS = {"entity_id", "revision_id", "body_sha256", "type", "body",
                    "context_revision_ids", "assertion_state", "validity"}
_TYPES = {"Problem", "Claim", "ProofArtifact", "DefinitionContext", "Assumption", "Reference"}


def _read(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("invalid or oversized fixture input")
        chunks, total = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ValueError("fixture input grew beyond its bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if total != before.st_size or any(getattr(before, name) != getattr(after, name) for name in fields):
            raise ValueError("fixture input changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate fixture JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite fixture JSON number")

    def strict(value):
        if value is None or type(value) in (bool, int):
            return
        if type(value) is str:
            value.encode("utf-8")
            return
        if type(value) is float and math.isfinite(value):
            return
        if type(value) is list:
            for item in value:
                strict(item)
            return
        if type(value) is dict:
            for key, item in value.items():
                strict(key)
                strict(item)
            return
        raise ValueError("invalid fixture JSON value")

    result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    strict(result)
    if type(result) is not dict:
        raise ValueError("fixture JSON must be an object")
    return result


def _readonly(root: Path) -> bool:
    try:
        (root / "forbidden-write").write_bytes(b"fixture")
    except OSError:
        pass
    else:
        return False
    for name in _FILES:
        try:
            descriptor = os.open(root / name, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError:
            continue
        else:
            os.close(descriptor)
            return False
    return True


def main():
    root = Path("/input")
    names = sorted(path.name for path in root.iterdir())
    if names != _FILES or list(Path("/work").iterdir()):
        raise ValueError("unexpected fixture input or nonempty work directory")
    raw_dto = _read(root / "context.json", 8388608)
    raw_use = _read(root / "use.json", 524288)
    raw_overlay = _read(root / "overlay.json", 524288)
    raw_driver = _read(root / "driver.py", 524288)
    dto = _json(raw_dto)
    _json(raw_use)
    _json(raw_overlay)
    if (set(dto) != _DTO_FIELDS or type(dto["version"]) is not str
            or type(dto["policy_hash"]) is not str or type(dto["roots"]) is not dict
            or set(dto["roots"]) != {"problem", "claim", "proof"}
            or type(dto["revisions"]) is not list or not 3 <= len(dto["revisions"]) <= 256
            or type(dto["relations"]) is not list or len(dto["relations"]) > 1024):
        raise ValueError("invalid fixture DTO shape")
    revisions, seen = [], set()
    for item in dto["revisions"]:
        if (type(item) is not dict or set(item) != _REVISION_FIELDS
                or any(type(item[key]) is not str or not item[key]
                       for key in ("entity_id", "revision_id", "type", "body", "body_sha256"))
                or item["type"] not in _TYPES or item["assertion_state"] != "candidate"
                or item["validity"] != "needs_review"
                or type(item["context_revision_ids"]) is not list
                or any(type(key) is not str for key in item["context_revision_ids"])
                or item["revision_id"] in seen):
            raise ValueError("invalid fixture revision shape")
        seen.add(item["revision_id"])
        body = item["body"].encode("utf-8")
        actual = hashlib.sha256(body).hexdigest()
        if len(body) > 131072 or actual != item["body_sha256"]:
            raise ValueError("fixture received incorrect or oversized body bytes")
        revisions.append({"entity_id": item["entity_id"], "revision_id": item["revision_id"],
                          "type": item["type"], "body_sha256": actual, "body_bytes": len(body)})
    if not _readonly(root):
        raise ValueError("fixture input is writable")
    result = {"version": "validity-observation-1", "input_files": names,
              "dto_sha256": hashlib.sha256(raw_dto).hexdigest(),
              "descriptor_sha256": hashlib.sha256(raw_use).hexdigest(),
              "overlay_sha256": hashlib.sha256(raw_overlay).hexdigest(),
              "driver_sha256": hashlib.sha256(raw_driver).hexdigest(),
              "revisions": revisions, "relations": len(dto["relations"]),
              "empty_work": True, "input_readonly": True, "mathematical_review": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
