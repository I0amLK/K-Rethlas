"""Deterministic, disposable retrieval projection, never knowledge or permission.

Callers must select an authorized project before loading/scoring this index and
recheck exact authoritative revisions and current withdrawal state before use.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math
import re
from typing import Any

from .embedding import DIMENSIONS, MODEL, EmbeddingError, validate_vector
from .private_fs import canonical_json, digest


VERSION = "retrieval-index-1"
INPUT_VERSION = "exact-markdown-utf8-chunks-1"
MAX_CHUNKS = 1024
MAX_REVISIONS = 1024
MAX_BODY_BYTES = 131072
MAX_INDEX_BYTES = 33554432
MAX_INPUT_BYTES = 8388608
MAX_QUERY_BYTES = 32768
_BINDING = {"service_id", "store_id", "store_config_sha256", "graph"}
_CHUNK = {"chunk_id", "entity_id", "revision_id", "body_sha256", "byte_start", "byte_end", "text", "text_sha256"}
_INDEX = {"version", "model", "dimensions", "input_version", "mode", "binding", "chunks", "index_sha256"}
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class RetrievalIndexError(ValueError):
    """An invalid or oversized disposable projection."""


class RetrievalIndexStale(RetrievalIndexError):
    """Explicit maintenance must rebuild for the current project checkpoint."""


def _hash(value: Any) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _id(value: Any) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        return len(value.encode("utf-8")) <= 4096
    except UnicodeError:
        return False


def _binding(value: Any) -> None:
    if type(value) is not dict or set(value) != _BINDING:
        raise RetrievalIndexError("invalid retrieval project binding")
    if any(not _hash(value[name]) for name in _BINDING - {"graph"}):
        raise RetrievalIndexError("invalid retrieval service/store binding")
    graph = value["graph"]
    if (type(graph) is not dict or set(graph) != {"head", "generation", "graph_sha256"}
            or not _hash(graph["head"]) or not _hash(graph["graph_sha256"])
            or type(graph["generation"]) is not int or graph["generation"] < 0):
        raise RetrievalIndexError("invalid retrieval graph checkpoint")


def _chunk_hash(chunk: dict) -> str:
    return digest({"input_version": INPUT_VERSION, **{key: value for key, value in chunk.items()
                                                      if key not in ("chunk_id", "vector")}})


def _chunk(value: Any, *, vector: bool = False) -> dict:
    expected = _CHUNK | ({"vector"} if vector else set())
    if type(value) is not dict or set(value) != expected:
        raise RetrievalIndexError("invalid retrieval chunk fields")
    if any(not _id(value[key]) for key in ("entity_id", "revision_id")):
        raise RetrievalIndexError("invalid retrieval revision identity")
    if any(not _hash(value[key]) for key in ("chunk_id", "body_sha256", "text_sha256")):
        raise RetrievalIndexError("invalid retrieval chunk hash")
    if type(value["text"]) is not str or not value["text"]:
        raise RetrievalIndexError("invalid retrieval chunk text")
    try:
        raw = value["text"].encode("utf-8")
    except UnicodeError:
        raise RetrievalIndexError("invalid retrieval chunk UTF-8") from None
    start, end = value["byte_start"], value["byte_end"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= MAX_BODY_BYTES
            or end - start != len(raw) or len(raw) > 32768
            or hashlib.sha256(raw).hexdigest() != value["text_sha256"]
            or _chunk_hash(value) != value["chunk_id"]):
        raise RetrievalIndexError("retrieval chunk bytes/hash mismatch")
    if vector:
        try:
            validate_vector(value["vector"])
        except EmbeddingError:
            raise RetrievalIndexError("invalid retrieval chunk vector") from None
    return value


def _chunks(values: Any, *, vector: bool = False) -> None:
    if type(values) is not list or len(values) > MAX_CHUNKS:
        raise RetrievalIndexError("retrieval chunk capacity exceeded")
    previous, identities, groups = None, set(), {}
    total = 0
    for item in values:
        _chunk(item, vector=vector)
        key = (item["entity_id"], item["revision_id"], item["byte_start"])
        if previous is not None and key <= previous or item["chunk_id"] in identities:
            raise RetrievalIndexError("retrieval chunks must be unique and canonically ordered")
        previous = key
        identities.add(item["chunk_id"])
        total += len(item["text"].encode("utf-8"))
        if total > MAX_INPUT_BYTES:
            raise RetrievalIndexError("retrieval input byte capacity exceeded")
        groups.setdefault((item["entity_id"], item["revision_id"]), []).append(item)
    for group in groups.values():
        position = 0
        body = []
        for item in group:
            if item["byte_start"] != position or item["body_sha256"] != group[0]["body_sha256"]:
                raise RetrievalIndexError("retrieval chunks do not cover an exact revision")
            position = item["byte_end"]
            body.append(item["text"])
        if hashlib.sha256("".join(body).encode("utf-8")).hexdigest() != group[0]["body_sha256"]:
            raise RetrievalIndexError("retrieval chunks do not bind the complete body")


def chunk_revisions(revisions: list[dict], *, chunk_bytes: int = 2048) -> list[dict]:
    if type(chunk_bytes) is not int or not 4 <= chunk_bytes <= 32768:
        raise RetrievalIndexError("invalid retrieval chunk byte limit")
    if type(revisions) is not list or len(revisions) > MAX_REVISIONS:
        raise RetrievalIndexError("retrieval revision capacity exceeded")
    chunks, seen = [], set()
    for revision in revisions:
        if (type(revision) is not dict or not {"entity_id", "revision_id", "body", "body_sha256"} <= set(revision)
                or not _id(revision["entity_id"]) or not _id(revision["revision_id"])
                or not _hash(revision["body_sha256"]) or type(revision["body"]) is not str
                or not revision["body"].strip()):
            raise RetrievalIndexError("invalid authoritative retrieval revision")
        identity = (revision["entity_id"], revision["revision_id"])
        if identity in seen:
            raise RetrievalIndexError("duplicate retrieval revision")
        seen.add(identity)
        try:
            raw = revision["body"].encode("utf-8")
        except UnicodeError:
            raise RetrievalIndexError("invalid authoritative revision UTF-8") from None
        if len(raw) > MAX_BODY_BYTES or hashlib.sha256(raw).hexdigest() != revision["body_sha256"]:
            raise RetrievalIndexError("authoritative revision body hash/size mismatch")
        start = 0
        while start < len(raw):
            end = min(start + chunk_bytes, len(raw))
            while end < len(raw) and raw[end] & 0xC0 == 0x80:
                end -= 1
            if end <= start:
                raise RetrievalIndexError("chunk limit cannot contain a UTF-8 character")
            text = raw[start:end].decode("utf-8")
            chunk = {"entity_id": revision["entity_id"], "revision_id": revision["revision_id"],
                     "body_sha256": revision["body_sha256"], "byte_start": start, "byte_end": end,
                     "text": text, "text_sha256": hashlib.sha256(raw[start:end]).hexdigest()}
            chunks.append({"chunk_id": _chunk_hash(chunk), **chunk})
            if len(chunks) > MAX_CHUNKS:
                raise RetrievalIndexError("retrieval chunk capacity exceeded")
            start = end
    chunks.sort(key=lambda item: (item["entity_id"], item["revision_id"], item["byte_start"]))
    _chunks(chunks)
    return chunks


def make_index(chunks: list[dict], vectors: list[list[float]] | None, *, binding: dict,
               mode: str) -> dict:
    _binding(binding)
    _chunks(chunks)
    if mode not in ("bge_cache_only", "bge_live", "lexical"):
        raise RetrievalIndexError("invalid retrieval index mode")
    if mode == "lexical":
        if vectors is not None:
            raise RetrievalIndexError("lexical index cannot contain synthetic vectors")
        entries = deepcopy(chunks)
    else:
        if type(vectors) is not list or len(vectors) != len(chunks):
            raise RetrievalIndexError("retrieval vector count mismatch")
        try:
            entries = [{**deepcopy(chunk), "vector": validate_vector(vector)} for chunk, vector in zip(chunks, vectors)]
        except EmbeddingError:
            raise RetrievalIndexError("invalid retrieval index vector") from None
    value = {"version": VERSION, "model": None if mode == "lexical" else MODEL,
             "dimensions": 0 if mode == "lexical" else DIMENSIONS, "input_version": INPUT_VERSION,
             "mode": mode, "binding": deepcopy(binding), "chunks": entries}
    result = {**value, "index_sha256": digest(value)}
    validate_index(result, binding=binding)
    return result


def validate_index(index: Any, *, binding: dict | None = None) -> dict:
    if type(index) is not dict or set(index) != _INDEX:
        raise RetrievalIndexError("invalid retrieval index fields")
    if index["version"] != VERSION or index["input_version"] != INPUT_VERSION:
        raise RetrievalIndexError("unsupported retrieval index version")
    mode = index["mode"]
    if mode not in ("bge_cache_only", "bge_live", "lexical"):
        raise RetrievalIndexError("invalid retrieval index mode")
    if (type(index["dimensions"]) is not int
            or index["dimensions"] != (0 if mode == "lexical" else DIMENSIONS)
            or index["model"] != (None if mode == "lexical" else MODEL)):
        raise RetrievalIndexError("retrieval model/dimension mismatch")
    _binding(index["binding"])
    if binding is not None:
        _binding(binding)
        if index["binding"] != binding:
            raise RetrievalIndexStale("retrieval index checkpoint or project mismatch")
    _chunks(index["chunks"], vector=mode != "lexical")
    try:
        raw = canonical_json(index)
        expected = digest({key: value for key, value in index.items() if key != "index_sha256"})
    except (ValueError, OverflowError, UnicodeError):
        raise RetrievalIndexError("invalid retrieval index JSON") from None
    if len(raw) > MAX_INDEX_BYTES or not _hash(index["index_sha256"]) or index["index_sha256"] != expected:
        raise RetrievalIndexError("retrieval index size/hash mismatch")
    return index


def _top(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_CHUNKS:
        raise RetrievalIndexError("invalid retrieval result limit")
    return value


def rank(index: dict, query_vector: list[float], *, top_k: int = 10) -> list[dict]:
    validate_index(index)
    _top(top_k)
    if index["mode"] == "lexical":
        raise RetrievalIndexError("lexical projection has no BGE vectors")
    try:
        query = validate_vector(query_vector)
    except EmbeddingError:
        raise RetrievalIndexError("invalid retrieval query vector") from None
    query_norm = math.hypot(*query)
    unit_query = [item / query_norm for item in query]
    hits = []
    for chunk in index["chunks"]:
        vector = chunk["vector"]
        norm = math.hypot(*vector)
        score = math.fsum(a * (b / norm) for a, b in zip(unit_query, vector))
        score = min(1.0, max(-1.0, score))
        hits.append({**{key: value for key, value in chunk.items() if key != "vector"}, "score": score})
    hits.sort(key=lambda item: (-item["score"], item["entity_id"], item["revision_id"], item["byte_start"], item["chunk_id"]))
    return hits[:top_k]


def tokenize(text: str) -> list[str]:
    if type(text) is not str:
        raise RetrievalIndexError("lexical input must be text")
    terms = re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
    result = []
    for term in terms:
        for group in re.findall(r"[㐀-鿿]+|[^㐀-鿿]+", term):
            if re.fullmatch(r"[㐀-鿿]+", group):
                result.extend(group)
                result.extend(group[index:index + 2] for index in range(len(group) - 1))
            else:
                result.append(group)
    return result


def lexical_rank(chunks: list[dict], query: str, *, top_k: int = 10) -> list[dict]:
    _top(top_k)
    if type(query) is not str or not query.strip():
        raise RetrievalIndexError("lexical query must be nonempty text")
    try:
        size = len(query.encode("utf-8"))
    except UnicodeError:
        raise RetrievalIndexError("invalid lexical query UTF-8") from None
    if size > MAX_QUERY_BYTES:
        raise RetrievalIndexError("lexical query exceeds byte limit")
    if type(chunks) is not list or len(chunks) > MAX_CHUNKS:
        raise RetrievalIndexError("retrieval chunk capacity exceeded")
    plain = [{key: value for key, value in item.items() if key != "vector"} if type(item) is dict else item for item in chunks]
    _chunks(plain)
    query_terms = Counter(tokenize(query))
    documents = [Counter(tokenize(item["text"])) for item in plain]
    frequency = Counter(term for document in documents for term in document)
    lengths = [sum(document.values()) for document in documents]
    average = sum(lengths) / len(lengths) if lengths else 0
    hits = []
    for chunk, document, length in zip(plain, documents, lengths):
        score = 0.0
        for term, query_count in query_terms.items():
            count = document.get(term, 0)
            if not count:
                continue
            inverse = math.log1p((len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            norm = 1.5 * (0.25 + 0.75 * length / average) if average else 1.5
            score += query_count * inverse * count * 2.5 / (count + norm)
        if score > 0:
            hits.append({**deepcopy(chunk), "score": score})
    hits.sort(key=lambda item: (-item["score"], item["entity_id"], item["revision_id"], item["byte_start"], item["chunk_id"]))
    return hits[:top_k]
