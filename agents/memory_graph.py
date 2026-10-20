"""Scope-local mathematical memory; graph.json is authoritative, index.json disposable.

One node's complete record_text is one embedding input (no truncation/chunking).
Channel limits select semantic seeds; explicit graph context may cross channels.
Pass the same api_lock_path to generation and verification for shared API serialization.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request
from uuid import uuid4

MODEL = "BAAI/bge-m3"
DIMENSION = 1024
RELATIONS = {"requires", "contrasts", "refuted_by", "supersedes"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def record_text(record):
    """Exact encoder input and fixture key; only the original content is embedded."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _cosine(first, second):
    if len(first) != len(second) or not first:
        raise ValueError("Vectors have different or empty dimensions")
    denominator = math.sqrt(sum(v * v for v in first) * sum(v * v for v in second))
    return sum(a * b for a, b in zip(first, second)) / denominator if denominator else 0.0


@contextmanager
def _locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".memory-", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


class MemoryGraph:
    def __init__(self, root, encoder=None, *, api_lock_path=None):
        self.root = Path(root).resolve()
        self.encoder = encoder
        self.api_lock_path = (Path(api_lock_path).resolve() if api_lock_path is not None
                              else self.root.parent / ".embedding-api.lock")

    def _path(self, scope, filename):
        directory = (self.root / scope).resolve()
        if directory == self.root or not directory.is_relative_to(self.root):
            raise ValueError("scope must name a directory inside memory root")
        return directory / filename

    @contextmanager
    def _graph(self, scope, write=False):
        path = self._path(scope, "graph.json")
        with _locked(path.with_name(".graph.lock")):
            graph = _read(path, {"meta": {"scope": scope, "created_at_utc": _now()},
                                 "nodes": {}, "edges": []})
            yield graph
            if write:
                graph["meta"]["updated_at_utc"] = _now()
                _write(path, graph)

    def init(self, scope, meta=None):
        with self._graph(scope, write=True) as graph:
            graph["meta"].update(meta or {})
        path = self._path(scope, "graph.json")
        return {"scope": scope, "memory_dir": str(path.parent),
                "graph_path": str(path), "meta": graph["meta"]}

    def _add(self, graph, channel, record, metadata):
        if not isinstance(record, dict):
            raise ValueError("record must be a JSON object")
        node_id = metadata.get("node_id") or uuid4().hex
        if node_id in graph["nodes"]:
            raise ValueError("node_id already exists; append a revision with supersedes")
        node = {"id": node_id, "channel": channel, "content": record,
                "type": metadata.get("type", record.get("type", channel)),
                "source": metadata.get("source", record.get("source")),
                "conditions": metadata.get("conditions", record.get("conditions", [])),
                "branch": metadata.get("branch", record.get("branch", record.get("branch_id"))),
                "status": metadata.get("status", record.get("status", "active")),
                "time": _now()}
        for relation in metadata.get("relations", []):
            if relation["type"] not in RELATIONS:
                raise ValueError("Unsupported relation type: " + relation["type"])
            graph["edges"].append({"source": node_id, "type": relation["type"],
                                   "target": relation["target_id"]})
        graph["nodes"][node_id] = node
        return node

    def append(self, scope, channel, record, metadata=None):
        with self._graph(scope, write=True) as graph:
            node = self._add(graph, channel, record, metadata or {})
        return node

    def branch_update(self, scope, branch_id, state):
        """Save a branch-state record and update that branch's node statuses, not content."""
        with self._graph(scope, write=True) as graph:
            if "status" in state:
                for node in graph["nodes"].values():
                    if node["branch"] == branch_id:
                        node["status"] = state["status"]
            node = self._add(graph, "branch_states", {"branch_id": branch_id, "state": state},
                             {"branch": branch_id, "status": state.get("status", "active")})
        return node

    @staticmethod
    def _relations(graph, node_id):
        return [{"type": edge["type"], "target_id": edge["target"]}
                for edge in graph["edges"] if edge["source"] == node_id]

    def get(self, scope, ids, include_dependencies=True):
        """Return requested full nodes and full one-hop outgoing relation targets."""
        with self._graph(scope) as graph:
            ids = list(dict.fromkeys(ids))
            nodes = [{**graph["nodes"][key], "relations": self._relations(graph, key)} for key in ids]
            targets = list(dict.fromkeys(edge["target"] for edge in graph["edges"]
                           if edge["source"] in ids and edge["target"] not in ids))
            dependencies = [{**graph["nodes"][key], "relations": self._relations(graph, key)}
                            for key in targets if key in graph["nodes"]] if include_dependencies else []
        return {"scope": scope, "count": len(nodes), "nodes": nodes,
                "direct_dependencies": dependencies}

    def query(self, scope, channel, filters=None, contains=None, limit=100, reverse=True):
        """Filters apply to envelope top-level keys, matching the old query contract."""
        if limit <= 0:
            raise ValueError("limit must be > 0")
        with self._graph(scope) as graph:
            items = [{"timestamp_utc": node["time"], "channel": node["channel"], "record": node["content"]}
                     for node in graph["nodes"].values() if node["channel"] == channel]
        if filters:
            items = [item for item in items if all(item.get(key) == value for key, value in filters.items())]
        if contains:
            items = [item for item in items if contains.lower() in json.dumps(item, ensure_ascii=False).lower()]
        items = (list(reversed(items)) if reverse else items)[:limit]
        return {"scope": scope, "channel": channel, "count": len(items), "items": items}

    def _request_embeddings(self, texts):
        key = os.environ.get("RETHLAS_EMBEDDING_API_KEY", "").strip()
        key_file = os.environ.get("RETHLAS_EMBEDDING_KEY_FILE")
        if not key and key_file:
            for line in Path(key_file).read_text(encoding="utf-8").splitlines():
                name, separator, value = line.strip().partition("=")
                if name.strip() == "SILICONFLOW_API_KEY" and separator:
                    key = value.strip().strip("\"'")
                    break
        if not key:
            raise ValueError("Set RETHLAS_EMBEDDING_API_KEY or RETHLAS_EMBEDDING_KEY_FILE "
                             "containing SILICONFLOW_API_KEY=...")
        endpoint = os.environ.get("RETHLAS_EMBEDDING_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings")
        payload = {"model": MODEL, "input": texts, "encoding_format": "float"}
        request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json"}, method="POST")
        with _locked(self.api_lock_path):
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
        return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]

    def _encode(self, texts):
        vectors = self.encoder(texts) if self.encoder is not None else self._request_embeddings(texts)
        if len(vectors) != len(texts) or any(len(vector) != DIMENSION for vector in vectors):
            raise ValueError("BAAI/bge-m3 must return one 1024-dimensional dense vector per input")
        return vectors

    def _vectors(self, scope, nodes):
        path = self._path(scope, "index.json")
        lock = path.with_name(".index.lock")
        texts = {node["id"]: record_text(node["content"]) for node in nodes}
        keys = {key: hashlib.sha256((MODEL + "\0" + text).encode("utf-8")).hexdigest()
                for key, text in texts.items()}
        empty = {"model": MODEL, "dimension": DIMENSION, "vectors": {}}
        with _locked(lock):
            cache = _read(path, empty)
        if cache["model"] != MODEL or cache["dimension"] != DIMENSION:
            raise ValueError("index.json model/dimension mismatch; delete the disposable index to rebuild")
        missing = {keys[key]: text for key, text in texts.items() if keys[key] not in cache["vectors"]}
        if missing:
            encoded = self._encode(list(missing.values()))
            with _locked(lock):
                cache = _read(path, empty)
                cache["vectors"].update(zip(missing, encoded))
                _write(path, cache)
        return {node_id: cache["vectors"][key] for node_id, key in keys.items()}

    def index(self, scope):
        with self._graph(scope) as graph:
            nodes = list(graph["nodes"].values())
        vectors = self._vectors(scope, nodes)
        return {"scope": scope, "model": MODEL, "dimension": DIMENSION, "node_count": len(vectors)}

    def search(self, scope, query, channels=None, top_k=10, limit_per_channel=10):
        """Rank full-record vectors, then expand outgoing explicit edges twice, at most 20 nodes."""
        if not query.strip() or top_k <= 0 or limit_per_channel <= 0:
            raise ValueError("query must be non-empty; top_k and limit_per_channel must be > 0")
        with self._graph(scope) as graph:
            nodes = graph["nodes"]
        candidates = [key for key, node in nodes.items() if channels is None or node["channel"] in channels]
        reasons, scores = {}, {}
        if candidates:
            vectors = self._vectors(scope, list(nodes.values()))
            query_vector = self._encode([query])[0]
            scores = {key: _cosine(query_vector, vector) for key, vector in vectors.items()}
            counts = Counter()
            for key in sorted(candidates, key=lambda key: (-scores[key], key)):
                channel = nodes[key]["channel"]
                if counts[channel] >= limit_per_channel:
                    continue
                reasons[key] = {"type": "semantic", "depth": 0}
                counts[channel] += 1
                if len(reasons) >= min(top_k, 20):
                    break
            adjacency = defaultdict(list)
            for edge in graph["edges"]:
                adjacency[edge["source"]].append(edge)
            pending = deque((key, 0) for key in reasons)
            while pending and len(reasons) < 20:
                source, depth = pending.popleft()
                if depth >= 2:
                    continue
                for edge in adjacency[source]:
                    target = edge["target"]
                    if target in reasons or target not in nodes:
                        continue
                    reasons[target] = {"type": "graph", "depth": depth + 1,
                                       "source_id": source, "relation": edge["type"]}
                    pending.append((target, depth + 1))
                    if len(reasons) >= 20:
                        break
        results = []
        for key, reason in reasons.items():
            node = nodes[key]
            summary = (node["content"].get("summary") or record_text(node["content"]))[:400]
            results.append({"id": key, "channel": node["channel"], "score": scores[key],
                            "summary": summary, "conditions": node["conditions"], "source": node["source"],
                            "status": node["status"], "relations": self._relations(graph, key),
                            "include_reason": reason})
        return {"scope": scope, "query": query, "channels": channels, "top_k": min(top_k, 20),
                "limit_per_channel": limit_per_channel, "count": len(results), "results": results}
