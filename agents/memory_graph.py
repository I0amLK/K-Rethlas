"""Scope-local SQLite memory with transactional nodes, edges and vector cache.

One node's complete record_text is one embedding input (no truncation/chunking).
Legacy graph.json/index.json are imported once and left untouched. Similarity,
channel selection and two-hop graph expansion retain their original semantics.
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
import sqlite3
import struct
import urllib.request
import warnings
from uuid import uuid4

MODEL = "BAAI/bge-m3"
DIMENSION = 1024
RELATIONS = {"requires", "contrasts", "refuted_by", "supersedes"}
DATABASE = "memory.sqlite3"
SCHEMA_VERSION = 1
_VECTOR = struct.Struct("<" + "d" * DIMENSION)


def _now():
    return datetime.now(timezone.utc).isoformat()


def record_text(record):
    """Exact encoder input and cache key; only the original content is embedded."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def _pack_vector(vector):
    if len(vector) != DIMENSION or any(not math.isfinite(value) for value in vector):
        raise ValueError("Expected one finite 1024-dimensional embedding vector")
    return _VECTOR.pack(*vector)


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

    def database_path(self, scope):
        return self._path(scope, DATABASE)

    @contextmanager
    def _connection(self, scope, write=False):
        path = self.database_path(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                # Serialize only initial setup/import, including concurrent first readers.
                with _locked(path.with_name(".graph.lock")):
                    if connection.execute("PRAGMA user_version").fetchone()[0] == 0:
                        connection.execute("PRAGMA journal_mode=WAL")
                        connection.execute("PRAGMA synchronous=FULL")
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            self._initialize(connection, scope, path.parent)
                            connection.commit()
                        except BaseException:
                            connection.rollback()
                            raise
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported memory database schema version: {version}")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        finally:
            connection.close()

    def _initialize(self, db, scope, directory):
        statements = (
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            "CREATE TABLE nodes (seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, "
            "channel TEXT NOT NULL, content TEXT NOT NULL, type TEXT NOT NULL, "
            "source TEXT NOT NULL, conditions TEXT NOT NULL, branch TEXT NOT NULL, "
            "status TEXT NOT NULL, time TEXT NOT NULL, text_key TEXT NOT NULL)",
            "CREATE INDEX nodes_channel ON nodes(channel, seq)",
            "CREATE INDEX nodes_branch ON nodes(branch)",
            "CREATE TABLE edges (seq INTEGER PRIMARY KEY, source TEXT NOT NULL, "
            "type TEXT NOT NULL, target TEXT NOT NULL)",
            "CREATE INDEX edges_source ON edges(source, seq)",
            "CREATE INDEX edges_target ON edges(target)",
            "CREATE TABLE embeddings (model TEXT NOT NULL, dimension INTEGER NOT NULL, "
            "text_key TEXT NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY (model, dimension, text_key))",
        )
        for statement in statements:
            db.execute(statement)
        legacy = directory / "graph.json"
        graph = json.loads(legacy.read_text(encoding="utf-8")) if legacy.exists() else {
            "meta": {"scope": scope, "created_at_utc": _now()}, "nodes": {}, "edges": []}
        for key, value in graph["meta"].items():
            self._set_meta(db, key, value)
        for identifier, node in graph["nodes"].items():
            if identifier != node["id"]:
                raise ValueError("Legacy node key/id mismatch")
            self._insert_node(db, node)
        for edge in graph["edges"]:
            self._insert_edge(db, edge)
        cache = directory / "index.json"
        if legacy.exists() and cache.exists():
            # A disposable bad cache must not block authoritative graph import.
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data["model"] != MODEL or data["dimension"] != DIMENSION:
                    raise ValueError("Legacy embedding model/dimension mismatch")
                packed = [(MODEL, DIMENSION, key, _pack_vector(vector))
                          for key, vector in data["vectors"].items()]
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
                warnings.warn(f"Ignoring disposable legacy index.json ({type(exc).__name__}); vectors will rebuild", RuntimeWarning)
                packed = []
            db.executemany("INSERT INTO embeddings VALUES (?, ?, ?, ?)", packed)
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def _set_meta(db, key, value):
        db.execute("INSERT INTO metadata VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, _json(value)))

    def _touch(self, db):
        self._set_meta(db, "updated_at_utc", _now())

    @staticmethod
    def _text_key(content):
        return hashlib.sha256((MODEL + "\0" + record_text(content)).encode("utf-8")).hexdigest()

    def _insert_node(self, db, node):
        if not isinstance(node["content"], dict):
            raise ValueError("record must be a JSON object")
        if db.execute("SELECT 1 FROM nodes WHERE id=?", (node["id"],)).fetchone():
            raise ValueError("node_id already exists; append a revision with supersedes")
        db.execute("INSERT INTO nodes (id,channel,content,type,source,conditions,branch,status,time,text_key) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (node["id"], node["channel"], _json(node["content"]), _json(node["type"]),
                    _json(node["source"]), _json(node["conditions"]), _json(node["branch"]),
                    _json(node["status"]), node["time"], self._text_key(node["content"])))

    @staticmethod
    def _insert_edge(db, edge):
        if edge["type"] not in RELATIONS:
            raise ValueError("Unsupported relation type: " + edge["type"])
        db.execute("INSERT INTO edges (source,type,target) VALUES (?,?,?)",
                   (edge["source"], edge["type"], edge["target"]))

    @staticmethod
    def _node(row):
        return {"id": row["id"], "channel": row["channel"], "content": json.loads(row["content"]),
                **{key: json.loads(row[key]) for key in ("type", "source", "conditions", "branch", "status")},
                "time": row["time"]}

    def init(self, scope, meta=None):
        with self._connection(scope, write=True) as db:
            for key, value in (meta or {}).items():
                self._set_meta(db, key, value)
            self._touch(db)
            metadata = {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM metadata")}
        path = self.database_path(scope)
        return {"scope": scope, "memory_dir": str(path.parent), "graph_path": str(path), "meta": metadata}

    def export_snapshot(self, scope):
        """Return a consistent JSON-compatible graph for diagnostics or export."""
        with self._connection(scope) as db:
            return {"meta": {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM metadata")},
                    "nodes": {row["id"]: self._node(row) for row in db.execute("SELECT * FROM nodes ORDER BY seq")},
                    "edges": [dict(row) for row in db.execute("SELECT source,type,target FROM edges ORDER BY seq")]}

    def backup(self, scope, destination):
        """Use SQLite's backup API so a live WAL is included; never overwrite files."""
        destination = Path(destination).resolve()
        if destination == self.database_path(scope):
            raise ValueError("Backup destination must differ from the live database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb"):
            pass
        try:
            with self._connection(scope) as source:
                target = sqlite3.connect(destination)
                try:
                    source.backup(target)
                finally:
                    target.close()
        except BaseException:
            destination.unlink()
            raise
        return {"scope": scope, "backup_path": str(destination)}

    def _add(self, db, channel, record, metadata):
        if not isinstance(record, dict):
            raise ValueError("record must be a JSON object")
        node = {"id": metadata.get("node_id") or uuid4().hex, "channel": channel, "content": record,
                "type": metadata.get("type", record.get("type", channel)),
                "source": metadata.get("source", record.get("source")),
                "conditions": metadata.get("conditions", record.get("conditions", [])),
                "branch": metadata.get("branch", record.get("branch", record.get("branch_id"))),
                "status": metadata.get("status", record.get("status", "active")), "time": _now()}
        self._insert_node(db, node)
        for relation in metadata.get("relations", []):
            self._insert_edge(db, {"source": node["id"], "type": relation["type"], "target": relation["target_id"]})
        return node

    def append(self, scope, channel, record, metadata=None):
        with self._connection(scope, write=True) as db:
            node = self._add(db, channel, record, metadata or {})
            self._touch(db)
        return node

    def branch_update(self, scope, branch_id, state):
        """Atomically update branch statuses and append its state record."""
        with self._connection(scope, write=True) as db:
            if "status" in state:
                db.execute("UPDATE nodes SET status=? WHERE branch=?", (_json(state["status"]), _json(branch_id)))
            node = self._add(db, "branch_states", {"branch_id": branch_id, "state": state},
                             {"branch": branch_id, "status": state.get("status", "active")})
            self._touch(db)
        return node

    @staticmethod
    def _relations(db, node_id):
        return [{"type": row["type"], "target_id": row["target"]}
                for row in db.execute("SELECT type,target FROM edges WHERE source=? ORDER BY seq", (node_id,))]

    def get(self, scope, ids, include_dependencies=True):
        """Return requested full nodes and full one-hop outgoing relation targets."""
        with self._connection(scope) as db:
            ids = list(dict.fromkeys(ids))
            nodes, targets = [], []
            for identifier in ids:
                row = db.execute("SELECT * FROM nodes WHERE id=?", (identifier,)).fetchone()
                if row is None:
                    raise KeyError(identifier)
                nodes.append({**self._node(row), "relations": self._relations(db, identifier)})
            if include_dependencies:
                # Preserve edge order even when requested IDs use a different order.
                for start in range(0, len(ids), 400):
                    batch = ids[start:start + 400]
                    placeholders = ",".join("?" for _ in batch)
                    targets.extend(db.execute(f"SELECT seq,target FROM edges WHERE source IN ({placeholders}) ORDER BY seq", batch))
            dependencies = []
            seen = set(ids)
            for target in sorted(targets, key=lambda row: row["seq"]):
                identifier = target["target"]
                if identifier in seen:
                    continue
                seen.add(identifier)
                row = db.execute("SELECT * FROM nodes WHERE id=?", (identifier,)).fetchone()
                if row is not None:
                    dependencies.append({**self._node(row), "relations": self._relations(db, identifier)})
        return {"scope": scope, "count": len(nodes), "nodes": nodes, "direct_dependencies": dependencies}

    def query(self, scope, channel, filters=None, contains=None, limit=100, reverse=True):
        """Keep the original envelope-only filters and substring matching."""
        if limit <= 0:
            raise ValueError("limit must be > 0")
        order = "DESC" if reverse else "ASC"
        items = []
        with self._connection(scope) as db:
            for row in db.execute(f"SELECT time,channel,content FROM nodes WHERE channel=? ORDER BY seq {order}", (channel,)):
                item = {"timestamp_utc": row["time"], "channel": row["channel"], "record": json.loads(row["content"])}
                if filters and not all(item.get(key) == value for key, value in filters.items()):
                    continue
                if contains and contains.lower() not in json.dumps(item, ensure_ascii=False).lower():
                    continue
                items.append(item)
                if len(items) >= limit:
                    break
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
        for vector in vectors:
            _pack_vector(vector)
        return vectors

    def _vectors(self, scope, nodes):
        texts = {node["id"]: record_text(node["content"]) for node in nodes}
        keys = {key: hashlib.sha256((MODEL + "\0" + text).encode("utf-8")).hexdigest()
                for key, text in texts.items()}
        cached = {}
        unique = list(dict.fromkeys(keys.values()))
        with self._connection(scope) as db:
            for start in range(0, len(unique), 400):
                batch = unique[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                for row in db.execute(f"SELECT text_key,vector FROM embeddings WHERE model=? AND dimension=? "
                                      f"AND text_key IN ({placeholders})", [MODEL, DIMENSION, *batch]):
                    cached[row["text_key"]] = list(_VECTOR.unpack(row["vector"]))
        missing = {keys[key]: text for key, text in texts.items() if keys[key] not in cached}
        if missing:
            # Never hold a SQLite transaction while waiting for the external API.
            encoded = self._encode(list(missing.values()))
            packed = [(MODEL, DIMENSION, key, _pack_vector(vector)) for key, vector in zip(missing, encoded)]
            with self._connection(scope, write=True) as db:
                db.executemany("INSERT OR IGNORE INTO embeddings VALUES (?,?,?,?)", packed)
                for key in missing:
                    row = db.execute("SELECT vector FROM embeddings WHERE model=? AND dimension=? AND text_key=?",
                                     (MODEL, DIMENSION, key)).fetchone()
                    cached[key] = list(_VECTOR.unpack(row["vector"]))
        return {node_id: cached[key] for node_id, key in keys.items()}

    def clear_index(self, scope):
        """Discard rebuildable vectors, never nodes/edges/metadata or legacy files."""
        with self._connection(scope, write=True) as db:
            count = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            db.execute("DELETE FROM embeddings")
        return {"scope": scope, "removed_vectors": count}

    def index(self, scope):
        with self._connection(scope) as db:
            nodes = [self._node(row) for row in db.execute("SELECT * FROM nodes ORDER BY seq")]
        vectors = self._vectors(scope, nodes)
        return {"scope": scope, "model": MODEL, "dimension": DIMENSION, "node_count": len(vectors)}

    def search(self, scope, query, channels=None, top_k=10, limit_per_channel=10):
        """Rank full-record vectors, then expand outgoing explicit edges twice, at most 20 nodes."""
        if not query.strip() or top_k <= 0 or limit_per_channel <= 0:
            raise ValueError("query must be non-empty; top_k and limit_per_channel must be > 0")
        with self._connection(scope) as db:
            nodes = {row["id"]: self._node(row) for row in db.execute("SELECT * FROM nodes ORDER BY seq")}
            edges = [dict(row) for row in db.execute("SELECT source,type,target FROM edges ORDER BY seq")]
        candidates = [key for key, node in nodes.items() if channels is None or node["channel"] in channels]
        reasons, scores = {}, {}
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge["source"]].append(edge)
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
            relations = [{"type": edge["type"], "target_id": edge["target"]} for edge in adjacency[key]]
            results.append({"id": key, "channel": node["channel"], "score": scores[key],
                            "summary": summary, "conditions": node["conditions"], "source": node["source"],
                            "status": node["status"], "relations": relations, "include_reason": reason})
        return {"scope": scope, "query": query, "channels": channels, "top_k": min(top_k, 20),
                "limit_per_channel": limit_per_channel, "count": len(results), "results": results}
