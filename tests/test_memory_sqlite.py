"""SQLite regression tests: no network calls or external research fixtures."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib import util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

AGENTS = Path(__file__).resolve().parents[1] / "agents"
sys.path.insert(0, str(AGENTS))
from memory_graph import DATABASE, DIMENSION, MODEL, MemoryGraph, record_text


def basis(index):
    vector = [0.0] * DIMENSION
    vector[index] = 1.0
    return vector


class Encoder:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [self.mapping.get(text, basis(0)) for text in texts]


def append(memory, identifier, **metadata):
    return memory.append("scope", "proof_steps", {"summary": identifier}, {"node_id": identifier, **metadata})


def legacy_node(identifier, **extra):
    return {"id": identifier, "channel": "proof_steps", "content": {"summary": identifier},
            "type": "lemma", "source": {"path": "fixture.md"}, "conditions": ["finite"],
            "branch": "route", "status": "active", "time": "2026-01-01T00:00:00+00:00", **extra}


def write_legacy(root, cache=None):
    directory = root / "scope"
    directory.mkdir(parents=True)
    graph = {"meta": {"scope": "scope", "custom": {"nested": [1, 2]}},
             "nodes": {key: legacy_node(key) for key in ("a", "b")},
             "edges": [{"source": "b", "type": "requires", "target": "a"}]}
    (directory / "graph.json").write_text(json.dumps(graph))
    if cache is not None:
        (directory / "index.json").write_text(cache if isinstance(cache, str) else json.dumps(cache))
    return graph


def test_roundtrip_metadata_no_json_writes(tmp_path):
    memory = MemoryGraph(tmp_path)
    result = memory.init("scope", {"title": "fixture"})
    assert Path(result["graph_path"]).name == DATABASE
    node = memory.append("scope", "proof_steps", {"proof": "complete proof " * 2000},
                         {"node_id": "a", "conditions": ["compact"], "source": {"location": "Lemma 2"}})
    assert MemoryGraph(tmp_path).get("scope", ["a"])["nodes"] == [{**node, "relations": []}]
    assert memory.init("scope")["meta"]["title"] == "fixture"
    assert not list(tmp_path.rglob("*.json"))
    with sqlite3.connect(memory.database_path("scope")) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_node_and_edges_rollback_together(tmp_path):
    memory = MemoryGraph(tmp_path)
    append(memory, "base")
    with pytest.raises(ValueError, match="Unsupported relation"):
        append(memory, "bad", relations=[{"type": "requires", "target_id": "base"},
                                          {"type": "invalid", "target_id": "base"}])
    snapshot = memory.export_snapshot("scope")
    assert list(snapshot["nodes"]) == ["base"] and snapshot["edges"] == []
    with pytest.raises(ValueError, match="already exists"):
        append(memory, "base")
    with pytest.raises(ValueError, match="JSON object"):
        memory.append("scope", "proof_steps", "not an object")


def test_branch_atomic_update_and_content_immutable(tmp_path, monkeypatch):
    memory = MemoryGraph(tmp_path)
    a, b = append(memory, "a", branch="route"), append(memory, "b", branch="other")
    state = memory.branch_update("scope", "route", {"status": "paused", "reason": "review"})
    full = memory.get("scope", ["a", "b"])["nodes"]
    assert full[0]["status"] == "paused" and full[0]["content"] == a["content"]
    assert full[1]["status"] == b["status"]
    assert state["content"]["state"]["reason"] == "review"
    def fail(*args):
        raise RuntimeError("injected append failure")
    monkeypatch.setattr(memory, "_add", fail)
    with pytest.raises(RuntimeError, match="injected"):
        memory.branch_update("scope", "route", {"status": "refuted"})
    assert memory.get("scope", ["a"])["nodes"][0]["status"] == "paused"


def test_get_order_dedup_one_hop_and_missing_targets(tmp_path):
    memory = MemoryGraph(tmp_path)
    for key in ("a", "b", "c"):
        append(memory, key)
    append(memory, "x", relations=[{"type": "requires", "target_id": "a"},
                                   {"type": "requires", "target_id": "missing"}])
    append(memory, "y", relations=[{"type": "contrasts", "target_id": "b"},
                                   {"type": "requires", "target_id": "a"}])
    full = memory.get("scope", ["y", "x", "y"])
    assert [node["id"] for node in full["nodes"]] == ["y", "x"]
    assert [node["id"] for node in full["direct_dependencies"]] == ["a", "b"]
    assert memory.get("scope", [], True)["nodes"] == []
    assert memory.get("scope", ["x"], False)["direct_dependencies"] == []
    with pytest.raises(KeyError):
        memory.get("scope", ["missing"])


def test_many_ids_are_batched_below_sqlite_parameter_limit(tmp_path):
    memory = MemoryGraph(tmp_path)
    with memory._connection("scope", write=True) as db:
        for i in range(450):
            memory._add(db, "proof_steps", {"summary": str(i)}, {"node_id": str(i)})
    assert memory.get("scope", [str(i) for i in range(450)])["count"] == 450
    memory.encoder = Encoder()
    assert memory.index("scope")["node_count"] == 450
    assert memory.index("scope")["node_count"] == 450
    assert len(memory.encoder.batches) == 1


def test_query_contract(tmp_path):
    memory = MemoryGraph(tmp_path)
    for i in range(5):
        memory.append("scope", "checks", {"summary": f"item-{i}", "status": "wrong"})
    assert [item["record"]["summary"] for item in memory.query("scope", "checks", limit=2)["items"]] == ["item-4", "item-3"]
    assert memory.query("scope", "checks", reverse=False, contains="ITEM-1")["count"] == 1
    assert memory.query("scope", "checks", filters={"status": "wrong"})["count"] == 0
    assert memory.query("scope", "checks", filters={"channel": "checks"})["count"] == 5
    with pytest.raises(ValueError):
        memory.query("scope", "checks", limit=0)


def test_two_hop_search_and_cache_lifecycle(tmp_path):
    records = {key: {"summary": key} for key in ("a", "b", "c")}
    encoder = Encoder({**{record_text(records[key]): basis(i) for i, key in enumerate(records)}, "find c": basis(2)})
    memory = MemoryGraph(tmp_path, encoder)
    append(memory, "a")
    append(memory, "b", relations=[{"type": "requires", "target_id": "a"}])
    memory.append("scope", "target", records["c"], {"node_id": "c", "relations": [{"type": "requires", "target_id": "b"}]})
    assert encoder.batches == []
    result = memory.search("scope", "find c", channels=["target"], top_k=1)
    assert [hit["id"] for hit in result["results"]] == ["c", "b", "a"]
    assert [hit["include_reason"]["depth"] for hit in result["results"]] == [0, 1, 2]
    assert [len(batch) for batch in encoder.batches] == [3, 1]
    assert MemoryGraph(tmp_path, encoder).search("scope", "find c", channels=["target"], top_k=1) == result
    assert [len(batch) for batch in encoder.batches] == [3, 1, 1]
    append(memory, "d")
    memory.index("scope")
    assert len(encoder.batches[-1]) == 1
    assert memory.clear_index("scope")["removed_vectors"] == 4
    memory.index("scope")
    assert len(encoder.batches[-1]) == 4
    with sqlite3.connect(memory.database_path("scope")) as db:
        assert db.execute("SELECT length(vector) FROM embeddings LIMIT 1").fetchone()[0] == 1024 * 8


def test_identical_content_shares_vector(tmp_path):
    encoder = Encoder()
    memory = MemoryGraph(tmp_path, encoder)
    for key in ("a", "b"):
        memory.append("scope", "proof_steps", {"summary": "same"}, {"node_id": key})
    assert memory.index("scope")["node_count"] == 2
    assert [len(batch) for batch in encoder.batches] == [1]


def test_embedding_api_not_inside_database_transaction(tmp_path):
    def encoder(texts):
        # A second instance must be able to commit while the first is encoding.
        append(MemoryGraph(tmp_path), "other-writer")
        return [basis(0) for _ in texts]
    memory = MemoryGraph(tmp_path, encoder)
    append(memory, "a")
    memory.index("scope")
    assert memory.get("scope", ["other-writer"])["count"] == 1


def test_concurrent_cache_inserts_do_not_lose_vectors(tmp_path):
    memory = MemoryGraph(tmp_path)
    a, b = append(memory, "a"), append(memory, "b")
    barrier = threading.Barrier(2)
    errors = []
    def encode(texts):
        barrier.wait(timeout=5)
        return [basis(0) for _ in texts]
    def index(node):
        try:
            MemoryGraph(tmp_path, encode)._vectors("scope", [node])
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=index, args=(node,)) for node in (a, b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors and all(not thread.is_alive() for thread in threads)
    with sqlite3.connect(memory.database_path("scope")) as db:
        assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2


@pytest.mark.parametrize("bad", [basis(0)[:-1], [float("nan")] * DIMENSION, [float("inf")] * DIMENSION])
def test_invalid_embedding_not_cached(tmp_path, bad):
    memory = MemoryGraph(tmp_path, lambda texts: [bad for _ in texts])
    append(memory, "a")
    with pytest.raises(ValueError):
        memory.index("scope")
    assert memory.clear_index("scope")["removed_vectors"] == 0


def test_legacy_migration_preserves_all_and_only_once(tmp_path):
    from hashlib import sha256
    key = sha256((MODEL + "\0" + record_text({"summary": "a"})).encode()).hexdigest()
    graph = write_legacy(tmp_path, {"model": MODEL, "dimension": DIMENSION, "vectors": {key: basis(0)}})
    originals = {path.name: path.read_bytes() for path in (tmp_path / "scope").glob("*.json")}
    encoder = Encoder()
    memory = MemoryGraph(tmp_path, encoder)
    assert memory.export_snapshot("scope") == graph
    assert memory._vectors("scope", [graph["nodes"]["a"]]) == {"a": basis(0)}
    assert encoder.batches == []
    append(memory, "new")
    assert all((tmp_path / "scope" / name).read_bytes() == content for name, content in originals.items())
    # After migration the DB is authoritative, even if the legacy source changes.
    (tmp_path / "scope/graph.json").write_text("{broken")
    assert MemoryGraph(tmp_path).get("scope", ["new"])["count"] == 1
    assert memory.clear_index("scope")["removed_vectors"] == 1
    assert MemoryGraph(tmp_path).clear_index("scope")["removed_vectors"] == 0


@pytest.mark.parametrize("cache", ["{broken", {"model": "other", "dimension": 1, "vectors": {}}])
def test_disposable_bad_legacy_cache_does_not_block_graph(tmp_path, cache):
    graph = write_legacy(tmp_path, cache)
    with pytest.warns(RuntimeWarning, match="legacy index"):
        assert MemoryGraph(tmp_path).export_snapshot("scope") == graph


def test_migration_rolls_back_then_retries(tmp_path):
    graph = write_legacy(tmp_path)
    graph["edges"].append({"source": "a", "type": "bad", "target": "b"})
    path = tmp_path / "scope/graph.json"
    path.write_text(json.dumps(graph))
    with pytest.raises(ValueError):
        MemoryGraph(tmp_path).export_snapshot("scope")
    with sqlite3.connect(tmp_path / "scope" / DATABASE) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 0
    graph["edges"].pop()
    path.write_text(json.dumps(graph))
    assert MemoryGraph(tmp_path).export_snapshot("scope") == graph


def test_backup_and_export(tmp_path):
    memory = MemoryGraph(tmp_path / "root")
    append(memory, "a")
    encoder = Encoder()
    memory.encoder = encoder
    memory.index("scope")
    destination = tmp_path / "backup" / DATABASE
    memory.backup("scope", destination)
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with pytest.raises(FileExistsError):
        memory.backup("scope", destination)
    with pytest.raises(ValueError):
        memory.backup("new-scope", memory.database_path("new-scope"))


def test_backup_includes_committed_live_wal(tmp_path):
    memory = MemoryGraph(tmp_path / "root")
    append(memory, "a")
    # Keep a connection open: SQLite cannot remove the live WAL on last-close.
    live = sqlite3.connect(memory.database_path("scope"))
    try:
        live.execute("SELECT COUNT(*) FROM nodes").fetchone()
        append(memory, "b")
        assert Path(str(memory.database_path("scope")) + "-wal").stat().st_size > 0
        path = tmp_path / "backup.sqlite3"
        memory.backup("scope", path)
        with sqlite3.connect(path) as backup:
            assert backup.execute("SELECT id FROM nodes ORDER BY seq").fetchall() == [("a",), ("b",)]
    finally:
        live.close()


def test_maintenance_cli(tmp_path):
    tool = AGENTS.parent / "tools/memory_store.py"
    root = tmp_path / "memory"
    graph = write_legacy(root)
    base = [sys.executable, str(tool), "--root", str(root), "--scope", "scope"]
    migrated = subprocess.run([*base, "migrate"], capture_output=True, text=True, check=True)
    assert json.loads(migrated.stdout)["node_count"] == 2
    exported = tmp_path / "snapshot.json"
    subprocess.run([*base, "export", str(exported)], capture_output=True, check=True)
    assert json.loads(exported.read_text()) == graph
    backup = tmp_path / "backup.sqlite3"
    subprocess.run([*base, "backup", str(backup)], capture_output=True, check=True)
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
    failed = subprocess.run([*base, "export", str(exported)], capture_output=True)
    assert failed.returncode != 0 and json.loads(exported.read_text()) == graph
    result = subprocess.run([*base, "clear-index"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["removed_vectors"] == 0


def worker(task):
    root, identifier = task
    memory = MemoryGraph(root)
    for i in range(25):
        append(memory, f"worker-{identifier}-{i}")
    return True


def test_concurrent_first_open_migration_and_appends(tmp_path):
    write_legacy(tmp_path)
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(worker, [(str(tmp_path), identifier) for identifier in range(4)]))
    snapshot = MemoryGraph(tmp_path).export_snapshot("scope")
    assert len(snapshot["nodes"]) == 102 and len(snapshot["edges"]) == 1
    with sqlite3.connect(tmp_path / "scope" / DATABASE) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_reader_sees_consistent_snapshot_during_writer(tmp_path):
    memory = MemoryGraph(tmp_path)
    append(memory, "a")
    ready, finish = threading.Event(), threading.Event()
    errors = []
    def writer():
        try:
            with memory._connection("scope", write=True) as db:
                memory._add(db, "proof_steps", {"summary": "b"}, {"node_id": "b"})
                ready.set()
                assert finish.wait(5)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert ready.wait(5)
        assert list(MemoryGraph(tmp_path).export_snapshot("scope")["nodes"]) == ["a"]
    finally:
        finish.set()
        thread.join(5)
    assert not errors and not thread.is_alive()
    assert list(memory.export_snapshot("scope")["nodes"]) == ["a", "b"]


def test_uncommitted_process_exit_is_rolled_back(tmp_path):
    memory = MemoryGraph(tmp_path)
    append(memory, "saved")
    code = """
from memory_graph import MemoryGraph
import os, sys
m = MemoryGraph(sys.argv[1])
with m._connection('scope', write=True) as db:
    m._add(db, 'proof_steps', {'summary':'uncommitted'}, {'node_id':'uncommitted'})
    os._exit(23)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env={**os.environ, "PYTHONPATH": str(AGENTS)})
    assert result.returncode == 23
    assert list(MemoryGraph(tmp_path).export_snapshot("scope")["nodes"]) == ["saved"]


@pytest.mark.parametrize("scope", ["../escape", ".", "/tmp/outside-memory"])
def test_scope_boundary(tmp_path, scope):
    with pytest.raises(ValueError):
        MemoryGraph(tmp_path).init(scope)


def test_unknown_schema_rejected(tmp_path):
    memory = MemoryGraph(tmp_path)
    memory.init("scope")
    with sqlite3.connect(memory.database_path("scope")) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema version"):
        memory.get("scope", [])


def test_real_mcp_modules_paths_and_role_isolation(tmp_path):
    servers = {}
    for role in ("generation", "verification"):
        spec = util.spec_from_file_location(f"sqlite_test_{role}", AGENTS / role / "mcp/server.py")
        server = util.module_from_spec(spec)
        sys.modules[spec.name] = server
        spec.loader.exec_module(server)
        assert server.APP is not None
        server.MEMORY_ROOT = tmp_path / role
        server._memory_store = partial(MemoryGraph, server.MEMORY_ROOT, Encoder(), api_lock_path=tmp_path / "api.lock")
        servers[role] = server
    generation, verification = servers["generation"], servers["verification"]
    initialized = generation.memory_init("nested/problem")
    node = generation.memory_append("nested/problem", "proof_steps", {"summary": "full"})
    assert node["path"] == initialized["graph_path"] and node["path"].endswith(DATABASE)
    assert generation.memory_search("nested/problem", "full")["count"] == 1
    check = verification.memory_append("run", "failed_checks", {"summary": "gap"})
    assert check["path"].endswith(DATABASE)
    assert verification.memory_query("run", "failed_checks")["count"] == 1
    assert verification.memory_query("other", "failed_checks")["count"] == 0
