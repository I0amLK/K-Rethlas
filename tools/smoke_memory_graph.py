#!/usr/bin/env python3
"""Offline Graph smoke using real MCP modules; function calls, not wire protocol.

Stored corpus vectors are fixture mappings, not new encodings of record JSON.
"""
import argparse
from functools import partial
from hashlib import sha256
from importlib import import_module, util
import json
from pathlib import Path
import sys
import sqlite3
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    workspace = parser.parse_args().workspace.resolve()
    repo = workspace / "Rethlas"
    fixture = workspace / "experiments" / "openviking"
    cache = fixture / "gateway" / "embedding-cache"
    for dependency in ("requests", "fastmcp", "jsonschema"):
        try:
            import_module(dependency)
        except ImportError as exc:
            raise SystemExit(f"Missing dependency {dependency}: {exc}; no install attempted.") from exc
    sys.path.insert(0, str(repo / "agents"))
    core = import_module("memory_graph")
    original_graph = core.MemoryGraph

    data = json.loads((fixture / "dataset.json").read_text(encoding="utf-8"))
    docs = {doc["id"]: doc for doc in data["documents"] if doc["id"] in ("d01", "d02")}
    bodies = {key: (fixture / doc["path"]).read_bytes().decode("utf-8") for key, doc in docs.items()}
    records = {key: {"body": text} for key, text in bodies.items()}
    query = next(q["text"] for q in data["queries"] if q["id"] == "q02")
    edge = next(e for e in data["edges"] if e["source"] == "d02" and e["target"] == "d01")
    relation = {"type": edge["relation"], "target_id": edge["target"]}
    assert relation == {"type": "requires", "target_id": "d01"}

    def cached_vector(text):
        key = sha256((core.MODEL + "\0" + text).encode("utf-8")).hexdigest()
        entry = json.loads((cache / (key + ".json")).read_text(encoding="utf-8"))
        assert (entry["model"] == core.MODEL and entry["input_sha256"] == key
                and entry["text"] == text), f"Exact-text fixture mismatch: {key}"
        assert len(entry["vector"]) == core.DIMENSION, f"Fixture dimension mismatch: {key}"
        return entry["vector"]

    vectors = {core.record_text(records[key]): cached_vector(text) for key, text in bodies.items()}
    vectors[query] = cached_vector(query)
    encoder_batches = []

    def fixture_encoder(texts):
        encoder_batches.append(len(texts))
        return [vectors[text] for text in texts]  # Unknown input raises KeyError; no API fallback.

    def load_server(role, memory_root):
        path = (repo / "agents" / role / "mcp" / "server.py").resolve()
        spec = util.spec_from_file_location(f"smoke_memory_{role}", path)
        server = util.module_from_spec(spec)
        sys.modules[spec.name] = server
        spec.loader.exec_module(server)
        if server.APP is None:
            raise SystemExit(f"{role}: FastMCP unavailable; real tools were not registered.")
        server.MemoryGraph = partial(original_graph, encoder=fixture_encoder)
        server.MEMORY_ROOT = memory_root / role
        return server

    with TemporaryDirectory(prefix="rethlas-memory-smoke-") as temporary:
        root = Path(temporary)
        generation = load_server("generation", root)
        verification = load_server("verification", root)
        problem, run = "smoke-compact-continuity", "smoke-statement-check"
        graph_path = Path(generation.memory_init(problem)["graph_path"])
        for identifier, channel in (("d01", "immediate_conclusions"), ("d02", "proof_steps")):
            doc = docs[identifier]
            metadata = {"node_id": identifier, "source": doc["path"], "type": doc["kind"],
                        "conditions": doc["assumptions"]}
            if identifier == "d02":
                metadata["relations"] = [relation]
            result = generation.memory_append(problem, channel, records[identifier], metadata=metadata)
            assert result["node_id"] == identifier and result["entry"]["record"] == records[identifier]
        assert encoder_batches == [], "Graph append must not encode"
        # q02's stored all-channel lead is d01; restrict semantic seeds to proofs explicitly.
        search = generation.memory_search(problem, query, channels=["proof_steps"],
                                          top_k=1, limit_per_channel=1)
        hits = search["results"]
        assert search["count"] == len(hits) == 2 and [hit["id"] for hit in hits] == ["d02", "d01"]
        assert hits[0]["include_reason"] == {"type": "semantic", "depth": 0}
        assert hits[0]["relations"] == [relation]
        assert hits[1]["include_reason"] == {"type": "graph", "depth": 1,
                                             "source_id": "d02", "relation": "requires"}
        full = generation.memory_get(problem, ["d02"], include_dependencies=True)
        nodes = full["nodes"] + full["direct_dependencies"]
        assert {node["id"]: node["content"] for node in nodes} == records
        assert full["count"] == 1 and len(full["direct_dependencies"]) == 1
        assert all(node["conditions"] == docs[node["id"]]["assumptions"] for node in nodes)
        assert all(node["source"] == docs[node["id"]]["path"] for node in nodes)
        assert graph_path.name == "memory.sqlite3" and graph_path.is_file()
        with sqlite3.connect(graph_path) as db:
            assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2
        disk_graph = generation._memory_store().export_snapshot(problem)
        assert len(disk_graph["nodes"]) == 2 and disk_graph["edges"] == [
            {"source": "d02", "type": "requires", "target": "d01"}]

        verification.memory_init(run)
        check = verification.memory_append(run, "statement_checks", records["d02"],
                                           metadata={"node_id": "statement-check"})
        queried = verification.memory_query(run, "statement_checks",
                                            filters={"channel": "statement_checks"},
                                            contains="sequential proof", limit=1, reverse=False)
        assert queried["count"] == 1 and queried["items"] == [check["entry"]]
        verified = verification.memory_get(run, ["statement-check"])
        assert verified["count"] == 1 and verified["nodes"][0]["content"] == records["d02"]
        assert verified["direct_dependencies"] == []

        generation = load_server("generation", root)
        verification = load_server("verification", root)
        assert generation.memory_get(problem, ["d02"], include_dependencies=True) == full
        assert verification.memory_get(run, ["statement-check"]) == verified
        assert encoder_batches == [2, 1], "Only two fixture records and the exact query should encode"
        print(json.dumps({
            "status": "ok",
            "counts": {"generation_nodes": len(disk_graph["nodes"]), "edges": len(disk_graph["edges"]),
                       "search_hits": search["count"], "verification_query_items": queried["count"],
                       "verification_nodes": verified["count"], "fixture_vectors": len(vectors)},
            "ids": {"generation": [node["id"] for node in nodes], "verification": check["node_id"]},
            "semantic_channels": ["proof_steps"],
            "search_to_graph": [{"id": hit["id"], "reason": hit["include_reason"]} for hit in hits],
            "full_text_equal": True,
            "reopen_equal": True,
            "reopen_mode": "fresh server modules and MemoryGraph instances; same process, not process restart",
            "mcp": "real FastMCP APP registration; function-level calls, not MCP wire protocol",
            "fixture_source": str(cache),
            "fixture_model": core.MODEL,
            "query_id": "q02",
            "query_cache_key": sha256((core.MODEL + "\0" + query).encode("utf-8")).hexdigest(),
            "fixture_mapping": "fixture映射，不验证新序列化输入编码；record_text({'body': 原文}) -> 原文已存真vector",
            "encoder_batches": encoder_batches,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
