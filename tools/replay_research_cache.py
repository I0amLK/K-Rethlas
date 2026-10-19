#!/usr/bin/env python3
"""Public pilot -> actual research CLI and MCP, strictly cache-only.

Run only after local execution has been authorized. This driver creates a new
private temporary project, never reads provider credentials, never starts the old
experiment gateway and never installs dependencies. The existing real BGE cache
is read-only. A new evidence directory is reserved before any app invocation.
The twenty reused pilot queries and relevance labels are frozen before ranking;
this is application cache replay, not a new live or held-out model benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-m3"
QUERY_IDS = ("q02", "q03", "q04", "q07", "q08", "q09", "q12", "q13", "q14", "q17",
             "q18", "q19", "q22", "q23", "q27", "q28", "q32", "q33", "q37", "q38")
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
KINDS = {"definition": "DefinitionContext", "theorem": "Claim",
         "counterexample": "Counterexample", "warning": "Failure"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def strict(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def constant(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def regular(path, maximum):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("input must not traverse symlinks")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
        raise ValueError("input must be a bounded single-link regular file")
    return path.read_bytes()


def write(path, value):
    with path.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(encoded(value) + b"\n")


def redact(raw):
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"eyJ[A-Za-z0-9_-]{77,}\.[A-Za-z0-9_-]{43}", "[REDACTED capability]", text)
    return re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)


def source_map():
    paths = [ROOT / "tools/research.py", Path(__file__).resolve()]
    paths += list((ROOT / "agents/shared/src/rethlas_core").rglob("*.py"))
    paths += list((ROOT / "agents/shared/src/rethlas_core/schemas").glob("*.json"))
    return {str(p.relative_to(ROOT)): sha(regular(p, 1048576)) for p in sorted(set(paths))}


def fixture(dataset_path, cache_path):
    raw = regular(dataset_path, 1048576)
    value = strict(raw)
    documents = value["documents"]
    if len(documents) != 32 or len({d["id"] for d in documents}) != 32:
        raise ValueError("expected the frozen public thirty-two-document pilot")
    queries = {q["id"]: q for q in value["queries"]}
    selected = [queries[key] for key in QUERY_IDS]
    if any(q["split"] != "test" for q in selected):
        raise ValueError("replay selection must use original test labels")
    texts, inputs, hashes = {}, [], {}
    for document in documents:
        relative = Path(document["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "corpus":
            raise ValueError("pilot document locator must remain inside corpus")
        body = regular(dataset_path.parent / relative, 2048)
        texts[document["id"]] = body.decode("utf-8")
        inputs.append(texts[document["id"]])
        hashes[str(relative)] = sha(body)
    inputs.extend(q["text"] for q in selected)
    cache_hashes = {}
    for text in inputs:
        key = sha((MODEL + "\0" + text).encode("utf-8"))
        cached_raw = regular(cache_path / (key + ".json"), 262144)
        cached = strict(cached_raw)
        if (set(cached) != {"model", "input_sha256", "text", "vector"}
                or cached["model"] != MODEL or cached["input_sha256"] != key or cached["text"] != text):
            raise ValueError("exact legacy BGE cache binding mismatch")
        vector = cached["vector"]
        if (type(vector) is not list or len(vector) != 1024
                or any(type(v) not in (int, float) for v in vector)
                or not all(math.isfinite(v) for v in vector)
                or not math.isfinite(math.hypot(*vector)) or math.hypot(*vector) == 0):
            raise ValueError("invalid real cache vector")
        cache_hashes[key + ".json"] = sha(cached_raw)
    frozen = {"version": "research-cache-replay-fixture-1", "recorded_at": now(),
        "dataset_sha256": sha(raw), "model": MODEL, "dimensions": 1024,
        "query_selection": "fixed IDs in driver; 16 assumption/counterexample and 4 cross-language queries",
        "labels_reused_from_previous_pilot": True, "new_heldout_benchmark": False,
        "documents": documents, "document_sha256": hashes, "queries": selected,
        "edges": value["edges"], "cache_sha256": cache_hashes,
        "external_provider_requests": 0, "mathematical_gold_certified": False}
    return frozen, texts


def graph(frozen, texts):
    result = {"entities": [], "revisions": [], "relations": [], "evidence": []}
    refs, revisions, types = {}, {}, {}
    for document in frozen["documents"]:
        key = document["id"]
        entity_id, revision_id = "pilot." + key, "pilot." + key + ".0"
        origin = {"source": "synthetic-public-pilot:" + document["path"],
                  "actor": "trusted-cache-replay-driver", "recorded_at": now()}
        evidence_id = "pilot.source." + key
        types[key] = KINDS[document["kind"]]
        result["entities"].append({"entity_id": entity_id, "type": types[key],
            "name": document["title"], "provenance": origin})
        revision = {"entity_id": entity_id, "revision_id": revision_id, "body": texts[key],
            "body_sha256": sha(texts[key].encode("utf-8")), "provenance": origin,
            "assertion_state": "candidate", "validity": "needs_review",
            "context_revision_ids": [], "evidence_ids": [evidence_id]}
        result["revisions"].append(revision)
        revisions[revision_id] = revision
        refs[key] = {k: revision[k] for k in ("entity_id", "revision_id", "body_sha256")}
        result["evidence"].append({"evidence_id": evidence_id,
            "locator": "public-pilot:" + document["path"], "sha256": revision["body_sha256"], "provenance": origin})
    for edge in frozen["edges"]:
        if edge["relation"] == "requires" and types[edge["target"]] == "DefinitionContext":
            kind = "in_context"
        elif edge["relation"] == "contrasts":
            kind = "contrasts"
        else:
            raise ValueError("pilot edge needs explicit typed mapping")
        result["relations"].append({"relation_id": "pilot." + edge["id"], "type": kind,
            "source": {k: refs[edge["source"]][k] for k in ("entity_id", "revision_id")},
            "target": {k: refs[edge["target"]][k] for k in ("entity_id", "revision_id")},
            "provenance": {"source": "synthetic-public-pilot:" + edge["id"],
                           "actor": "trusted-cache-replay-driver", "recorded_at": now()},
            "assertion_state": "candidate", "context_revision_ids": [], "evidence_ids": []})
    return result, refs, revisions


def measure(query, result, refs, revisions):
    if result.get("retrieval_mode") != "bge_cache_only" or result.get("fallback_reason") is not None:
        raise ValueError("cache replay must not silently become lexical")
    reverse = {ref["revision_id"]: key for key, ref in refs.items()}
    ranking, disclosed, sources = [], set(), 0
    for item in result["results"]:
        key = reverse[item["ref"]["revision_id"]]
        if item["ref"] != refs[key] or item["current"]["math_blocked"] or not item["context"]["overlay"]["input_eligible"]:
            raise ValueError("result failed authoritative current/ref checks")
        ranking.append(key)
        for revision in item["context"]["dto"]["revisions"]:
            expected = revisions[revision["revision_id"]]
            if revision["body"] != expected["body"] or revision["body_sha256"] != expected["body_sha256"]:
                raise ValueError("context must preserve exact public Markdown")
            disclosed.add(reverse[revision["revision_id"]])
        own = next(r for r in item["context"]["dto"]["revisions"] if r["revision_id"] == item["ref"]["revision_id"])
        evidence = {e["evidence_id"]: e for e in item["context"]["dto"]["evidence"]}
        if any(evidence[e]["sha256"] == item["ref"]["body_sha256"] for e in own["evidence_ids"]):
            sources += 1
    relevant = {key for key, grade in query["relevance"].items() if grade > 0}
    gain = lambda grades: sum((2 ** grade - 1) / math.log2(pos + 2) for pos, grade in enumerate(grades))
    ideal = gain(sorted(query["relevance"].values(), reverse=True)[:10])
    required = set(query["required_context"])
    return {"query_id": query["id"], "ranked_document_ids": ranking,
        "recall_at_10": len(set(ranking) & relevant) / len(relevant),
        "ndcg_at_10": gain([query["relevance"].get(key, 0) for key in ranking[:10]]) / ideal,
        "required_context_found": sorted(required & disclosed), "required_context": sorted(required),
        "context_coverage": len(required & disclosed) / len(required) if required else None,
        "exact_source_coverage": sources / len(ranking) if ranking else 0.0,
        "mathematical_correctness_proved": False}


class Replay:
    def __init__(self, evidence, private):
        self.evidence, self.private = evidence, private
        self.records, self.counter, self.checks = [], 0, []
        self.config = private / "client.json"

    def cli(self, command, *arguments, expected=0):
        self.counter += 1
        started = time.monotonic()
        process = subprocess.run([sys.executable, "-I", "-B", "-m", "rethlas_core.research_cli", command, *map(str, arguments)],
            cwd=self.private, stdin=subprocess.DEVNULL, capture_output=True, timeout=60, env=ENV)
        stem = f"cli-{self.counter:03d}-{command}"
        (self.evidence / (stem + ".stdout.log")).write_text(redact(process.stdout), encoding="utf-8")
        (self.evidence / (stem + ".stderr.log")).write_text(redact(process.stderr), encoding="utf-8")
        value = strict(process.stdout)
        self.records.append({"entrypoint": "research_cli", "command": command,
            "arguments": [str(a).replace(str(self.private), "<private>") for a in arguments],
            "elapsed_seconds": time.monotonic() - started, "exit_code": process.returncode,
            "expected_exit_code": expected, "stdout": stem + ".stdout.log", "stderr": stem + ".stderr.log"})
        if process.returncode != expected or process.stderr:
            raise ValueError("actual CLI result differs from expected protocol")
        return value

    def request(self, value):
        path = self.private / ("input-" + sha(encoded(value)) + ".json")
        if not path.exists():
            write(path, value)
        return path

    def invoke(self, name, value=None, *, expected=0, **arguments):
        options = ["--config", self.config]
        if value is not None:
            options += ["--request", self.request(value)]
        for key, val in arguments.items():
            options += ["--" + key.replace("_", "-"), val]
        return self.cli(name, *options, expected=expected)

    async def mcp(self, queries, refs, revisions, *, withdrawn=False):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=sys.executable,
            args=["-I", "-B", "-m", "rethlas_core.project_mcp", "--config", str(self.config)],
            cwd=str(self.private), env={**ENV, "HOME": str(self.private)})
        suffix = "after-withdrawal" if withdrawn else "before-withdrawal"
        rows = []
        with (self.evidence / ("mcp-" + suffix + ".stderr.log")).open("x", encoding="utf-8") as errors:
            async with stdio_client(params, errlog=errors) as (read, send):
                async with ClientSession(read, send, read_timeout_seconds=timedelta(seconds=60)) as client:
                    await client.initialize()
                    async def call(name, arguments):
                        started = time.monotonic()
                        result = await client.call_tool(name, arguments)
                        value = result.structuredContent
                        if not isinstance(value, dict):
                            value = strict(result.content[0].text.encode("utf-8"))
                        number = len(self.records) + 1
                        output = f"mcp-{number:03d}-{name}.json"
                        write(self.evidence / output, value)
                        elapsed = time.monotonic() - started
                        self.records.append({"entrypoint": "project_mcp", "tool": name, "arguments": arguments,
                            "elapsed_seconds": elapsed, "result": output, "phase": suffix})
                        return value, elapsed
                    for query in queries:
                        result, elapsed = await call("project_search", {"query": query["text"], "top_k": 10})
                        if withdrawn:
                            if any(hit["ref"] in [refs[key] for key in ("d01", "d02", "d03", "d04")] for hit in result["results"]):
                                raise ValueError("withdrawn mathematical context was recommended")
                        else:
                            rows.append({**measure(query, result, refs, revisions), "elapsed_seconds": elapsed})
                    context, _ = await call("get_revision", {"ref": refs["d02"]})
                    if withdrawn:
                        if context.get("status") != "rejected" or "dto" in context:
                            raise ValueError("withdrawn support must not remain a current context")
                    elif not {refs["d01"]["revision_id"], refs["d02"]["revision_id"]} <= {
                            r["revision_id"] for r in context["dto"]["revisions"]}:
                        raise ValueError("MCP omitted required compactness definition context")
        return rows


def run(driver, frozen, texts, cache):
    data, refs, revisions = graph(frozen, texts)
    root, credentials = driver.private / "project", driver.private / "credentials"
    credentials.mkdir(mode=0o700)
    driver.cli("init", "--root", root, "--owner", "cache-replay")
    config = {"version": "research-client-1", "root": str(root), "reader": None, "writer": None,
              "indexer": None, "runner": None, "embedding": {"mode": "bge_cache_only", "cache_dir": str(cache)},
              "text_provider": None}
    for field, role in (("reader", "project_reader"), ("writer", "project_writer"), ("runner", "research_runner")):
        token_file = credentials / (field + ".token")
        issued = driver.cli("issue", "--root", root, "--role", role, "--token-file", token_file)
        config[field] = {"run_id": issued["run_id"], "token_file": str(token_file)}
    config["indexer"] = dict(config["writer"])
    write(driver.config, config)
    status = driver.invoke("status")
    driver.invoke("commit", {"version": 1, "request_id": "public-pilot-seed",
                            "expected_head": status["binding"]["graph"]["head"], **data})
    index = driver.invoke("index")
    if index["mode"] != "bge_cache_only" or index["cache_miss_lexical_fallback"] or index["chunks"] != 32:
        raise ValueError("all exact original corpus inputs must use real cached vectors")
    cli_rows = []
    for query in frozen["queries"]:
        result = driver.invoke("search", query=query["text"], top_k=10)
        cli_rows.append({**measure(query, result, refs, revisions), "elapsed_seconds": driver.records[-1]["elapsed_seconds"]})
    mcp_rows = asyncio.run(asyncio.wait_for(driver.mcp(frozen["queries"], refs, revisions), timeout=240))
    if [r["ranked_document_ids"] for r in cli_rows] != [r["ranked_document_ids"] for r in mcp_rows]:
        raise ValueError("CLI and MCP must share exact authorized BGE ranking")
    driver.checks.append("twenty_actual_cli_and_mcp_rankings_equal")
    context = driver.invoke("get", refs["d02"])
    if refs["d01"]["revision_id"] not in {r["revision_id"] for r in context["dto"]["revisions"]}:
        raise ValueError("CLI omitted mathematical definition context")
    driver.invoke("expand", {"refs": [refs["d02"]], "relation_types": ["in_context", "contrasts"], "depth": 1})
    checkpoint = driver.invoke("status")["binding"]
    index_path = root / "index/current.json"
    regular(index_path, 33554432)
    index_path.unlink()  # Only this newly created disposable private index.
    driver.invoke("search", expected=2, query=frozen["queries"][0]["text"])
    driver.invoke("index")
    if driver.invoke("status")["binding"] != checkpoint:
        raise ValueError("deleting and rebuilding index changed graph authority")
    replay = driver.invoke("search", query=frozen["queries"][0]["text"])
    if measure(frozen["queries"][0], replay, refs, revisions)["ranked_document_ids"] != cli_rows[0]["ranked_document_ids"]:
        raise ValueError("index rebuild changed exact cache ranking")
    driver.checks.append("index_delete_rebuild_preserves_graph_and_ranking")
    bodies = {p.name: sha(regular(p, 131072)) for p in (root / "memory/bodies").iterdir()}
    driver.invoke("withdraw", {"version": "project-1", "request_id": "withdraw-public-context",
                              "targets": [refs["d01"]], "reason": "Synthetic withdrawal application replay"})
    current = driver.invoke("status")["current"]
    if not all(current[refs[key]["revision_id"]]["math_blocked"] for key in ("d01", "d02", "d03", "d04")):
        raise ValueError("withdrawal did not propagate through exact mathematical context")
    driver.invoke("get", refs["d02"], expected=2)
    driver.invoke("get", refs["d05"])
    result = driver.invoke("search", query=frozen["queries"][0]["text"], top_k=10)
    if any(hit["ref"] in [refs[key] for key in ("d01", "d02", "d03", "d04")] for hit in result["results"]):
        raise ValueError("CLI recommended withdrawn mathematical support")
    asyncio.run(asyncio.wait_for(driver.mcp(frozen["queries"][:1], refs, revisions, withdrawn=True), timeout=60))
    if bodies != {p.name: sha(regular(p, 131072)) for p in (root / "memory/bodies").iterdir()}:
        raise ValueError("withdrawal rewrote historical Markdown")
    driver.checks.append("withdrawal_blocks_current_cli_mcp_and_preserves_history")
    def metrics(rows):
        times = sorted(row["elapsed_seconds"] for row in rows)
        contexts = [row["context_coverage"] for row in rows if row["context_coverage"] is not None]
        return {"recall_at_10": sum(r["recall_at_10"] for r in rows) / len(rows),
                "ndcg_at_10": sum(r["ndcg_at_10"] for r in rows) / len(rows),
                "required_context_coverage": sum(contexts) / len(contexts) if contexts else None,
                "exact_source_coverage": sum(r["exact_source_coverage"] for r in rows) / len(rows),
                "latency_p50_seconds": times[math.ceil(len(times) * 0.50) - 1],
                "latency_p95_seconds": times[math.ceil(len(times) * 0.95) - 1]}
    return {"index": index, "cli_queries": cli_rows, "mcp_queries": mcp_rows,
        "metrics": {"cli": metrics(cli_rows), "mcp": metrics(mcp_rows),
                    "timing_scope": "CLI includes interpreter/startup; MCP includes tool roundtrip after initialization; no remote latency"},
        "query_count_per_entrypoint": len(cli_rows), "document_count": len(texts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.absolute()
    if any(p.is_symlink() for p in (evidence, *evidence.parents)):
        raise ValueError("evidence path must not traverse symlinks")
    evidence.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = {"version": "research-cache-replay-1", "started_at": now(), "result": "failed",
              "mode": "bge_cache_only", "external_provider_requests": 0, "new_live_model_requests": 0,
              "mathematical_correctness_proved": False, "human_review": "pending",
              "final_astra_acceptance": "pending", "records": [], "checks": []}
    before = source_map()
    write(evidence / "source-before.json", before)
    driver = None
    try:
        frozen, texts = fixture(args.dataset.absolute(), args.cache_dir.absolute())
        write(evidence / "frozen-fixture.json", frozen)
        report["fixture_sha256"] = sha(regular(evidence / "frozen-fixture.json", 1048576))
        with tempfile.TemporaryDirectory(prefix="rethlas-cache-app-replay-") as temporary:
            private = Path(temporary)
            private.chmod(0o700)
            driver = Replay(evidence, private)
            report.update(run(driver, frozen, texts, args.cache_dir.absolute()))
        report["result"] = "passed"
    except Exception as exc:
        report["error"] = type(exc).__name__
    finally:
        if driver is not None:
            report.update(records=driver.records, checks=driver.checks)
        after = source_map()
        write(evidence / "source-after.json", after)
        report["changed_source"] = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        report["warning_lines"] = sum("Warning:" in line for path in evidence.glob("*.log")
                                      for line in path.read_text(encoding="utf-8").splitlines())
        if report["changed_source"] or report["warning_lines"]:
            report["result"] = "failed"
        report["finished_at"] = now()
        write(evidence / "report.json", report)
    print(encoded({"result": report["result"], "report": str(evidence / "report.json")}).decode("utf-8"))
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
