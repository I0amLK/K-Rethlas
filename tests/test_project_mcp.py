"""Actual local MCP stdio acceptance; no provider, network or legacy writes."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from rethlas_core.context_views import exact_ref
from rethlas_core.private_fs import canonical_json
from rethlas_core.project_mcp import _safe, build_mcp_app
from rethlas_core.project_service import ProjectService
from rethlas_core.research_records import candidate, relation
from rethlas_core.retrieval import RetrievalService
from rethlas_core.worker import WorkerUnavailable


ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"memory_init", "memory_append", "memory_search", "project_search", "branch_update",
         "get_revision", "expand", "research_run", "research_revise", "research_review", "research_status", "research_history"}
COMMAND_RECORDS = []


@asynccontextmanager
async def connected(target, error_path):
    if not isinstance(target, StdioServerParameters):
        async with create_connected_server_and_client_session(target) as session:
            yield session
        return
    with error_path.open("a", encoding="utf-8") as errors:
        async with stdio_client(target, errlog=errors) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                yield session


def load_legacy(role):
    path = ROOT / "agents" / role / "mcp/server.py"
    spec = importlib.util.spec_from_file_location("managed_mcp_" + role, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectMCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rethlas-mcp-test-")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.service = ProjectService.initialize(self.parent / "project", owner="mcp-test")
        self.credentials = self.parent / "credentials"
        self.credentials.mkdir(mode=0o700)
        self.grants = {role: self.service.issue(role=role) for role in
                       ("project_reader", "project_writer", "project_withdrawer")}
        self.config = {"version": "research-client-1", "root": str(self.service.root),
                       "reader": None, "writer": None, "indexer": None, "runner": None,
                       "embedding": {"mode": "lexical"}, "text_provider": None}
        for field, role in (("reader", "project_reader"), ("writer", "project_writer"), ("indexer", "project_writer")):
            grant = self.grants[role]
            path = self.credentials / (field + ".token")
            path.write_text(grant["token"], encoding="ascii")
            path.chmod(0o600)
            self.config[field] = {"run_id": grant["run_id"], "token_file": str(path)}
        self.config_path = self.parent / "client.json"
        self.write_config()
        assumption, assumption_r = candidate("math.a", "math.a.1", "Assumption", "Synthetic premise", "Assume x is positive. 正数前提。")
        claim, claim_r = candidate("math.c", "math.c.1", "Claim", "Synthetic conclusion", "Then x + 1 is positive. 正数加一仍为正数。")
        self.premise, self.claim = exact_ref(assumption_r), exact_ref(claim_r)
        self.commit("seed", {"entities": [assumption, claim], "revisions": [assumption_r, claim_r],
            "relations": [relation("depends_on", claim_r, assumption_r)], "evidence": []})
        self.index()

    def write_config(self):
        self.config_path.write_bytes(canonical_json(self.config))
        self.config_path.chmod(0o600)

    def commit(self, request_id, graph):
        writer = self.grants["project_writer"]
        with self.service._locked(writer["token"], writer["run_id"], "project.commit") as (_, state, _, _):
            head = state["accepted_graph"]["head"]
        return self.service.commit(writer["token"], {"version": 1, "request_id": request_id,
            "expected_head": head, **graph}, run_id=writer["run_id"])

    def index(self):
        writer = self.grants["project_writer"]
        return RetrievalService(self.service).build_index(writer["token"], run_id=writer["run_id"])

    def transport(self, *, role=None, config_path=None):
        if role is None:
            args = ["-I", "-B", "-m", "rethlas_core.project_mcp", "--config", str(config_path or self.config_path)]
        else:
            args = ["-I", "-B", str(ROOT / "agents" / role / "mcp/server.py"), "--managed-config", str(self.config_path)]
        return StdioServerParameters(command=sys.executable, args=args,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(self.parent),
                 "LOGNAME": "", "SHELL": "", "TERM": "", "USER": ""}, cwd=str(ROOT.parent))

    def client(self, target):
        return connected(target, self.parent / "stdio.stderr.log")

    def call(self, client, name, arguments, *, transport="stdio", provider_mode="lexical"):
        async def operation():
            result = await client.call_tool(name, arguments)
            value = result.structuredContent
            if not isinstance(value, dict):
                value = json.loads(result.content[0].text)
            COMMAND_RECORDS.append({"tool": name, "arguments": arguments, "result": value,
                                    "transport": transport, "provider_mode": provider_mode})
            return value
        return operation()

    def run_async(self, operation):
        return asyncio.run(asyncio.wait_for(operation, timeout=60))

    def test_real_stdio_tools_and_complete_mathematical_context(self):
        async def scenario():
            async with self.client(self.transport()) as client:
                tools = (await client.list_tools()).tools
                self.assertEqual({tool.name for tool in tools}, TOOLS)
                self.assertFalse({"approve_text", "approve_online", "research_approve", "issue", "rebuild_index"}
                                 & {tool.name for tool in tools})
                for tool in tools:
                    properties = tool.inputSchema.get("properties", {})
                    self.assertFalse({"root", "token", "token_file", "endpoint", "config", "key_file", "model", "provider"} & set(properties))
                initialized = await self.call(client, "memory_init", {"problem_id": "分析/一致连续"})
                self.assertEqual(initialized["problem_id"], "分析/一致连续")
                self.assertFalse(initialized["legacy_jsonl_writable"])
                found = await self.call(client, "project_search", {"query": "正数加一", "top_k": 5})
                self.assertEqual(found["retrieval_mode"], "lexical")
                self.assertTrue(found["results"])
                self.assertIn(self.claim, [item["ref"] for item in found["results"]])
                context = await self.call(client, "get_revision", {"ref": self.claim})
                ids = {r["revision_id"] for r in context["dto"]["revisions"]}
                self.assertEqual(ids, {self.premise["revision_id"], self.claim["revision_id"]})
                navigation = await self.call(client, "expand", {"refs": [self.claim], "depth": 1})
                self.assertEqual(len(navigation["revisions"]), 2)
                self.assertTrue(context["overlay"]["input_eligible"])
                self.assertTrue(all(r["assertion_state"] == "candidate" and r["validity"] == "needs_review"
                                    for r in context["dto"]["revisions"]))
        self.run_async(scenario())

    def test_stdio_append_unicode_events_source_and_exact_problem_filter(self):
        async def append():
            async with self.client(self.transport()) as client:
                for problem in ("中文/命题一", "中文/命题二"):
                    result = await self.call(client, "memory_append", {"problem_id": problem, "channel": "events",
                        "record": {"source": "https://example.invalid/public-synthetic", "text": "一致连续来源"}})
                    self.assertEqual(result["status"], "committed_candidate")
                    self.assertEqual(result["assertion_state"], "candidate")
                branch = await self.call(client, "branch_update", {"problem_id": "中文/命题一", "branch_id": "branch-a",
                    "state": {"progress": "candidate"}})
                self.assertEqual(branch["channel"], "branch_states")
        self.run_async(append())
        self.index()
        async def query():
            async with self.client(self.transport()) as client:
                result = await self.call(client, "memory_search", {"problem_id": "中文/命题一", "query": "一致连续来源", "channels": ["events"]})
                self.assertEqual(result["count"], 1)
                self.assertEqual(result["results"][0]["metadata"], {"problem_id": "中文/命题一", "channel": "events"})
                self.assertEqual(result["results"][0]["type"], "Reference")
                missing = await self.call(client, "memory_search", {"problem_id": "中文/不存在", "query": "一致连续来源"})
                self.assertEqual(missing["count"], 0)
        self.run_async(query())
        self.assertFalse(list(self.parent.rglob("*.jsonl")))

    def test_stdio_withdrawal_blocks_current_use_without_rebuilding_index(self):
        selected = self.grants["project_withdrawer"]
        self.service.withdraw(selected["token"], {"version": "project-1", "request_id": "withdraw-premise",
            "targets": [self.premise], "reason": "Synthetic negative case"}, run_id=selected["run_id"])
        async def scenario():
            async with self.client(self.transport()) as client:
                found = await self.call(client, "project_search", {"query": "正数", "top_k": 5})
                self.assertEqual(found["count"], 0)
                body = await self.call(client, "get_revision", {"ref": self.claim})
                self.assertEqual(body["status"], "rejected")
                self.assertNotIn("dto", body)
        self.run_async(scenario())

    def test_stdio_readonly_config_cannot_append_or_fall_back_to_jsonl(self):
        self.config["writer"] = self.config["indexer"] = None
        self.write_config()
        async def scenario():
            async with self.client(self.transport()) as client:
                result = await self.call(client, "memory_append", {"problem_id": "p", "channel": "events", "record": {"text": "candidate"}})
                self.assertEqual(result, {"status": "rejected", "error": "ValidityError"})
                research = await self.call(client, "research_run", {"request_id": "no-configuration", "question": "synthetic"})
                self.assertEqual(research["status"], "rejected")
                self.assertNotIn("root", json.dumps(research))
        self.run_async(scenario())
        self.assertFalse(list(self.parent.rglob("*.jsonl")))

    def test_legacy_builders_default_deny_and_explicit_managed_route(self):
        for role in ("generation", "verification"):
            module = load_legacy(role)
            with self.subTest(role=role), self.assertRaises(WorkerUnavailable):
                module.build_mcp_app()
            with patch("rethlas_core.project_mcp.build_mcp_app", return_value="managed-app") as builder:
                self.assertEqual(module.build_mcp_app(managed_config=self.config_path), "managed-app")
                builder.assert_called_once_with(self.config_path)

    def test_legacy_generation_and_verification_launch_real_managed_stdio(self):
        for role in ("generation", "verification"):
            async def scenario():
                async with self.client(self.transport(role=role)) as client:
                    tools = (await client.list_tools()).tools
                    self.assertEqual({tool.name for tool in tools}, TOOLS)
                    result = await self.call(client, "memory_init", {"problem_id": "启动/中文"})
                    self.assertEqual(result["problem_id"], "启动/中文")
            with self.subTest(role=role):
                self.run_async(scenario())

    def test_mcp_errors_do_not_echo_exception_content(self):
        result = _safe(lambda: (_ for _ in ()).throw(ValueError("private token and path")))
        self.assertEqual(result, {"status": "rejected", "error": "ValueError"})
        from rethlas_core.provider import ProviderTimeout
        uncertain = _safe(lambda: (_ for _ in ()).throw(ProviderTimeout("private provider context")))
        self.assertEqual(uncertain, {"status": "indeterminate", "error": "ProviderTimeout"})

    def test_stdio_stale_index_and_unknown_revision_fail_explicitly(self):
        artifact, artifact_r = candidate("math.extra", "math.extra.1", "ResearchArtifact", "Public synthetic note", "新增公开合成数学笔记。")
        self.commit("new-note", {"entities": [artifact], "revisions": [artifact_r], "relations": [], "evidence": []})
        async def stale():
            async with self.client(self.transport()) as client:
                result = await self.call(client, "project_search", {"query": "正数"})
                self.assertEqual(result, {"status": "rejected", "error": "RetrievalIndexStale"})
                missing = await self.call(client, "get_revision", {"ref": {**self.claim, "revision_id": "not-committed"}})
                self.assertEqual(missing["status"], "rejected")
                self.assertNotIn("dto", missing)
        self.run_async(stale())
        self.index()
        async def rebuilt():
            async with self.client(self.transport()) as client:
                result = await self.call(client, "project_search", {"query": "合成数学笔记"})
                self.assertIn(exact_ref(artifact_r), [row["ref"] for row in result["results"]])
        self.run_async(rebuilt())

    def test_real_stdio_mock_research_persists_restart_and_withdrawal_current(self):
        runner = self.service.issue(role="research_runner")
        token_path = self.credentials / "runner.token"
        token_path.write_text(runner["token"], encoding="ascii")
        token_path.chmod(0o600)
        self.config["runner"] = {"run_id": runner["run_id"], "token_file": str(token_path)}
        self.write_config()
        # The synthetic transport is fixed in a trusted test host, never selected
        # by production configuration or an MCP model request.
        driver = self.parent / "synthetic_research_stdio.py"
        driver.write_bytes((ROOT / "tests/project_mcp_mock_driver.py").read_bytes())
        parameters = self.transport()
        parameters.args = ["-I", "-B", str(driver), str(self.config_path)]
        request = {"request_id": "stdio-research", "question": "Then x + 1 is positive.", "selected_refs": [self.claim]}
        async def run():
            async with self.client(parameters) as client:
                result = await self.call(client, "research_run", request, provider_mode="mock")
                self.assertIn("request_status", result, result)
                self.assertEqual(result["request_status"], "accepted")
                self.assertEqual(result["historical"]["mode"], "mock")
                self.assertEqual(result["historical"]["verdict"], "wrong")
                self.assertFalse(result["historical"]["mathematical_correctness_proved"])
                return result
        original = self.run_async(run())
        async def restarted_status():
            async with self.client(self.transport()) as client:
                return await self.call(client, "research_status", {"request_id": "stdio-research"}, provider_mode="mock")
        restarted = self.run_async(restarted_status())
        self.assertEqual(restarted["historical"], original["historical"])
        proof = original["historical"]["proof"]
        context = self.service.get_revision(self.grants["project_reader"]["token"], proof,
            run_id=self.grants["project_reader"]["run_id"])
        draft = next(row for row in context["dto"]["revisions"] if row["revision_id"] == proof["revision_id"])
        self.assertEqual((draft["assertion_state"], draft["validity"]), ("candidate", "needs_review"))
        selected = self.grants["project_withdrawer"]
        self.service.withdraw(selected["token"], {"version": "project-1", "request_id": "withdraw-research-premise",
            "targets": [self.premise], "reason": "Synthetic current-input regression"}, run_id=selected["run_id"])
        withdrawn = self.run_async(restarted_status())
        self.assertEqual(withdrawn["historical"], original["historical"])
        self.assertTrue(withdrawn["current"]["current"][proof["revision_id"]]["math_blocked"])
        self.assertFalse(list(self.parent.rglob("*.jsonl")))

    def test_offline_stdio_tools_do_not_require_live_embedding_approval(self):
        self.config["embedding"] = {"mode": "bge_live", "endpoint": "https://example.invalid/v1/embeddings",
                                    "key_file": str(self.parent / "absent-private-secret")}
        self.write_config()
        async def scenario():
            async with self.client(self.transport()) as client:
                initialized = await self.call(client, "memory_init", {"problem_id": "离线/中文"})
                self.assertEqual(initialized["problem_id"], "离线/中文")
                body = await self.call(client, "get_revision", {"ref": self.claim})
                self.assertTrue(body["overlay"]["input_eligible"])
                found = await self.call(client, "project_search", {"query": "正数"})
                self.assertEqual(found, {"status": "rejected", "error": "FileNotFoundError"})
                self.assertNotIn("retrieval_mode", found)
        self.run_async(scenario())
        self.assertFalse((self.parent / "absent-private-secret").exists())

    def test_explicit_research_tools_return_pending_approval_without_self_approval(self):
        selected = {"run_id": "fixed-test-run", "token": "private-test-token"}
        cases = {
            "research_run": ("run", {"request_id": "new", "question": "public synthetic", "selected_refs": [self.claim]}),
            "research_revise": ("revise", {"request_id": "new", "parent_request_id": "parent", "instructions": "address the gap"}),
            "research_review": ("review", {"request_id": "new", "proof_ref": self.claim, "statement": "public synthetic"}),
        }
        for tool, (method, arguments) in cases.items():
            expected = {"request_status": "bound", "historical": {"stage": "awaiting_review_approval" if method == "review" else "awaiting_generation_approval"},
                "approval_required": {"purpose": "review" if method == "review" else "generation", "messages": [], "input_sha256": "0" * 64}}
            async def scenario():
                with patch("rethlas_core.project_mcp.grant", return_value=selected), \
                     patch("rethlas_core.project_mcp.configured_text", return_value="mock-configured-provider"), \
                     patch("rethlas_core.project_mcp.configured_embedding", return_value=None), \
                     patch("rethlas_core.research.ResearchPipeline") as pipeline:
                    getattr(pipeline.return_value, method).return_value = expected
                    async with self.client(build_mcp_app(self.config_path)) as client:
                        listed = {item.name for item in (await client.list_tools()).tools}
                        self.assertFalse(any("approv" in name or "issue" in name for name in listed))
                        result = await self.call(client, tool, arguments, transport="in_process_mock_routing")
                        self.assertEqual(result, expected)
                    getattr(pipeline.return_value, method).assert_called_once_with(selected, **arguments)
            with self.subTest(tool=tool):
                self.run_async(scenario())

    def test_status_does_not_construct_a_text_or_embedding_provider(self):
        selected = {"run_id": "fixed-test-run", "token": "private-test-token"}
        expected = {"request_status": "bound", "historical": {"stage": "awaiting_review_approval"}}
        async def scenario():
            with patch("rethlas_core.project_mcp.grant", return_value=selected), \
                 patch("rethlas_core.project_mcp.configured_text", side_effect=AssertionError("offline status")), \
                 patch("rethlas_core.project_mcp.configured_embedding", side_effect=AssertionError("offline status")), \
                 patch("rethlas_core.research.ResearchPipeline") as pipeline:
                pipeline.return_value.status.return_value = expected
                async with self.client(build_mcp_app(self.config_path)) as client:
                    self.assertEqual(await self.call(client, "research_status", {"request_id": "research-one"},
                                                     transport="in_process_mock_routing"), expected)
                self.assertIsNone(pipeline.call_args.args[2])
                pipeline.return_value.status.assert_called_once_with(selected, request_id="research-one")
        self.run_async(scenario())

    def test_parser_errors_do_not_echo_private_arguments(self):
        for arguments in (["--config", str(self.config_path), "--api-key", "private-value"], []):
            process = subprocess.run([sys.executable, "-I", "-B", "-m", "rethlas_core.project_mcp", *arguments],
                cwd=ROOT.parent, capture_output=True, timeout=20, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
            self.assertEqual(process.returncode, 2)
            self.assertEqual(process.stdout, b"")
            self.assertEqual(json.loads(process.stderr), {"status": "rejected", "error": "ValueError"})
            self.assertNotIn(b"private-value", process.stderr)

    def test_unavailable_config_cli_fails_closed_without_stdout(self):
        process = subprocess.run([sys.executable, "-I", "-B", "-m", "rethlas_core.project_mcp", "--config",
            str(self.parent / "absent-private-config")], cwd=ROOT.parent, capture_output=True, timeout=20,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, b"")
        self.assertEqual(json.loads(process.stderr), {"status": "rejected", "error": "FileNotFoundError"})
        self.assertNotIn(b"absent-private-config", process.stderr)


if __name__ == "__main__":
    unittest.main()
