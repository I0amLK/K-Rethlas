"""M1 policy/math wire examples and identity/review confusion regressions."""
from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from importlib.resources import files
from unittest.mock import patch

from jsonschema import Draft202012Validator, FormatChecker

from rethlas_core.contracts import validate_math_memory, validate_policy


PROVENANCE = {"source": "fixture:contracts", "actor": "fixture-author", "recorded_at": "2026-10-16T00:00:00Z"}
MINIMAL_POLICY = {
    "schema_version": "1.0", "default": "deny", "max_model_concurrency": 3,
    "nested_agents": False, "network": "disabled",
    "rules": [{"role": "generation", "action": "verify", "domains": ["fixture"]}],
}


def digest(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def canonical_relation_digest(graph, relation):
    revisions = {item["revision_id"]: item for item in graph["revisions"]}
    pending = [relation["source"]["revision_id"], relation["target"]["revision_id"], *relation["context_revision_ids"]]
    snapshots = {}
    evidence_ids = set(relation["evidence_ids"])
    while pending:
        revision_id = pending.pop()
        if revision_id in snapshots:
            continue
        revision = revisions[revision_id]
        snapshots[revision_id] = {key: value for key, value in revision.items() if key != "review_bindings"}
        pending.extend(revision["context_revision_ids"])
        evidence_ids.update(revision["evidence_ids"])
    evidence = {item["evidence_id"]: item for item in graph["evidence"] if item["evidence_id"] in evidence_ids}
    binding = {"relation": relation, "revisions": snapshots, "evidence": evidence}
    return digest(json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def add_entity(graph, entity_id, entity_type, name, body, **revision_fields):
    revision_id = entity_id + ":r1"
    graph["entities"].append({"entity_id": entity_id, "type": entity_type, "name": name, "provenance": deepcopy(PROVENANCE)})
    revision = {
        "entity_id": entity_id, "revision_id": revision_id, "body": body, "body_sha256": digest(body),
        "provenance": deepcopy(PROVENANCE), "assertion_state": "candidate", "validity": "needs_review",
        "context_revision_ids": [], "evidence_ids": [],
    }
    revision.update(revision_fields)
    graph["revisions"].append(revision)
    return {"entity_id": entity_id, "revision_id": revision_id}


def add_relation(graph, relation_id, relation_type, source, target, **fields):
    relation = {
        "relation_id": relation_id, "type": relation_type, "source": deepcopy(source), "target": deepcopy(target),
        "provenance": deepcopy(PROVENANCE), "assertion_state": "candidate", "context_revision_ids": [], "evidence_ids": [],
    }
    relation.update(fields)
    graph["relations"].append(relation)
    return relation


def minimal_math_memory():
    """Nonempty legal interchange example; locator is metadata, not a store path."""
    graph = {"schema_version": "1.0", "entities": [], "revisions": [], "relations": [], "evidence": [
        {"evidence_id": "ev:1", "locator": "fixture:note#p1", "sha256": digest("原始证据"), "provenance": deepcopy(PROVENANCE)}
    ]}
    add_entity(graph, "claim:1", "Claim", "中文命题", "# 中文命题\n对所有 n ∈ ℕ，n + 0 = n。", evidence_ids=["ev:1"])
    return graph


# Similar/same display names never establish identity, scope or equivalence.
ANTI_CONFUSION_FIXTURES = (
    ("紧致 / compact", "拓扑 / topology", "每个开覆盖有有限子覆盖。", "逻辑 / logic", "Every finitely satisfiable set of sentences is satisfiable."),
    ("正规 / normal", "群论 / group theory", "gHg^{-1} = H for every g ∈ G.", "概率 / probability", "X has density exp(-x²/2)/√(2π)."),
    ("有界 / bounded", "逐点 / pointwise", "∀x ∃M: |f(x)| ≤ M", "一致 / uniform", "∃M ∀x: |f(x)| ≤ M"),
)


def reviewed_graph():
    graph = minimal_math_memory()
    target = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
    reviewer = add_entity(graph, "review:1", "Review", "人工审查 / human review", "逐项核对定义与量词。",
                          assertion_state="reviewed", review_kind="human_review", evidence_ids=["ev:1"])
    relation = add_relation(graph, "rel:review", "reviews", reviewer, target,
                            assertion_state="reviewed", evidence_ids=["ev:1"])
    graph["revisions"][0]["assertion_state"] = "reviewed"
    graph["revisions"][1]["review_bindings"] = [{"relation_id": "rel:review", "sha256": canonical_relation_digest(graph, relation)}]
    return graph


class PolicyTests(unittest.TestCase):
    def test_minimal_policy_and_all_allowed_actions(self):
        self.assertEqual(validate_policy(deepcopy(MINIMAL_POLICY)), {"valid": True, "errors": []})
        for role, actions in {"generation": ["verify", "status"], "verification": ["finding.append", "report.submit", "status"], "publisher": ["publish", "status"], "memory": ["memory.read", "memory.commit", "memory.maintain"]}.items():
            for action in actions:
                policy = deepcopy(MINIMAL_POLICY)
                policy["rules"] = [{"role": role, "action": action, "domains": ["fixture", "production"]}]
                self.assertTrue(validate_policy(policy)["valid"])

    def test_memory_extension_is_finite_and_policy_authority_matrices_match(self):
        from rethlas_core.authorization import ROLE_ACTIONS

        schema = json.loads(files("rethlas_core").joinpath("schemas", "policy.schema.json").read_text(encoding="utf-8"))
        properties = schema["properties"]["rules"]["items"]["properties"]
        all_actions = set().union(*ROLE_ACTIONS.values())
        self.assertEqual(set(properties["role"]["enum"]), set(ROLE_ACTIONS))
        self.assertEqual(set(properties["action"]["enum"]), all_actions)
        for role, allowed in ROLE_ACTIONS.items():
            for action in all_actions | {"memory.unknown", "unknown.action"}:
                policy = deepcopy(MINIMAL_POLICY)
                policy["rules"] = [{"role": role, "action": action, "domains": ["fixture"]}]
                with self.subTest(role=role, action=action):
                    self.assertEqual(validate_policy(policy)["valid"], action in allowed)
        for role in ("unknown", "Memory", "memory.admin"):
            policy = deepcopy(MINIMAL_POLICY)
            policy["rules"][0].update(role=role, action="memory.read")
            with self.subTest(role=role):
                self.assertFalse(validate_policy(policy)["valid"])
        for field in ("resource_scope", "scope", "authority", "unknown"):
            for rule_field in (False, True):
                policy = deepcopy(MINIMAL_POLICY)
                policy["rules"][0].update(role="memory", action="memory.read")
                (policy["rules"][0] if rule_field else policy)[field] = "not allowed"
                with self.subTest(field=field, rule_field=rule_field):
                    self.assertFalse(validate_policy(policy)["valid"])

    def test_closed_fields_and_permissions(self):
        for path, value in [("default", "allow"), ("nested_agents", True), ("network", "enabled"), ("max_model_concurrency", 4), ("max_model_concurrency", 0), ("max_model_concurrency", True), ("authority", "forged")]:
            with self.subTest(path=path, value=value):
                policy = deepcopy(MINIMAL_POLICY)
                policy[path] = value
                self.assertFalse(validate_policy(policy)["valid"])
        for role, action in [("generation", "publish"), ("verification", "verify"), ("publisher", "report.submit"), ("publisher", "approval.issue")]:
            policy = deepcopy(MINIMAL_POLICY)
            policy["rules"][0].update(role=role, action=action)
            self.assertFalse(validate_policy(policy)["valid"])
        for domains in [[], ["fixture", "fixture"], ["unknown"]]:
            policy = deepcopy(MINIMAL_POLICY)
            policy["rules"][0]["domains"] = domains
            self.assertFalse(validate_policy(policy)["valid"])


class MathMemoryTests(unittest.TestCase):
    def assert_valid(self, graph):
        result = validate_math_memory(graph)
        self.assertTrue(result["valid"], result["errors"])

    def assert_invalid(self, graph, contains):
        result = validate_math_memory(graph)
        self.assertFalse(result["valid"])
        self.assertTrue(any(contains in error for error in result["errors"]), result["errors"])

    def test_resources_use_draft_202012(self):
        for filename in ["policy.schema.json", "math_memory.schema.json", "verification_output.schema.json"]:
            schema = json.loads(files("rethlas_core").joinpath("schemas", filename).read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            Draft202012Validator.check_schema(schema)

    def test_minimal_example_and_no_mutation(self):
        graph = minimal_math_memory()
        before = deepcopy(graph)
        self.assert_valid(graph)
        self.assertEqual(graph, before)
        self.assertEqual(graph["entities"][0]["name"], "中文命题")

    def test_three_bilingual_anti_confusion_fixtures(self):
        for index, (name, scope_a, body_a, scope_b, body_b) in enumerate(ANTI_CONFUSION_FIXTURES):
            with self.subTest(name=name):
                graph = minimal_math_memory()
                a = add_entity(graph, f"ctx:{index}:a", "DefinitionContext", scope_a, body_a)
                b = add_entity(graph, f"ctx:{index}:b", "DefinitionContext", scope_b, body_b)
                left = add_entity(graph, f"def:{index}:a", "Claim", name, body_a, context_revision_ids=[a["revision_id"]])
                right = add_entity(graph, f"def:{index}:b", "Claim", name, body_b, context_revision_ids=[b["revision_id"]])
                add_relation(graph, "contrast", "contrasts", left, right)
                self.assert_valid(graph)
                self.assertNotEqual(left, right)
                self.assertEqual([e["name"] for e in graph["entities"][-2:]], [name, name])
                self.assertNotEqual(graph["revisions"][-1]["body_sha256"], graph["revisions"][-2]["body_sha256"])

    def test_multiple_revisions_preserve_entity_identity(self):
        graph = minimal_math_memory()
        revision = deepcopy(graph["revisions"][0])
        revision.update(revision_id="claim:1:r2", body="修订后的不同公式")
        revision["body_sha256"] = digest(revision["body"])
        graph["revisions"].append(revision)
        self.assert_valid(graph)
        graph["revisions"][-1]["revision_id"] = "claim:1"
        self.assert_invalid(graph, "separate identities")

    def test_unknown_fields_types_missing_provenance_and_hashes(self):
        for section in [None, "entities", "revisions", "relations", "evidence"]:
            graph = minimal_math_memory()
            endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
            add_relation(graph, "citation", "cites", endpoint, endpoint)
            item = graph if section is None else graph[section][0]
            item["unexpected"] = True
            self.assert_invalid(graph, "Additional properties")
        for field in ["provenance", "body_sha256"]:
            graph = minimal_math_memory()
            del graph["revisions"][0][field]
            self.assert_invalid(graph, field)
        for field in ["locator", "sha256"]:
            graph = minimal_math_memory()
            del graph["evidence"][0][field]
            self.assert_invalid(graph, field)
        graph = minimal_math_memory()
        graph["entities"][0]["type"] = "AnalysisNode"
        self.assert_invalid(graph, "schema error")
        graph = minimal_math_memory()
        graph["revisions"][0]["body"] += " "
        self.assert_invalid(graph, "body_sha256")
        graph = minimal_math_memory()
        graph["entities"][0]["provenance"]["recorded_at"] = "yesterday"
        self.assert_invalid(graph, "date-time")

    def test_provenance_date_time_accepts_explicit_rfc3339_subset(self):
        timestamps = [
            PROVENANCE["recorded_at"],
            "2024-02-29T23:59:59Z",
            "2000-02-29t00:00:00z",
            "2026-10-16T12:34:56+08:00",
            "2026-10-16T12:34:56-05:30",
            "2026-10-16T12:34:56+00:00",
            "2026-10-16T12:34:56-00:00",  # RFC 3339 unknown local offset.
            "2026-10-16T12:34:56.0Z",
            "2026-10-16T12:34:56.123456789+05:45",
            "2026-10-16T12:34:56.000000000000000001-23:59",
            "0001-01-01T00:00:00+23:59",
            "9999-12-31T23:59:59-23:59",
        ]
        for timestamp in timestamps:
            with self.subTest(timestamp=timestamp):
                graph = minimal_math_memory()
                endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
                add_relation(graph, "citation", "cites", endpoint, endpoint)
                for section in ["entities", "revisions", "relations", "evidence"]:
                    graph[section][0]["provenance"]["recorded_at"] = timestamp
                before = deepcopy(graph)
                self.assert_valid(graph)
                self.assertEqual(graph, before)

    def test_provenance_date_time_rejects_malformed_or_impossible_values(self):
        timestamps = [
            "yesterday", "", "2026-10-16", "2026-10-16T00:00:00",
            "2026-02-29T00:00:00Z", "1900-02-29T00:00:00Z",
            "2024-02-30T00:00:00Z", "2026-04-31T00:00:00Z",
            "2026-00-16T00:00:00Z", "2026-13-16T00:00:00Z",
            "2026-10-00T00:00:00Z", "2026-10-32T00:00:00Z",
            "0000-01-01T00:00:00Z", "10000-01-01T00:00:00Z",
            "2026-10-16T24:00:00Z", "2026-10-16T12:60:00Z",
            "2026-10-16T12:00:60Z", "2016-12-31T23:59:60Z",
            "2026-10-16T12:34:56+24:00", "2026-10-16T12:34:56-24:00",
            "2026-10-16T12:34:56+00:60", "2026-10-16T12:34:56-00:60",
            "2026-10-16T12:34:56+08", "2026-10-16T12:34:56+0800",
            "2026-10-16T12:34:56+08:00:00", "2026-10-16T12:34:56+8:00",
            "2026-10-16T12:34:56.Z", "2026-10-16T12:34:56,5Z",
            "2026-10-16 12:34:56Z", "20261016T123456Z",
            "2026-W42-5T12:34:56Z", "2026-289T12:34:56Z",
            "2026-1-16T12:34:56Z", "2026-10-16T1:34:56Z",
            "２０２６-10-16T12:34:56Z", "2026-10-16T12:34:56.５Z",
            " 2026-10-16T12:34:56Z", "2026-10-16T12:34:56Z ",
            "2026-10-16T12:34:56Z\n", "2026-10-16T12:34:56Zjunk",
        ]
        # Check every provenance-bearing schema location independently, not only
        # the first entity that originally exposed the missing format checker.
        for section in ["entities", "revisions", "relations", "evidence"]:
            for timestamp in timestamps:
                with self.subTest(section=section, timestamp=timestamp):
                    graph = minimal_math_memory()
                    endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
                    add_relation(graph, "citation", "cites", endpoint, endpoint)
                    graph[section][0]["provenance"]["recorded_at"] = timestamp
                    self.assert_invalid(graph, "date-time")
            for value in [None, True, 20261016, [], {}]:
                with self.subTest(section=section, value=value):
                    graph = minimal_math_memory()
                    endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
                    add_relation(graph, "citation", "cites", endpoint, endpoint)
                    graph[section][0]["provenance"]["recorded_at"] = value
                    self.assert_invalid(graph, "not of type 'string'")

    def test_date_time_checker_is_independent_of_optional_global_checker(self):
        from rethlas_core.contracts import _validator

        original = FormatChecker.checkers.copy()
        for checker in [None, (lambda value: True, ())]:
            checkers = {} if checker is None else {"date-time": checker}
            with self.subTest(checker="absent" if checker is None else "permissive"):
                with patch.dict(FormatChecker.checkers, checkers, clear=True):
                    _validator.cache_clear()
                    graph = minimal_math_memory()
                    self.assert_valid(graph)
                    graph["entities"][0]["provenance"]["recorded_at"] = "yesterday"
                    self.assert_invalid(graph, "date-time")
                    self.assertEqual(FormatChecker.checkers, checkers)
        self.assertEqual(FormatChecker.checkers, original)

    def test_dangling_exact_revision_context_and_evidence(self):
        graph = minimal_math_memory()
        endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
        relation = add_relation(graph, "citation", "cites", endpoint, {"entity_id": "claim:1", "revision_id": "missing"})
        self.assert_invalid(graph, "exact existing")
        relation["target"] = {"entity_id": "other", "revision_id": "claim:1:r1"}
        self.assert_invalid(graph, "exact existing")
        for field, value, message in [("context_revision_ids", ["missing"], "dangling context"), ("evidence_ids", ["missing"], "dangling evidence")]:
            graph = minimal_math_memory()
            graph["revisions"][0][field] = value
            self.assert_invalid(graph, message)
        graph = minimal_math_memory()
        graph["revisions"].append(deepcopy(graph["revisions"][0]))
        self.assert_invalid(graph, "duplicate revision_id")
        for section, key in [("entities", "entity_id"), ("evidence", "evidence_id")]:
            graph = minimal_math_memory()
            graph[section].append(deepcopy(graph[section][0]))
            self.assert_invalid(graph, "duplicate " + key)
        graph = minimal_math_memory()
        endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
        add_relation(graph, "citation", "cites", endpoint, endpoint)
        graph["relations"].append(deepcopy(graph["relations"][0]))
        self.assert_invalid(graph, "duplicate relation_id")
        graph = minimal_math_memory()
        graph["revisions"][0]["entity_id"] = "missing"
        self.assert_invalid(graph, "unknown entity_id")

    def test_only_mathematical_dependency_subgraph_is_dag(self):
        for kind in ["depends_on", "cites", "contrasts", "derived_from", "has_subgoal"]:
            graph = minimal_math_memory()
            a = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
            b = add_entity(graph, "claim:2", "Claim", "另一命题", "B")
            add_relation(graph, "ab", kind, a, b)
            add_relation(graph, "ba", kind, b, a)
            if kind == "depends_on":
                self.assert_invalid(graph, "DAG")
            else:
                self.assert_valid(graph)
        for kind, entity_type in [("assumes", "Assumption"), ("in_context", "DefinitionContext")]:
            graph = minimal_math_memory()
            a = add_entity(graph, "premise:a", entity_type, "前提甲", "A")
            b = add_entity(graph, "premise:b", entity_type, "前提乙", "B")
            add_relation(graph, "ab", kind, a, b)
            add_relation(graph, "ba", kind, b, a)
            self.assert_invalid(graph, "DAG")
        graph = minimal_math_memory()
        endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
        add_relation(graph, "self", "depends_on", endpoint, endpoint)
        self.assert_invalid(graph, "DAG")

    def test_review_requires_exact_binding_and_evidence_not_candidate(self):
        graph = reviewed_graph()
        self.assert_valid(graph)  # Structural completeness only, not trusted promotion.
        for mutation in ["candidate-review", "unknown-kind", "no-review-evidence", "no-binding", "changed-endpoint", "no-edge-evidence"]:
            graph = reviewed_graph()
            review = graph["revisions"][1]
            if mutation == "candidate-review":
                review["assertion_state"] = "candidate"
            elif mutation == "unknown-kind":
                review["review_kind"] = "unknown"
            elif mutation == "no-review-evidence":
                review["evidence_ids"] = []
            elif mutation == "no-binding":
                del review["review_bindings"]
            elif mutation == "changed-endpoint":
                graph["relations"][0]["target"]["revision_id"] = "missing"
            else:
                graph["relations"][0]["evidence_ids"] = []
            with self.subTest(mutation=mutation):
                self.assertFalse(validate_math_memory(graph)["valid"])
        graph = minimal_math_memory()
        graph["revisions"][0]["assertion_state"] = "reviewed"
        self.assert_invalid(graph, "exact reviews")

    def test_relation_review_does_not_follow_from_source_revision(self):
        graph = reviewed_graph()
        endpoint = {"entity_id": "claim:1", "revision_id": "claim:1:r1"}
        add_relation(graph, "uncovered-citation", "cites", endpoint, endpoint, assertion_state="reviewed", evidence_ids=["ev:1"])
        self.assert_invalid(graph, "exact Review digest binding")
        graph = reviewed_graph()
        graph["relations"][0]["context_revision_ids"] = ["missing"]
        self.assert_invalid(graph, "binding sha256 mismatch")
        for mutation in ["endpoint", "evidence", "provenance", "body", "evidence_hash"]:
            graph = reviewed_graph()
            relation = graph["relations"][0]
            if mutation == "endpoint":
                relation["target"] = deepcopy(relation["source"])
            elif mutation == "evidence":
                relation["evidence_ids"] = []
            elif mutation == "body":
                graph["revisions"][0]["body"] = "同ID内容已变 / changed body"
                graph["revisions"][0]["body_sha256"] = digest(graph["revisions"][0]["body"])
            elif mutation == "evidence_hash":
                graph["evidence"][0]["sha256"] = digest("different evidence")
            else:
                relation["provenance"]["actor"] = "another-reviewer"
            self.assert_invalid(graph, "binding sha256 mismatch")

    def test_progress_review_kind_assertion_and_validity_are_separate(self):
        graph = minimal_math_memory()
        add_entity(graph, "research:1", "ResearchArtifact", "历史已解决 / legacy solved", "一次研究尝试。",
                   research_progress="completed", historical_status="solved")
        self.assert_valid(graph)
        revision = graph["revisions"][-1]
        self.assertEqual(revision["assertion_state"], "candidate")
        self.assertEqual(revision["validity"], "needs_review")
        revision["validity"] = "verified"
        self.assert_invalid(graph, "schema error")
        graph = minimal_math_memory()
        graph["revisions"][0]["review_kind"] = "human_review"
        self.assert_invalid(graph, "only allowed on Review")

    def test_non_json_values_fail_closed(self):
        for payload in [None, [], float("nan"), {"data": object()}, {1: "bad key"}]:
            self.assertFalse(validate_policy(payload)["valid"])
            self.assertFalse(validate_math_memory(payload)["valid"])
        graph = minimal_math_memory()
        graph["entities"][0]["name"] = "\ud800"
        self.assertFalse(validate_math_memory(graph)["valid"])
        cyclic = []
        cyclic.append(cyclic)
        self.assertFalse(validate_math_memory(cyclic)["valid"])


if __name__ == "__main__":
    unittest.main()
