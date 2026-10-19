"""Pure T09 semantic checks; execution requires separate runtime authorization."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from rethlas_core.context_views import (
    closure, exact_ref, mathematical_projection, minimal_view,
)
from rethlas_core.contracts import _relation_sha256
from rethlas_core.private_fs import canonical_json, digest
from rethlas_core.validity import (
    POLICY, POLICY_SHA256, POLICY_VERSION, VERSION, ValidityCapacity,
    ValidityConflict, ValidityError, project_use, review_queue,
    use_overlay, validity_states,
)

from test_memory_store import provenance


def add_revision(graph, identity, kind="Claim", *, body=None):
    body = body if body is not None else "合成正文 " + identity + "\r\nα ≠ β\n"
    graph["entities"].append({"entity_id": identity, "type": kind, "name": "PRIVATE_SAME_NAME",
                              "provenance": provenance("PRIVATE_ACTOR")})
    revision = {"entity_id": identity, "revision_id": identity + ":0", "body": body,
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "provenance": provenance("PRIVATE_ACTOR"), "assertion_state": "candidate",
                "validity": "needs_review", "context_revision_ids": [], "evidence_ids": []}
    if kind == "Review":
        revision["review_kind"] = "unknown"
    graph["revisions"].append(revision)
    return revision


def add_edge(graph, source, target, kind="depends_on", *, contexts=None):
    relation = {"relation_id": source + "/" + kind + "/" + target, "type": kind,
                "source": {"entity_id": source, "revision_id": source + ":0"},
                "target": {"entity_id": target, "revision_id": target + ":0"},
                "provenance": provenance("PRIVATE_EDGE_ACTOR"), "assertion_state": "candidate",
                "context_revision_ids": list(contexts or ()), "evidence_ids": []}
    graph["relations"].append(relation)
    return relation


def dependency_fixture():
    graph = {"schema_version": "1.0", "entities": [], "revisions": [], "relations": [], "evidence": []}
    for key, kind in (("problem", "Problem"), ("A", "Assumption"), ("B", "Claim"), ("C", "Claim"),
                      ("proofC", "ProofArtifact"), ("D", "Claim"), ("proofD", "ProofArtifact")):
        add_revision(graph, key, kind)
    add_edge(graph, "B", "A")
    add_edge(graph, "C", "B")
    add_edge(graph, "proofC", "C", "attempts_to_prove")
    add_edge(graph, "D", "A", "cites")
    add_edge(graph, "proofD", "D", "attempts_to_prove")
    return graph


def by_revision(graph):
    return {r["revision_id"]: r for r in graph["revisions"]}


def arguments(graph, claim="C"):
    index = by_revision(graph)
    roots = {"problem": exact_ref(index["problem:0"]), "claim": exact_ref(index[claim + ":0"]),
             "proof": exact_ref(index["proof" + claim + ":0"])}
    selected, _ = closure(graph, {r["revision_id"] for r in roots.values()},
                          proof_pair=(roots["proof"]["revision_id"], roots["claim"]["revision_id"]))
    return {"roots": roots, "dependencies": [exact_ref(index[key]) for key in sorted(selected - {r["revision_id"] for r in roots.values()})],
            "references": [exact_ref(index[key]) for key in sorted(selected)
                           if next(e for e in graph["entities"] if e["entity_id"] == index[key]["entity_id"])["type"] == "Reference"]}


def withdraw(graph, *targets, event="withdraw-1"):
    index = by_revision(graph)
    return {"event_id": event, "targets": [exact_ref(index[key + ":0"]) for key in targets],
            "reason": "PRIVATE_REASON", "actor": "PRIVATE_WITHDRAWER"}


def ids(values):
    return {r["revision_id"] for r in values}


class ValidityProjectionTests(unittest.TestCase):
    def setUp(self):
        self.graph = dependency_fixture()

    def project(self, claim="C"):
        return project_use(self.graph, **arguments(self.graph, claim))

    def test_t09_dependency_chain_blocks_b_c_not_citation_d(self):
        events = [withdraw(self.graph, "A")]
        before = deepcopy(self.graph)
        states = validity_states(self.graph, events)
        for key in ("A:0", "B:0", "C:0"):
            self.assertTrue(states[key]["math_blocked"])
            self.assertEqual(ids(states[key]["blockers"]), {"A:0"})
        self.assertTrue(states["A:0"]["directly_withdrawn"])
        self.assertFalse(states["D:0"]["math_blocked"])
        self.assertFalse(states["proofC:0"]["math_blocked"])
        self.assertFalse(use_overlay(self.project("C"), events)["fixture_input_eligible"])
        citation = use_overlay(self.project("D"), events)
        self.assertTrue(citation["fixture_input_eligible"])
        self.assertEqual(citation["citation_only"], [{**exact_ref(by_revision(self.graph)["A:0"]),
            "withdrawn": True, "use": "historical_citation_only", "not_authorized_as_premise": True}])
        self.assertEqual(self.graph, before)

    def test_root_proof_withdrawal_does_not_reverse_prove_or_block_claim(self):
        events = [withdraw(self.graph, "proofC")]
        states = validity_states(self.graph, events)
        self.assertFalse(states["C:0"]["math_blocked"])
        self.assertTrue(states["proofC:0"]["math_blocked"])
        self.assertFalse(use_overlay(self.project(), events)["fixture_input_eligible"])
        self.assertEqual(POLICY_SHA256, digest(POLICY))
        self.assertEqual((VERSION, POLICY_VERSION), ("validity-1", "current-fixture-1"))
        self.assertFalse(POLICY["mathematical_review"])
        self.assertFalse(POLICY["production_completed"])

    def test_revision_context_and_math_edge_context_are_support(self):
        for placement in ("revision", "relation"):
            graph = dependency_fixture()
            context = add_revision(graph, "context", "DefinitionContext")
            if placement == "revision":
                by_revision(graph)["B:0"]["context_revision_ids"] = [context["revision_id"]]
            else:
                graph["relations"][0]["context_revision_ids"] = [context["revision_id"]]
            with self.subTest(placement=placement):
                states = validity_states(graph, [withdraw(graph, "context")])
                self.assertTrue(states["B:0"]["math_blocked"])
                self.assertTrue(states["C:0"]["math_blocked"])
                self.assertFalse(states["D:0"]["math_blocked"])
                projection = project_use(graph, **arguments(graph))
                self.assertIn("context:0", ids(projection["support"]))
                self.assertFalse(use_overlay(projection, [withdraw(graph, "context")])["fixture_input_eligible"])

    def test_citation_edge_context_and_descendants_stay_citation_only(self):
        context = add_revision(self.graph, "citation-context", "DefinitionContext")
        base = add_revision(self.graph, "citation-base", "DefinitionContext")
        context["context_revision_ids"] = [base["revision_id"]]
        edge = next(r for r in self.graph["relations"] if r["type"] == "cites")
        edge["context_revision_ids"] = [context["revision_id"]]
        projection = self.project("D")
        self.assertEqual(ids(projection["citation_only"]), {"A:0", "citation-context:0", "citation-base:0"})
        for target in ("citation-context", "citation-base"):
            with self.subTest(target=target):
                events = [withdraw(self.graph, target)]
                self.assertTrue(use_overlay(projection, events)["fixture_input_eligible"])
                self.assertFalse(validity_states(self.graph, events)["D:0"]["math_blocked"])

    def test_cited_revision_own_mathematical_dependency_is_not_promoted(self):
        context = add_revision(self.graph, "context", "DefinitionContext")
        by_revision(self.graph)["A:0"]["context_revision_ids"] = [context["revision_id"]]
        projection = self.project("D")
        self.assertEqual(ids(projection["citation_only"]), {"A:0", "context:0"})
        self.assertTrue(use_overlay(projection, [withdraw(self.graph, "context")])["fixture_input_eligible"])

    def test_independent_support_path_wins_over_citation(self):
        add_edge(self.graph, "D", "A")
        projection = self.project("D")
        self.assertIn("A:0", ids(projection["support"]))
        self.assertNotIn("A:0", ids(projection["citation_only"]))
        self.assertFalse(use_overlay(projection, [withdraw(self.graph, "A")])["fixture_input_eligible"])

    def test_attempt_relation_context_supports_use_not_global_claim(self):
        context = add_revision(self.graph, "attempt-context", "DefinitionContext")
        attempt = next(r for r in self.graph["relations"] if r["source"]["entity_id"] == "proofD")
        attempt["context_revision_ids"] = [context["revision_id"]]
        events = [withdraw(self.graph, "attempt-context")]
        states = validity_states(self.graph, events)
        self.assertFalse(states["D:0"]["math_blocked"])
        self.assertFalse(states["proofD:0"]["math_blocked"])
        projection = self.project("D")
        self.assertIn("attempt-context:0", ids(projection["support"]))
        self.assertFalse(use_overlay(projection, events)["fixture_input_eligible"])

    def test_nonpremise_relations_never_become_global_dependencies(self):
        for kind in ("cites", "contrasts", "derived_from", "has_subgoal", "refutes", "attempts_to_prove"):
            graph = dependency_fixture()
            graph["relations"] = [r for r in graph["relations"] if r["source"]["entity_id"] != "D"]
            add_edge(graph, "D", "A", kind)
            with self.subTest(kind=kind):
                self.assertFalse(validity_states(graph, [withdraw(graph, "A")])["D:0"]["math_blocked"])

    def test_unsupported_disclosure_relation_rejects_instead_of_deleting(self):
        for kind in ("contrasts", "derived_from", "has_subgoal", "refutes"):
            graph = dependency_fixture()
            add_edge(graph, "D", "B", kind)
            roots = arguments(dependency_fixture(), "D")["roots"]
            dependencies = [exact_ref(by_revision(graph)[key]) for key in ("A:0", "B:0")]
            with self.subTest(kind=kind), self.assertRaises(ValidityError):
                project_use(graph, roots=roots, dependencies=dependencies, references=[])

    def test_missing_or_extra_disclosure_dependencies_reject(self):
        for changed in ([], [exact_ref(by_revision(self.graph)["D:0"]) ]):
            args = arguments(self.graph)
            args["dependencies"] = changed
            with self.subTest(dependencies=changed), self.assertRaises(ValidityError):
                project_use(self.graph, **args)

    def test_reference_body_needs_exact_explicit_approval(self):
        source = add_revision(self.graph, "source", "Reference")
        add_edge(self.graph, "D", "source", "cites")
        args = arguments(self.graph, "D")
        self.assertEqual(args["references"], [exact_ref(source)])
        projection = project_use(self.graph, **args)
        self.assertIn(source["revision_id"], ids(projection["citation_only"]))
        for references in ([], [{**exact_ref(source), "body_sha256": "0" * 64}]):
            with self.subTest(references=references), self.assertRaises(ValidityError):
                project_use(self.graph, **{**args, "references": references})

    def test_legacy_wrapper_and_nonmath_dependency_reject(self):
        for kind in ("Review", "ResearchArtifact", "Failure"):
            graph = dependency_fixture()
            revision = add_revision(graph, "history", kind)
            if kind == "ResearchArtifact":
                revision["research_progress"] = "completed"
                revision["historical_status"] = "solved"
            add_edge(graph, "D", "history", "cites")
            with self.subTest(kind=kind), self.assertRaises(ValidityError):
                project_use(graph, **arguments(graph, "D"))
        for section in ("entities", "revisions"):
            graph = dependency_fixture()
            next(item for item in graph[section] if item["entity_id"] == "A")["provenance"]["source"] = "legacy-jsonl://snapshot/old"
            with self.subTest(section=section), self.assertRaises(ValidityError):
                project_use(graph, **arguments(graph, "D"))

    def test_projection_matches_existing_minimal_view_without_workspace_fabrication(self):
        args = arguments(self.graph)
        manifest = {"version": "context-1", "request_id": "historical-view", "workspace_id": "1" * 64,
                    "snapshot_sha256": "2" * 64, "join_sha256": "3" * 64,
                    "policy_version": "minimal-math-1", **args}
        old = minimal_view(self.graph, manifest)
        core = mathematical_projection(self.graph, **args)
        projection = self.project()
        self.assertEqual(core, {key: old[key] for key in ("roots", "revisions", "relations")})
        self.assertEqual(core, {key: projection[key] for key in core})
        self.assertEqual(old["manifest_sha256"], digest(manifest))
        self.assertFalse({"workspace_id", "join_sha256", "manifest_sha256"} & set(projection))

    def test_utf8_crlf_and_opaque_identity_are_preserved(self):
        body = "# 合成\r\nα ≠ β\n\né / é\n"
        by_revision(self.graph)["A:0"].update(body=body, body_sha256=hashlib.sha256(body.encode()).hexdigest())
        projection = self.project()
        self.assertEqual(next(r for r in projection["revisions"] if r["revision_id"] == "A:0")["body"], body)
        self.assertNotIn("PRIVATE_", canonical_json(projection).decode())
        self.assertNotIn("PRIVATE_", canonical_json(use_overlay(projection, [withdraw(self.graph, "A")])).decode())
        bad = deepcopy(self.graph)
        by_revision(bad)["A:0"]["body"] = body.replace("\r\n", "\n")
        with self.assertRaises(ValidityError):
            project_use(bad, **arguments(self.graph))

    def test_withdrawal_exact_identity_hash_unknown_and_duplicates_reject(self):
        event = withdraw(self.graph, "A")
        for target in ({**event["targets"][0], "body_sha256": "0" * 64},
                       {**event["targets"][0], "entity_id": "different"},
                       {**event["targets"][0], "revision_id": "A:unknown"}):
            with self.subTest(target=target), self.assertRaises(ValidityError):
                validity_states(self.graph, [{**event, "targets": [target]}])
        with self.assertRaises(ValidityConflict):
            validity_states(self.graph, [event, deepcopy(event)])
        with self.assertRaises(ValidityError):
            validity_states(self.graph, [{**event, "targets": event["targets"] * 2}])

    def test_equal_names_and_bodies_do_not_merge_new_identities(self):
        old = by_revision(self.graph)["A:0"]
        add_revision(self.graph, "otherA", "Assumption", body=old["body"])
        add_edge(self.graph, "D", "otherA")
        events = [withdraw(self.graph, "A")]
        self.assertFalse(validity_states(self.graph, events)["otherA:0"]["math_blocked"])
        self.assertTrue(use_overlay(self.project("D"), events)["fixture_input_eligible"])

    def test_new_candidate_depending_old_withdrawal_is_immediately_blocked(self):
        events = [withdraw(self.graph, "A")]
        add_revision(self.graph, "new")
        add_edge(self.graph, "new", "A")
        state = validity_states(self.graph, events)["new:0"]
        self.assertTrue(state["math_blocked"])
        self.assertFalse(state["directly_withdrawn"])
        self.assertEqual(by_revision(self.graph)["new:0"]["validity"], "needs_review")
        queue = review_queue(self.graph, events, {})
        self.assertTrue(any(item["kind"] == "math_dependency_blocked" and item["revision"]["revision_id"] == "new:0" for item in queue))

    def test_unrelated_withdrawal_does_not_change_use_overlay_digest(self):
        projection = self.project("D")
        before = use_overlay(projection, [])
        after = use_overlay(projection, [withdraw(self.graph, "B")])
        self.assertEqual(before, after)
        self.assertEqual(digest(before), digest(after))
        cited = use_overlay(projection, [withdraw(self.graph, "A")])
        self.assertNotEqual(digest(before), digest(cited))
        self.assertTrue(cited["fixture_input_eligible"])
        self.assertFalse({"control_sequence", "graph_head", "reason", "actor"} & set(after))

    def test_forged_partition_body_and_extra_frozen_fields_reject(self):
        original = self.project()
        bad_partition = deepcopy(original)
        bad_partition["citation_only"] = [*bad_partition["citation_only"], bad_partition["support"].pop()]
        bad_body = deepcopy(original)
        bad_body["revisions"][0]["body"] += "changed"
        extra = {**original, "mathematically_usable": True}
        for bad in (bad_partition, bad_body, extra):
            with self.subTest(bad=bad), self.assertRaises(ValidityError):
                use_overlay(bad, [])

    def test_source_larger_than_view_is_allowed_but_entry_limit_is_not(self):
        graph = dependency_fixture()
        for number in range(260):
            add_revision(graph, "unrelated" + str(number))
        projection = project_use(graph, **arguments(graph, "D"))
        self.assertEqual(len(projection["revisions"]), 4)
        for number in range(260, 1018):
            add_revision(graph, "unrelated" + str(number))
        with self.assertRaises(ValidityCapacity):
            validity_states(graph, [])

    def test_depth_and_disclosure_size_are_bounded_without_truncation(self):
        for count in (66, 257):
            graph = dependency_fixture()
            previous = "C"
            for number in range(count):
                key = "long" + str(number)
                add_revision(graph, key)
                add_edge(graph, previous, key)
                previous = key
            roots = arguments(dependency_fixture())["roots"]
            dependencies = [exact_ref(r) for r in graph["revisions"] if r["entity_id"] not in {"problem", "C", "proofC", "D", "proofD"}]
            with self.subTest(count=count), self.assertRaises(ValidityError):
                project_use(graph, roots=roots, dependencies=dependencies, references=[])

    def test_withdrawal_and_use_counts_reject_over_bound(self):
        events = [withdraw(self.graph, "A", event="withdraw-" + str(number)) for number in range(33)]
        with self.assertRaises(ValidityCapacity):
            validity_states(self.graph, events)
        with self.assertRaises(ValidityCapacity):
            review_queue(self.graph, [], {"use" + str(n): self.project("D") for n in range(9)})

    def test_context_cycle_is_bounded_and_propagates_without_recursion(self):
        left = add_revision(self.graph, "left", "DefinitionContext")
        right = add_revision(self.graph, "right", "DefinitionContext")
        left["context_revision_ids"] = [right["revision_id"]]
        right["context_revision_ids"] = [left["revision_id"]]
        by_revision(self.graph)["C:0"]["context_revision_ids"] = [left["revision_id"]]
        states = validity_states(self.graph, [withdraw(self.graph, "right")])
        self.assertTrue(states["C:0"]["math_blocked"])
        self.assertEqual(ids(states["C:0"]["blockers"]), {"right:0"})


class ValidityReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.graph = dependency_fixture()

    def test_review_target_obsolescence_is_one_way_and_not_mathematical_authority(self):
        review = add_revision(self.graph, "review", "Review")
        add_edge(self.graph, "review", "C", "reviews")
        states = validity_states(self.graph, [withdraw(self.graph, "A")])
        self.assertTrue(states[review["revision_id"]]["review_input_obsolete"])
        self.assertFalse(states[review["revision_id"]]["math_blocked"])
        self.assertEqual(ids(states[review["revision_id"]]["review_blockers"]), {"A:0"})
        withdrawn_review = validity_states(self.graph, [withdraw(self.graph, "review")])
        self.assertFalse(withdrawn_review["C:0"]["math_blocked"])
        self.assertEqual(by_revision(self.graph)["C:0"]["assertion_state"], "candidate")
        self.assertEqual(review["review_kind"], "unknown")

    def test_bound_relation_snapshot_inputs_and_contexts_obsolete_review(self):
        review = add_revision(self.graph, "review", "Review")
        context = add_revision(self.graph, "context", "DefinitionContext")
        target_edge = next(r for r in self.graph["relations"] if r["source"]["entity_id"] == "D")
        target_edge["context_revision_ids"] = [context["revision_id"]]
        index = by_revision(self.graph)
        review["review_bindings"] = [{"relation_id": target_edge["relation_id"],
                                      "sha256": _relation_sha256(target_edge, index, {})}]
        for target in ("A", "D", "context"):
            with self.subTest(target=target):
                states = validity_states(self.graph, [withdraw(self.graph, target)])
                self.assertTrue(states["review:0"]["review_input_obsolete"])
                self.assertFalse(states["review:0"]["math_blocked"])
        self.assertFalse(validity_states(self.graph, [withdraw(self.graph, "B")])["review:0"]["review_input_obsolete"])

    def test_review_of_obsolete_review_propagates_only_input_obsolescence(self):
        add_revision(self.graph, "review", "Review")
        add_revision(self.graph, "review2", "Review")
        add_edge(self.graph, "review", "C", "reviews")
        add_edge(self.graph, "review2", "review", "reviews")
        states = validity_states(self.graph, [withdraw(self.graph, "A")])
        self.assertTrue(states["review:0"]["review_input_obsolete"])
        self.assertTrue(states["review2:0"]["review_input_obsolete"])
        self.assertFalse(states["review2:0"]["math_blocked"])
        self.assertFalse(states["D:0"]["math_blocked"])

    def test_bad_review_binding_digest_rejects_not_ignored(self):
        review = add_revision(self.graph, "review", "Review")
        relation = self.graph["relations"][0]
        review["review_bindings"] = [{"relation_id": relation["relation_id"], "sha256": "0" * 64}]
        with self.assertRaises(ValidityError):
            validity_states(self.graph, [])

    def test_queue_distinguishes_direct_math_review_and_citation_refresh(self):
        add_revision(self.graph, "review", "Review")
        add_edge(self.graph, "review", "C", "reviews")
        use = project_use(self.graph, **arguments(self.graph, "D"))
        events = [withdraw(self.graph, "A")]
        queue = review_queue(self.graph, events, {"D-use": use})
        self.assertEqual({item["kind"] for item in queue}, {
            "direct_withdrawal", "math_dependency_blocked", "review_input_obsolete", "citation_refresh"})
        self.assertEqual([item["key"] for item in queue], sorted(item["key"] for item in queue))
        self.assertEqual(len(queue), len({item["key"] for item in queue}))
        self.assertTrue(any(item["kind"] == "citation_refresh" and item["use_id"] == "D-use"
                            and item["revision"]["revision_id"] == "A:0" for item in queue))
        self.assertNotIn("PRIVATE_", canonical_json(queue).decode())

    def test_queue_is_order_independent_and_stable_per_withdrawal_event(self):
        events = [withdraw(self.graph, "A", event="first"), withdraw(self.graph, "B", event="second")]
        first = review_queue(self.graph, events, {})
        graph = deepcopy(self.graph)
        for key in ("entities", "revisions", "relations", "evidence"):
            graph[key].reverse()
        self.assertEqual(first, review_queue(graph, list(reversed(events)), {}))
        self.assertEqual({item["withdrawal_event"] for item in first}, {"first", "second"})


if __name__ == "__main__":
    unittest.main()
