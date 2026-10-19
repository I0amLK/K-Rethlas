"""One research project, one visibility scope and one owned graph authority.

This trusted host API reuses the fixture-tested graph fence and withdrawal journal.
It does not expose fixture publication or give model workers a store credential.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import secrets

from .authorization import AuthorizationError, ROLE_ACTIONS
from .context_views import exact_ref
from .private_fs import canonical_json, digest
from .project_protocol import VERSION, protocol
from .project_views import PROJECT_POLICY_SHA256, project_context
from .validity import ValidityError, ValidityCapacity, ValidityConflict, validity_states
from .validity_service import CurrentValidityService, _floor, _guard, INTERNAL_RUN
from .validity_protocol import META_BYTES, DTO_BYTES


PROJECT_ROLES = frozenset({"project_reader", "project_writer", "project_withdrawer", "research_runner"})


class ProjectService(CurrentValidityService):
    DOMAIN = "research"
    VERSION = VERSION
    POLICY_HASH = PROJECT_POLICY_SHA256
    PROTOCOL = staticmethod(protocol)
    COMMIT_ACTION = "project.commit"
    WITHDRAW_ACTION = "project.withdraw"
    DIRECTORIES = ("authority", "control", "memory", "uses", "operations", "index", "research")

    @classmethod
    def initialize(cls, root, *, owner, domain="research"):
        if domain != "research":
            raise ValidityError("research projects require an explicit research domain")
        return super().initialize(root, owner=owner, domain=domain)

    @contextmanager
    def _locked(self, token, run_id, action, cancel_event=None):
        with self.authority._registry() as registry:
            claims = self.authority._check_locked(registry, token, owner=self.owner, run_id=run_id,
                domain=self.DOMAIN, action=action, now=self.clock())
            if claims["role"] not in PROJECT_ROLES or action not in ROLE_ACTIONS[claims["role"]]:
                raise AuthorizationError("research project role differs")
            if len(registry["runs"]) > 32 or len(registry["tokens"]) > 256:
                raise ValidityCapacity("research authority capacity reached")
            with self.directory.lock():
                state = self._state()
                resource = state["runs"].get(run_id)
                if resource != {"role": claims["role"], "resource": "owned-store"}:
                    raise AuthorizationError("research project run resource differs")
                _guard(claims, self.clock, cancel_event)
                yield registry, state, claims, resource["resource"]

    def issue(self, *, role, actions=None, ttl_seconds=300):
        if role not in PROJECT_ROLES:
            raise AuthorizationError("unknown research project role")
        if actions is None:
            actions = sorted(ROLE_ACTIONS[role])
        if (type(actions) is not list or not actions or any(type(a) is not str for a in actions)
                or len(set(actions)) != len(actions) or not set(actions) <= ROLE_ACTIONS[role]):
            raise AuthorizationError("actions exceed the research project role")
        run_id = "p." + digest({"service": self.config["service_id"], "role": role})
        with self._operator() as (registry, state):
            self._admit(state, "service.tokens", 256)
            if run_id not in state["runs"]:
                if len(state["runs"]) >= 32:
                    raise ValidityCapacity("research run capacity reached")
                self._change(state, "RUN_BOUND", lambda n: n["runs"].update({run_id: {"role": role, "resource": "owned-store"}}))
            if run_id not in registry["runs"]:
                registry["runs"][run_id] = {"owner": self.owner, "domain": self.DOMAIN, "state": "open", "epoch": 0}
            if len(registry["tokens"]) >= 256:
                raise ValidityCapacity("research capability capacity reached")
            def reserve(proposed):
                raw = canonical_json(proposed)
                if len(raw) > 1048576:
                    raise ValidityCapacity("research registry byte capacity reached")
                self._reserve(len(raw))
            token = self.authority._issue_locked(registry, owner=self.owner, run_id=run_id, domain=self.DOMAIN,
                role=role, actions=actions, ttl_seconds=ttl_seconds, now=self.clock(), before_save=reserve)
            return {"token": token, "run_id": run_id}

    @contextmanager
    def snapshot(self, token, *, run_id, action="project.read", cancel_event=None):
        """Authenticate before hydration; retain the store checkpoint through return."""
        with self._locked(token, run_id, action, cancel_event) as (_, state, claims, _):
            if state["pending_graph"] is not None:
                from .validity import ValidityRecoveryRequired
                raise ValidityRecoveryRequired("research graph fence requires explicit recovery")
            store, internal = self._internal(state)
            exported = store.export(internal, run_id=INTERNAL_RUN)
            with self._owned_checkpoint(state, store, internal) as point:
                if _floor(point) != state["accepted_graph"] or exported["head"] != point["head"]:
                    raise ValidityConflict("research graph differs from accepted checkpoint")
                withdrawals = self._withdrawals(state)
                yield state, exported["graph"], withdrawals, claims
                _guard(claims, self.clock, cancel_event)

    def binding(self, state):
        return {"service_id": self.config["service_id"], "store_id": self.config["store_id"],
                "store_config_sha256": self.config["store_config_sha256"], "graph": deepcopy(state["accepted_graph"])}

    def status(self, token, *, run_id, refs=None, cancel_event=None):
        if refs is not None:
            refs = protocol(refs, "refs")
        with self.snapshot(token, run_id=run_id, cancel_event=cancel_event) as (state, graph, withdrawals, _):
            current = validity_states(graph, withdrawals)
            selected = set(current) if refs is None else {item["revision_id"] for item in refs}
            by_id = {r["revision_id"]: r for r in graph["revisions"]}
            if refs is not None and any(key not in by_id or exact_ref(by_id[key]) != ref for key, ref in
                                       ((r["revision_id"], r) for r in refs)):
                raise ValidityError("status requires exact committed references")
            return {"status": "research_project", "binding": self.binding(state), "checked_at": self._checked_at(state),
                    "current": {key: current[key] for key in sorted(selected)},
                    "mathematical_correctness_proved": False, "production_completed": False}

    def prepare_context(self, token, selected, *, run_id, reference_refs=None, cancel_event=None):
        with self.snapshot(token, run_id=run_id, cancel_event=cancel_event) as (state, graph, withdrawals, _):
            value = project_context(graph, withdrawals, selected, reference_refs=reference_refs)
            if not value["overlay"]["input_eligible"]:
                raise ValidityError("mathematical context support is withdrawn")
            if len(canonical_json(value)) > DTO_BYTES:
                raise ValidityCapacity("complete research input byte capacity reached")
            return {**value, "binding": self.binding(state), "checked_at": self._checked_at(state)}

    def get_revision(self, token, ref, *, run_id, cancel_event=None):
        return self.prepare_context(token, [ref], run_id=run_id, cancel_event=cancel_event)

    def expand(self, token, selected, *, run_id, relation_types=None, depth=1, cancel_event=None):
        selected = protocol(selected, "refs")
        if type(depth) is not int or not 0 <= depth <= 8 or not 1 <= len(selected) <= 16:
            raise ValidityError("invalid bounded project navigation")
        allowed = {"depends_on", "assumes", "in_context", "cites", "attempts_to_prove", "reviews", "refutes", "contrasts", "derived_from", "has_subgoal"}
        if relation_types is not None and (type(relation_types) is not list or not relation_types
                or len(relation_types) > len(allowed) or any(type(k) is not str for k in relation_types)):
            raise ValidityError("bounded relation types must be a list")
        kinds = allowed if relation_types is None else set(relation_types)
        if not kinds or not kinds <= allowed:
            raise ValidityError("unknown project navigation relation")
        with self.snapshot(token, run_id=run_id, cancel_event=cancel_event) as (state, graph, withdrawals, _):
            from .context_views import refs as exact_refs
            index = {r["revision_id"]: r for r in graph["revisions"]}
            keys = set(exact_refs(selected, index))
            initial = set(keys)
            edges = {}
            frontier = keys
            for _ in range(depth):
                following = set()
                for edge in graph["relations"]:
                    if edge["source"]["revision_id"] in frontier and edge["type"] in kinds:
                        edges[edge["relation_id"]] = edge
                        following.add(edge["target"]["revision_id"])
                        following.update(edge["context_revision_ids"])
                for key in frontier:
                    following.update(index[key]["context_revision_ids"])
                frontier = following - keys
                keys |= following
                if len(keys) > 64 or len(edges) > 64:
                    raise ValidityCapacity("navigation capacity reached, no truncation")
            states = validity_states(graph, withdrawals)
            return {"status": "navigation_not_complete_mathematical_context", "checked_at": self._checked_at(state),
                    "revisions": [{**deepcopy(index[key]), "current": states[key],
                                   "not_authorized_as_premise": key not in initial} for key in sorted(keys)],
                    "relations": [deepcopy(edges[key]) for key in sorted(edges)]}

    @contextmanager
    def _prepared_condition(self, state, transition):
        if transition["kind"] in ("RESEARCH_STARTED", "RESEARCH_PROGRESS", "RESEARCH_FINISHED"):
            # A prepared business stage has not been accepted. It may only be
            # cancelled, retaining a terminal request binding. A new explicit
            # request must rerun the pipeline's exact current-input checks.
            raise ValidityConflict("unaccepted research stages require cancellation, not completion")
        with super()._prepared_condition(state, transition):
            yield

    def _fixture_disabled(self, *args, **kwargs):
        raise AuthorizationError("fixture use and publication do not authorize research")

    register_use = read = lookup = probe = publish = _fixture_disabled
