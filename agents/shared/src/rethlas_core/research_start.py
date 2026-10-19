"""Thin trusted-operator native start/resume entry; no automatic consent."""
from __future__ import annotations

from pathlib import Path
import secrets
import sys

from .codex_native_provider import LIMITS, native_profile
from .memory_io import read_input
from .online_round import NativeRound
from .private_fs import PrivateDirectory, canonical_json, digest, strict_json_loads
from .project_client import connect, configured_text, grant, read_configuration
from .project_service import ProjectService
from .research import ResearchPipeline
from .validity import ValidityError


def _confirm(value, label):
    print("\n" + label + " (default: no)")
    print(canonical_json(value).decode())
    if value.get("schema") == "rethlas.codex-native-execution":
        print("--- BEGIN EXACT RETHLAS STDIN (UTF-8; not upstream HTTP) ---")
        print(value["stdin"])
        print("--- END EXACT RETHLAS STDIN ---")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    try:
        return input("Approve exactly this? Type yes: ").strip() == "yes"
    except (EOFError, KeyboardInterrupt):
        return False


def _selection(args, purpose):
    value = {"type": "codex_cli_native"}
    for attr, key in (("model", "model"), ("effort", "reasoning_effort")):
        choice = getattr(args, purpose + "_" + attr)
        if choice is not None:
            value[key] = choice
    return native_profile(value)


def _initialize(root, args):
    if args.question_file is None:
        raise ValidityError("new native project requires a question file")
    question = read_input(args.question_file, 8192, single_link=True, stable=True).decode("utf-8")
    if not question.strip():
        raise ValidityError("empty native question")
    selections = {p: _selection(args, p) for p in ("generation", "review")}
    # Never repair an existing unknown or partially initialized directory. The
    # marker is written first; only a final manifest makes the directory usable.
    root.mkdir(mode=0o700)
    directory = PrivateDirectory(root)
    directory.write_bytes("initializing.json", canonical_json({"version": "native-start-initializing-1"}), exclusive=True)
    credentials = directory.child("credentials", create=True)
    service = ProjectService.initialize(root / "project", owner="native-operator")
    config = {"version": "research-client-4", "root": str(root / "project"), "embedding": {"mode": "lexical"},
        "indexer": None, **{p + "_provider": v for p, v in selections.items()}}
    for name, role in (("reader", "project_reader"), ("writer", "project_writer"), ("runner", "research_runner")):
        issued = service.issue(role=role, ttl_seconds=3600)
        credentials.write_bytes(name + ".token", issued["token"].encode("ascii"), exclusive=True)
        config[name] = {"run_id": issued["run_id"], "token_file": str(credentials.root / (name + ".token"))}
    directory.write_bytes("client.json", canonical_json(config), exclusive=True)
    request = {"request_id": "native." + secrets.token_hex(12), "question": question, "question_only": True}
    directory.write_bytes("request.json", canonical_json(request), exclusive=True)
    requests = directory.child("requests", create=True)
    requests.write_bytes(request["request_id"] + ".json", canonical_json(request), exclusive=True)
    directory.write_bytes("start.json", canonical_json({"version": "native-start-1",
        "service_id": service.config["service_id"], "request_id": request["request_id"]}), exclusive=True)
    return directory


def _remaining(service, selected):
    claims = service.authority.authenticate(selected["token"], action="research.run", now=service.clock())
    return claims["expires_at"] - service.clock()


def _renew(directory, service, config):
    # issue's deterministic role run ID preserves the original request and all
    # journal counts. Renewal is never treated as policy or input approval.
    for name, role in (("reader", "project_reader"), ("writer", "project_writer"), ("runner", "research_runner")):
        issued = service.issue(role=role, ttl_seconds=3600)
        if issued["run_id"] != config[name]["run_id"]:
            raise ValidityError("native renewal must preserve run identity")
        credentials = PrivateDirectory(Path(config[name]["token_file"]).parent)
        credentials.write_bytes(name + ".token", issued["token"].encode("ascii"))


def _display_result(pipeline, selected, result, request_id):
    history = result.get("historical", {})
    if history.get("stage") == "completed":
        retained = {}
        key = digest({"service_id": pipeline.service.config["service_id"], "request_id": request_id})
        for purpose in ("generation", "review"):
            if purpose + "_sha256" in history:
                retained[purpose] = pipeline._load(key, purpose + "_response", history[purpose + "_sha256"])["content"]
        print(canonical_json({"candidate_and_model_report": retained, "mathematical_correctness_proved": False}).decode())
    return result


def _new_round(directory, args, manifest):
    if args.question_file is None:
        raise ValidityError("explicit new native round requires a question file")
    if any(getattr(args, p + "_" + k) is not None for p in ("generation", "review") for k in ("model", "effort")):
        raise ValidityError("new round preserves saved provider choices")
    config, service, retrieval, memory = connect(directory.root / "client.json", providers=False)
    selected = grant(config, "runner", required=True)
    current = ResearchPipeline(service, retrieval).status(selected, request_id=manifest["request_id"])
    if current["request_status"] != "accepted":
        raise ValidityError("pending or uncertain round cannot be replaced by a new request")
    question = read_input(args.question_file, 8192, single_link=True, stable=True).decode("utf-8")
    if not question.strip():
        raise ValidityError("empty native question")
    request = {"request_id": "native." + secrets.token_hex(12), "question": question, "question_only": True}
    requests = directory.child("requests")
    requests.write_bytes(request["request_id"] + ".json", canonical_json(request), exclusive=True)
    # The immutable request is durable first; the small pointer is committed last.
    # An interrupted pointer update leaves an unused request, never a new send.
    following = {**manifest, "request_id": request["request_id"]}
    directory.write_bytes("start.json", canonical_json(following))
    return following


def _refresh_stage(pipeline, selected, request_id):
    current = pipeline.status(selected, request_id=request_id)
    envelope = current.get("approval_required", {})
    if (current["request_status"] != "bound" or envelope.get("schema") != "rethlas.codex-native-execution"
            or current["historical"].get("stage") not in ("awaiting_generation_approval", "awaiting_review_approval")):
        raise ValidityError("only an unstarted native exact-input stage can be refreshed")
    key, progress = envelope["request_key"], current["historical"]
    with pipeline.service._locked(selected["token"], selected["run_id"], "research.run") as (_, state, claims, _):
        binding = state["requests"][key]
        if binding["status"] != "bound" or binding["result"] != progress:
            raise ValidityError("native stage changed before explicit refresh")
        pipeline._provider_for(envelope["purpose"]).cleanup(envelope)
        following = {k: v for k, v in progress.items() if k not in ("native_execution_sha256", "approval_prompt_sha256")}
        following["stage"] = "native_stage_refresh_requested"
        def refresh(next_state):
            next_state["requests"][key].update(result=following)
        pipeline.service._change(state, "RESEARCH_PROGRESS", refresh, key=key, claims=claims)


def start_project(args):
    from .research_cli import _path
    root = _path(args.project)
    repository = _path(Path(__file__).resolve().parents[4])
    if root == repository or repository in root.parents or root in repository.parents:
        raise ValidityError("private native project must be outside repository")
    existed = root.exists()
    if args.refresh_stage and (not args.continue_pending or args.new_round):
        raise ValidityError("refresh-stage requires explicit continue of the original pending round")
    if existed:
        directory = PrivateDirectory(root)
        manifest = strict_json_loads(directory.read_bytes("start.json", max_bytes=4096))
        if type(manifest) is not dict or set(manifest) != {"version", "service_id", "request_id"} or manifest["version"] != "native-start-1":
            raise ValidityError("unknown native project directory")
        if args.new_round:
            if args.renew:
                raise ValidityError("renew and create a new round as separate explicit actions")
            with directory.lock("start.lock"):
                if strict_json_loads(directory.read_bytes("start.json", max_bytes=4096)) != manifest:
                    raise ValidityError("native project pointer changed")
                manifest = _new_round(directory, args, manifest)
        elif args.question_file is not None or any(getattr(args, p + "_" + k) is not None
                for p in ("generation", "review") for k in ("model", "effort")):
            raise ValidityError("existing project opens unchanged; use explicit new-round with a new question")
    else:
        if args.new_round:
            raise ValidityError("new-round requires an existing completed native project")
        directory = _initialize(root, args)
        manifest = strict_json_loads(directory.read_bytes("start.json", max_bytes=4096))
    config = read_configuration(root / "client.json")
    if (config["version"] != "research-client-4" or config["root"] != str(root / "project")
            or any(type(config[name]) is not dict or config[name].get("token_file") != str(root / "credentials" / (name + ".token"))
                   for name in ("reader", "writer", "runner"))):
        raise ValidityError("native start configuration differs")
    service = ProjectService(Path(config["root"]))
    if service.config["service_id"] != manifest["service_id"]:
        raise ValidityError("native start project identity differs")
    if args.renew:
        if not _confirm({"ttl_seconds": 3600, "same_run_request_budget": True}, "Explicit capability renewal"):
            return {"status": "renewal_pending", "request_id": manifest["request_id"], "request_sent": False}
        _renew(directory, service, config)
    config, service, retrieval, memory = connect(root / "client.json", providers=False)
    selected = grant(config, "runner", required=True)
    offline = ResearchPipeline(service, retrieval)
    from .authorization import AuthorizationError
    try:
        remaining = _remaining(service, selected)
    except AuthorizationError:
        return {"status": "renewal_required", "request_id": manifest["request_id"],
                "command": "start PROJECT --renew --continue", "request_sent": False}
    if existed and not args.continue_pending and not args.new_round:
        history = offline.history(selected)
        return {"status": "native_history_only", "history": history,
                "research": offline.status(selected, request_id=manifest["request_id"]) if history["count"] else None,
                "request_id": manifest["request_id"], "continue": "start PROJECT --continue", "request_sent": False}
    if remaining < LIMITS["minimum_ttl_seconds"]:
        return {"status": "renewal_required", "request_id": manifest["request_id"],
                "command": "start PROJECT --renew --continue", "request_sent": False}
    history = offline.history(selected)
    if any(item["request_id"] == manifest["request_id"] and item["request_status"] in ("started", "accepted")
           for item in history["items"]):
        # No installation/provider construction is needed for historical state,
        # even when --continue is supplied to a started/unknown request.
        result = offline.status(selected, request_id=manifest["request_id"])
        return {"status": "native_no_resend" if result["request_status"] == "started" else "native_completed",
                "research": result, "request_sent": False}
    pipeline = ResearchPipeline(service, retrieval,
        generation_provider=configured_text(service, config, "generation", native_operator=True),
        review_provider=configured_text(service, config, "review", native_operator=True))
    request = strict_json_loads(directory.child("requests").read_bytes(manifest["request_id"] + ".json", max_bytes=16384))
    if request.get("request_id") != manifest["request_id"] or request.get("question_only") is not True:
        raise ValidityError("native retained request differs")
    history = offline.history(selected)
    known = any(item["request_id"] == manifest["request_id"] for item in history["items"])
    if args.refresh_stage:
        _refresh_stage(pipeline, selected, request["request_id"])
    result = pipeline.resume(selected, request_id=request["request_id"]) if known else pipeline.run(selected, **request)
    # Exactly one policy and one actual input per purpose. A denial/EOF keeps the
    # pending object. Reopening does not run this loop without --continue.
    enough = _remaining(pipeline.service, selected) >= LIMITS["minimum_ttl_seconds"]
    for _ in range(3):
        approval = result.get("approval_required")
        if approval is None:
            break
        if not enough:
            return {"status": "renewal_required", "request_id": request["request_id"],
                    "command": "start PROJECT --renew --continue", "request_sent": False}
        if not _confirm(approval, "Native policy / exact " + approval.get("purpose", "round policy")):
            return {"status": "native_approval_pending", "research": result, "request_id": request["request_id"],
                    "continue": "start PROJECT --continue", "request_sent": False}
        NativeRound(service).approve(approval, materials="public_or_synthetic")
        previous = result["historical"].get("stage")
        result = pipeline.resume(selected, request_id=request["request_id"])
        enough = _remaining(pipeline.service, selected) >= LIMITS["minimum_ttl_seconds"]
        if result["historical"].get("stage") == previous and "approval_required" in result:
            raise ValidityError("native approval did not advance its exact retained stage")
    return _display_result(pipeline, selected, result, request["request_id"])
