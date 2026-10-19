"""Explicit operator configuration for managed project entrypoints.

Native Codex authentication stays in the CLI. HTTP credentials are only read
from the operator's explicit key-file at send; no global secret scan occurs.
"""
from __future__ import annotations

from pathlib import Path

from .memory_io import read_input, read_token
from .private_fs import strict_json_loads
from .project_service import ProjectService
from .retrieval import RetrievalService
from .managed_memory import ManagedMemory
from .embedding import EmbeddingProvider
from .online_round import OnlineRound
from .validity import ValidityError


def read_configuration(path):
    config = strict_json_loads(read_input(Path(path), 65536, single_link=True, stable=True))
    base = {"version", "root", "reader", "writer", "indexer", "runner", "embedding"}
    if type(config) is not dict:
        raise ValidityError("invalid explicit research client configuration")
    version = config.get("version")
    required = base | ({"text_provider"} if version == "research-client-1" else
                       {"generation_provider", "review_provider"})
    if version not in ("research-client-1", "research-client-2", "research-client-3", "research-client-4") or set(config) != required:
        raise ValidityError("invalid explicit research client configuration")
    if type(config["root"]) is not str or not Path(config["root"]).is_absolute():
        raise ValidityError("research service root must be explicit and absolute")
    if version == "research-client-4":
        from .codex_native_provider import native_profile
        for purpose in ("generation", "review"):
            native_profile(config[purpose + "_provider"])
    if version == "research-client-3":
        from .codex_provider import native_profile
        for purpose in ("generation", "review"):
            value = config[purpose + "_provider"]
            if type(value) is not dict or value.get("type") not in ("codex_cli", "openai_chat"):
                raise ValidityError("invalid explicit provider selection")
            if value["type"] == "codex_cli":
                native_profile(value)
    return config


def grant(config, name, *, required=False):
    value = config[name]
    if value is None and not required:
        return None
    if (type(value) is not dict or set(value) != {"run_id", "token_file"}
            or type(value["run_id"]) is not str or type(value["token_file"]) is not str
            or not Path(value["token_file"]).is_absolute()):
        raise ValidityError("invalid fixed research credential reference")
    return {"run_id": value["run_id"], "token": read_token(Path(value["token_file"]))}


def configured_embedding(service, config):
    value = config["embedding"]
    if value is None or value == {"mode": "lexical"}:
        return None
    if type(value) is not dict or "mode" not in value:
        raise ValidityError("invalid embedding configuration")
    if value["mode"] == "bge_cache_only" and set(value) == {"mode", "cache_dir"}:
        return EmbeddingProvider(mode="bge_cache_only", cache_dir=Path(value["cache_dir"]))
    if value["mode"] == "bge_live" and set(value) == {"mode", "endpoint", "key_file"}:
        approved = OnlineRound(service)
        return EmbeddingProvider(mode="bge_live", endpoint=value["endpoint"], key_file=Path(value["key_file"]),
                                 admit=approved.admit_embedding, allowed_sha256=approved.allowed)
    raise ValidityError("unknown or incomplete embedding configuration")


def provider_configuration(config, purpose="generation"):
    if purpose not in ("generation", "review"):
        raise ValidityError("unknown configured provider purpose")
    return config["text_provider"] if config["version"] == "research-client-1" else config[purpose + "_provider"]


def configured_text(service, config, purpose="generation", *, native_operator=False):
    from .provider import TextProvider
    value = provider_configuration(config, purpose)
    legacy = config["version"] == "research-client-1"
    if type(value) is not dict:
        raise ValidityError("explicit text provider configuration is required")
    kind = "openai_chat" if legacy else value.get("type")
    if kind == "codex_cli_native":
        if config["version"] != "research-client-4" or not native_operator:
            raise ValidityError("native CLI execution is available only through the trusted operator start entry")
        from .codex_native_provider import CodexNativeProvider
        provider = CodexNativeProvider(selection=value)
    elif kind == "codex_cli" and not legacy:
        from .codex_provider import CodexCLIProvider, native_profile
        if config["version"] == "research-client-2":
            if set(value) != {"type", "model"}:
                raise ValidityError("invalid legacy native Codex configuration")
            from .provider import _model
            settings = {"model": _model(value["model"])}
        else:
            settings = {"selection": native_profile(value)}
        # Construction is offline. V3 preserves omitted selection fields rather
        # than inventing defaults; unresolved native settings remain fail-closed.
        provider = CodexCLIProvider(**settings, admit=lambda messages, target:
            OnlineRound(service).admit_text(messages, target, provider=provider.binding()))
    elif kind == "openai_chat":
        required = {"mode", "endpoint", "key_file", "model"} | (set() if legacy else {"type"})
        optional = set() if legacy else {"allowed_response_models"}
        if (set(value) - required - optional or not required <= value.keys()
                or value["mode"] != "live" or type(value["key_file"]) is not str
                or not Path(value["key_file"]).is_absolute()):
            raise ValidityError("invalid explicit HTTPS text provider configuration")
        approved = OnlineRound(service)
        provider = TextProvider(mode="live", endpoint=value["endpoint"], key_file=Path(value["key_file"]),
            model=value["model"], allowed_response_models=value.get("allowed_response_models"),
            admit=lambda messages, target: approved.admit_text(messages, target,
                provider=None if legacy else provider.binding()))
    else:
        raise ValidityError("unknown text provider configuration type")
    return provider


def connect(path, *, providers=True):
    config = read_configuration(path)
    service = ProjectService(Path(config["root"]))
    retrieval = RetrievalService(service, provider=configured_embedding(service, config) if providers else None)
    memory = ManagedMemory(service, retrieval, reader=grant(config, "reader", required=True),
                           writer=grant(config, "writer"), indexer=grant(config, "indexer"))
    return config, service, retrieval, memory
