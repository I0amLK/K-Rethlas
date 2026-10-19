"""Strict native Codex execution facts, not resolver or isolation certification.

The effort vocabulary is observed in the installed CLI. Membership does not
certify account/model support, and a valid binding does not open the live gate.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import unquote, urlsplit

from .private_fs import digest
from .validity import ValidityError
from .validity_protocol import copied


VERSION = "codex-provider-binding-1"
CLI_VERSION = "0.163.0"
OBSERVED_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
_FIELDS = {"version", "provider_type", "mode", "selection", "resolved", "cli_version",
           "execution_sha256", "configuration_sha256"}
_HASH = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", re.ASCII)
_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", re.ASCII)
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", re.ASCII)


def _invalid():
    raise ValidityError("invalid native Codex provider binding")


def _model(value):
    if (type(value) is not str or not _MODEL.fullmatch(value)
            or value.lower() in {"default", "unknown", "auto"}
            or any(part in (".", "..") for part in value.split("/"))):
        _invalid()


def _effort(value):
    if type(value) is not str or value not in OBSERVED_EFFORTS:
        _invalid()


def _destination(value):
    if (type(value) is not str or not value or len(value) > 2048
            or not value.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in value)
            or any(char in value for char in ("\\", "?", "#"))):
        _invalid()
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        _invalid()
    if (parsed.scheme != "https" or not host or parsed.username is not None
            or parsed.password is not None or port not in (None, 443) or parsed.netloc.endswith(":")
            or any(part in (".", "..") for part in unquote(parsed.path).split("/"))):
        _invalid()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if (len(host) > 253 or "." not in host or all(char in "0123456789." for char in host)
                or host.endswith((".localhost", ".local", ".internal"))
                or any(not _LABEL.fullmatch(label) for label in host.split("."))):
            _invalid()
    else:
        if not address.is_global:
            _invalid()


def validate_codex_binding(value):
    """Validate exact versioned facts and their digest; return detached JSON.

    This verifies structure and self-consistency only. The trusted provider must
    separately establish resolution provenance and effective execution isolation.
    """
    value = copied(value)
    if (type(value) is not dict or set(value) != _FIELDS
            or value["version"] != VERSION or value["provider_type"] != "codex_cli"
            or type(value["mode"]) is not str or value["mode"] not in ("live", "mock")
            or value["cli_version"] != CLI_VERSION):
        _invalid()
    selection, resolved = value["selection"], value["resolved"]
    if (type(selection) is not dict or not set(selection) <= {"model", "reasoning_effort"}
            or type(resolved) is not dict or set(resolved) != {"model", "reasoning_effort", "provider"}):
        _invalid()
    if "model" in selection:
        _model(selection["model"])
    if "reasoning_effort" in selection:
        _effort(selection["reasoning_effort"])
    _model(resolved["model"])
    _effort(resolved["reasoning_effort"])
    if any(selection[name] != resolved[name] for name in selection):
        _invalid()
    provider = resolved["provider"]
    if (type(provider) is not dict or set(provider) != {"id", "destination"}
            or type(provider["id"]) is not str or not _PROVIDER.fullmatch(provider["id"])
            or provider["id"].lower() in {"default", "unknown", "auto"}):
        _invalid()
    _destination(provider["destination"])
    for name in ("execution_sha256", "configuration_sha256"):
        if type(value[name]) is not str or not _HASH.fullmatch(value[name]):
            _invalid()
    if value["configuration_sha256"] != digest({k: v for k, v in value.items() if k != "configuration_sha256"}):
        _invalid()
    return value


def make_codex_binding(*, mode, selection, resolved, cli_version, execution_sha256):
    """Build a binding only from explicit trusted resolution/execution facts."""
    value = {"version": VERSION, "provider_type": "codex_cli", "mode": mode,
             "selection": selection, "resolved": resolved, "cli_version": cli_version,
             "execution_sha256": execution_sha256}
    value["configuration_sha256"] = digest(value)
    return validate_codex_binding(value)
