"""Constrained local client; no arbitrary endpoints, redirects, or proxy inheritance."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from urllib.parse import urlsplit

import httpx

from .private_fs import strict_json_loads
from .publication import text_hash
from .verification import require_valid_output


def local_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/verify"
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.port is None):
        raise ValueError("verification endpoint must be a fixed IPv4 loopback /verify URL")
    return endpoint


def read_local_credential(path: Path) -> str:
    if not path.is_absolute() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("credential must be an absolute, non-symlink path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("credential must be an owner-only regular file")
        token = os.read(descriptor, 8193).decode("ascii").strip()
        if not token or len(token) > 8192 or any(character.isspace() for character in token):
            raise ValueError("invalid credential file")
        return token
    finally:
        os.close(descriptor)


def verify_via_local_service(statement: str, proof: str, *, endpoint: str,
                             token_file: Path, owner: str, run_id: str, domain: str,
                             request_id: str, timeout: float = 30) -> dict:
    local_endpoint(endpoint)
    if not 0 < timeout <= 30:
        raise ValueError("verification timeout must be bounded")
    token = read_local_credential(token_file)
    request = {"statement": statement, "proof": proof, "owner": owner,
               "run_id": run_id, "domain": domain, "request_id": request_id}
    with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
        response = client.post(endpoint, json=request, headers={"Authorization": "Bearer " + token})
        if response.is_redirect:
            raise ValueError("verification redirects are forbidden")
        response.raise_for_status()
        body = strict_json_loads(response.content)
    require_valid_output({key: body.get(key) for key in ("verification_report", "verdict", "repair_hints")})
    envelope = body.get("envelope", {})
    binding = envelope.get("binding", {})
    expected = {"owner": owner, "run_id": run_id, "domain": domain, "request_id": request_id,
                "statement_sha256": text_hash(statement), "proof_sha256": text_hash(proof)}
    if any(binding.get(key) != value for key, value in expected.items()) or not body.get("receipt"):
        raise ValueError("verification response does not bind the requested inputs")
    # Keep the full envelope/receipt. A worker receiving these fields cannot
    # cryptographically confer production completion; only controller status can.
    return body
