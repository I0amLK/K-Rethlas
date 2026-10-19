"""Explicitly approved, single-connection public HTTPS source retrieval.

DNS is bounded in a credential-free owned process. Every result must be public;
the verified IP is used directly for the sole TLS connection with host checking.
No proxy, cookies, redirects, compression, retry, or ambient authentication.
"""
from __future__ import annotations

import http.client
import ipaddress
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import threading
import time
from urllib.parse import quote, urlsplit

from .private_fs import canonical_json, strict_json_loads
from .source_pdf import SourceCancelled, SourceError, _cancelled
from .worker import terminate_group


TEXT_BYTES = 2097152
PDF_BYTES = 20971520
MIME_TYPES = {"text/plain", "text/markdown", "text/html", "application/pdf"}
FETCH_SECONDS = 30


def parse_url(url):
    if (type(url) is not str or not 0 < len(url.encode("utf-8")) <= 8192
            or any(ord(char) <= 32 or ord(char) == 127 for char in url)
            or "\\" in url):
        raise SourceError("source URL must be a bounded HTTPS address")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.fragment or parsed.port not in (None, 443)):
            raise SourceError("source URL requires public HTTPS port 443 without credentials or fragment")
        host = parsed.hostname.encode("idna").decode("ascii")
        if (len(host) > 253 or "%" in host or host.endswith(".")
                or any(not part or len(part) > 63 for part in host.split("."))):
            raise SourceError("source URL hostname is invalid")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for char in host):
                raise SourceError("source URL requires a public hostname")
        else:
            if not _public(address):
                raise SourceError("source URL address must be public")
        target = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
        if parsed.query:
            target += "?" + quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
        return {"url": url, "host": host, "port": 443, "target": target}
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, SourceError):
            raise
        raise SourceError("source URL is invalid") from exc


def _public(address):
    return (address.is_global and not address.is_multicast and not address.is_unspecified
            and not address.is_reserved and not address.is_loopback and not address.is_link_local
            and not address.is_private and not getattr(address, "is_site_local", False)
            and (not isinstance(address, ipaddress.IPv6Address)
                 or address.ipv4_mapped is None and address.sixtofour is None and address.teredo is None))


def _resolve(host, port, *, deadline, cancel_event=None):
    """No getaddrinfo call can outlive our owned resolver deadline."""
    if time.monotonic() >= deadline:
        raise SourceError("source DNS deadline exceeded")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return [{"family": socket.AF_INET6 if address.version == 6 else socket.AF_INET,
                 "address": str(address)}]
    _cancelled(cancel_event)
    command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("source_dns_worker.py"))]
    try:
        process = subprocess.Popen(command, cwd="/", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    except OSError as exc:
        raise SourceError("source DNS worker could not start") from exc
    pending = canonical_json({"host": host, "port": port})
    try:
        while True:
            _cancelled(cancel_event)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SourceError("source DNS deadline exceeded")
            try:
                raw, _ = process.communicate(pending, timeout=min(remaining, 0.05))
                break
            except subprocess.TimeoutExpired:
                pending = None
        _cancelled(cancel_event)
        if process.returncode != 0 or len(raw) > 16384:
            raise SourceError("source DNS lookup failed")
        values = strict_json_loads(raw)
        if (type(values) is not list or not 1 <= len(values) <= 64
                or any(type(item) is not dict or set(item) != {"family", "address"}
                       or type(item["family"]) is not int or item["family"] not in (socket.AF_INET, socket.AF_INET6)
                       or type(item["address"]) is not str for item in values)):
            raise SourceError("source DNS returned invalid addresses")
        return values
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, SourceError):
            raise
        raise SourceError("source DNS returned invalid data") from exc
    finally:
        terminate_group(process)


def _verified_addresses(values):
    try:
        if type(values) is not list or not 1 <= len(values) <= 64:
            raise SourceError("source DNS returned no address")
        for item in values:
            if (type(item) is not dict or set(item) != {"family", "address"}
                    or type(item["family"]) not in (int, socket.AddressFamily) or type(item["address"]) is not str):
                raise ValueError("invalid DNS address")
            address = ipaddress.ip_address(item["address"])
            family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            if not _public(address) or item["family"] != family:
                raise ValueError("not public")
    except (KeyError, ValueError, TypeError) as exc:
        if isinstance(exc, SourceError):
            raise
        raise SourceError("source DNS includes a non-public or invalid address") from exc
    return values


def _https_bytes(parsed, *, cancel_event=None):
    """Production transport; callers cannot inject headers, roots or addresses."""
    deadline = time.monotonic() + FETCH_SECONDS
    values = _verified_addresses(_resolve(parsed["host"], parsed["port"], deadline=deadline,
                                          cancel_event=cancel_event))
    _cancelled(cancel_event)
    item = values[0]  # One chosen verified address, never retry another result.
    raw = socket.socket(item["family"], socket.SOCK_STREAM)
    connection, secure = None, None
    expired = threading.Event()

    def stop():
        expired.set()
        target = secure if secure is not None else raw
        try:
            target.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        target.close()

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raw.close()
        raise SourceError("source fetch deadline exceeded")
    timer = threading.Timer(remaining, stop)
    timer.daemon = True
    timer.start()
    try:
        raw.settimeout(remaining)
        raw.connect((item["address"], parsed["port"]))
        _cancelled(cancel_event)
        secure = ssl.create_default_context().wrap_socket(raw, server_hostname=parsed["host"],
                                                          do_handshake_on_connect=False)
        # Publish the SSL socket before handshaking so the absolute deadline
        # timer can close it even while OpenSSL is waiting on the peer.
        secure.do_handshake()
        connection = http.client.HTTPSConnection(parsed["host"], parsed["port"], timeout=remaining)
        connection.sock = secure
        _cancelled(cancel_event)
        connection.request("GET", parsed["target"], headers={
            "User-Agent": "Rethlas-Source/1", "Accept": "text/plain,text/markdown,text/html,application/pdf",
            "Accept-Encoding": "identity", "Connection": "close"})
        response = connection.getresponse()
        if response.status != 200:
            if 300 <= response.status < 400:
                raise SourceError("source redirect requires a separately confirmed URL")
            raise SourceError("source HTTPS response was not successful")
        mime = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if mime not in MIME_TYPES or response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise SourceError("source response type or encoding is unsupported")
        maximum = PDF_BYTES if mime == "application/pdf" else TEXT_BYTES
        length = response.getheader("Content-Length")
        if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > maximum):
            raise SourceError("source response size exceeds its limit")
        chunks, total = [], 0
        while True:
            _cancelled(cancel_event)
            if expired.is_set() or time.monotonic() >= deadline:
                raise SourceError("source fetch deadline exceeded")
            chunk = response.read1(min(65536, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise SourceError("source response size exceeds its limit")
            chunks.append(chunk)
        _cancelled(cancel_event)
        if expired.is_set() or time.monotonic() >= deadline:
            raise SourceError("source fetch deadline exceeded")
        data = b"".join(chunks)
        if not data or length is not None and total != int(length):
            raise SourceError("source response body is empty or incomplete")
        return {"mime": mime, "data": data}
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise SourceError("source HTTPS transport failed or timed out") from exc
    finally:
        timer.cancel()
        if connection is not None:
            connection.close()
        if secure is not None:
            secure.close()
        raw.close()
