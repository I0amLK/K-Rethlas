"""Fixed, credential-free DNS worker for one explicitly approved source host."""
from __future__ import annotations

import json
import resource
import socket
import sys


def main():
    resource.setrlimit(resource.RLIMIT_AS, (134217728, 134217728))
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    request = json.loads(sys.stdin.buffer.read(16385))
    values = socket.getaddrinfo(request["host"], request["port"], type=socket.SOCK_STREAM,
                                proto=socket.IPPROTO_TCP)
    addresses = []
    for family, _, _, _, address in values:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        item = {"family": int(family), "address": address[0]}
        if item not in addresses:
            addresses.append(item)
        if len(addresses) > 64:
            raise ValueError("DNS result capacity reached")
    if not addresses:
        raise ValueError("no usable address")
    sys.stdout.buffer.write(json.dumps(addresses, separators=(",", ":")).encode("ascii"))


if __name__ == "__main__":
    main()
