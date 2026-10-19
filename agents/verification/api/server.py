"""Compatibility ASGI entrypoint; no implicit state creation or model execution.

Use `rethlas-control serve` for the configured loopback service. Importing this
module creates only a disabled app, never authority keys or an unrestricted worker.
"""
from rethlas_core.api import create_app

app = create_app()
