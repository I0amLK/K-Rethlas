"""Loopback-only service with authentication before body/resource allocation."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from .authorization import AuthorizationError
from .controller import Controller
from .private_fs import digest, strict_json_loads
from .publication import PublicationConflict
from .worker import WorkerCancelled, WorkerTimeout, WorkerUnavailable


class Admission:
    def __init__(self, app, controller: Controller | None, *, max_body: int, max_concurrent: int):
        self.app, self.controller, self.max_body = app, controller, max_body
        self.slots = threading.BoundedSemaphore(max_concurrent)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health":
            return await self.app(scope, receive, send)
        if self.controller is None:
            return await JSONResponse({"detail": "controller is not configured"}, 503)(scope, receive, send)
        headers = dict(scope["headers"])
        credential = headers.get(b"authorization", b"")
        if not credential.startswith(b"Bearer ") or len(credential) > 8192:
            return await JSONResponse({"detail": "authentication required"}, 401)(scope, receive, send)
        if not self.slots.acquire(blocking=False):
            return await JSONResponse({"detail": "concurrency limit reached"}, 429)(scope, receive, send)
        try:
            from starlette.concurrency import run_in_threadpool
            try:
                token = credential[7:].decode("ascii")
                await run_in_threadpool(self.controller.authority.authenticate, token,
                    action="verify" if scope["path"] == "/verify" else "status")
            except (AuthorizationError, ValueError, UnicodeError):
                return await JSONResponse({"detail": "invalid authorization"}, 403)(scope, receive, send)
            try:
                if int(headers.get(b"content-length", b"0")) > self.max_body:
                    return await JSONResponse({"detail": "body too large"}, 413)(scope, receive, send)
                data, deadline = bytearray(), time.monotonic() + 10
                while True:
                    message = await asyncio.wait_for(receive(), timeout=max(0.001, deadline - time.monotonic()))
                    if message["type"] == "http.disconnect":
                        return
                    data.extend(message.get("body", b""))
                    if len(data) > self.max_body:
                        return await JSONResponse({"detail": "body too large"}, 413)(scope, receive, send)
                    if not message.get("more_body", False):
                        break
            except (ValueError, asyncio.TimeoutError):
                return await JSONResponse({"detail": "invalid or timed-out body"}, 400)(scope, receive, send)
            scope["rethlas.token"] = token
            delivered = False

            async def limited_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(data), "more_body": False}
                return await receive()

            return await self.app(scope, limited_receive, send)
        finally:
            self.slots.release()


def create_app(root: Path | None = None, *, driver: str = "disabled", timeout: float = 5,
               max_body: int = 131072, max_concurrent: int = 1) -> FastAPI:
    if not 1 <= max_concurrent <= 2 or not 1024 <= max_body <= 1048576:
        raise ValueError("invalid service limits")
    controller = Controller(root, driver=driver, timeout=timeout) if root is not None else None
    app = FastAPI(title="Rethlas trusted verification protocol", version="0.1.0",
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(Admission, controller=controller, max_body=max_body, max_concurrent=max_concurrent)
    app.state.controller = controller

    @app.get("/health")
    def health():
        return {"status": "ok", "driver": driver, "live_available": False}

    @app.post("/verify")
    async def verify(request: Request):
        try:
            payload = strict_json_loads(await request.body())
            # Hold the admission slot until cleanup, including on client disconnect.
            from starlette.concurrency import run_in_threadpool
            cancelled = threading.Event()
            task = asyncio.create_task(run_in_threadpool(controller.verify,
                request.scope["rethlas.token"], payload, cancel_event=cancelled))

            async def monitor_disconnect():
                while True:
                    if (await request.receive())["type"] == "http.disconnect":
                        cancelled.set()
                        return

            monitor = asyncio.create_task(monitor_disconnect())
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled.set()
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                raise
            finally:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass
            logging.getLogger("rethlas.audit").info("verification_complete trace=%s status=%s",
                digest({"run_id": payload["run_id"], "request_id": payload["request_id"]})[:16], result["status"])
            return result
        except AuthorizationError:
            raise HTTPException(403, "authorization rejected") from None
        except PublicationConflict:
            raise HTTPException(409, "run is sealed or input binding changed") from None
        except WorkerUnavailable:
            raise HTTPException(503, "live worker unavailable; no unsafe fallback") from None
        except WorkerTimeout:
            raise HTTPException(504, "driver deadline exceeded") from None
        except WorkerCancelled:
            raise HTTPException(409, "verification request cancelled") from None
        except (ValueError, KeyError, TypeError, RecursionError):
            raise HTTPException(422, "invalid request or report contract") from None

    @app.post("/status")
    async def status(request: Request):
        try:
            payload = strict_json_loads(await request.body())
            if not isinstance(payload, dict) or set(payload) - {"statement", "proof"}:
                raise ValueError("invalid status request")
            if any(not isinstance(value, str) for value in payload.values()):
                raise ValueError("invalid text")
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(controller.status, request.scope["rethlas.token"], **payload)
        except AuthorizationError:
            raise HTTPException(403, "authorization rejected") from None
        except (ValueError, TypeError, RecursionError):
            raise HTTPException(422, "invalid status contract") from None

    return app
