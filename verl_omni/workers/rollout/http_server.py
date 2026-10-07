# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Owned HTTP server lifetime; auto-port startup follows verl.workers.rollout.utils."""

import asyncio
from argparse import Namespace
from typing import Any

import uvicorn


class _UvicornServer(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self._startup_done = asyncio.Event()
        self.actual_port = None

    async def startup(self, sockets=None):
        try:
            await super().startup(sockets=sockets)
            if self.servers:
                self.actual_port = self.servers[0].sockets[0].getsockname()[1]
        finally:
            self._startup_done.set()


async def run_uvicorn(app: Any, args: Namespace, address: str) -> tuple[int, asyncio.Task[None], uvicorn.Server]:
    """Start HTTP serving and retain the Server needed for graceful shutdown."""
    app.server_args = args
    server = _UvicornServer(uvicorn.Config(app, host=address, port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    task.add_done_callback(lambda _: server._startup_done.set())
    await server._startup_done.wait()
    if server.actual_port is None:
        await task
        raise RuntimeError("HTTP server did not bind a listening port")
    return server.actual_port, task, server


def request_http_shutdown(worker: Any) -> None:
    """Ask Uvicorn to close listeners, requests and the application lifespan."""
    task = getattr(worker, "_server_task", None)
    server = getattr(worker, "_http_server", None)
    if server is None and task is not None:
        # Pinned verl starts Server.serve() directly but returns only its Task.
        # Retain that live Server; an unknown/cancelled task requires actor termination.
        frame = getattr(task.get_coro(), "cr_frame", None)
        server = frame.f_locals.get("self") if frame is not None else None
        if not isinstance(server, uvicorn.Server):
            raise RuntimeError("Cannot identify the live Uvicorn Server")
        worker._http_server = server
    if server is not None:
        task.get_loop().call_soon_threadsafe(setattr, server, "should_exit", True)


def http_shutdown_complete(worker: Any) -> bool:
    """Confirm serving completed normally, listeners closed and lifespan ended."""
    task = getattr(worker, "_server_task", None)
    server = getattr(worker, "_http_server", None)
    if task is None:
        if server is not None:
            raise RuntimeError("HTTP Server exists without its serving task")
        return True
    if not task.done():
        return False
    if task.cancelled():
        raise RuntimeError("HTTP serving task was cancelled without graceful shutdown")
    task.result()
    if server is None or any(listener.is_serving() for listener in server.servers):
        raise RuntimeError("HTTP listening sockets have not closed")
    if server.started:
        lifespan = server.lifespan
        if getattr(lifespan, "shutdown_failed", False):
            raise RuntimeError("HTTP application lifespan shutdown failed")
        event = getattr(lifespan, "shutdown_event", None)
        if event is not None and not event.is_set():
            raise RuntimeError("HTTP application lifespan shutdown has not finished")
    return True
