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
"""Real local HTTP checks for graceful closure and cancelled serving tasks."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from verl_omni.workers.rollout.http_server import http_shutdown_complete, request_http_shutdown, run_uvicorn


async def _status(port):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await writer.drain()
        return await reader.readline()
    finally:
        writer.close()
        await writer.wait_closed()


def _app(events):
    @asynccontextmanager
    async def lifespan(app):
        events.append("startup")
        yield
        events.append("lifespan shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return {"ok": True}

    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream", [False, True])
async def test_graceful_http_shutdown_closes_port_and_runs_lifespan(upstream):
    events = []
    app = _app(events)
    args = SimpleNamespace()
    if upstream:
        from verl.workers.rollout.utils import run_uvicorn as upstream_run

        port, task = await upstream_run(app, args, "127.0.0.1")
        worker = SimpleNamespace(_server_task=task)
    else:
        port, task, server = await run_uvicorn(app, args, "127.0.0.1")
        worker = SimpleNamespace(_server_task=task, _http_server=server)
    try:
        assert b"200" in await _status(port)
        assert not http_shutdown_complete(worker)
        request_http_shutdown(worker)
        await asyncio.wait_for(asyncio.shield(task), timeout=3)
        assert http_shutdown_complete(worker)
        assert events == ["startup", "lifespan shutdown"]
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", port)
    finally:
        if not task.done():
            worker._http_server.should_exit = True
            await asyncio.wait_for(asyncio.shield(task), timeout=3)


@pytest.mark.asyncio
async def test_cancelled_serve_task_is_not_successful_http_shutdown():
    events = []
    port, task, server = await run_uvicorn(_app(events), SimpleNamespace(), "127.0.0.1")
    worker = SimpleNamespace(_server_task=task, _http_server=server)
    try:
        assert b"200" in await _status(port)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert b"200" in await _status(port)
        assert events == ["startup"]
        with pytest.raises(RuntimeError, match="cancelled without graceful"):
            http_shutdown_complete(worker)
    finally:
        await server.shutdown()
    assert events == ["startup", "lifespan shutdown"]


@pytest.mark.asyncio
async def test_unidentified_upstream_http_task_fails_closed():
    task = asyncio.create_task(asyncio.sleep(0))
    worker = SimpleNamespace(_server_task=task)
    with pytest.raises(RuntimeError, match="Cannot identify"):
        request_http_shutdown(worker)
    await task


def test_partially_initialized_worker_without_http_server_is_safe():
    worker = SimpleNamespace()
    request_http_shutdown(worker)
    assert http_shutdown_complete(worker)
