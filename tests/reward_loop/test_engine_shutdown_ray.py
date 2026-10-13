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
"""Real Ray/HTTP reward shutdown with owned subprocesses and shared memory."""

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import httpx
import psutil
import pytest
import ray

from verl_omni.reward_loop import engine_shutdown as shutdown


@ray.remote(num_cpus=1)
class _Journal:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)

    def read(self):
        return self.events


@ray.remote(num_cpus=1)
class _Consumer:
    def __init__(self, journal, failure):
        self.journal = journal
        self.failure = failure
        code = """
import json, multiprocessing as mp, sys
from multiprocessing.shared_memory import SharedMemory
memory = SharedMemory(create=True, size=4096)
semaphore = mp.get_context('spawn').Semaphore()
print(json.dumps([memory.name, semaphore._semlock.name]), flush=True)
sys.stdin.read(1)
memory.close()
memory.unlink()
"""
        self.child = subprocess.Popen(
            [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.child_pid = self.child.pid
        self.memory_name, self.semaphore_name = json.loads(self.child.stdout.readline())

    async def start(self):
        from fastapi import FastAPI
        from verl.workers.rollout.utils import run_uvicorn

        @asynccontextmanager
        async def lifespan(app):
            yield
            await self.journal.record.remote("lifespan shutdown")

        app = FastAPI(lifespan=lifespan)

        @app.get("/")
        def ready():
            return {"ready": True}

        port, self._server_task = await run_uvicorn(app, SimpleNamespace(), "127.0.0.1")
        shutdown.retain_http_server(self)
        self.engine = SimpleNamespace(
            output_handler=asyncio.create_task(asyncio.sleep(3600)), shutdown=self.stop_engine
        )
        return port, os.getpid(), self.child_pid, self.memory_name, self.semaphore_name

    async def collective_rpc(self, method, timeout):
        assert method == "shutdown"
        await self.journal.record.remote("worker shutdown")

    def stop_engine(self, timeout):
        assert shutdown.http_shutdown_complete(self)
        assert self.engine.output_handler.done()
        ray.get(self.journal.record.remote("engine shutdown"))
        if self.failure == "error":
            raise RuntimeError("injected shutdown failure")
        if self.failure == "timeout":
            time.sleep(60)
        self.child.stdin.write("x")
        self.child.stdin.flush()
        self.child.wait(timeout=5)


def _retain_semaphore(worker):
    import multiprocessing

    worker.semaphore = multiprocessing.get_context("spawn").Semaphore()
    return worker.semaphore._semlock.name


@ray.remote(num_cpus=1)
class _AsyncOmniShutdownProbe:
    async def start(self):
        self.output_finished = False

        async def output():
            try:
                await asyncio.sleep(60)
            finally:
                self.output_finished = True

        import multiprocessing
        import weakref

        class Engine:
            def __init__(self):
                self.semaphore = multiprocessing.get_context("spawn").Semaphore()

            def shutdown(self):
                self.final_output_task.cancel()
                self.final_output_task = None

        engine = Engine()
        self.semaphore_name = engine.semaphore._semlock.name
        self.engine_ref = weakref.ref(engine)

        # The real output coroutine retains its engine through self.
        async def retained_output(owner):
            try:
                await output()
            finally:
                assert owner is not None

        task = asyncio.create_task(retained_output(engine))
        await asyncio.sleep(0)
        engine.final_output_task = task
        self.engine = engine

    async def close_engine(self):
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

        await vLLMOmniHttpServer.close_engine(self)
        return (
            self.output_finished
            and self.engine is None
            and self.engine_ref() is None
            and not Path("/dev/shm", "sem." + self.semaphore_name.lstrip("/")).exists()
        )


def test_ray_awaits_omni_engine_close_method():
    with TemporaryDirectory(prefix="ca") as ray_temp:
        try:
            ray.init(
                num_cpus=1, num_gpus=0, include_dashboard=False, _temp_dir=ray_temp, object_store_memory=128 * 1024**2
            )
            actor = _AsyncOmniShutdownProbe.remote()
            ray.get(actor.start.remote(), timeout=60)
            assert ray.get(actor.close_engine.remote(), timeout=60)
        finally:
            ray.shutdown()


def test_graceful_actor_exit_preserves_unrelated_actor():
    from verl_omni.utils.ray_lifecycle import close_actor_gracefully

    with TemporaryDirectory(prefix="cg") as ray_temp:
        try:
            ray.init(
                num_cpus=2, num_gpus=0, include_dashboard=False, _temp_dir=ray_temp, object_store_memory=128 * 1024**2
            )
            owned = _Journal.remote()
            unrelated = _Journal.remote()
            semaphore_name = ray.get(owned.__ray_call__.remote(_retain_semaphore), timeout=60)
            identity = ray.get(owned.__ray_call__.remote(shutdown.actor_process_identity), timeout=60)
            close_actor_gracefully(owned, identity, timeout=10)
            assert not Path("/dev/shm", "sem." + semaphore_name.lstrip("/")).exists()
            errors = "\n".join(path.read_text() for path in Path(ray_temp).rglob("worker-*.err"))
            assert "resource_tracker: process died unexpectedly" not in errors
            assert "KeyError:" not in errors
            assert ray.get(unrelated.read.remote(), timeout=10) == []
            if psutil.pid_exists(identity["pid"]):
                process = psutil.Process(identity["pid"])
                assert process.create_time() != identity["created"] or process.status() == psutil.STATUS_ZOMBIE
        finally:
            ray.shutdown()


@pytest.mark.parametrize("failure", ["none", "error", "timeout"])
def test_owned_reward_server_router_and_children_stop_without_affecting_other_actors(monkeypatch, failure):
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "0")
    monkeypatch.setenv("RAY_num_server_call_thread", "2")
    monkeypatch.setenv("RAY_num_grpc_internal_threads", "2")
    monkeypatch.setenv("RAY_worker_num_grpc_internal_threads", "2")
    monkeypatch.setattr(shutdown, "_ENGINE_SHUTDOWN_TIMEOUT", 0.2 if failure == "timeout" else 10)
    ray_temp = TemporaryDirectory(prefix="rs")
    try:
        ray.init(
            num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2, _temp_dir=ray_temp.name
        )
        journal = _Journal.remote()
        unrelated = _Journal.remote()
        placement_worker = _Journal.remote()
        worker_identity = ray.get(placement_worker.__ray_call__.remote(shutdown.actor_process_identity), timeout=60)
        consumer = _Consumer.remote(journal, failure)
        port, pid, child_pid, memory_name, semaphore_name = ray.get(consumer.start.remote(), timeout=60)
        actor_created, child_created = psutil.Process(pid).create_time(), psutil.Process(child_pid).create_time()
        manager = object.__new__(shutdown.RewardEngineManager)
        manager.config = SimpleNamespace(rollout=SimpleNamespace(name="vllm", engine_kwargs={}))
        manager.rollout_replicas = [SimpleNamespace(servers=[consumer], workers=[placement_worker])]
        manager.server_addresses = [f"127.0.0.1:{port}"]
        manager._router_process = None
        manager._initialize_router()
        router_pid = manager._router_process.pid
        with httpx.Client(trust_env=False, timeout=3) as client:
            assert client.get(f"http://127.0.0.1:{port}/").status_code == 200
        manager.shutdown()
        # Check before Ray shutdown so cluster teardown cannot hide leaked actors.
        with httpx.Client(trust_env=False, timeout=2) as client, pytest.raises(httpx.ConnectError):
            client.get(f"http://127.0.0.1:{port}/")
        assert ray.get(journal.read.remote()) == ["lifespan shutdown", "worker shutdown", "engine shutdown"]
        for owned_pid, created in [
            (pid, actor_created),
            (child_pid, child_created),
            (worker_identity["pid"], worker_identity["created"]),
        ]:
            if psutil.pid_exists(owned_pid):
                process = psutil.Process(owned_pid)
                assert process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE
        assert not psutil.pid_exists(router_pid)
        assert not Path("/dev/shm", memory_name).exists()
        assert not Path("/dev/shm", "sem." + semaphore_name.lstrip("/")).exists()
        assert ray.get(unrelated.read.remote()) == []
        manager.shutdown()
    finally:
        ray.shutdown()
        ray_temp.cleanup()
