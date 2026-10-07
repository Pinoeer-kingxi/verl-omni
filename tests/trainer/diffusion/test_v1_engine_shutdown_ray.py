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
"""Real Ray consumers with HTTP serving and an engine child process; no GPUs."""

import asyncio
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from types import SimpleNamespace

import psutil
import pytest
import ray

from verl_omni.trainer import main_diffusion_v1 as entrypoint
from verl_omni.utils.ray_lifecycle import actor_process_identity, wait_for_actor_exit
from verl_omni.workers.rollout.http_server import run_uvicorn


@ray.remote(num_cpus=1)
class Consumer:
    async def start(self, failure):
        from fastapi import FastAPI

        self.lifespan_closed = False
        self.failure = failure
        self.workers_released = False

        @asynccontextmanager
        async def lifespan(app):
            yield
            self.lifespan_closed = True

        app = FastAPI(lifespan=lifespan)

        @app.get("/")
        async def root():
            return {"ok": True}

        self.child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)", "verl-omni-teardown-test"]
        )

        def shutdown(**kwargs):
            assert self.lifespan_closed, "HTTP lifespan must finish before engine shutdown"
            if kwargs:
                assert kwargs == {"timeout": 90} and self.workers_released
            if failure == "error":
                raise RuntimeError("injected engine shutdown error")
            if failure == "timeout":
                time.sleep(3600)
            self.child.terminate()
            self.child.wait(timeout=5)

        self.engine = SimpleNamespace(shutdown=shutdown)
        port, self._server_task, self._http_server = await run_uvicorn(app, None, "127.0.0.1")
        return port, os.getpid(), self.child.pid

    async def collective_rpc(self, method, timeout):
        assert self.lifespan_closed and method == "shutdown" and timeout == 90
        if self.failure == "prepare_error":
            raise RuntimeError("injected worker resource release error")
        if self.failure == "prepare_timeout":
            await asyncio.sleep(3600)
        self.workers_released = True


@ray.remote(num_cpus=0)
class Producer:
    def identity(self):
        return actor_process_identity(self)

    def start_child(self):
        code = """
import subprocess
import sys
import time
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(3600)', 'verl-omni-producer-exit-test'],
    start_new_session=True,
)
print(child.pid, flush=True)
time.sleep(3600)
"""
        self.child = subprocess.Popen(
            [sys.executable, "-c", code, "verl-omni-producer-exit-test"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.child.stdout.readline()
        return self.identity()

    def detach_child(self):
        self.child.terminate()
        self.child.wait(timeout=3)


@pytest.fixture(scope="module")
def ray_cluster(tmp_path_factory):
    ray.init(
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _temp_dir=str(tmp_path_factory.mktemp("rt")),
    )
    yield
    ray.shutdown()


def running(pid):
    try:
        return psutil.Process(pid).status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def test_graceful_producer_exit_is_confirmed_with_real_ray(ray_cluster):
    producer = Producer.remote()
    try:
        identity = ray.get(producer.identity.remote(), timeout=60)
        task = entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
        task._trainer_initialized = True
        task.trainer = SimpleNamespace(
            actor_rollout_wg=SimpleNamespace(
                workers=[
                    SimpleNamespace(
                        __ray_call__=SimpleNamespace(remote=lambda callback: producer.identity.remote()),
                        __ray_terminate__=producer.__ray_terminate__,
                    )
                ]
            )
        )
        started = time.monotonic()
        task._shutdown_vllm_engines()
        assert time.monotonic() - started < 35
        assert not running(identity["pid"])
    finally:
        ray.kill(producer, no_restart=True)


def test_graceful_exit_confirmation_is_bounded_and_does_not_kill_live_actor(ray_cluster):
    producer = Producer.remote()
    try:
        identity = ray.get(producer.identity.remote(), timeout=60)
        with pytest.raises((TimeoutError, ray.exceptions.GetTimeoutError)):
            wait_for_actor_exit(identity, 0.2)
        assert ray.get(producer.identity.remote(), timeout=10) == identity
        # A reused PID is a different process; confirmation must never signal it.
        wait_for_actor_exit({**identity, "created": identity["created"] - 1}, 10)
        assert ray.get(producer.identity.remote(), timeout=10) == identity
    finally:
        ray.kill(producer, no_restart=True)


def test_actor_exit_does_not_hide_a_surviving_owned_child(ray_cluster):
    producer = Producer.remote()
    identity = None
    try:
        identity = ray.get(producer.start_child.remote(), timeout=60)
        assert len(identity["children"]) == 2
        ray.get(producer.detach_child.remote(), timeout=10)
        try:
            ray.get(producer.__ray_terminate__.remote(), timeout=30)
        except (ray.exceptions.ActorDiedError, ray.exceptions.ActorUnavailableError):
            pass
        with pytest.raises((TimeoutError, ray.exceptions.GetTimeoutError)):
            wait_for_actor_exit(identity, 0.2)
        assert any(running(owned["pid"]) for owned in identity["children"])
    finally:
        ray.kill(producer, no_restart=True)
        if identity is not None:
            for owned in identity["children"]:
                if running(owned["pid"]):
                    child = psutil.Process(owned["pid"])
                    assert child.create_time() == owned["created"]
                    assert child.cmdline()[-1] == "verl-omni-producer-exit-test"
                    child.terminate()
                    child.wait(timeout=3)


@pytest.mark.parametrize(
    ("backend", "failure"),
    [
        ("vllm_omni", None),
        ("vllm_omni", "error"),
        ("vllm_omni", "timeout"),
        ("vllm", None),
        ("vllm", "error"),
        ("vllm", "timeout"),
        ("vllm", "prepare_error"),
        ("vllm", "prepare_timeout"),
    ],
)
def test_http_and_engine_consumers_stop_before_producer_cleanup(ray_cluster, monkeypatch, backend, failure):
    consumer = Consumer.remote()
    port, actor_pid, child_pid = ray.get(consumer.start.remote(failure), timeout=30)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
        assert response.status == 200
    cleaned = []
    forced_terminations = []
    original_terminate = entrypoint.terminate_actor_and_wait

    def terminate(*args):
        forced_terminations.append(True)
        return original_terminate(*args)

    monkeypatch.setattr(entrypoint, "terminate_actor_and_wait", terminate)

    def producer_cleanup(callback):
        assert not running(child_pid)
        if failure:
            assert not running(actor_pid)
        with pytest.raises(OSError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
        cleaned.append(True)

    task = entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
    task._trainer_initialized = True
    task.trainer = SimpleNamespace(
        llm_server_manager=SimpleNamespace(
            rollout_config=SimpleNamespace(name=backend), server_handles=[consumer], rollout_replicas=[]
        ),
        actor_rollout_wg=SimpleNamespace(
            workers=[SimpleNamespace(__ray_call__=SimpleNamespace(remote=producer_cleanup))]
        ),
    )
    monkeypatch.setattr(
        entrypoint, "_ENGINE_SHUTDOWN_TIMEOUT", 0.1 if failure in ("timeout", "prepare_timeout") else 10
    )
    monkeypatch.setattr(entrypoint, "_ACTOR_TERMINATION_TIMEOUT", 30)
    original_get = entrypoint.ray.get
    monkeypatch.setattr(
        entrypoint.ray, "get", lambda ref, timeout: None if ref is None else original_get(ref, timeout=timeout)
    )
    try:
        started = time.monotonic()
        task._shutdown_vllm_engines()
        assert cleaned == [True]
        assert len(forced_terminations) == (0 if failure is None else 1)
        assert time.monotonic() - started < 45
    finally:
        ray.kill(consumer, no_restart=True)
        if running(child_pid):
            child = psutil.Process(child_pid)
            assert child.cmdline()[-1] == "verl-omni-teardown-test"
            child.kill()
            psutil.wait_procs([child], timeout=3)


@pytest.mark.parametrize("failure", [None, "error", "timeout"])
def test_remote_ray_consumer_outside_server_process_tree_blocks_ipc_cleanup(ray_cluster, monkeypatch, failure):
    consumer = Consumer.remote()
    remote_consumer = Producer.remote()
    child_pid = None
    remote_identity = None
    cleaned = []
    try:
        port, actor_pid, child_pid = ray.get(consumer.start.remote(failure), timeout=60)
        identity = ray.get(consumer.__ray_call__.remote(actor_process_identity), timeout=60)
        remote_identity = ray.get(remote_consumer.identity.remote(), timeout=60)
        assert remote_identity["pid"] not in {identity["pid"], *(p["pid"] for p in identity["children"])}

        def producer_cleanup(callback):
            cleaned.append(True)
            return ray.put(None)

        task = entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
        task._trainer_initialized = True
        task.trainer = SimpleNamespace(
            llm_server_manager=SimpleNamespace(
                rollout_config=SimpleNamespace(
                    name="vllm_omni", engine_kwargs={"vllm_omni": {"distributed_executor_backend": "ray"}}
                ),
                server_handles=[consumer],
                rollout_replicas=[],
            ),
            actor_rollout_wg=SimpleNamespace(
                workers=[SimpleNamespace(__ray_call__=SimpleNamespace(remote=producer_cleanup))]
            ),
        )
        monkeypatch.setattr(entrypoint, "_ENGINE_SHUTDOWN_TIMEOUT", 0.1 if failure == "timeout" else 10)
        monkeypatch.setattr(entrypoint, "_ACTOR_TERMINATION_TIMEOUT", 30)
        with pytest.raises(RuntimeError, match="IPC cleanup was skipped") as captured:
            task._shutdown_vllm_engines()
        assert "executor" in str(captured.value.__cause__)
        assert cleaned == []
        assert not running(actor_pid) and not running(child_pid)
        with pytest.raises(OSError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
        assert ray.get(remote_consumer.identity.remote(), timeout=10) == remote_identity
    finally:
        ray.kill(consumer, no_restart=True)
        ray.kill(remote_consumer, no_restart=True)
        if child_pid is not None and running(child_pid):
            child = psutil.Process(child_pid)
            assert child.cmdline()[-1] == "verl-omni-teardown-test"
            child.kill()
            psutil.wait_procs([child], timeout=3)
