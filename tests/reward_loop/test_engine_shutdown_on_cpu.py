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
"""Reward server shutdown ordering and real HTTP shutdown contracts."""

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import psutil
import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf
from verl.workers.rollout.utils import run_uvicorn

from verl_omni.reward_loop import engine_shutdown as shutdown
from verl_omni.reward_loop import reward_loop as loops
from verl_omni.reward_loop import reward_model as models


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_serve", [False, True])
async def test_real_http_shutdown_closes_socket_and_lifespan(cancel_serve):
    calls = []

    @asynccontextmanager
    async def lifespan(app):
        calls.append("startup")
        yield
        calls.append("shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def ready():
        return {"ready": True}

    port, task = await run_uvicorn(app, SimpleNamespace(), "127.0.0.1")
    worker = SimpleNamespace(_server_task=task)
    shutdown.retain_http_server(worker)
    server = worker._http_server
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            url = f"http://127.0.0.1:{port}/"
            assert (await client.get(url)).status_code == 200
            if cancel_serve:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (await client.get(url)).status_code == 200
                with pytest.raises(RuntimeError, match="cancelled"):
                    shutdown.http_shutdown_complete(worker)
                assert calls == ["startup"]
            else:
                shutdown.request_http_shutdown(worker)
                await asyncio.wait_for(task, 5)
                assert shutdown.http_shutdown_complete(worker)
                assert calls == ["startup", "shutdown"]
                with pytest.raises(httpx.ConnectError):
                    await client.get(url)
    finally:
        if cancel_serve:
            await server.shutdown()
        elif not task.done():
            server.should_exit = True
            await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_unknown_http_task_cannot_be_declared_closed():
    task = asyncio.create_task(asyncio.sleep(10))
    try:
        with pytest.raises(RuntimeError, match="Cannot identify"):
            shutdown.request_http_shutdown(SimpleNamespace(_server_task=task))
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class _Ref:
    def __init__(self, call):
        self.call = call


def _get(ref, timeout):
    return ref.call()


def _actor(worker, actor_id="owned"):
    return SimpleNamespace(
        _actor_id=actor_id,
        __ray_call__=SimpleNamespace(remote=lambda callback: _Ref(lambda: callback(worker))),
        collective_rpc=SimpleNamespace(remote=lambda method, timeout: _Ref(lambda: worker.events.append(method))),
    )


@pytest.mark.parametrize("failure", ["none", "engine_error", "engine_timeout", "partial", "http_timeout"])
def test_server_shutdown_confirms_owned_termination_after_grace_or_failure(monkeypatch, failure):
    events = []
    worker = SimpleNamespace(events=events)

    def stop_engine(timeout):
        events.append("engine")
        if failure == "engine_error":
            raise ValueError("engine shutdown failed")
        if failure == "engine_timeout":
            raise shutdown.ray.exceptions.GetTimeoutError("engine shutdown timed out")

    worker.engine = None if failure == "partial" else SimpleNamespace(output_handler=None, shutdown=stop_engine)
    server = _actor(worker)
    identity = {"owned": True}
    monkeypatch.setattr(shutdown.ray, "get", _get)
    monkeypatch.setattr(shutdown, "actor_process_identity", lambda _: identity)
    monkeypatch.setattr(shutdown, "request_http_shutdown", lambda _: events.append("http"))

    def http_complete(_):
        if failure == "http_timeout":
            raise TimeoutError("HTTP still serving")
        events.append("lifespan stopped")
        return True

    monkeypatch.setattr(shutdown, "http_shutdown_complete", http_complete)

    def confirm(actor, captured, timeout):
        assert actor is server and captured is identity
        events.append("consumers stopped")

    monkeypatch.setattr(shutdown, "terminate_actor_and_wait", confirm)
    shutdown._shutdown_server(server)
    expected = ["http"]
    if failure != "http_timeout":
        expected.append("lifespan stopped")
        if failure != "partial":
            expected += ["shutdown", "engine"]
    assert events == expected + ["consumers stopped"]


def _manager(servers):
    manager = object.__new__(shutdown.RewardEngineManager)
    manager.config = OmegaConf.create({"rollout": {"name": "vllm", "engine_kwargs": {}}})
    manager.rollout_replicas = [SimpleNamespace(servers=list(servers), workers=[])]
    manager._router_process = None
    return manager


def test_manager_reports_unconfirmed_termination_and_still_closes_other_engines(monkeypatch):
    failed, healthy = _actor(None, "failed"), _actor(None, "healthy")
    manager = _manager([failed, healthy])
    calls = []

    def stop(actor):
        calls.append(actor._actor_id)
        if actor is failed:
            raise TimeoutError("consumer still alive")

    monkeypatch.setattr(shutdown, "_shutdown_server", stop)
    with pytest.raises(RuntimeError, match="not confirmed"):
        manager.shutdown()
    assert calls == ["failed", "healthy"]
    assert manager.rollout_replicas[0].servers == [failed]


def test_manager_repeated_shutdown_does_not_repeat_side_effects(monkeypatch):
    actor = _actor(None)
    manager = _manager([actor])
    calls = []
    monkeypatch.setattr(shutdown, "_shutdown_server", calls.append)
    manager.shutdown()
    manager.shutdown()
    assert calls == [actor]
    assert all(not replica.servers for replica in manager.rollout_replicas)


def test_manager_retains_and_confirms_router_before_stopping_engines(monkeypatch):
    calls = []

    class Process:
        alive = True
        pid = 12345

        def start(self):
            calls.append("start router")

        def is_alive(self):
            return self.alive

        def terminate(self):
            calls.append("terminate router")

        def join(self, timeout):
            calls.append("join router")

        def kill(self):
            calls.append("kill router")
            self.alive = False

        def close(self):
            calls.append("close router")

    process = Process()
    manager = _manager([_actor(None)])
    manager.server_addresses = ["127.0.0.1:12345"]
    monkeypatch.setattr(shutdown.multiprocessing, "Process", lambda **kwargs: process)
    monkeypatch.setattr(shutdown.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(shutdown, "get_free_port", lambda _: (12345, None))
    monkeypatch.setattr(shutdown, "time", SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setattr(shutdown, "_shutdown_server", lambda _: calls.append("engine"))
    manager._initialize_router()
    assert manager._router_process is process
    manager.shutdown()
    assert calls == [
        "start router",
        "terminate router",
        "join router",
        "kill router",
        "join router",
        "close router",
        "engine",
    ]
    manager.shutdown()
    assert manager._router_process is None


@pytest.mark.parametrize("failure", ["start", "exit"])
def test_router_startup_failure_closes_process_handle_and_preserves_error(monkeypatch, failure):
    error = OSError("cannot fork router")
    calls = []

    class Process:
        pid = None

        def start(self):
            if failure == "start":
                raise error
            self.pid = 12345

        def is_alive(self):
            return False

        def join(self, timeout):
            assert self.pid is not None
            calls.append("join")

        def close(self):
            calls.append("close")

    def initialize(manager, config, resource_pool):
        manager.server_addresses = ["127.0.0.1:9"]
        manager._initialize_router()

    monkeypatch.setattr(shutdown.multiprocessing, "Process", lambda **kwargs: Process())
    monkeypatch.setattr(shutdown.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(shutdown, "get_free_port", lambda _: (12345, None))
    monkeypatch.setattr(shutdown, "time", SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setattr(shutdown.RewardModelManager, "__init__", initialize)
    with pytest.raises(OSError if failure == "start" else RuntimeError) as captured:
        shutdown.RewardEngineManager(_manager([]).config)
    if failure == "start":
        assert captured.value is error
    else:
        assert "exited during startup" in str(captured.value)
    assert not getattr(captured.value, "__notes__", [])
    assert calls == (["close"] if failure == "start" else ["join", "close"])


def test_router_startup_interrupt_closes_real_listener_and_process(monkeypatch):
    error = KeyboardInterrupt("router startup interrupted")
    owned = []
    original_process = shutdown.multiprocessing.Process

    def create_process(**kwargs):
        process = original_process(**kwargs)
        owned.append({"process": process, "port": kwargs["args"][1]})
        return process

    def initialize(manager, config, resource_pool):
        manager.server_addresses = ["127.0.0.1:9"]
        manager._initialize_router()

    def interrupt(seconds):
        assert seconds == 3
        record = owned[0]
        record["pid"] = record["process"].pid
        record["created"] = psutil.Process(record["pid"]).create_time()
        deadline = time.monotonic() + 15
        while True:
            try:
                with socket.create_connection(("127.0.0.1", record["port"]), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Router did not start listening") from None
                time.sleep(0.05)
        raise error

    monkeypatch.setattr(shutdown.multiprocessing, "Process", create_process)
    monkeypatch.setattr(shutdown.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(shutdown, "time", SimpleNamespace(sleep=interrupt))
    monkeypatch.setattr(shutdown.RewardModelManager, "__init__", initialize)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            shutdown.RewardEngineManager(_manager([]).config)
        assert captured.value is error and not getattr(error, "__notes__", [])
        record = owned[0]
        assert not psutil.pid_exists(record["pid"])
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", record["port"]), timeout=1)
    finally:
        for record in owned:
            try:
                process = psutil.Process(record["pid"])
                if process.create_time() == record["created"]:
                    process.terminate()
                    process.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


def test_partial_initialization_preserves_error_and_reports_cleanup_failure(monkeypatch):
    error = ValueError("second replica init failed")
    cleanup_error = TimeoutError("first replica cannot stop")

    def initialize(manager, config, resource_pool):
        manager.rollout_replicas = [SimpleNamespace(servers=[_actor(None)], workers=[])]
        raise error

    monkeypatch.setattr(shutdown.RewardModelManager, "__init__", initialize)
    monkeypatch.setattr(shutdown, "_shutdown_server", lambda _: (_ for _ in ()).throw(cleanup_error))
    with pytest.raises(ValueError) as captured:
        shutdown.RewardEngineManager(_manager([]).config)
    assert captured.value is error
    assert "cleanup also failed" in captured.value.__notes__[0]


def test_named_model_initialization_failure_closes_previously_created_engine(monkeypatch):
    config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False},
                "models": {"first": {"backend": "engine"}, "second": {"backend": "engine"}},
            }
        }
    )
    calls = []

    def initialize(model, name, *args):
        if name == "second":
            raise ValueError("second model init failed")
        model.spec = SimpleNamespace(name=name)
        model.reward_model_manager = SimpleNamespace(config=_manager([]).config, shutdown=lambda: calls.append(name))

    monkeypatch.setattr(models.EngineManagedRewardModel, "__init__", initialize)
    monkeypatch.setattr(models.MultiRewardModelManager, "_split_model_resource_pools", lambda *args: ({}, {}))
    with pytest.raises(ValueError, match="second model"):
        models.MultiRewardModelManager(config)
    assert calls == ["first"]


def test_reward_worker_initialization_failure_closes_created_model_manager(monkeypatch):
    calls = []
    config = OmegaConf.create(
        {"reward": {"models": {}, "reward_functions": {"ocr": {}}, "reward_model": {"enable": False}}}
    )
    owned = SimpleNamespace(models={"ocr": object()}, shutdown=lambda: calls.append("shutdown"))
    monkeypatch.setattr(loops, "MultiRewardModelManager", lambda *args, **kwargs: owned)
    monkeypatch.setattr(loops, "accelerator_workers_enabled", lambda config: False)
    monkeypatch.setattr(loops, "resolve_reward_manager_cls", lambda config: object)
    error = ValueError("reward worker init failed")

    def initialize_workers(manager):
        raise error

    monkeypatch.setattr(loops.OmniRewardLoopManager, "_init_reward_loop_workers", initialize_workers)
    with pytest.raises(ValueError) as captured:
        loops.OmniRewardLoopManager(config)
    assert captured.value is error
    assert calls == ["shutdown"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distributed-executor-backend": "ray"},
        {"data_parallel_backend": "ray"},
        {"distributed_executor_backend": "mp", "distributed-executor-backend": "ray"},
        {"data_parallel_backend": "mp", "data-parallel-backend": "ray"},
    ],
)
def test_remote_executor_is_not_mistaken_for_an_owned_local_process_tree(kwargs):
    config = _manager([]).config
    config.rollout.engine_kwargs = {"vllm": kwargs}
    assert not shutdown.supports_local_shutdown(config)


@pytest.mark.parametrize("consumer_stopped", [True, False])
def test_owned_placement_workers_stop_only_after_all_replica_consumers(monkeypatch, consumer_stopped):
    server, worker = _actor(None, "consumer"), _actor(None, "placement")
    manager = _manager([server])
    manager.rollout_replicas[0].workers = [worker]
    events = []

    def stop_consumer(actor):
        events.append("consumer")
        if not consumer_stopped:
            raise TimeoutError("consumer stop unconfirmed")

    monkeypatch.setattr(shutdown, "_shutdown_server", stop_consumer)
    monkeypatch.setattr(shutdown, "_shutdown_worker", lambda actor: events.append("placement"))
    if consumer_stopped:
        manager.shutdown()
        assert events == ["consumer", "placement"]
        assert manager.rollout_replicas[0].workers == []
    else:
        with pytest.raises(RuntimeError, match="not confirmed"):
            manager.shutdown()
        assert events == ["consumer"]
        assert manager.rollout_replicas[0].workers == [worker]


def test_partial_replica_without_servers_still_closes_its_owned_workers(monkeypatch):
    worker = _actor(None, "placement")
    manager = _manager([])
    manager.rollout_replicas[0].workers = [worker]
    calls = []
    monkeypatch.setattr(shutdown, "_shutdown_worker", calls.append)
    manager.shutdown()
    manager.shutdown()
    assert calls == [worker]
