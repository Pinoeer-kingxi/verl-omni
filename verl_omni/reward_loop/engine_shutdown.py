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
"""Final shutdown of owned named vLLM reward servers and their router."""

import gc
import logging
import multiprocessing
import time
from typing import Any

import ray
import uvicorn
from verl.experimental.reward_loop.reward_model import RewardModelManager
from verl.experimental.reward_loop.router.naive_router import run_router
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

from verl_omni.utils.ray_lifecycle import actor_process_identity, terminate_actor_and_wait

logger = logging.getLogger(__name__)

_HTTP_SHUTDOWN_TIMEOUT = 30
_ENGINE_SHUTDOWN_TIMEOUT = 120
_ENGINE_PROCESS_GRACE = 90
_ACTOR_TERMINATION_TIMEOUT = 30


def supports_local_shutdown(config) -> bool:
    """Return whether all vLLM consumers live in the owned process trees."""
    kwargs = config.rollout.engine_kwargs.get("vllm", {}) or {}
    executors = [
        kwargs[key] for key in ("distributed_executor_backend", "distributed-executor-backend") if key in kwargs
    ]
    dp_backends = [kwargs[key] for key in ("data_parallel_backend", "data-parallel-backend") if key in kwargs]
    return (
        config.rollout.name == "vllm"
        and all(value in {"mp", "uni"} for value in executors)
        and all(value == "mp" for value in dp_backends)
    )


def retain_http_server(worker: Any) -> None:
    """Retain the Server exposed by the pinned upstream serve coroutine."""
    task = getattr(worker, "_server_task", None)
    if task is None:
        return
    server = getattr(worker, "_http_server", None)
    if server is None:
        # Pinned verl returns only Server.serve()'s task, whose live frame owns
        # the Server. A cancelled or unidentified task needs process termination.
        frame = task.get_coro().cr_frame
        server = frame.f_locals.get("self") if frame is not None else None
        if not isinstance(server, uvicorn.Server):
            raise RuntimeError("Cannot identify the owned Uvicorn Server")
        worker._http_server = server


def request_http_shutdown(worker: Any) -> None:
    """Request normal HTTP shutdown on the server's event loop."""
    retain_http_server(worker)
    task = getattr(worker, "_server_task", None)
    if task is not None:
        task.get_loop().call_soon_threadsafe(setattr, worker._http_server, "should_exit", True)


def http_shutdown_complete(worker: Any) -> bool:
    """Confirm listeners and lifespan closed, rather than task cancellation."""
    task = getattr(worker, "_server_task", None)
    if task is None:
        return True
    if not task.done():
        return False
    if task.cancelled():
        raise RuntimeError("HTTP task was cancelled without graceful shutdown")
    task.result()
    server = worker._http_server
    if any(listener.is_serving() for listener in server.servers):
        raise RuntimeError("HTTP listening sockets are still open")
    if server.started:
        if server.lifespan.shutdown_failed or not server.lifespan.shutdown_event.is_set():
            raise RuntimeError("HTTP application lifespan shutdown did not complete")
    return True


def _cancel_output_handler(worker: Any) -> bool:
    engine = getattr(worker, "engine", None)
    if engine is None:
        return False
    handler = engine.output_handler
    if handler is not None:
        handler.get_loop().call_soon_threadsafe(handler.cancel)
    return True


def _shutdown_engine(worker: Any) -> None:
    engine = getattr(worker, "engine", None)
    if engine is not None:
        handler = engine.output_handler
        if handler is not None and not handler.done():
            raise RuntimeError("vLLM output handler cancellation did not finish")
        engine.shutdown(timeout=_ENGINE_PROCESS_GRACE)
    worker.engine = None
    worker._server_task = None
    worker._http_server = None
    gc.collect()


def close_http_server(server: ray.actor.ActorHandle, timeout: float) -> None:
    """Close listening sockets and application lifespan before engine shutdown."""
    deadline = time.monotonic() + timeout
    ray.get(server.__ray_call__.remote(request_http_shutdown), timeout=timeout)
    while not ray.get(
        server.__ray_call__.remote(http_shutdown_complete), timeout=max(0.001, deadline - time.monotonic())
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError("Reward HTTP shutdown deadline expired")
        time.sleep(0.01)


def _shutdown_server(server: ray.actor.ActorHandle) -> None:
    identity = None
    try:
        identity = ray.get(server.__ray_call__.remote(actor_process_identity), timeout=_HTTP_SHUTDOWN_TIMEOUT)
        close_http_server(server, _HTTP_SHUTDOWN_TIMEOUT)
        initialized = ray.get(server.__ray_call__.remote(_cancel_output_handler), timeout=_HTTP_SHUTDOWN_TIMEOUT)
        if initialized:
            # Release worker resources before vLLM's outer process manager exits.
            ray.get(
                server.collective_rpc.remote("shutdown", timeout=_ENGINE_PROCESS_GRACE),
                timeout=_ENGINE_SHUTDOWN_TIMEOUT,
            )
        ray.get(server.__ray_call__.remote(_shutdown_engine), timeout=_ENGINE_SHUTDOWN_TIMEOUT)
    except Exception:
        logger.warning("Graceful reward engine shutdown failed; terminating its owned processes", exc_info=True)
    # Closing the actor also closes remaining tracker pipes. Both normal and
    # failed engine shutdown must confirm all owned consumer processes stopped.
    terminate_actor_and_wait(server, identity, _ACTOR_TERMINATION_TIMEOUT)


def _shutdown_worker(worker: ray.actor.ActorHandle) -> None:
    identity = None
    try:
        identity = ray.get(worker.__ray_call__.remote(actor_process_identity), timeout=_ACTOR_TERMINATION_TIMEOUT)
    except Exception:
        logger.warning("Cannot capture reward placement worker identity", exc_info=True)
    terminate_actor_and_wait(worker, identity, _ACTOR_TERMINATION_TIMEOUT)


class RewardEngineManager(RewardModelManager):
    """Keep ownership of the upstream reward router and initialized replicas.

    Normal construction and serving use the pinned upstream manager. Explicit
    final shutdown currently covers vLLM with local mp/uni executors.
    """

    def __init__(self, config, resource_pool=None):
        self.config = config
        self.rollout_replicas = []
        self._router_process = None
        try:
            super().__init__(config, resource_pool)
        except BaseException as error:
            if not supports_local_shutdown(config):
                raise
            try:
                self.shutdown()
            except Exception as cleanup_error:
                error.add_note(f"Reward engine initialization cleanup also failed: {cleanup_error}")
                logger.exception("Reward engine initialization cleanup failed")
            raise

    def _initialize_router(self):
        urls = [f"http://{address}" for address in self.server_addresses]
        router_ip = ray.util.get_node_ip_address().strip("[]")
        router_port, _ = get_free_port(router_ip)
        self.router_address = (
            f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
        )
        # Retain ownership before the pinned upstream startup wait can be interrupted.
        self._router_process = multiprocessing.Process(target=run_router, args=(router_ip, router_port, urls))
        self._router_process.daemon = True
        self._router_process.start()
        time.sleep(3)
        if not self._router_process.is_alive():
            raise RuntimeError("Reward router exited during startup")

    def _initialize_llm_servers(self):
        super()._initialize_llm_servers()
        if supports_local_shutdown(self.config):
            ray.get(
                [
                    server.__ray_call__.remote(retain_http_server)
                    for replica in self.rollout_replicas
                    for server in replica.servers
                ],
                timeout=_HTTP_SHUTDOWN_TIMEOUT,
            )

    def shutdown(self) -> None:
        """Stop admission, engines and actors; report unconfirmed termination."""
        if not supports_local_shutdown(self.config):
            raise NotImplementedError("Reward shutdown requires vLLM with local mp/uni and mp data-parallel workers")
        failures = []
        if self._router_process is not None:
            process = self._router_process
            try:
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=_HTTP_SHUTDOWN_TIMEOUT)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=_ACTOR_TERMINATION_TIMEOUT)
                    if process.is_alive():
                        raise TimeoutError("Reward router did not stop before its deadline")
                process.close()
                self._router_process = None
            except Exception as error:
                failures.append(error)

        for replica in self.rollout_replicas:
            for server in list(replica.servers):
                try:
                    _shutdown_server(server)
                    replica.servers.remove(server)
                except Exception as error:
                    failures.append(error)
                    logger.exception("Cannot confirm reward engine consumer termination")
            if not replica.servers:
                for worker in list(replica.workers):
                    try:
                        _shutdown_worker(worker)
                        replica.workers.remove(worker)
                    except Exception as error:
                        failures.append(error)
                        logger.exception("Cannot confirm reward placement worker termination")
        if failures:
            raise RuntimeError("Reward engine cleanup failed; consumer termination was not confirmed") from failures[0]
