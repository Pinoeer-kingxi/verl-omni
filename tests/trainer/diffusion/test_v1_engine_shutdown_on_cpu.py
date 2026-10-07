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
"""V1 consumer shutdown, forced termination and producer cleanup ordering."""

import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.reward_loop.reward_model import EngineManagedRewardModel
from verl_omni.trainer import main_diffusion_v1 as entrypoint


def runner():
    task = entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
    task._trainer_initialized = True
    return task


class Task:
    def __init__(self):
        self.finished = False

    def get_loop(self):
        return SimpleNamespace(call_soon_threadsafe=lambda fn, *args: fn(*args))

    def cancel(self):
        self.finished = True

    def done(self):
        return self.finished

    def cancelled(self):
        return False

    def result(self):
        return None


class Server:
    def __init__(self, task, calls, ignore_exit=False):
        self.task = task
        self.calls = calls
        self.ignore_exit = ignore_exit
        self.started = True
        self.servers = [SimpleNamespace(is_serving=lambda: not task.finished)]
        self.lifespan = SimpleNamespace(shutdown_event=SimpleNamespace(is_set=lambda: task.finished))

    @property
    def should_exit(self):
        return self.task.finished

    @should_exit.setter
    def should_exit(self, value):
        self.calls.append("http exit")
        if not self.ignore_exit:
            self.task.finished = True


def manager(calls, name="vllm", error=None, rollout=False, ignore_exit=False, partial=False):
    task = Task()
    worker = SimpleNamespace(_server_task=task, _http_server=Server(task, calls, ignore_exit))

    def shutdown(**kwargs):
        assert kwargs == ({"timeout": 90} if name == "vllm" else {})
        assert task.done()
        assert worker._http_server.lifespan.shutdown_event.is_set()
        if name == "vllm":
            assert worker.engine.output_handler.done()
        calls.append("shutdown")
        if error:
            raise error

    worker.engine = None if partial else SimpleNamespace(shutdown=shutdown, output_handler=Task())
    server = SimpleNamespace(
        __ray_call__=SimpleNamespace(remote=lambda callback: callback(worker)),
        __ray_terminate__=SimpleNamespace(remote=lambda: None),
        collective_rpc=SimpleNamespace(remote=lambda method, timeout: None),
    )
    result = SimpleNamespace(server_handles=[server], rollout_replicas=[])
    if rollout:
        result.rollout_config = SimpleNamespace(name=name)
    else:
        result.config = SimpleNamespace(rollout=SimpleNamespace(name=name))
    return result


@pytest.fixture(autouse=True)
def ray_callbacks(monkeypatch):
    monkeypatch.setattr(entrypoint.ray, "get", lambda ref, timeout: ref)
    monkeypatch.setattr(entrypoint, "actor_process_identity", lambda worker: {"owned": True})
    monkeypatch.setattr(entrypoint, "wait_for_actor_exit", lambda identity, timeout: None)


def test_closes_rollout_standalone_legacy_and_named_engines():
    calls = []
    task = runner()
    named = object.__new__(EngineManagedRewardModel)
    named.reward_model_manager = manager(calls)
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", rollout=True),
        standalone_server_manager=manager(calls, name="vllm_omni", rollout=True),
        reward_loop_manager=SimpleNamespace(
            reward_model_manager=manager(calls),
            multi_reward_model_manager=SimpleNamespace(models={"ocr": named, "native": object()}),
        ),
    )
    task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown"] * 4


@pytest.mark.parametrize("trainer", [None, SimpleNamespace()])
def test_partial_initialization_without_managers(trainer):
    task = runner()
    task.trainer = trainer
    task._shutdown_vllm_engines()


def test_partial_replica_server_is_closed_even_without_head_handle():
    calls = []
    owned = manager(calls, name="vllm_omni", rollout=True, partial=True)
    owned.rollout_replicas = [SimpleNamespace(servers=owned.server_handles)]
    owned.server_handles = []
    task = runner()
    task.trainer = SimpleNamespace(llm_server_manager=owned)
    task._shutdown_vllm_engines()
    assert calls == ["http exit"]


def test_deduplicates_replica_head_handles():
    calls = []
    owned = manager(calls, name="vllm_omni", rollout=True)
    owned.rollout_replicas = [SimpleNamespace(servers=owned.server_handles)]
    task = runner()
    task.trainer = SimpleNamespace(llm_server_manager=owned)
    task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown"]


def test_graceful_consumer_exit_must_be_confirmed_before_producer_cleanup(monkeypatch):
    calls = []
    task = runner()
    owned = manager(calls, name="vllm_omni", rollout=True)
    owned.server_handles[0].__ray_terminate__.remote = lambda: calls.append("exit")

    def confirm(identity, timeout):
        assert calls == ["http exit", "shutdown", "exit"]
        calls.append("confirm")

    monkeypatch.setattr(entrypoint, "wait_for_actor_exit", confirm)
    task.trainer = SimpleNamespace(
        llm_server_manager=owned,
        actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]),
    )
    task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown", "exit", "confirm", "producer cleanup"]


def test_unconfirmed_graceful_consumer_exit_skips_producer_cleanup(monkeypatch):
    calls = []
    task = runner()

    def fail_confirmation(*args):
        calls.append("confirm")
        raise TimeoutError("consumer child still alive")

    monkeypatch.setattr(entrypoint, "wait_for_actor_exit", fail_confirmation)
    monkeypatch.setattr(entrypoint, "terminate_actor_and_wait", fail_confirmation)
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", rollout=True),
        actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]),
    )
    with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
        task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown", "confirm", "confirm"]


def test_unknown_engine_backend_reports_failure_and_skips_producer_cleanup():
    calls = []
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="sglang", rollout=True),
        actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]),
    )
    with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
        task._shutdown_vllm_engines()
    assert calls == []


@pytest.mark.parametrize("executor", ["ray", "external_launcher", "custom.Executor"])
@pytest.mark.parametrize("backend", ["vllm", "vllm_omni"])
@pytest.mark.parametrize("failure", [None, "shutdown", "timeout", "partial"])
def test_external_executor_skips_ipc_even_after_local_server_exits(monkeypatch, executor, backend, failure):
    calls = []
    owned = manager(
        calls,
        name=backend,
        rollout=True,
        error=(
            RuntimeError("shutdown failed")
            if failure == "shutdown"
            else TimeoutError("shutdown timed out")
            if failure == "timeout"
            else None
        ),
        partial=failure == "partial",
    )
    owned.rollout_config.engine_kwargs = {backend: {"distributed_executor_backend": executor}}
    monkeypatch.setattr(entrypoint, "terminate_actor_and_wait", lambda *args: calls.append("local stopped"))
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=owned, actor_rollout_wg=SimpleNamespace(workers=[producer(calls)])
    )
    with pytest.raises(RuntimeError, match="IPC cleanup was skipped") as captured:
        task._shutdown_vllm_engines()
    assert "executor" in str(captured.value.__cause__)
    assert "producer cleanup" not in calls
    assert calls[0] == "http exit"


@pytest.mark.parametrize("executor", [None, "mp", "uni"])
@pytest.mark.parametrize("backend", ["vllm", "vllm_omni"])
def test_local_executor_allows_ipc_after_confirmed_exit(executor, backend):
    calls = []
    owned = manager(calls, name=backend, rollout=True)
    owned.rollout_config.engine_kwargs = {backend: {"distributed_executor_backend": executor}}
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=owned, actor_rollout_wg=SimpleNamespace(workers=[producer(calls)])
    )
    task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown", "producer cleanup"]


@pytest.mark.parametrize("backend", ["vllm", "vllm_omni"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"distributed-executor-backend": "ray"},
        {"distributed_executor_backend": "mp", "distributed-executor-backend": "ray"},
        {"data_parallel_backend": "ray"},
        {"data-parallel-backend": "ray"},
    ],
)
def test_remote_executor_cli_aliases_and_data_parallel_backend_block_ipc(backend, overrides):
    calls = []
    owned = manager(calls, name=backend, rollout=True)
    owned.rollout_config.engine_kwargs = {backend: overrides}
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=owned, actor_rollout_wg=SimpleNamespace(workers=[producer(calls)])
    )
    with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
        task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"worker_backend": "ray"},
        {"deploy_config": "custom.yaml"},
        {"stage_overrides": '{"0": {"distributed_executor_backend": "ray"}}'},
        {"stage-overrides": '{"0": {"distributed_executor_backend": "ray"}}'},
        {"worker-backend": "ray"},
        {"worker_backend": "multi_process", "worker-backend": "ray"},
    ],
)
def test_unverified_omni_stage_ownership_skips_producer_cleanup(overrides):
    calls = []
    owned = manager(calls, name="vllm_omni", rollout=True)
    owned.rollout_config.engine_kwargs = {"vllm_omni": overrides}
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=owned, actor_rollout_wg=SimpleNamespace(workers=[producer(calls)])
    )
    with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
        task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown"]


def producer(calls):
    return SimpleNamespace(__ray_call__=SimpleNamespace(remote=lambda callback: calls.append("producer cleanup")))


@pytest.mark.parametrize(
    ("failure", "confirmed"),
    [
        (None, True),
        (RuntimeError("worker release failed"), True),
        (TimeoutError("worker release timed out"), True),
        (TimeoutError("worker release timed out"), False),
    ],
)
def test_worker_resource_release_precedes_engine_shutdown(monkeypatch, failure, confirmed):
    calls = []
    owned = manager(calls, name="vllm", rollout=True)

    def release(method, timeout):
        assert method == "shutdown" and timeout == 90
        calls.append("worker release")
        if failure:
            raise failure

    def terminate(*args):
        calls.append("terminate")
        if not confirmed:
            raise TimeoutError("consumer still running")
        calls.append("stopped")

    owned.server_handles[0].collective_rpc.remote = release
    monkeypatch.setattr(entrypoint, "terminate_actor_and_wait", terminate)
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=owned, actor_rollout_wg=SimpleNamespace(workers=[producer(calls)])
    )
    if not confirmed:
        with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
            task._shutdown_vllm_engines()
        assert calls == ["http exit", "worker release", "terminate"]
    else:
        task._shutdown_vllm_engines()
        expected = ["shutdown"] if failure is None else ["terminate", "stopped"]
        assert calls == ["http exit", "worker release", *expected, "producer cleanup"]


@pytest.mark.parametrize("error", [RuntimeError("shutdown failed"), TimeoutError("shutdown timed out")])
@pytest.mark.parametrize("confirmed", [False, True])
def test_shutdown_failure_requires_confirmed_consumer_exit_before_ipc(monkeypatch, error, confirmed):
    calls = []

    def terminate(server, identity, timeout):
        assert identity == {"owned": True}
        assert calls == ["http exit", "shutdown"]
        calls.append("terminate")
        if not confirmed:
            raise TimeoutError("consumer still running")
        calls.append("stopped")

    monkeypatch.setattr(entrypoint, "terminate_actor_and_wait", terminate)
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", error=error, rollout=True),
        actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]),
    )
    if confirmed:
        task._shutdown_vllm_engines()
        assert calls == ["http exit", "shutdown", "terminate", "stopped", "producer cleanup"]
    else:
        with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
            task._shutdown_vllm_engines()
        assert calls == ["http exit", "shutdown", "terminate"]


def test_http_timeout_terminates_consumer_without_calling_engine_shutdown(monkeypatch):
    calls = []
    monkeypatch.setattr(entrypoint, "_HTTP_SHUTDOWN_TIMEOUT", 0.02)
    monkeypatch.setattr(entrypoint, "terminate_actor_and_wait", lambda *args: calls.append("stopped"))
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", rollout=True, ignore_exit=True),
        actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]),
    )
    task._shutdown_vllm_engines()
    assert calls == ["http exit", "stopped", "producer cleanup"]


def test_unknown_actor_identity_fails_closed_and_other_engines_still_close(monkeypatch):
    calls = []
    owned = manager(calls, name="vllm_omni", rollout=True)

    def unavailable(callback):
        raise RuntimeError("actor unavailable")

    owned.server_handles[0].__ray_call__.remote = unavailable
    monkeypatch.setattr(
        entrypoint,
        "terminate_actor_and_wait",
        lambda *args: (_ for _ in ()).throw(RuntimeError("unknown identity")),
    )
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=owned,
        reward_loop_manager=SimpleNamespace(reward_model_manager=manager(calls)),
        actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]),
    )
    with pytest.raises(RuntimeError, match="IPC cleanup was skipped"):
        task._shutdown_vllm_engines()
    assert calls == ["http exit", "shutdown"]


def test_incomplete_trainer_init_does_not_release_unknown_consumers_ipc():
    calls = []
    task = runner()
    task._trainer_initialized = False
    task.trainer = SimpleNamespace(actor_rollout_wg=SimpleNamespace(workers=[producer(calls)]))
    with pytest.raises(RuntimeError, match="initialization incomplete"):
        task._shutdown_vllm_engines()
    assert calls == []


@pytest.mark.parametrize("first_worker_fails", [False, True])
@pytest.mark.parametrize("rocm", [False, True])
def test_actor_ipc_is_collected_after_receivers_shutdown(monkeypatch, first_worker_fails, rocm):
    import torch
    from verl.utils import device

    calls = []
    runtime = SimpleNamespace(cudaError_t=SimpleNamespace(cudaSuccess=0), cudaDeviceReset=lambda: (0,))
    monkeypatch.setattr(torch.version, "hip", "simulated-rocm" if rocm else None)
    monkeypatch.setitem(sys.modules, "cuda", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cuda.bindings", None if rocm else SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(device, "is_cuda_available", True)
    monkeypatch.setattr(
        device,
        "get_torch_device",
        lambda: SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=lambda: None,
            ipc_collect=lambda: calls.append("ipc"),
            empty_cache=lambda: None,
        ),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def remote(callback):
        assert calls[:2] == ["http exit", "shutdown"]
        if first_worker_fails and len(calls) == 2:
            calls.append("failed actor")
            raise RuntimeError("actor unavailable")
        return callback(SimpleNamespace(worker_dict={}))

    worker = SimpleNamespace(
        __ray_call__=SimpleNamespace(remote=remote),
        __ray_terminate__=SimpleNamespace(remote=lambda: calls.append("exit")),
    )
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", rollout=True),
        actor_rollout_wg=SimpleNamespace(workers=[worker, worker]),
    )
    if first_worker_fails:
        with pytest.raises(RuntimeError, match="Training worker cleanup failed"):
            task._shutdown_vllm_engines()
        assert calls == ["http exit", "shutdown", "failed actor", "ipc", "exit"]
    else:
        task._shutdown_vllm_engines()
        assert calls == ["http exit", "shutdown", "ipc", "ipc", "exit", "exit"]


@pytest.mark.parametrize("error", [RuntimeError("worker exit failed"), TimeoutError("worker exit timed out")])
def test_producer_exit_failure_is_reported_after_confirmed_consumer_shutdown(monkeypatch, error):
    calls = []
    exit_ref = object()

    def get(ref, timeout):
        if ref is exit_ref:
            assert calls == ["http exit", "shutdown", "ipc", "exit"]
            assert timeout == entrypoint._ACTOR_TERMINATION_TIMEOUT
            raise error
        return ref

    def collect(callback):
        calls.append("ipc")
        return {"owned": True}

    def terminate():
        calls.append("exit")
        return exit_ref

    monkeypatch.setattr(entrypoint.ray, "get", get)
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", rollout=True),
        actor_rollout_wg=SimpleNamespace(
            workers=[
                SimpleNamespace(
                    __ray_call__=SimpleNamespace(remote=collect),
                    __ray_terminate__=SimpleNamespace(remote=terminate),
                )
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="Training worker cleanup failed") as captured:
        task._shutdown_vllm_engines()
    assert captured.value.__cause__ is error


def test_submits_all_producer_exits_before_waiting(monkeypatch):
    submitted = []

    def worker(index):
        return SimpleNamespace(
            __ray_call__=SimpleNamespace(remote=lambda callback: {"owned": True}),
            __ray_terminate__=SimpleNamespace(remote=lambda: submitted.append(index) or ("exit", index)),
        )

    def get(ref, timeout):
        if isinstance(ref, tuple):
            assert submitted == [0, 1]
            raise entrypoint.ray.exceptions.ActorDiedError()
        return ref

    monkeypatch.setattr(entrypoint.ray, "get", get)
    task = runner()
    task.trainer = SimpleNamespace(actor_rollout_wg=SimpleNamespace(workers=[worker(0), worker(1)]))
    task._shutdown_vllm_engines()


@pytest.mark.parametrize("rpc_error", ["died", "unavailable", None])
@pytest.mark.parametrize("confirmed", [False, True])
def test_producer_exit_requires_process_confirmation(monkeypatch, rpc_error, confirmed):
    calls = []
    exit_ref = object()

    def get(ref, timeout):
        if ref is exit_ref and rpc_error is not None:
            if rpc_error == "died":
                raise entrypoint.ray.exceptions.ActorDiedError()
            raise entrypoint.ray.exceptions.ActorUnavailableError("temporarily unavailable", None)
        return ref

    def confirm(identity, timeout):
        assert identity == {"owned": True}
        assert timeout == entrypoint._ACTOR_TERMINATION_TIMEOUT
        calls.append("confirm exit")
        if not confirmed:
            raise TimeoutError("producer is still alive")

    monkeypatch.setattr(entrypoint.ray, "get", get)
    monkeypatch.setattr(entrypoint, "wait_for_actor_exit", confirm)
    worker = SimpleNamespace(
        __ray_call__=SimpleNamespace(remote=lambda callback: {"owned": True}),
        __ray_terminate__=SimpleNamespace(remote=lambda: exit_ref),
    )
    task = runner()
    task.trainer = SimpleNamespace(actor_rollout_wg=SimpleNamespace(workers=[worker]))
    if confirmed:
        task._shutdown_vllm_engines()
    else:
        with pytest.raises(RuntimeError, match="Training worker cleanup failed"):
            task._shutdown_vllm_engines()
    assert calls == ["confirm exit"]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_no_cuda_context_does_not_run_terminal_cuda_cleanup(monkeypatch, cuda_available):
    from verl.utils import device

    calls = []
    monkeypatch.setattr(device, "is_cuda_available", cuda_available)
    monkeypatch.setattr(device, "get_torch_device", lambda: SimpleNamespace(is_initialized=lambda: False))

    def remote(callback):
        calls.append("actor")
        callback(SimpleNamespace())

    task = runner()
    task.trainer = SimpleNamespace(
        actor_rollout_wg=SimpleNamespace(workers=[SimpleNamespace(__ray_call__=SimpleNamespace(remote=remote))])
    )
    task._shutdown_vllm_engines()
    assert calls == ["actor"]


def test_submits_all_actor_ranks_before_waiting(monkeypatch):
    from verl.utils import device

    submitted = []
    completed = []
    monkeypatch.setattr(device, "is_cuda_available", False)

    def remote(callback):
        submitted.append(callback)
        return len(submitted) - 1

    def get(index, timeout):
        assert len(submitted) == 2
        assert timeout == 120
        submitted[index](SimpleNamespace())
        completed.append(index)

    monkeypatch.setattr(entrypoint.ray, "get", get)
    worker = SimpleNamespace(__ray_call__=SimpleNamespace(remote=remote))
    task = runner()
    task.trainer = SimpleNamespace(actor_rollout_wg=SimpleNamespace(workers=[worker, worker]))
    task._shutdown_vllm_engines()
    assert completed == [0, 1]


@pytest.mark.parametrize("trainer_mode", ["sync", "separate_async"])
@pytest.mark.parametrize("failure_phase", [None, "init", "fit"])
def test_run_limits_engine_cleanup_to_sync_and_closes_transfer_queue(monkeypatch, failure_phase, trainer_mode):
    import transfer_queue as tq

    from verl_omni.trainer.diffusion import v1

    calls = []
    failure = ValueError("training failed")

    class Trainer:
        def __init__(self, config):
            pass

        def init(self):
            if failure_phase == "init":
                raise failure

        def fit(self, agent):
            if failure_phase == "fit":
                raise failure

    monkeypatch.setattr(tq, "init", lambda config: calls.append("queue init"))
    monkeypatch.setattr(tq, "close", lambda: calls.append("queue close"))
    monkeypatch.setattr(v1, "get_diffusion_trainer_cls", lambda mode: Trainer)
    task = runner()
    monkeypatch.setattr(task, "init_agent_loop_manager", lambda: None)
    monkeypatch.setattr(task, "_shutdown_vllm_engines", lambda: calls.append("engines closed"))
    config = OmegaConf.create({"transfer_queue": {"enable": False}, "trainer": {"v1": {"trainer_mode": trainer_mode}}})
    if failure_phase is None:
        task.run(config)
    else:
        with pytest.raises(ValueError) as captured:
            task.run(config)
        assert captured.value is failure
    expected = (
        ["queue init", "engines closed", "queue close"] if trainer_mode == "sync" else ["queue init", "queue close"]
    )
    assert calls == expected


@pytest.mark.parametrize("failure_phase", [None, "init", "fit"])
def test_cleanup_failure_is_reported_without_losing_training_error_or_queue_close(monkeypatch, failure_phase):
    import transfer_queue as tq

    from verl_omni.trainer.diffusion import v1

    calls = []
    original = ValueError("training failed")

    class Trainer:
        def __init__(self, config):
            pass

        def init(self):
            if failure_phase == "init":
                raise original

        def fit(self, agent):
            if failure_phase == "fit":
                raise original

    def failed_cleanup():
        raise RuntimeError("consumer termination unconfirmed; IPC skipped")

    monkeypatch.setattr(tq, "init", lambda config: None)
    monkeypatch.setattr(tq, "close", lambda: calls.append("queue closed"))
    monkeypatch.setattr(v1, "get_diffusion_trainer_cls", lambda mode: Trainer)
    task = runner()
    monkeypatch.setattr(task, "init_agent_loop_manager", lambda: None)
    monkeypatch.setattr(task, "_shutdown_vllm_engines", failed_cleanup)
    config = OmegaConf.create({"transfer_queue": {"enable": False}, "trainer": {"v1": {"trainer_mode": "sync"}}})
    if failure_phase is None:
        with pytest.raises(RuntimeError, match="termination unconfirmed"):
            task.run(config)
    else:
        with pytest.raises(ValueError) as captured:
            task.run(config)
        assert captured.value is original
        assert original.__notes__ == ["V1 cleanup also failed: consumer termination unconfirmed; IPC skipped"]
    assert calls == ["queue closed"]
