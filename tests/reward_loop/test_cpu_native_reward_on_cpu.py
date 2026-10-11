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
"""CPU-native reward placement and executor contracts."""

import asyncio
import time
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop import reward_loop as loop_module
from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import (
    EngineManagedRewardModel,
    MultiRewardModelManager,
    NativeManagedRewardModel,
)
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec, parse_reward_model_config, reward_role_required


def test_cpu_native_role_resolves_placement_interpolations():
    config = OmegaConf.create(
        {
            "resource": "cpu",
            "reward": {
                "reward_model": {"enable": False},
                "models": {"quality": {"backend": "native", "placement": {"resource": "${resource}"}}},
            },
        }
    )
    assert not reward_role_required(config)
    config.resource = "accelerator"
    assert reward_role_required(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["thread", "async", "awaitable"])
@pytest.mark.parametrize("trigger", ["cancel", "timeout"])
async def test_cancelled_native_inference_keeps_model_until_actual_completion(monkeypatch, kind, trigger):
    entered, release, finished, closed = (Event() for _ in range(4))

    async def async_infer(self, value):
        entered.set()
        try:
            async with asyncio.timeout(3):
                while not release.is_set():
                    await asyncio.sleep(0.001)
            return value + 1
        finally:
            finished.set()

    class Model:
        def __init__(self, **kwargs):
            del kwargs

        def infer(self, value):
            if kind == "awaitable":
                return async_infer(self, value)
            entered.set()
            try:
                if not release.wait(3):
                    raise TimeoutError("Controlled inference was not released")
                return value + 1
            finally:
                finished.set()

        def close(self):
            assert finished.is_set()
            closed.set()

    if kind == "async":
        Model.infer = async_infer
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: Model)
    executor = NativeRewardExecutor(
        RewardModelSpec(name="quality", backend="native", device_type="cpu", executor_config={"model": "test:model"})
    )
    await executor.wake_up()
    operation = executor.infer(1)
    caller = asyncio.create_task(asyncio.wait_for(operation, 0.02) if trigger == "timeout" else operation)
    sleeping = None
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        if trigger == "cancel":
            caller.cancel()
        await asyncio.wait({caller}, timeout=0.04)
        assert not caller.done()
        assert executor._inflight == 1
        assert not finished.is_set()
        sleeping = asyncio.create_task(executor.sleep())
        await asyncio.wait({sleeping}, timeout=0.02)
        assert not sleeping.done()
        assert not closed.is_set()
        release.set()
        await asyncio.wait_for(sleeping, 1)
        with pytest.raises(asyncio.CancelledError if trigger == "cancel" else TimeoutError):
            await asyncio.wait_for(asyncio.shield(caller), 1)
        assert finished.is_set() and closed.is_set()
        assert executor._inflight == 0 and executor._model is None
    finally:
        release.set()
        await asyncio.gather(caller, *([sleeping] if sleeping else []), return_exceptions=True)
        await executor.sleep()


@pytest.mark.asyncio
async def test_cpu_native_executor_uses_unindexed_cpu_device(monkeypatch):
    spec = RewardModelSpec(
        name="quality",
        backend="native",
        model_path="/models/quality",
        executor_config={"model": "tests.fake:CpuModel"},
        device_type="cpu",
    )
    executor = NativeRewardExecutor(spec)
    closed = []
    model = SimpleNamespace(infer=lambda value: value + 1, close=lambda: closed.append(True))

    def build_model(model_path, device):
        assert model_path == "/models/quality"
        assert device == torch.device("cpu")
        return model

    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: build_model)
    monkeypatch.setattr(executor_module, "_empty_accelerator_cache", lambda: pytest.fail("CPU sleep cleared GPU cache"))
    await executor.wake_up()
    assert await executor.infer(1) == 2
    await executor.sleep()
    assert closed == [True]
    assert executor._model is None


@pytest.mark.asyncio
async def test_native_executor_explicit_cpu_device_does_not_probe_accelerator(monkeypatch):
    executor = NativeRewardExecutor(
        RewardModelSpec(
            name="quality",
            backend="native",
            executor_config={"model": "tests.fake:CpuModel", "kwargs": {"device": "cpu"}},
            device_type="cpu",
        )
    )
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: lambda device: SimpleNamespace(device=device))
    monkeypatch.setattr(executor_module, "get_device_name", lambda: pytest.fail("Explicit device was ignored"))
    await executor.wake_up()
    assert executor._model.device == "cpu"


@pytest.mark.parametrize(
    "placement, message",
    [
        ({"resource": "gpu"}, "placement.resource"),
        ({"resource": "cpu", "cpus_per_worker": 0}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": -1}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": True}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": 1.5}, "positive integer"),
        ({"resource": "cpu", "cpus_per_worker": "2"}, "positive integer"),
        ({"resource": "accelerator", "cpus_per_worker": 2}, "only supported for resource='cpu'"),
        ({"resource": "cpu", "num_replicas": 0}, "placement.num_replicas"),
        ({"resource": "cpu", "num_replicas": True}, "placement.num_replicas"),
        ({"resource": "cpu", "num_replicas": 1.5}, "placement.num_replicas"),
        ({"resource": "cpu", "num_replicas": None}, "placement.num_replicas"),
        ({"resource": "cpu", "devices": [0]}, "remove placement.devices"),
        ({"resource": "accelerator", "num_replicas": 2}, "only supported for resource='cpu'"),
    ],
)
def test_cpu_native_rejects_invalid_resource_reservations(placement, message):
    with pytest.raises(ValueError, match=message):
        parse_reward_model_config(
            "quality",
            {
                "backend": "native",
                "placement": {
                    **({"num_replicas": 1} if placement["resource"] == "cpu" else {"devices": [0]}),
                    **placement,
                },
                "executor": {"model": "tests.fake:CpuModel"},
            },
        )


@pytest.mark.parametrize(
    "resource,device,valid",
    [
        ("cpu", "cpu", True),
        ("cpu", torch.device("cpu"), True),
        ("cpu", "cpu:0", True),
        ("cpu", "cuda:0", False),
        ("cpu", torch.device("cuda:1"), False),
        ("accelerator", "cpu", False),
        ("accelerator", "cuda:0", True),
        ("accelerator", "meta", False),
        ("cpu", "invalid-device", False),
        ("cpu", None, False),
        ("cpu", 0, False),
    ],
)
def test_explicit_native_device_matches_reserved_resource(monkeypatch, resource, device, valid):
    from verl_omni.workers.config import reward as config_module

    monkeypatch.setattr(config_module, "get_device_name", lambda: "cuda")
    value = {
        "backend": "native",
        "placement": {"resource": resource, **({"num_replicas": 2} if resource == "cpu" else {"devices": [0]})},
        "executor": {"model": "tests.fake:CpuModel", "kwargs": {"device": device}},
    }
    if valid:
        parsed = parse_reward_model_config("quality", value)
        assert parsed.executor.kwargs["device"] == device
    else:
        with pytest.raises(ValueError, match=r"executor.kwargs.device"):
            MultiRewardModelManager(
                OmegaConf.create(
                    {"reward": {"models": {"quality": value}, "reward_model": {"enable": False}}},
                    flags={"allow_objects": True},
                )
            )
    assert not torch.cuda.is_initialized()


@pytest.mark.asyncio
async def test_cpu_phase_submission_failure_drains_accepted_rpc_before_sleep():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    accepted = []

    async def score(data):
        entered.set()
        await release.wait()
        calls.append("score_finished")
        return [{"reward_score": 1.0} for _ in range(len(data))]

    def submit(data):
        request = asyncio.create_task(score(data))
        accepted.append(request)
        return request

    def fail_submission(data):
        raise ValueError("intentional RPC submission failure")

    async def wake_up():
        calls.append("wake_up")

    async def sleep(*, native_timeout=None):
        calls.append("sleep")

    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager._scoring_unusable = False
    manager._shared_worker_process_identities = {}
    manager._reward_dispatch_batch_sizes = {}
    manager._reward_worker_groups = {
        "quality": [
            SimpleNamespace(compute_score_batch=SimpleNamespace(remote=submit)),
            SimpleNamespace(compute_score_batch=SimpleNamespace(remote=fail_submission)),
        ]
    }
    manager.multi_reward_model_manager = SimpleNamespace(models={"quality": object()}, wake_up=wake_up, sleep=sleep)
    scoring = asyncio.create_task(manager.async_compute_rm_score(DataProto.from_dict(tensors={"id": torch.arange(2)})))
    try:
        await entered.wait()
        await asyncio.sleep(0)
        returned_before_release = scoring.done()
    finally:
        release.set()
        result = (await asyncio.gather(scoring, return_exceptions=True))[0]
        await asyncio.gather(*accepted)
    assert not returned_before_release, "A submission failure returned before an accepted RPC finished"
    assert isinstance(result, ValueError)
    assert "intentional RPC submission failure" in str(result)
    assert calls == ["wake_up", "score_finished", "sleep"]


def test_shared_worker_identity_is_recorded_before_native_cpus_are_reserved(monkeypatch):
    model_config = {
        "backend": "native",
        "placement": {"resource": "cpu", "num_replicas": 1},
        "executor": {"model": "tests.fake:CpuModel"},
    }
    model = NativeManagedRewardModel("quality", model_config)
    config = OmegaConf.create(
        {
            "reward": {
                "num_workers": 1,
                "models": {"quality": model_config},
                "reward_functions": {
                    "quality": {"path": "tests.fake.py", "name": "quality"},
                    "rule": {"path": "tests.fake.py", "name": "rule"},
                },
                "reward_model": {"enable": False},
            }
        }
    )
    calls = []
    identity = {"pid": 17, "created": 23.0, "node_id": "owned-node"}

    def capture(callback):
        calls.append("shared_identity")
        return identity

    shared = SimpleNamespace(__ray_call__=SimpleNamespace(remote=capture))

    def reserve_native(*args):
        calls.append("native_cpu_reserved")
        return [object()]

    manager = object.__new__(OmniRewardLoopManager)
    manager.config = config
    manager.reward_router_address = None
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"quality": model},
        reward_model_specs={"quality": model.spec},
        bind_native_workers=lambda name, workers: model.bind_workers(workers),
    )
    manager._create_node_affinity_workers = lambda *args: [shared]
    manager._create_native_workers = reserve_native
    monkeypatch.setattr(loop_module.ray, "remote", lambda cls: cls)
    monkeypatch.setattr(loop_module.ray, "get", lambda refs: refs)
    manager._init_reward_loop_workers()
    assert calls == ["shared_identity", "native_cpu_reserved"]
    assert manager._shared_worker_process_identities == {id(shared): identity}


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["cancel", "timeout", "rpc_error", "submission_error"])
@pytest.mark.parametrize("confirmed", [False, True])
async def test_never_returning_rpc_has_bounded_cleanup_and_retains_model_until_stopped(monkeypatch, trigger, confirmed):
    calls = []
    entered = asyncio.Event()
    request = None

    async def score(data):
        entered.set()
        await asyncio.Event().wait()

    def submit(data):
        nonlocal request
        request = asyncio.create_task(score(data))
        return request

    async def failed(data):
        raise ValueError("scoring failed")

    def failed_submit(data):
        raise ValueError("submission failed")

    worker = SimpleNamespace(compute_score_batch=SimpleNamespace(remote=submit))
    workers = [worker]
    if trigger not in ("cancel", "timeout"):
        workers.append(
            SimpleNamespace(
                compute_score_batch=SimpleNamespace(remote=failed if trigger == "rpc_error" else failed_submit)
            )
        )
    model = NativeManagedRewardModel(
        "quality",
        {
            "backend": "native",
            "placement": {"resource": "cpu", "num_replicas": 1},
            "executor": {"model": "tests.fake:CpuModel"},
        },
    )
    model.bind_workers(workers)
    model._worker_process_identities[id(worker)] = {"owned": True}

    async def wake():
        pass

    async def sleep(*, native_timeout=None):
        assert calls == ["terminate", "stopped"]
        assert worker not in model._workers
        calls.append("sleep")

    def terminate(actor, identity, timeout):
        assert actor is worker
        assert identity == {"owned": True}
        calls.append("terminate")
        time.sleep(0.005)
        if not confirmed:
            raise TimeoutError("worker termination unconfirmed")
        calls.append("stopped")

    monkeypatch.setattr(loop_module, "_SCORING_DRAIN_TIMEOUT", 0.02)
    monkeypatch.setattr(loop_module, "terminate_actor_and_wait", terminate)
    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager._scoring_unusable = False
    manager._shared_worker_process_identities = {}
    manager._reward_worker_groups = {"quality": workers}
    manager._reward_dispatch_batch_sizes = {}
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"quality": model},
        wake_up=wake,
        sleep=sleep,
        has_engine_model=False,
    )
    operation = manager.async_compute_rm_score(DataProto.from_dict(tensors={"id": torch.arange(2)}))
    scoring = asyncio.create_task(asyncio.wait_for(operation, timeout=0.02) if trigger == "timeout" else operation)
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    started = time.monotonic()
    if trigger == "cancel":
        scoring.cancel()
        await asyncio.sleep(0.005)
        scoring.cancel()
    completed, _ = await asyncio.wait({scoring}, timeout=0.25)
    assert completed, "Cleanup deadline must bound a scoring RPC that never returns"
    result = (await asyncio.gather(scoring, return_exceptions=True))[0]
    assert time.monotonic() - started < 0.25
    assert not manager._score_lock.locked()
    assert request.cancelled()
    if confirmed:
        assert calls == ["terminate", "stopped", "sleep"]
        expected_error = {"cancel": asyncio.CancelledError, "timeout": TimeoutError}.get(trigger, ValueError)
        assert isinstance(result, expected_error)
    else:
        assert calls == ["terminate"]
        assert isinstance(result, RuntimeError)
        assert "termination could not be confirmed" in str(result)
    with pytest.raises(RuntimeError, match="recreate the reward manager"):
        await manager.async_compute_rm_score(DataProto.from_dict(tensors={"id": torch.arange(2)}))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "sleep"])
async def test_engine_lifecycle_awaits_existing_replica_methods_concurrently(method):
    entered = set()
    all_entered = asyncio.Event()

    def replica(index):
        async def lifecycle():
            entered.add(index)
            if len(entered) == 2:
                all_entered.set()
            await all_entered.wait()

        return SimpleNamespace(**{method: lifecycle})

    model = object.__new__(EngineManagedRewardModel)
    model.offload = True
    model.reward_model_manager = SimpleNamespace(rollout_replicas=[replica(0), replica(1)])
    await asyncio.wait_for(getattr(model, method)(), timeout=0.5)
    assert entered == {0, 1}


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_sync_scoring_engine_sleep_failure_is_bounded_without_executor_threads(monkeypatch, failure):
    import threading

    calls = []
    replica_sleep_finished = []

    async def wake():
        calls.append("wake")

    async def sleep(*, native_timeout=None):
        calls.append("sleep")
        try:
            if failure == "error":
                raise ValueError("injected replica sleep failure")
            await asyncio.Event().wait()
        finally:
            replica_sleep_finished.append(True)

    def synchronous_sleep():
        # The previous bridge outlived wait_for and delayed asyncio.run exit.
        time.sleep(0.6)

    engine = object.__new__(EngineManagedRewardModel)
    engine.offload = True
    engine.spec = RewardModelSpec(name="ocr", backend="engine")
    engine.reward_model_manager = SimpleNamespace(
        rollout_replicas=[SimpleNamespace(wake_up=wake, sleep=sleep)],
        wake_up=lambda: None,
        sleep=synchronous_sleep,
    )
    models = object.__new__(MultiRewardModelManager)
    models.models = {"ocr": engine}
    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager._scoring_unusable = False
    manager.multi_reward_model_manager = models

    async def score(data):
        calls.append("score")
        return data

    manager._compute_named_model_scores = score
    monkeypatch.setattr(loop_module, "_MODEL_SLEEP_TIMEOUT", 0.02)
    threads_before = set(threading.enumerate())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Reward model cleanup failed") as caught:
        manager.compute_rm_score(object())
    assert time.monotonic() - started < 0.3
    assert isinstance(caught.value.__cause__, ValueError if failure == "error" else TimeoutError)
    assert calls == ["wake", "score", "sleep"]
    assert replica_sleep_finished == [True]
    assert set(threading.enumerate()) <= threads_before
    assert not manager._score_lock.locked()
    assert manager._scoring_unusable
    with pytest.raises(RuntimeError, match="recreate the reward manager"):
        manager.compute_rm_score(object())


@pytest.mark.parametrize("blocked", ["lock", "rpc"])
def test_sync_native_sleep_deadline_remains_bounded_with_owned_cancellation(monkeypatch, blocked):
    finished = []

    async def stalled_sleep():
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    model = NativeManagedRewardModel(
        "quality",
        {
            "backend": "native",
            "placement": {"resource": "cpu", "num_replicas": 1},
            "executor": {"model": "tests.fake:CpuModel"},
        },
    )
    model.bind_workers([SimpleNamespace(sleep_reward_model=SimpleNamespace(remote=lambda name: stalled_sleep()))])
    if blocked == "lock":
        model._lifecycle_lock = asyncio.Lock()
        asyncio.run(model._lifecycle_lock.acquire())
    models = object.__new__(MultiRewardModelManager)
    models.models = {"quality": model}
    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager._scoring_unusable = False
    manager.multi_reward_model_manager = models
    manager._compute_named_model_scores = lambda data: asyncio.sleep(0, result=data)
    models.wake_up = lambda: asyncio.sleep(0)
    monkeypatch.setattr(loop_module, "_MODEL_SLEEP_TIMEOUT", 0.02)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Reward model cleanup failed"):
        manager.compute_rm_score(object())
    assert time.monotonic() - started < 0.3
    assert manager._scoring_unusable and not manager._score_lock.locked()
    assert model._workers
    assert finished == ([] if blocked == "lock" else [True])
    if blocked == "lock":
        model._lifecycle_lock.release()
