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
"""V1 engine teardown ownership and error preservation."""

import asyncio
import inspect
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.reward_loop.reward_model import EngineManagedRewardModel
from verl_omni.trainer import main_diffusion_v1 as entrypoint


def runner():
    return entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()


def manager(calls, name="vllm", error=False, rollout=False):
    worker = None

    def remote(callback):
        nonlocal worker
        if worker is None:
            calls.append(name)
        if error:
            raise RuntimeError("server unavailable")

        def shutdown(**kwargs):
            assert kwargs == ({"timeout": 90} if name == "vllm" else {})
            assert worker._server_task is None
            if name == "vllm":
                assert worker.engine.output_handler.done()
            calls.append("shutdown")

        async def execute():
            nonlocal worker

            async def serving():
                await asyncio.Event().wait()

            if worker is None:
                worker = SimpleNamespace(
                    engine=SimpleNamespace(
                        shutdown=shutdown,
                        output_handler=asyncio.create_task(serving()) if name == "vllm" else None,
                    ),
                    _server_task=asyncio.create_task(serving()),
                )
            result = callback(worker)
            assert not inspect.isawaitable(result), "Ray __ray_call__ does not await callback coroutines"
            await asyncio.sleep(0)

        asyncio.run(execute())
        return None

    result = SimpleNamespace(server_handles=[SimpleNamespace(__ray_call__=SimpleNamespace(remote=remote))])
    if rollout:
        result.rollout_config = SimpleNamespace(name=name)
        result.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=result.rollout_config))
    else:
        result.config = SimpleNamespace(rollout=SimpleNamespace(name=name))
    return result


def test_closes_rollout_standalone_legacy_and_named_engines(monkeypatch):
    calls = []
    timeouts = []
    monkeypatch.setattr(entrypoint.ray, "get", lambda ref, timeout: timeouts.append(timeout))
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
    assert calls == ["vllm_omni", "shutdown", "vllm_omni", "shutdown", "vllm", "shutdown", "vllm", "shutdown"]
    assert timeouts == [120] * 8


@pytest.mark.parametrize("trainer", [None, SimpleNamespace()])
def test_partial_initialization_without_managers(trainer):
    task = runner()
    task.trainer = trainer
    task._shutdown_vllm_engines()


def test_skips_other_engine_backends():
    calls = []
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="sglang", rollout=True),
        reward_loop_manager=SimpleNamespace(reward_model_manager=manager(calls, name="sglang")),
    )
    task._shutdown_vllm_engines()
    assert calls == []


def test_failed_server_does_not_skip_other_owned_engines(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(entrypoint.ray, "get", lambda ref, timeout: None)
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", error=True, rollout=True),
        reward_loop_manager=SimpleNamespace(reward_model_manager=manager(calls)),
    )
    task._shutdown_vllm_engines()
    assert calls == ["vllm_omni", "vllm", "shutdown"]
    assert "Failed to shut down vLLM server engine" in caplog.text


@pytest.mark.parametrize("first_worker_fails", [False, True])
@pytest.mark.parametrize("rocm", [False, True])
def test_actor_ipc_is_collected_after_receivers_shutdown(monkeypatch, caplog, first_worker_fails, rocm):
    import torch
    from verl.utils import device

    calls = []
    runtime = SimpleNamespace(cudaError_t=SimpleNamespace(cudaSuccess=0), cudaDeviceReset=lambda: (0,))
    monkeypatch.setattr(torch.version, "hip", "simulated-rocm" if rocm else None)
    monkeypatch.setitem(sys.modules, "cuda", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cuda.bindings", None if rocm else SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(entrypoint.ray, "get", lambda ref, timeout: None)
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
        assert calls[:2] == ["vllm_omni", "shutdown"]
        if first_worker_fails and len(calls) == 2:
            calls.append("failed actor")
            raise RuntimeError("actor unavailable")
        callback(SimpleNamespace(worker_dict={}))

    worker = SimpleNamespace(__ray_call__=SimpleNamespace(remote=remote))
    task = runner()
    task.trainer = SimpleNamespace(
        llm_server_manager=manager(calls, name="vllm_omni", rollout=True),
        actor_rollout_wg=SimpleNamespace(workers=[worker, worker]),
    )
    task._shutdown_vllm_engines()
    expected = (
        ["vllm_omni", "shutdown", "failed actor", "ipc"]
        if first_worker_fails
        else ["vllm_omni", "shutdown", "ipc", "ipc"]
    )
    assert calls == expected
    if first_worker_fails:
        assert "Failed to collect actor IPC after engine shutdown" in caplog.text


@pytest.mark.parametrize("cuda_available", [False, True])
def test_no_cuda_context_does_not_run_terminal_cuda_cleanup(monkeypatch, cuda_available):
    from verl.utils import device

    calls = []
    monkeypatch.setattr(entrypoint.ray, "get", lambda ref, timeout: None)
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
