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
"""Sync teardown ordering across inference consumers and IPC producers."""

import sys
from types import SimpleNamespace

import pytest

from verl_omni.trainer.diffusion.v1 import shutdown


@pytest.mark.parametrize("failure", [None, "error", "timeout"])
def test_consumer_exit_gates_producer_cleanup(monkeypatch, failure):
    events = []

    def close_reward():
        events.append("reward")
        if failure == "error":
            raise ValueError("consumer still running")
        if failure == "timeout":
            raise TimeoutError("consumer exit unconfirmed")

    def remote(callback):
        events.append("identity" if callback is shutdown.actor_process_identity else "ipc")
        return {"owned": True}

    worker = SimpleNamespace(__ray_call__=SimpleNamespace(remote=remote))
    trainer = SimpleNamespace(
        reward_loop_manager=SimpleNamespace(multi_reward_model_manager=SimpleNamespace(shutdown=close_reward)),
        llm_server_manager=SimpleNamespace(
            get_replicas=lambda: [SimpleNamespace(shutdown=lambda: events.append("rollout"))]
        ),
        actor_rollout_wg=SimpleNamespace(workers=[worker]),
    )
    monkeypatch.setattr(shutdown.ray, "get", lambda refs, timeout: refs)
    monkeypatch.setattr(shutdown, "close_actor_gracefully", lambda *args, **kwargs: events.append("worker stopped"))
    if failure:
        with pytest.raises(ValueError if failure == "error" else TimeoutError) as captured:
            shutdown.close_sync_trainer(trainer, "vllm_omni")
        assert "producer IPC is retained" in captured.value.__notes__[0]
        assert events == ["reward", "rollout"]
    else:
        shutdown.close_sync_trainer(trainer, "vllm_omni")
        assert events == ["reward", "rollout", "identity", "ipc", "worker stopped"]


def test_partial_rollout_initialization_retains_producer_ipc():
    shutdown.close_sync_trainer(SimpleNamespace(), "vllm_omni")
    with pytest.raises(RuntimeError, match="initialization was incomplete"):
        shutdown.close_sync_trainer(SimpleNamespace(actor_rollout_wg=SimpleNamespace(workers=[object()])), "vllm_omni")


@pytest.mark.parametrize("forced_failure", [False, True])
def test_failed_graceful_worker_exit_requires_confirmed_fallback(monkeypatch, forced_failure):
    events = []
    worker = SimpleNamespace(__ray_call__=SimpleNamespace(remote=lambda callback: {"owned": True}))
    trainer = SimpleNamespace(
        llm_server_manager=SimpleNamespace(get_replicas=lambda: []),
        actor_rollout_wg=SimpleNamespace(workers=[worker]),
    )
    monkeypatch.setattr(shutdown.ray, "get", lambda refs, timeout: refs)

    def grace(*args, **kwargs):
        events.append("graceful")
        raise TimeoutError("actor still running")

    def forced(*args, **kwargs):
        events.append("forced")
        if forced_failure:
            raise TimeoutError("process tree still running")

    monkeypatch.setattr(shutdown, "close_actor_gracefully", grace)
    monkeypatch.setattr(shutdown, "terminate_actor_and_wait", forced)
    if forced_failure:
        with pytest.raises(RuntimeError, match="Training worker exit was not confirmed"):
            shutdown.close_sync_trainer(trainer, "vllm_omni")
    else:
        shutdown.close_sync_trainer(trainer, "vllm_omni")
    assert events == ["graceful", "forced"]


def test_other_rollout_backends_keep_existing_worker_lifecycle():
    shutdown.close_sync_trainer(SimpleNamespace(actor_rollout_wg=SimpleNamespace(workers=[object()])), "diffusers")


def test_training_state_is_released_after_distributed_groups_stop(monkeypatch):
    events = []
    monkeypatch.setattr(shutdown.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(shutdown.torch.distributed, "destroy_process_group", lambda: events.append("groups stopped"))
    accelerator = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        ipc_collect=lambda: events.append("ipc"),
        empty_cache=lambda: events.append("cache"),
    )
    monkeypatch.setattr(shutdown, "get_torch_device", lambda: accelerator)
    monkeypatch.setattr(shutdown, "is_cuda_available", False)
    worker = SimpleNamespace(worker_dict={"actor": object()})
    shutdown._collect_worker_ipc(worker)
    assert events == ["groups stopped", "synchronize", "ipc", "cache"]
    assert worker.worker_dict == {}


@pytest.mark.parametrize("status", [0, 1])
def test_context_reset_failure_is_reported_after_model_and_ipc_release(monkeypatch, status):
    events = []
    monkeypatch.setattr(shutdown.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        shutdown,
        "get_torch_device",
        lambda: SimpleNamespace(
            synchronize=lambda: events.append("sync"),
            ipc_collect=lambda: events.append("ipc"),
            empty_cache=lambda: events.append("cache"),
        ),
    )
    monkeypatch.setattr(shutdown, "is_cuda_available", True)
    monkeypatch.setattr(shutdown.torch.version, "hip", None)
    worker = SimpleNamespace(worker_dict={"actor": object()})

    def reset():
        assert worker.worker_dict == {}
        events.append("reset")
        return (status,)

    runtime = SimpleNamespace(cudaDeviceReset=reset, cudaError_t=SimpleNamespace(cudaSuccess=0))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(runtime=runtime))
    if status:
        with pytest.raises(RuntimeError, match="CUDA context shutdown failed"):
            shutdown._collect_worker_ipc(worker)
    else:
        shutdown._collect_worker_ipc(worker)
    assert events == ["sync", "ipc", "cache", "reset"]
