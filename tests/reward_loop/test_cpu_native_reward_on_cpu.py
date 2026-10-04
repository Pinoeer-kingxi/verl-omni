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
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec, parse_reward_model_config, reward_role_required


def test_cpu_native_schema_does_not_require_reward_gpu_pool():
    model = parse_reward_model_config(
        "quality",
        {
            "backend": "native",
            "placement": {"resource": "cpu", "devices": [0], "cpus_per_worker": 2},
            "executor": {"model": "tests.fake:CpuModel"},
        },
    )
    config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False},
                "models": {"quality": {"backend": "native", "placement": {"resource": "cpu"}}},
            }
        }
    )
    assert model.placement.is_cpu
    assert model.placement.cpus_per_worker == 2
    assert not reward_role_required(config)


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
@pytest.mark.parametrize("device_type", ["cpu", "accelerator"])
async def test_native_executor_explicit_device_does_not_probe_accelerator(monkeypatch, device_type):
    executor = NativeRewardExecutor(
        RewardModelSpec(
            name="quality",
            backend="native",
            executor_config={"model": "tests.fake:CpuModel", "kwargs": {"device": "cpu"}},
            device_type=device_type,
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
    ],
)
def test_cpu_native_rejects_invalid_resource_reservations(placement, message):
    with pytest.raises(ValueError, match=message):
        parse_reward_model_config(
            "quality",
            {
                "backend": "native",
                "placement": {"devices": [0], **placement},
                "executor": {"model": "tests.fake:CpuModel"},
            },
        )


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

    async def sleep():
        calls.append("sleep")

    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
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
