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
"""V1 sync reward shutdown during successful and failed training."""

import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer import main_diffusion_v1 as entrypoint


@pytest.mark.parametrize("failure", [None, "constructor", "init", "fit", "cleanup", "fit_and_cleanup"])
def test_v1_sync_shutdown_preserves_training_failure_and_closes_transfer_queue(monkeypatch, failure):
    from verl_omni.trainer.diffusion import v1

    calls = []
    training_error = ValueError("training failed")
    cleanup_error = TimeoutError("shutdown failed")

    def close_rewards():
        calls.append("reward shutdown")
        if failure in {"cleanup", "fit_and_cleanup"}:
            raise cleanup_error

    class Trainer:
        def __init__(self, config):
            if failure == "constructor":
                raise training_error
            self.reward_loop_manager = SimpleNamespace(
                multi_reward_model_manager=SimpleNamespace(shutdown=close_rewards)
            )

        def init(self):
            if failure == "init":
                raise training_error

        def fit(self, agent_loop):
            calls.append("fit")
            if failure in {"fit", "fit_and_cleanup"}:
                raise training_error

    monkeypatch.setattr(v1, "get_diffusion_trainer_cls", lambda mode: Trainer)
    monkeypatch.setitem(
        sys.modules, "transfer_queue", SimpleNamespace(init=lambda _: None, close=lambda: calls.append("tq close"))
    )
    runner = entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
    monkeypatch.setattr(runner, "init_agent_loop_manager", lambda: None)
    config = OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": "sync"}},
            "transfer_queue": {"enable": False},
            "actor_rollout_ref": {"rollout": {"name": "vllm_omni"}},
        }
    )
    if failure is None:
        runner.run(config)
    else:
        expected = cleanup_error if failure == "cleanup" else training_error
        with pytest.raises(type(expected)) as captured:
            runner.run(config)
        assert captured.value is expected
        if failure == "fit_and_cleanup":
            assert "shutdown also failed" in captured.value.__notes__[0]
    assert calls[-1] == "tq close"
    assert ("reward shutdown" in calls) is (failure != "constructor")


def test_v1_async_training_keeps_existing_lifecycle(monkeypatch):
    from verl_omni.trainer.diffusion import v1

    calls = []
    trainer = SimpleNamespace(init=lambda: None, fit=lambda _: calls.append("fit"))
    monkeypatch.setattr(v1, "get_diffusion_trainer_cls", lambda mode: lambda config: trainer)
    monkeypatch.setitem(
        sys.modules, "transfer_queue", SimpleNamespace(init=lambda _: None, close=lambda: calls.append("tq close"))
    )
    runner = entrypoint.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
    monkeypatch.setattr(runner, "init_agent_loop_manager", lambda: None)
    runner.run(OmegaConf.create({"trainer": {"v1": {"trainer_mode": "async"}}, "transfer_queue": {"enable": False}}))
    assert calls == ["fit", "tq close"]
