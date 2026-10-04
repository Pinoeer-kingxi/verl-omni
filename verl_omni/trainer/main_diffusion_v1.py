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

import gc
import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_class_from_fqn

from verl_omni.utils.config import validate_config
from verl_omni.utils.diffusion_attention import validate_attention_consistency
from verl_omni.utils.rl_insight import enable_rl_insight

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def run_diffusion_v1(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed v1 diffusion training.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed diffusion training including Ray initialization
                settings, model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # TransferQueue is required for v1; force-enable it before ray.init() so
    # TRANSFER_QUEUE_ENABLE is exported to every worker through the runtime env.
    config.transfer_queue.enable = True
    enable_rl_insight(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = DiffusionTaskRunnerV1.options(num_cpus=1)

    if (
        is_cuda_available
        and OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and OmegaConf.select(config, "global_profiler.steps") is not None
        and len(OmegaConf.select(config, "global_profiler.steps")) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote
class DiffusionTaskRunnerV1:
    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl_omni.agent_loop import create_diffusion_agent_loop_manager

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
            self.agent_loop_manager = agent_loop_manager_cls.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )
        else:
            self.agent_loop_manager = create_diffusion_agent_loop_manager(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

    def run(self, config: DictConfig):
        """Run the v1 diffusion training process."""
        import transfer_queue as tq

        from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

        # TransferQueue is required for v1; force-enable it regardless of the yaml default.
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        # initialize transfer queue inside the Ray task runner
        tq.init(config.transfer_queue)
        try:
            trainer_cls = get_diffusion_trainer_cls(config.trainer.v1.trainer_mode)
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            try:
                # Async trainers need separate lifecycle validation before adopting this cleanup.
                if config.trainer.v1.trainer_mode == "sync":
                    self._shutdown_vllm_engines()
            finally:
                tq.close()

    def _shutdown_vllm_engines(self):
        """Close owned vLLM engines before Ray tears down their process trees."""
        from verl_omni.reward_loop.reward_model import EngineManagedRewardModel

        if self.trainer is None:
            return
        managers = []
        for attribute in ("llm_server_manager", "standalone_server_manager"):
            manager = getattr(self.trainer, attribute, None)
            if manager is not None and manager.rollout_config.name in ("vllm", "vllm_omni"):
                managers.append((manager, manager.rollout_config.name))
        rewards = getattr(self.trainer, "reward_loop_manager", None)
        if rewards is not None:
            legacy = getattr(rewards, "reward_model_manager", None)
            if legacy is not None and legacy.config.rollout.name in ("vllm", "vllm_omni"):
                managers.append((legacy, legacy.config.rollout.name))
            named = getattr(rewards, "multi_reward_model_manager", None)
            if named is not None:
                managers.extend(
                    (model.reward_model_manager, model.reward_model_manager.config.rollout.name)
                    for model in named.models.values()
                    if isinstance(model, EngineManagedRewardModel)
                    and model.reward_model_manager.config.rollout.name in ("vllm", "vllm_omni")
                )
        for manager, backend in managers:

            def cancel_requests(worker, backend=backend):
                server_task = getattr(worker, "_server_task", None)
                if server_task is not None:
                    server_task.get_loop().call_soon_threadsafe(server_task.cancel)
                if backend == "vllm" and (handler := getattr(worker.engine, "output_handler", None)) is not None:
                    handler.get_loop().call_soon_threadsafe(handler.cancel)

            def shutdown(worker, backend=backend):
                # Process task cancellation before blocking on engine shutdown in this actor.
                server_task = getattr(worker, "_server_task", None)
                if server_task is not None:
                    if not server_task.done():
                        raise RuntimeError("HTTP server task did not finish cancellation")
                    worker._server_task = None
                if backend == "vllm":
                    handler = getattr(worker.engine, "output_handler", None)
                    if handler is not None and not handler.done():
                        raise RuntimeError("vLLM output handler did not finish cancellation")
                    # Leave time for worker teardown before the outer process manager kills the engine core.
                    worker.engine.shutdown(timeout=90)
                else:
                    worker.engine.shutdown()
                worker.engine = None
                gc.collect()

            for server in getattr(manager, "server_handles", []):
                try:
                    ray.get(server.__ray_call__.remote(cancel_requests), timeout=120)
                    ray.get(server.__ray_call__.remote(shutdown), timeout=120)
                except Exception:
                    logger.exception("Failed to shut down vLLM server engine")

        actor_workers = getattr(self.trainer, "actor_rollout_wg", None)
        if actor_workers is not None:

            def collect_ipc(worker):
                from verl.utils.device import get_torch_device, is_cuda_available

                if not is_cuda_available:
                    return
                gc.collect()
                accelerator = get_torch_device()
                if not accelerator.is_initialized():
                    return
                import torch

                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                accelerator.synchronize()
                worker.worker_dict.clear()
                gc.collect()
                accelerator.ipc_collect()
                accelerator.empty_cache()
                if torch.version.hip is None:
                    from cuda.bindings import runtime

                    # The owned worker is finished; close its runtime IPC export table before Ray exits it.
                    (status,) = runtime.cudaDeviceReset()
                    if status != runtime.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"CUDA context shutdown failed: {status}")

            pending_cleanup = []
            for worker in actor_workers.workers:
                try:
                    pending_cleanup.append(worker.__ray_call__.remote(collect_ipc))
                except Exception:
                    logger.exception("Failed to collect actor IPC after engine shutdown")
            # Every rank must enter distributed teardown before the controller waits for one rank.
            for cleanup in pending_cleanup:
                try:
                    ray.get(cleanup, timeout=120)
                except Exception:
                    logger.exception("Failed to collect actor IPC after engine shutdown")


@hydra.main(config_path="./config", config_name="diffusion_trainer", version_base=None)
def main(config):
    """Main entry point for v1 diffusion training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    OmegaConf.resolve(config)
    validate_config(config)
    validate_attention_consistency(config)

    if config.trainer.get("use_v1", False):
        run_diffusion_v1(config)
    else:
        # Explicit opt-out of the (default since v0.3.0) V1 trainer: fall back
        # to the legacy v0 entrypoint, which emits the deprecation warning.
        from verl_omni.trainer.main_diffusion import run_diffusion

        run_diffusion(config)


if __name__ == "__main__":
    main()
