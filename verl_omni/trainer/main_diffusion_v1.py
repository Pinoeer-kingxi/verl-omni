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
import time
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_class_from_fqn

from verl_omni.utils.config import validate_config
from verl_omni.utils.diffusion_attention import validate_attention_consistency
from verl_omni.utils.ray_lifecycle import actor_process_identity, terminate_actor_and_wait, wait_for_actor_exit
from verl_omni.utils.rl_insight import enable_rl_insight

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_HTTP_SHUTDOWN_TIMEOUT = 30
_ENGINE_SHUTDOWN_TIMEOUT = 120
_ACTOR_TERMINATION_TIMEOUT = 30


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
        self._trainer_initialized = False

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
        self._trainer_initialized = False
        training_error = None
        try:
            trainer_cls = get_diffusion_trainer_cls(config.trainer.v1.trainer_mode)
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self._trainer_initialized = True
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        except BaseException as exc:
            training_error = exc
            raise
        finally:
            try:
                # Async trainers need separate lifecycle validation before adopting this cleanup.
                if config.trainer.v1.trainer_mode == "sync":
                    try:
                        self._shutdown_vllm_engines()
                    except Exception as exc:
                        if training_error is None:
                            raise
                        training_error.add_note(f"V1 cleanup also failed: {exc}")
                        logger.exception("V1 cleanup failed after training failed")
            finally:
                tq.close()

    def _shutdown_vllm_engines(self):
        """Stop consumers before releasing the training workers' IPC exports."""
        from verl_omni.reward_loop.reward_model import EngineManagedRewardModel
        from verl_omni.workers.rollout.http_server import http_shutdown_complete, request_http_shutdown

        if self.trainer is None:
            return
        managers = []
        for attribute in ("llm_server_manager", "standalone_server_manager"):
            manager = getattr(self.trainer, attribute, None)
            if manager is not None:
                managers.append((manager, manager.rollout_config))
        rewards = getattr(self.trainer, "reward_loop_manager", None)
        if rewards is not None:
            legacy = getattr(rewards, "reward_model_manager", None)
            if legacy is not None:
                managers.append((legacy, legacy.config.rollout))
            named = getattr(rewards, "multi_reward_model_manager", None)
            if named is not None:
                managers.extend(
                    (model.reward_model_manager, model.reward_model_manager.config.rollout)
                    for model in named.models.values()
                    if isinstance(model, EngineManagedRewardModel)
                )
        failures = []
        seen = set()
        for manager, rollout_config in managers:
            backend = rollout_config.name
            if backend not in ("vllm", "vllm_omni"):
                failures.append(RuntimeError(f"Cannot confirm consumer exit for engine backend {backend!r}"))
                continue
            engine_kwargs = (getattr(rollout_config, "engine_kwargs", None) or {}).get(backend, {}) or {}
            # The pinned launcher defaults to mp; remote executors are outside this actor's PID tree.
            for option, local_backends in (
                ("distributed_executor_backend", ("mp", "uni")),
                ("data_parallel_backend", ("mp",)),
            ):
                for spelling in (option, option.replace("_", "-")):
                    value = engine_kwargs.get(spelling)
                    if value is not None and value not in local_backends:
                        failures.append(RuntimeError(f"Cannot confirm all consumers for {spelling}={value!r}"))
            if backend == "vllm_omni":
                deployment_keys = (
                    "deploy_config",
                    "stage_configs_path",
                    "stage_overrides",
                    "strategy_config",
                    "pipeline_name",
                )
                external_stages = any(
                    engine_kwargs.get(key) or engine_kwargs.get(key.replace("_", "-")) for key in deployment_keys
                )
                local_stages = all(
                    engine_kwargs.get(key) in (None, "multi_process") for key in ("worker_backend", "worker-backend")
                )
                if external_stages or not local_stages:
                    failures.append(RuntimeError("Cannot confirm consumer ownership for custom Omni stages"))
            # Replica servers also cover non-head nodes and partially initialized managers.
            servers = list(getattr(manager, "server_handles", []))
            for replica in getattr(manager, "rollout_replicas", []):
                servers.extend(getattr(replica, "servers", []))
            for server in servers:
                actor_id = getattr(server, "_actor_id", id(server))
                if actor_id in seen:
                    continue
                seen.add(actor_id)
                identity = None
                try:
                    identity = ray.get(server.__ray_call__.remote(actor_process_identity), timeout=5)
                    deadline = time.monotonic() + _HTTP_SHUTDOWN_TIMEOUT
                    ray.get(server.__ray_call__.remote(request_http_shutdown), timeout=_HTTP_SHUTDOWN_TIMEOUT)
                    while not ray.get(
                        server.__ray_call__.remote(http_shutdown_complete),
                        timeout=max(0.001, deadline - time.monotonic()),
                    ):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("HTTP server did not stop before the shutdown deadline")
                        time.sleep(0.01)

                    def cancel_output_handler(worker, backend=backend):
                        engine = getattr(worker, "engine", None)
                        if backend == "vllm" and (handler := getattr(engine, "output_handler", None)) is not None:
                            handler.get_loop().call_soon_threadsafe(handler.cancel)
                        return engine is not None

                    def shutdown(worker, backend=backend):
                        engine = getattr(worker, "engine", None)
                        if engine is not None:
                            if backend == "vllm":
                                handler = getattr(engine, "output_handler", None)
                                if handler is not None and not handler.done():
                                    raise RuntimeError("vLLM output handler did not finish cancellation")
                                engine.shutdown(timeout=90)
                            else:
                                engine.shutdown()
                        worker.engine = None
                        worker._server_task = None
                        worker._http_server = None
                        gc.collect()

                    initialized = ray.get(server.__ray_call__.remote(cancel_output_handler), timeout=5)
                    if backend == "vllm" and initialized:
                        # Release model resources before the executor's five-second process-exit grace.
                        ray.get(
                            server.collective_rpc.remote("shutdown", timeout=90),
                            timeout=_ENGINE_SHUTDOWN_TIMEOUT,
                        )
                    ray.get(server.__ray_call__.remote(shutdown), timeout=_ENGINE_SHUTDOWN_TIMEOUT)
                    # Close the server's remaining multiprocessing pipes so its
                    # resource tracker can reclaim engine caches before Ray stops.
                    termination = server.__ray_terminate__.remote()
                    try:
                        ray.get(termination, timeout=_ACTOR_TERMINATION_TIMEOUT)
                    except (ray.exceptions.ActorDiedError, ray.exceptions.ActorUnavailableError):
                        pass
                    wait_for_actor_exit(identity, _ACTOR_TERMINATION_TIMEOUT)
                except Exception:
                    logger.exception("Graceful vLLM shutdown failed; terminating the owned consumer")
                    try:
                        terminate_actor_and_wait(server, identity, _ACTOR_TERMINATION_TIMEOUT)
                    except Exception as exc:
                        failures.append(exc)
                        logger.exception("Cannot confirm vLLM consumer termination")
        if failures:
            raise RuntimeError("Consumer cleanup failed; training worker CUDA IPC cleanup was skipped") from failures[0]

        actor_workers = getattr(self.trainer, "actor_rollout_wg", None)
        if actor_workers is not None:
            if not self._trainer_initialized:
                raise RuntimeError("Trainer initialization incomplete; CUDA IPC cleanup was skipped")

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
                return actor_process_identity(worker)

            pending_cleanup = []
            failures = []
            for worker in actor_workers.workers:
                try:
                    pending_cleanup.append((worker, worker.__ray_call__.remote(collect_ipc)))
                except Exception as exc:
                    failures.append(exc)
                    logger.exception("Failed to collect actor IPC after engine shutdown")
            # Every rank must enter distributed teardown before the controller waits for one rank.
            completed_workers = []
            for worker, cleanup in pending_cleanup:
                try:
                    identity = ray.get(cleanup, timeout=120)
                    if identity:
                        completed_workers.append((worker, identity))
                except Exception as exc:
                    failures.append(exc)
                    logger.exception("Failed to collect actor IPC after engine shutdown")
            # Graceful actor exit runs backend destructors that unlink IPC counter files.
            pending_exit = []
            for worker, identity in completed_workers:
                try:
                    pending_exit.append((worker.__ray_terminate__.remote(), identity))
                except Exception as exc:
                    failures.append(exc)
                    logger.exception("Failed to exit training worker after IPC cleanup")
            for termination, identity in pending_exit:
                try:
                    try:
                        ray.get(termination, timeout=_ACTOR_TERMINATION_TIMEOUT)
                    except (ray.exceptions.ActorDiedError, ray.exceptions.ActorUnavailableError):
                        # Ray can report intentional exit as temporarily unavailable.
                        pass
                    wait_for_actor_exit(identity, _ACTOR_TERMINATION_TIMEOUT)
                except Exception as exc:
                    failures.append(exc)
                    logger.exception("Failed to exit training worker after IPC cleanup")
            if failures:
                raise RuntimeError("Training worker cleanup failed") from failures[0]


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
