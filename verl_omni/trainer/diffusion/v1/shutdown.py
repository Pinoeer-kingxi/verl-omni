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
"""Close sync inference consumers before their training-side IPC producers."""

import gc

import ray
import torch
from verl.utils.device import get_torch_device, is_cuda_available

from verl_omni.utils.ray_lifecycle import actor_process_identity, close_actor_gracefully, terminate_actor_and_wait


def _collect_worker_ipc(worker):
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    accelerator = get_torch_device()
    accelerator.synchronize()
    worker.worker_dict.clear()
    gc.collect()
    collect = getattr(accelerator, "ipc_collect", None)
    if collect is not None:
        collect()
    accelerator.empty_cache()
    if is_cuda_available and torch.version.hip is None:
        from cuda.bindings import runtime

        # All consumers and distributed groups have stopped; this worker exits next.
        (status,) = runtime.cudaDeviceReset()
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA context shutdown failed: {status}")


def close_sync_trainer(trainer, rollout_name: str) -> None:
    """Confirm all initialized consumers stopped before releasing producer IPC.

    Partial initialization is supported. Unconfirmed consumer exit prevents
    producer cleanup; unrelated actors and shared GPU state are left alone.
    """
    rewards = getattr(trainer, "reward_loop_manager", None)
    rollout = getattr(trainer, "llm_server_manager", None)
    consumers = []
    if rewards is not None:
        consumers.append(rewards.multi_reward_model_manager)
    if rollout_name == "vllm_omni" and rollout is not None:
        consumers.extend(rollout.get_replicas())
    failures = []
    for consumer in consumers:
        try:
            consumer.shutdown()
        except Exception as error:
            failures.append(error)
    if failures:
        failures[0].add_note("Inference consumer exit was not confirmed; producer IPC is retained")
        raise failures[0]
    if rollout_name != "vllm_omni":
        return
    group = getattr(trainer, "actor_rollout_wg", None)
    if group is None:
        return
    if rollout is None:
        raise RuntimeError("Rollout initialization was incomplete; producer IPC is retained")
    workers = group.workers
    identities = ray.get([worker.__ray_call__.remote(actor_process_identity) for worker in workers], timeout=30)
    ray.get([worker.__ray_call__.remote(_collect_worker_ipc) for worker in workers], timeout=30)
    for worker, identity in zip(workers, identities, strict=True):
        try:
            close_actor_gracefully(worker, identity, timeout=30)
        except Exception:
            try:
                terminate_actor_and_wait(worker, identity, timeout=30)
            except Exception as error:
                failures.append(error)
    if failures:
        raise RuntimeError("Training worker exit was not confirmed") from failures[0]
