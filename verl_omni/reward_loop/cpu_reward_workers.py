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
"""Placement helper for native reward workers that run on CPU resources."""

from __future__ import annotations


def build_cpu_reward_workers(
    config,
    reward_loop_workers_class,
    reward_model_specs,
    worker_indices,
    cpus_per_worker,
    worker_name_prefix,
):
    """Create one Ray reward actor per configured CPU replica.

    CPU native deployments do not borrow the trainer's accelerator resource
    pool. ``worker_indices`` are logical replica slots from the named-model
    placement config and determine stable actor ordering; Ray schedules each
    actor with an explicit CPU reservation.
    """
    worker_indices = tuple(worker_indices)
    return [
        reward_loop_workers_class.options(
            num_cpus=cpus_per_worker,
            name=f"{worker_name_prefix}_{worker_index}",
        ).remote(config, None, reward_model_specs)
        for worker_index in worker_indices
    ]
