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
"""CPU tests for the LTX OmniNFT RMSNorm compatibility workaround."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.normalization import RMSNorm
from tensordict import TensorDict

from verl_omni.pipelines.ltx2_omni_nft import compat
from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ltx_workaround_preserves_rms_norm_outside_npu(monkeypatch, device):
    monkeypatch.setattr(compat, "get_device_name", lambda: device)
    module = torch.nn.Sequential(RMSNorm(4, eps=1e-6, elementwise_affine=False))
    original = module[0].forward.__func__

    compat.apply_ltx_npu_rms_norm_workaround(module)

    assert module[0].forward.__func__ is original


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ltx_npu_workaround_preserves_output_and_gradients(monkeypatch, dtype):
    monkeypatch.setattr(compat, "get_device_name", lambda: "npu")
    norm = RMSNorm(4, eps=1e-6, elementwise_affine=False)
    untouched_norm = RMSNorm(4, eps=1e-6, elementwise_affine=False)
    original = RMSNorm.forward
    inputs = torch.tensor([[0.25, -0.5, 1.0, 2.0], [-1.5, 0.0, 0.75, 0.5]], dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    before = inputs.detach().clone()
    state_keys = tuple(norm.state_dict())

    compat.apply_ltx_npu_rms_norm_workaround(norm)
    output = norm(inputs)
    reference = torch.nn.functional.rms_norm(reference_inputs.float(), (4,), eps=norm.eps).to(dtype)
    output.float().square().sum().backward()
    reference.float().square().sum().backward()

    assert output.dtype == dtype
    assert output.device == inputs.device
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(inputs, before)
    torch.testing.assert_close(
        inputs.grad, reference_inputs.grad, atol=2e-2 if dtype == torch.bfloat16 else 1e-6, rtol=1e-4
    )
    assert torch.isfinite(inputs.grad).all()
    assert tuple(norm.state_dict()) == state_keys
    assert untouched_norm.forward.__func__ is original
    assert RMSNorm.forward is original


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_ltx_adapter_scopes_workaround_to_model_instances(monkeypatch, device):
    monkeypatch.setattr(compat, "get_device_name", lambda: device)
    module = torch.nn.Sequential(RMSNorm(4, eps=1e-6, elementwise_affine=False), RMSNorm(4, eps=1e-6))
    untouched = RMSNorm(4, eps=1e-6, elementwise_affine=False)
    original_forward = RMSNorm.forward
    latents = torch.randn(1, 5, 4)
    embeds = torch.randn(1, 2, 4)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            guidance_scale=1.0,
            num_frames=9,
            height=64,
            width=64,
            frame_rate=24,
        )
    )
    batch = TensorDict({"video_latents_clean": latents[:, :3], "audio_prompt_embeds": embeds}, batch_size=[1])

    inputs, negative = LTX23OmniNFT.prepare_model_inputs(
        module,
        config,
        latents,
        torch.tensor([500.0]),
        embeds,
        torch.ones(1, 2, dtype=torch.bool),
        None,
        None,
        batch,
        0,
    )

    expected = compat.ltx_affine_free_rms_norm_forward if device == "npu" else original_forward
    assert module[0].forward.__func__ is expected
    assert module[1].forward.__func__ is original_forward
    assert untouched.forward.__func__ is original_forward
    assert RMSNorm.forward is original_forward
    assert negative is None
    torch.testing.assert_close(inputs["hidden_states"], latents[:, :3])
    torch.testing.assert_close(inputs["audio_hidden_states"], latents[:, 3:])
