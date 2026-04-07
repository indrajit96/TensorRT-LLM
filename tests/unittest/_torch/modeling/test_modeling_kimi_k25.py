# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Kimi K2.5 multimodal model (MoonViT + DeepSeek-V3).

Since Kimi K2.5 uses custom HF code (trust_remote_code) and the DeepSeek-V3
backbone is very large, we:
  * Skip HF inference comparison (skip_hf_inference=True)
  * Build the config manually with tiny dimensions (2 encoder + 2 LLM layers)
  * Skip weight loading (random weights for shape-only testing)
  * Create synthetic inputs (no tokenizer / HF processor needed)
  * Focus on verifying the full forward-pass pipeline works end-to-end:
    Synthetic pixel_values → MoonViT → PatchMerger → Projector → Fusion → LLM
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_kimi_k25 import KimiK25Model
from tensorrt_llm.inputs.multimodal import (
    MultimodalInput,
    MultimodalParams,
)

# ---------------------------------------------------------------------------
# Tiny config for testing — all dimensions drastically reduced
# ---------------------------------------------------------------------------

# Vision sub-config parameters (MoonViT-tiny)
_VT_HIDDEN = 64
_VT_INTERMEDIATE = 128
_VT_LAYERS = 2
_VT_HEADS = 4
_PATCH_SIZE = 14
_POS_EMB_H = 16
_POS_EMB_W = 16
_MERGE_K = 2  # 2×2 spatial merge

# Text sub-config parameters (DeepSeek-V3-tiny, dense-only)
_TEXT_HIDDEN = 128
_TEXT_INTERMEDIATE = 256
_TEXT_LAYERS = 2
_TEXT_HEADS = 4
_TEXT_KV_HEADS = 2
_VOCAB_SIZE = 200000
_PLACEHOLDER_TOKEN_ID = 163605

# Synthetic image layout (must be divisible by _MERGE_K and _PATCH_SIZE)
_IMG_GRID_T = 1
_IMG_GRID_H = 4  # in patches — i.e. 4 patches high
_IMG_GRID_W = 4  # in patches — i.e. 4 patches wide
_NUM_PATCHES = _IMG_GRID_T * _IMG_GRID_H * _IMG_GRID_W  # 16
_NUM_MM_TOKENS = (_IMG_GRID_H // _MERGE_K) * (_IMG_GRID_W // _MERGE_K)  # 4


def _make_tiny_kimi_k25_config() -> PretrainedConfig:
    """Build a minimal KimiK25-like config for unit testing.

    All dimensions are shrunk drastically so the model fits on a single GPU
    with random weights and the forward pass completes quickly.

    Key choices
    -----------
    * 2 MoonViT encoder layers (real model: 27)
    * 2 DeepSeek-V3 decoder layers (real model: 61)
    * All decoder layers are dense (no MoE) via ``first_k_dense_replace=999``
    * Eager attention in MoonViT (no flash_attn dependency)
    """

    # ── Vision sub-config (MoonViT-tiny) ─────────────────────────────────
    vision_config = PretrainedConfig(
        vt_hidden_size=_VT_HIDDEN,
        vt_intermediate_size=_VT_INTERMEDIATE,
        vt_num_hidden_layers=_VT_LAYERS,
        vt_num_attention_heads=_VT_HEADS,
        patch_size=_PATCH_SIZE,
        init_pos_emb_height=_POS_EMB_H,
        init_pos_emb_width=_POS_EMB_W,
        init_pos_emb_time=2,
        pos_emb_type="divided_fixed",
        merge_kernel_size=[_MERGE_K, _MERGE_K],
        merge_type="sd2_tpool",
        video_attn_type="spatial_temporal",
        # Projector sizing
        mm_hidden_size=_VT_HIDDEN,
        text_hidden_size=_TEXT_HIDDEN,
        projector_ln_eps=1e-5,
        # Use eager attention so we don't need flash_attn
        _attn_implementation="eager",
    )

    # ── Text sub-config (DeepSeek-V3-tiny, dense-only) ───────────────────
    text_config = PretrainedConfig(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        hidden_size=_TEXT_HIDDEN,
        intermediate_size=_TEXT_INTERMEDIATE,
        num_hidden_layers=_TEXT_LAYERS,
        num_attention_heads=_TEXT_HEADS,
        num_key_value_heads=_TEXT_KV_HEADS,
        vocab_size=_VOCAB_SIZE,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        torch_dtype=torch.bfloat16,
        # MLA parameters (tiny)
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        q_lora_rank=32,
        kv_lora_rank=32,
        v_head_dim=16,
        # MoE parameters — dense-only via first_k_dense_replace > num_layers
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=128,
        first_k_dense_replace=999,  # all layers are dense
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        # No MTP (matches Kimi K2.5)
        num_nextn_predict_layers=0,
        # RoPE
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "original_max_position_embeddings": 4096,
        },
    )

    # ── Top-level composite config ───────────────────────────────────────
    config = PretrainedConfig(
        architectures=["KimiK25ForConditionalGeneration"],
        model_type="kimi_k25",
        media_placeholder_token_id=_PLACEHOLDER_TOKEN_ID,
        torch_dtype=torch.bfloat16,
        text_config=text_config,
        vision_config=vision_config,
    )

    return config


# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------


def _make_synthetic_pixel_values(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Create synthetic pixel_values for one image.

    Returns shape ``(_NUM_PATCHES, 3, _PATCH_SIZE, _PATCH_SIZE)``.
    """
    return torch.randn(
        _NUM_PATCHES, 3, _PATCH_SIZE, _PATCH_SIZE,
        device=device, dtype=dtype,
    )


def _make_synthetic_grid_thws(device: torch.device) -> torch.Tensor:
    """Create grid_thws for one image: (1, 3) → [[t, h, w]]."""
    return torch.tensor(
        [[_IMG_GRID_T, _IMG_GRID_H, _IMG_GRID_W]],
        dtype=torch.int64, device=device,
    )


def _make_synthetic_inputs(
    device: torch.device,
    num_text_prefix: int = 8,
    num_text_suffix: int = 8,
) -> Tuple[torch.Tensor, List["MultimodalParams"]]:
    """Build synthetic input_ids and multimodal_params for one request.

    Token layout::

        [text_prefix ... | mm_placeholder×N | ... text_suffix]

    Where mm_placeholder tokens are ``vocab_size + 1`` (OOV sentinel),
    N = ``_NUM_MM_TOKENS``.
    """
    placeholder_id = _VOCAB_SIZE + 1

    prefix_ids = torch.randint(1, _VOCAB_SIZE, (num_text_prefix,))
    mm_ids = torch.full((_NUM_MM_TOKENS,), placeholder_id, dtype=torch.int64)
    suffix_ids = torch.randint(1, _VOCAB_SIZE, (num_text_suffix,))
    input_ids = torch.cat([prefix_ids, mm_ids, suffix_ids]).to(
        dtype=torch.int32, device=device)

    mm_start = num_text_prefix
    multimodal_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],  # dummy hash
        multimodal_positions=[mm_start],
        multimodal_lengths=[_NUM_MM_TOKENS],
    )

    pixel_values = _make_synthetic_pixel_values(device)
    grid_thws = _make_synthetic_grid_thws(device)

    multimodal_params = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            },
        },
        multimodal_input=multimodal_input,
    )

    return input_ids, [multimodal_params]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class TestKimiK25Scenario(MultimodalScenario):
    pass


class TestKimiK25(TestModelingMultimodal):
    """Unit test for Kimi K2.5 multimodal model.

    Only runs sanity-checking for the TRTLLM model forward pass with
    random weights.  HuggingFace inference is skipped since Kimi K2.5
    uses custom HF code and the DeepSeek-V3 backbone is too large.

    Unlike other VLM tests, this one creates synthetic inputs (pixel_values
    and input_ids) directly, so no tokenizer or HF processor is needed.
    """

    # ── Abstract method implementations ──────────────────────────────────

    def get_model_config(self) -> Dict:
        # Not used directly — create_hf_config is overridden.
        return {}

    def get_trtllm_model_class(self):
        return KimiK25Model

    def get_hf_model_class(self):
        return None

    def create_hf_model(self, pretrained_config):
        if self.skip_hf_inference:
            return None
        raise ValueError(
            "Kimi K2.5 does not support HuggingFace inference in this test."
        )

    def get_weight_mapper_class(self):
        # Weights are not loaded (random init) so mapper is not needed.
        return None

    def get_model_type(self) -> str:
        return "kimi_k25"

    def get_model_config_class(self):
        return PretrainedConfig

    @property
    def trust_remote_code(self) -> bool:
        return True

    @property
    def skip_hf_inference(self) -> bool:
        return True

    # ── Config creation ──────────────────────────────────────────────────

    def create_hf_config(self) -> PretrainedConfig:
        """Build a tiny config programmatically.

        We override the base class to avoid needing the real HF checkpoint
        or a ``from_dict`` call that doesn't handle nested sub-configs.
        """
        return _make_tiny_kimi_k25_config()

    # ── Input creation (synthetic — no tokenizer needed) ─────────────────

    def get_raw_trtllm_inputs(
        self, modality: str, prompt: List[str], media: List[str],
    ):
        """Create synthetic input_ids + multimodal_params.

        Bypasses the HF tokenizer / processor entirely.
        """
        return _make_synthetic_inputs(self.device)

    def get_hf_inputs(
        self, modality: str, prompt: List[str], media: List[str],
    ):
        """Return dummy — HF inference is always skipped for Kimi K2.5."""
        return {}

    # ── Scenarios ────────────────────────────────────────────────────────

    def get_scenarios(self) -> List[TestKimiK25Scenario]:
        return [
            # Basic image forward-pass sanity check
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # CUDA graph generation
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=True,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # Chunked prefill
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=True,
                kv_cache_reuse=False,
            ),
            # KV cache reuse
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=True,
            ),
        ]
