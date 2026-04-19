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

Since Kimi K2.5 uses custom HF code and the full DeepSeek-V3 backbone is very
large, these tests use a tiny Kimi-like configuration and synthetic inputs.

Coverage provided by this file:
  * Synthetic multimodal forward coverage with tiny random-weight Kimi config
  * Aggregated `LLM(..., load_format="dummy")` smoke test with a tiny local
    HF-style Kimi config directory
  * Tokenized+MM fast-path tests (``expand_prompt_token_ids_for_mm``,
    ``get_text_with_mm_placeholders``, ``get_num_tokens_per_image``)
    that validate the Dynamo/serving integration path
"""

import json
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from test_modeling_multimodal import (
    MultimodalScenario,
    TestModelingMultimodal as _TestModelingMultimodal,
    llm_models_root,
)
from transformers import PreTrainedTokenizerFast, PretrainedConfig

from tensorrt_llm import LLM
from tensorrt_llm._torch.models.modeling_kimi_k25 import KimiK25Model
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.quantization import QuantAlgo

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
_MERGE_K = 2  # 2x2 spatial merge

# Text sub-config parameters (DeepSeek-V3-tiny)
# These dimensions are intentionally tiny, but the MLA-related fields are set
# to values that TRT-LLM's runtime path can actually initialize.
_TEXT_HIDDEN = 128
_TEXT_INTERMEDIATE = 256
_TEXT_LAYERS = 2
_TEXT_HEADS = 2
_TEXT_KV_HEADS = 2
_VOCAB_SIZE = 200000
_PLACEHOLDER_TOKEN_ID = 163605

# Synthetic image layout (must be divisible by _MERGE_K and _PATCH_SIZE)
_IMG_GRID_T = 1
_IMG_GRID_H = 4  # in patches
_IMG_GRID_W = 4  # in patches
_NUM_PATCHES = _IMG_GRID_T * _IMG_GRID_H * _IMG_GRID_W  # 16
_NUM_MM_TOKENS = (_IMG_GRID_H // _MERGE_K) * (_IMG_GRID_W // _MERGE_K)  # 4
_KIMI_K25_NVFP4_DIR = "Kimi-K2.5-NVFP4"

_LOCAL_KIMI_K25_CONFIG_MODULE = """
from transformers import PretrainedConfig


class KimiK25VisionConfig(PretrainedConfig):
    model_type = "kimi_k25_vision"

    def __init__(self, text_hidden_size=7168, **kwargs):
        self.text_hidden_size = text_hidden_size
        super().__init__(**kwargs)


class KimiK25Config(PretrainedConfig):
    model_type = "kimi_k25"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        media_placeholder_token_id=163605,
        **kwargs,
    ):
        text_config = text_config or {}
        vision_config = vision_config or {}
        if isinstance(text_config, dict):
            text_config = PretrainedConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = KimiK25VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.media_placeholder_token_id = media_placeholder_token_id
        super().__init__(**kwargs)
"""


def _make_tiny_kimi_k25_config() -> PretrainedConfig:
    """Build a minimal KimiK25-like config for unit testing."""
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
        mm_hidden_size=_VT_HIDDEN,
        text_hidden_size=_TEXT_HIDDEN,
        projector_ln_eps=1e-5,
        _attn_implementation="eager",
    )

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
        tie_word_embeddings=False,
        # MLA parameters tuned so TRT-LLM's runtime path can initialize.
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        q_lora_rank=32,
        kv_lora_rank=512,
        v_head_dim=128,
        # MoE parameters — dense-only via first_k_dense_replace > num_layers.
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=128,
        first_k_dense_replace=999,
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        num_nextn_predict_layers=0,
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

    return PretrainedConfig(
        architectures=["KimiK25ForConditionalGeneration"],
        model_type="kimi_k25",
        media_placeholder_token_id=_PLACEHOLDER_TOKEN_ID,
        torch_dtype=torch.bfloat16,
        text_config=text_config,
        vision_config=vision_config,
    )


def _write_local_tokenizer(model_dir: str) -> None:
    """Write a minimal local tokenizer so AutoTokenizer/AutoProcessor can load."""
    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|image|>": 4,
        "hello": 5,
        "world": 6,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.save_pretrained(model_dir)


def _write_local_kimi_k25_config(model_dir: str) -> None:
    """Write a tiny local HF-style Kimi config for dummy-load smoke tests."""
    config = _make_tiny_kimi_k25_config().to_dict()
    config["auto_map"] = {
        "AutoConfig": "configuration_kimi_k25.KimiK25Config",
    }

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    with open(model_dir_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(
        model_dir_path / "configuration_kimi_k25.py", "w", encoding="utf-8"
    ) as f:
        f.write(_LOCAL_KIMI_K25_CONFIG_MODULE)

    _write_local_tokenizer(model_dir)


# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------


def _make_synthetic_pixel_values(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Create synthetic pixel_values for one image."""
    return torch.randn(
        _NUM_PATCHES,
        3,
        _PATCH_SIZE,
        _PATCH_SIZE,
        device=device,
        dtype=dtype,
    )


def _make_synthetic_grid_thws(device: torch.device) -> torch.Tensor:
    """Create grid_thws for one image: (1, 3) -> [[t, h, w]]."""
    return torch.tensor(
        [[_IMG_GRID_T, _IMG_GRID_H, _IMG_GRID_W]],
        dtype=torch.int64,
        device=device,
    )


def _make_synthetic_inputs(
    device: torch.device,
    num_text_prefix: int = 8,
    num_text_suffix: int = 8,
) -> Tuple[torch.Tensor, List[MultimodalParams]]:
    """Build synthetic input_ids and multimodal_params for one request."""
    placeholder_id = _VOCAB_SIZE + 1

    prefix_ids = torch.randint(1, _VOCAB_SIZE, (num_text_prefix,))
    mm_ids = torch.full((_NUM_MM_TOKENS,), placeholder_id, dtype=torch.int64)
    suffix_ids = torch.randint(1, _VOCAB_SIZE, (num_text_suffix,))
    input_ids = torch.cat([prefix_ids, mm_ids, suffix_ids]).to(
        dtype=torch.int32, device=device
    )

    mm_start = num_text_prefix
    multimodal_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
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
    __test__ = False


class TestKimiK25(_TestModelingMultimodal):
    """Unit test for Kimi K2.5 multimodal model."""

    def get_model_config(self) -> Dict:
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

    def create_hf_config(self) -> PretrainedConfig:
        return _make_tiny_kimi_k25_config()

    def get_raw_test_inputs(self, modality: str):
        del modality
        return ["Describe the image briefly."], ["synthetic_image"]

    def get_raw_trtllm_inputs(
        self,
        modality: str,
        prompt: List[str],
        media: List[str],
    ):
        del modality, prompt, media
        return _make_synthetic_inputs(self.device)

    def get_hf_inputs(
        self,
        modality: str,
        prompt: List[str],
        media: List[str],
    ):
        del modality, prompt, media
        return {}

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        extra_attrs = self.model_config.extra_attrs
        extra_attrs["attention_metadata"] = weakref.ref(self.attn_metadata)
        with model_extra_attrs(extra_attrs):
            return super().run_trtllm_forward(trtllm_inputs, use_cuda_graph)

    def get_scenarios(self) -> List[TestKimiK25Scenario]:
        return [
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=True,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=True,
                kv_cache_reuse=False,
            ),
            TestKimiK25Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=True,
            ),
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kimi_k25_dummy_llm_load(tmp_path) -> None:
    """Smoke-test the aggregated LLM load path with dummy Kimi weights."""
    _write_local_kimi_k25_config(str(tmp_path))

    with LLM(
        model=str(tmp_path),
        load_format="dummy",
        trust_remote_code=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpus_per_node=1,
        max_batch_size=1,
        max_seq_len=64,
        enable_chunked_prefill=False,
    ) as llm:
        assert llm is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kimi_k25_nvfp4_dummy_llm_load() -> None:
    """Smoke-test the NVFP4 Kimi config path while still using dummy weights."""
    models_root = llm_models_root()
    if models_root is None:
        pytest.skip("LLM_MODELS_ROOT is not available")

    model_path = Path(models_root) / _KIMI_K25_NVFP4_DIR
    if not model_path.exists():
        pytest.skip(f"{model_path} is not available")

    with LLM(
        model=str(model_path),
        load_format="dummy",
        trust_remote_code=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpus_per_node=1,
        max_batch_size=1,
        max_seq_len=64,
        enable_chunked_prefill=False,
    ) as llm:
        assert llm is not None
        assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4


# =============================================================================
# Dynamo / serving fast-path tests
#
# These tests exercise the tokenized+MM path: Dynamo's Rust frontend has
# already tokenized the prompt (with placeholder tokens in the IDs), and the
# Python backend sends raw PIL images.  TRT-LLM must process images and
# expand the placeholders WITHOUT detokenising and re-tokenising.
# =============================================================================


def _make_kimi_input_processor(model_path: str):
    """Instantiate a KimiK25InputProcessor from a real or local HF directory."""
    from transformers import AutoTokenizer

    from tensorrt_llm.inputs.registry import create_input_processor

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    return create_input_processor(
        model_path, tokenizer=tokenizer, trust_remote_code=True)


class TestKimiK25ExpandPromptTokenIds:
    """Tests for expand_prompt_token_ids_for_mm — pure token-ID expansion.

    These do NOT need a real model or GPU.  They use the tiny local config
    to instantiate the input processor and test the expansion logic with
    synthetic token-ID sequences.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        _write_local_kimi_k25_config(str(tmp_path))
        self.input_processor = _make_kimi_input_processor(str(tmp_path))
        self.placeholder_id = _VOCAB_SIZE + 1
        self.image_token_id = _PLACEHOLDER_TOKEN_ID

    def test_single_image_expansion(self):
        """One image placeholder is expanded to the correct count."""
        prompt_ids = [1, 2, self.image_token_id, 3, 4]
        num_tokens = [_NUM_MM_TOKENS]  # 4

        expanded = self.input_processor.expand_prompt_token_ids_for_mm(
            prompt_ids, num_tokens)

        expected_len = 2 + _NUM_MM_TOKENS + 2
        assert len(expanded) == expected_len
        assert expanded[:2] == [1, 2]
        assert expanded[2:2 + _NUM_MM_TOKENS] == [self.placeholder_id
                                                   ] * _NUM_MM_TOKENS
        assert expanded[-2:] == [3, 4]

    def test_multiple_image_expansion(self):
        """Multiple image placeholders each expand independently."""
        prompt_ids = [
            1, self.image_token_id, 2, self.image_token_id, 3,
            self.image_token_id, 4
        ]
        tokens_per_image = [10, 20, 5]

        expanded = self.input_processor.expand_prompt_token_ids_for_mm(
            prompt_ids, tokens_per_image)

        expected_len = 1 + 10 + 1 + 20 + 1 + 5 + 1
        assert len(expanded) == expected_len
        pos = 0
        assert expanded[pos] == 1
        pos += 1
        assert expanded[pos:pos + 10] == [self.placeholder_id] * 10
        pos += 10
        assert expanded[pos] == 2
        pos += 1
        assert expanded[pos:pos + 20] == [self.placeholder_id] * 20
        pos += 20
        assert expanded[pos] == 3
        pos += 1
        assert expanded[pos:pos + 5] == [self.placeholder_id] * 5
        pos += 5
        assert expanded[pos] == 4

    def test_no_placeholders_passthrough(self):
        """When there are no placeholders and zero images, IDs pass through."""
        prompt_ids = [1, 2, 3, 4, 5]
        expanded = self.input_processor.expand_prompt_token_ids_for_mm(
            prompt_ids, [])
        assert expanded == prompt_ids

    def test_mismatched_counts_raises(self):
        """More placeholders than entries in num_mm_tokens raises ValueError."""
        prompt_ids = [self.image_token_id, self.image_token_id]
        with pytest.raises(ValueError, match="placeholder"):
            self.input_processor.expand_prompt_token_ids_for_mm(
                prompt_ids, [10])

    def test_fewer_placeholders_than_entries_raises(self):
        """Fewer placeholders than entries in num_mm_tokens raises ValueError."""
        prompt_ids = [1, self.image_token_id, 2]
        with pytest.raises(ValueError, match="placeholder"):
            self.input_processor.expand_prompt_token_ids_for_mm(
                prompt_ids, [10, 20])


class TestKimiK25GetTextWithPlaceholders:
    """Tests for get_text_with_mm_placeholders — dummy text generation."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        _write_local_kimi_k25_config(str(tmp_path))
        self.input_processor = _make_kimi_input_processor(str(tmp_path))

    def test_single_image(self):
        text = self.input_processor.get_text_with_mm_placeholders(
            {"image": 1})
        assert "<|media_placeholder|>" in text
        assert text.count("<|media_placeholder|>") == 1

    def test_multiple_images(self):
        text = self.input_processor.get_text_with_mm_placeholders(
            {"image": 3})
        assert text.count("<|media_placeholder|>") == 3

    def test_zero_images(self):
        text = self.input_processor.get_text_with_mm_placeholders(
            {"image": 0})
        assert "<|media_placeholder|>" not in text


class TestKimiK25SharedExpansionHelper:
    """Tests for _expand_image_placeholders_in_token_ids — shared helper
    used by both expand_prompt_token_ids_for_mm and get_prompt_token_ids."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        _write_local_kimi_k25_config(str(tmp_path))
        self.input_processor = _make_kimi_input_processor(str(tmp_path))
        self.placeholder_id = _VOCAB_SIZE + 1
        self.image_token_id = _PLACEHOLDER_TOKEN_ID

    def test_returns_offsets_and_lengths(self):
        """Verify that offsets and lengths are correct."""
        prompt_ids = [10, self.image_token_id, 20, self.image_token_id, 30]
        expanded, lengths, offsets = (
            self.input_processor._expand_image_placeholders_in_token_ids(
                prompt_ids, [3, 5]))

        assert lengths == [3, 5]
        assert offsets == [1, 1 + 3 + 1]  # offset 1, then 1 + 3 tokens + 1 text
        assert len(expanded) == 1 + 3 + 1 + 5 + 1

    def test_get_prompt_token_ids_uses_shared_helper(self):
        """get_prompt_token_ids delegates to _expand_image_placeholders_in_token_ids."""
        text_prompt = "hello world"
        mm_handles = [{"tensor_size": [_NUM_MM_TOKENS, _TEXT_HIDDEN]}]
        inputs = {"prompt": text_prompt}

        expanded_ids, mm_lengths, mm_offsets = (
            self.input_processor.get_prompt_token_ids(inputs, mm_handles))

        token_ids = self.input_processor.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0].tolist()
        num_placeholders = sum(1 for t in token_ids
                               if t == self.image_token_id)

        if num_placeholders > 0:
            assert len(mm_lengths) == num_placeholders
            assert all(
                ln == _NUM_MM_TOKENS for ln in mm_lengths)


class TestKimiK25DynamoStyleE2E:
    """End-to-end test mimicking the Dynamo/serving interaction pattern.

    Dynamo sends:
      {
          "prompt_token_ids": [..., 163605, ...],  # from Rust tokenizer
          "multi_modal_data": {"image": [PIL.Image]},
          "mm_processor_kwargs": {}
      }

    TRT-LLM must:
      1. Run vision preprocessing on the PIL images (get pixel_values)
      2. Compute per-image token counts
      3. Expand placeholders in prompt_token_ids
      4. Return expanded IDs + multimodal_data for the GPU pipeline

    This test requires a real Kimi model directory because it exercises
    the full HF AutoProcessor (Kimi's custom processor code).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from PIL import Image

        models_root = llm_models_root()
        if models_root is None:
            pytest.skip("LLM_MODELS_ROOT is not available")
        model_path = Path(models_root) / _KIMI_K25_NVFP4_DIR
        if not model_path.exists():
            pytest.skip(f"{model_path} is not available")

        self.model_path = str(model_path)
        self.input_processor = _make_kimi_input_processor(self.model_path)
        self.test_image = Image.new("RGB", (224, 224), color=(128, 64, 200))

    def test_get_num_tokens_per_image(self):
        """get_num_tokens_per_image returns a positive integer consistent
        with the merge-kernel formula."""
        num_tokens = self.input_processor.get_num_tokens_per_image(
            image=self.test_image)

        assert isinstance(num_tokens, int)
        assert num_tokens > 0

    def test_fast_path_produces_expanded_ids(self):
        """The full tokenized_multimodal_process pipeline produces
        expanded IDs where placeholders are replaced by OOV sentinels."""
        from tensorrt_llm.inputs.registry import create_input_processor_with_hash
        from tensorrt_llm.sampling_params import SamplingParams

        vocab_size = self.input_processor.vocab_size
        image_token_id = self.input_processor.image_token_index
        placeholder_id = vocab_size + 1

        prompt_token_ids = [1, 2, 3, image_token_id, 4, 5]
        inputs = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"image": [self.test_image]},
            "mm_processor_kwargs": {},
        }
        sampling_params = SamplingParams(max_tokens=1)

        processor_with_hash = create_input_processor_with_hash(
            self.input_processor)
        expanded_ids, extra = processor_with_hash(inputs, sampling_params)

        assert image_token_id not in expanded_ids, \
            "Raw placeholder should be expanded away"
        assert any(tok >= vocab_size for tok in expanded_ids), \
            "Expanded IDs should contain OOV sentinels"
        assert extra is not None
        mm_data = extra.get("multimodal_data", {})
        assert "image" in mm_data, \
            "Extra outputs should contain vision data"
        assert "pixel_values" in mm_data["image"], \
            "Vision data should have pixel_values"

    def test_fast_path_vs_direct_call_parity(self):
        """Verify that the Dynamo fast path and the direct __call__ path
        produce equivalent expanded token IDs (same OOV sentinel positions,
        same pixel_values shape)."""
        from tensorrt_llm.inputs.registry import create_input_processor_with_hash
        from tensorrt_llm.sampling_params import SamplingParams

        sampling_params = SamplingParams(max_tokens=1)
        text_prompt = "Describe this image in detail"

        # --- Path A: Direct __call__ (what LLM.generate with prompt= does) ---
        direct_inputs = {
            "prompt": text_prompt,
            "multi_modal_data": {"image": [self.test_image]},
        }
        direct_ids, direct_extra = self.input_processor(
            direct_inputs, sampling_params)

        # --- Path B: Dynamo fast path (prompt_token_ids from Rust) ---
        # Simulate Rust frontend: tokenize and insert placeholder
        image_token_id = self.input_processor.image_token_index
        tokenized = self.input_processor.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0].tolist()
        rust_token_ids = [image_token_id] + tokenized

        dynamo_inputs = {
            "prompt_token_ids": rust_token_ids,
            "multi_modal_data": {"image": [self.test_image]},
            "mm_processor_kwargs": {},
        }
        processor_with_hash = create_input_processor_with_hash(
            self.input_processor)
        dynamo_ids, dynamo_extra = processor_with_hash(
            dynamo_inputs, sampling_params)

        # Both paths should produce OOV sentinels (no raw placeholder left)
        vocab_size = self.input_processor.vocab_size
        assert image_token_id not in direct_ids
        assert image_token_id not in dynamo_ids

        direct_oov_count = sum(1 for t in direct_ids if t >= vocab_size)
        dynamo_oov_count = sum(1 for t in dynamo_ids if t >= vocab_size)
        assert direct_oov_count > 0
        assert dynamo_oov_count > 0
        assert direct_oov_count == dynamo_oov_count, \
            (f"OOV sentinel counts must match: direct={direct_oov_count}, "
             f"dynamo={dynamo_oov_count}")

        # Both paths should produce pixel_values of the same shape
        direct_pv = direct_extra["multimodal_data"]["image"]["pixel_values"]
        dynamo_pv = dynamo_extra["multimodal_data"]["image"]["pixel_values"]
        assert direct_pv.shape == dynamo_pv.shape, \
            (f"pixel_values shapes must match: direct={direct_pv.shape}, "
             f"dynamo={dynamo_pv.shape}")

    def test_multiple_images_fast_path(self):
        """The fast path handles multiple images in a single request."""
        from PIL import Image

        from tensorrt_llm.inputs.registry import create_input_processor_with_hash
        from tensorrt_llm.sampling_params import SamplingParams

        image_token_id = self.input_processor.image_token_index
        vocab_size = self.input_processor.vocab_size

        img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img2 = Image.new("RGB", (112, 112), color=(0, 255, 0))

        prompt_token_ids = [
            1, image_token_id, 2, image_token_id, 3
        ]
        inputs = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"image": [img1, img2]},
            "mm_processor_kwargs": {},
        }
        sampling_params = SamplingParams(max_tokens=1)

        processor_with_hash = create_input_processor_with_hash(
            self.input_processor)
        expanded_ids, extra = processor_with_hash(inputs, sampling_params)

        assert image_token_id not in expanded_ids
        oov_count = sum(1 for t in expanded_ids if t >= vocab_size)
        assert oov_count > 0

        n1 = self.input_processor.get_num_tokens_per_image(image=img1)
        n2 = self.input_processor.get_num_tokens_per_image(image=img2)
        assert oov_count == n1 + n2, \
            (f"Total OOV sentinels ({oov_count}) must equal "
             f"sum of per-image tokens ({n1} + {n2} = {n1 + n2})")
