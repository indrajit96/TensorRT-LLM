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
"""Kimi K2.5 multimodal model — MoonViT vision encoder + projector + DeepSeek-V3 LLM.

This follows the LlavaNext composite VLM pattern:
  * KimiK25InputProcessor  — CPU-side preprocessing (tokenize, image→pixel_values)
  * KimiK25VisionModel     — GPU vision tower (MoonViT) + PatchMergerMLP projector
  * KimiK25Model           — orchestrator that fuses vision embeddings into the LLM

The vision tower is imported from ``modeling_moonvit.MoonViT3dModel``, analogous
to how LlavaNext imports ``CLIPVisionModel`` from ``modeling_clip``.
"""

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.kimi_k25_weight_mapper import KimiK25HfWeightMapper
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_moonvit import MoonViT3dModel
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_utils import register_auto_model, register_vision_encoder

# Flash attention availability (for vision encoder fallback decision)
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


# =============================================================================
# Multi-modal Projector
# =============================================================================

class KimiK25PatchMergerMLP(nn.Module):
    """Multi-modal projector: LayerNorm → Linear → GELU → Linear.

    Attribute names match HF checkpoint: mm_projector.pre_norm.*, mm_projector.proj.*
    """

    def __init__(
        self,
        mm_hidden_size: int = 1152,
        text_hidden_size: int = 7168,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        projector_ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = mm_hidden_size * (
            merge_kernel_size[0] * merge_kernel_size[1])
        self.pre_norm = nn.LayerNorm(mm_hidden_size, eps=projector_ln_eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, text_hidden_size),
        )

    def forward(
        self, x: Union[List[torch.Tensor], torch.Tensor], *args, **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(x, (list, tuple)):
            return [
                self.proj(self.pre_norm(item).view(item.shape[0], -1))
                for item in x
            ]
        B = x.shape[0]
        return self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))


# =============================================================================
# CPU Input Processor
# =============================================================================

class KimiK25InputProcessor(BaseMultimodalInputProcessor,
                            BaseMultimodalDummyInputsBuilder):
    """Pre-processes raw images + text into token IDs and multimodal data.

    Responsibilities:
      * Tokenise the text prompt via HF AutoProcessor.
      * Convert images to ``pixel_values`` tensors + ``grid_thws``.
      * Replace the media-placeholder token with out-of-vocab sentinel IDs
        so the fusion step can scatter vision embeddings at those positions.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._config = config
        self._tokenizer = (
            tokenizer if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast,
            ))
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast,
        )
        self._model_path = model_path
        self._dtype = (
            self.config.text_config.torch_dtype or self.config.torch_dtype)

        self.image_token_index = getattr(
            config, "media_placeholder_token_id", 163605)
        self.vocab_size = config.text_config.vocab_size

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # -- Helpers -------------------------------------------------------------

    def _postprocess(
        self,
        input_ids: torch.Tensor,
        mm_features: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace media-placeholder tokens with OOV sentinel IDs."""
        mm_tokens = torch.tensor([self.image_token_index]).to(
            input_ids.device)
        model_hidden_size = self.config.text_config.hidden_size
        start_len = end_len = 0

        mm_token_positions = torch.where(
            torch.isin(input_ids, mm_tokens))[0]
        num_medias = num_mm_tokens = len(mm_token_positions)
        if num_medias > 1 and isinstance(mm_features, torch.Tensor):
            mm_features = list(
                mm_features.split(mm_features.shape[0] // num_medias))

        if isinstance(mm_features, torch.Tensor):
            num_frames, mm_feature_length, mm_hidden_dim = mm_features.shape
            mm_lengths_per_split = [mm_feature_length * num_frames]
            mm_lengths_per_frame = [mm_feature_length]
        elif isinstance(mm_features, list):
            num_frames = (
                len(mm_features) if mm_features[0].dim() == 2 else sum(
                    f.shape[0] for f in mm_features))
            mm_lengths_per_split = [
                f.shape[0] if f.dim() == 2 else f.shape[0] * f.shape[1]
                for f in mm_features
            ]
            mm_lengths_per_frame = [
                f.shape[0] if f.dim() == 2 else f.shape[1]
                for f in mm_features
            ]
            mm_hidden_dim = mm_features[0].shape[-1]
            mm_features = torch.cat(mm_features, dim=0)
        else:
            raise ValueError(
                f"Invalid multimodal features type: {type(mm_features)}")
        mm_total_length = sum(mm_lengths_per_split)
        assert mm_hidden_dim == model_hidden_size, \
            "Multimodal embedding_dim must match model hidden_size"

        mm_split_positions = torch.cat(
            [mm_token_positions, mm_token_positions + 1]).unique()
        input_ids_splits = list(
            input_ids.tensor_split(mm_split_positions.cpu()))
        mm_ids_splits = list(
            torch.arange(
                self.vocab_size,
                self.vocab_size + mm_total_length,
                device=input_ids.device,
            ).split(mm_lengths_per_split))

        for i, mm_ids in enumerate(mm_ids_splits):
            mm_ids = mm_ids.reshape(-1, mm_lengths_per_frame[i])
            mm_ids_splits[i] = mm_ids.flatten()

        mm_split_idx = 0
        for i, split in enumerate(input_ids_splits):
            if torch.isin(split, mm_tokens).any().item():
                input_ids_splits[i] = mm_ids_splits[mm_split_idx]
                mm_split_idx += 1
        assert mm_split_idx == len(mm_ids_splits)

        fused_input_ids = torch.cat(input_ids_splits).to(
            device=input_ids.device)
        fused_length = (
            len(input_ids) + mm_total_length
            + num_frames * (start_len + end_len) - num_medias)
        assert len(fused_input_ids) == fused_length

        mm_features = mm_features.view(-1, mm_features.shape[-1])
        return fused_input_ids, mm_features

    # -- Tokenized+MM fast path (Dynamo / serving) ---------------------------

    def get_text_with_mm_placeholders(self, mm_counts: Dict[str, int]) -> str:
        """Return minimal dummy text so the HF processor can run vision-only.

        The ``tokenized_multimodal_process`` pipeline calls ``__call__`` with
        this dummy text + the real PIL images to obtain ``pixel_values`` /
        ``grid_thws`` without re-tokenising the actual prompt.
        """
        num_images = mm_counts.get("image", 0)
        placeholder = getattr(self.config, "media_placeholder_token",
                              "<|media_placeholder|>")
        return placeholder * num_images

    def get_num_tokens_per_image(
        self, *, image: Image.Image, **kwargs
    ) -> int:
        """Compute the number of LLM tokens one image produces after merge.

        Runs the HF processor on a single image to obtain ``grid_thws``,
        then applies the merge-kernel downsampling formula.
        """
        processed = self.processor(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": "x"},
            ]}],
            return_tensors="pt",
        )
        grid_thws = processed.get(
            'grid_thws', processed.get('image_grid_thw'))
        merge_k = getattr(
            self.config.vision_config, 'merge_kernel_size', [2, 2])
        _, h, w = grid_thws[0].tolist()
        return (h // merge_k[0]) * (w // merge_k[1])

    def _expand_image_placeholders_in_token_ids(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Replace each image placeholder token with the right number of OOV sentinels.

        Returns (expanded_ids, mm_token_lengths, mm_token_offsets).
        """
        placeholder_id = self.vocab_size + 1

        expanded: List[int] = []
        mm_token_lengths: List[int] = []
        mm_token_offsets: List[int] = []
        image_idx = 0
        for tok in prompt_token_ids:
            if tok == self.image_token_index:
                if image_idx >= len(num_mm_tokens_per_placeholder):
                    raise ValueError(
                        f"More image placeholder tokens in prompt than "
                        f"num_mm_tokens_per_placeholder entries: "
                        f"found {image_idx + 1} placeholders, "
                        f"have {len(num_mm_tokens_per_placeholder)} entries.")
                n = num_mm_tokens_per_placeholder[image_idx]
                mm_token_offsets.append(len(expanded))
                expanded.extend([placeholder_id] * n)
                mm_token_lengths.append(n)
                image_idx += 1
            else:
                expanded.append(tok)

        if image_idx != len(num_mm_tokens_per_placeholder):
            raise ValueError(
                f"Expected {len(num_mm_tokens_per_placeholder)} image "
                f"placeholders, found {image_idx}.")
        return expanded, mm_token_lengths, mm_token_offsets

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
        hf_processor_mm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Expand MM placeholder tokens in pre-tokenised IDs.

        Used by the ``tokenized_multimodal_process`` pipeline so that
        Dynamo/serving can send ``prompt_token_ids`` (from the Rust
        frontend, already containing single placeholder tokens) and raw
        PIL images, without detokenising and re-tokenising.
        """
        expanded, _, _ = self._expand_image_placeholders_in_token_ids(
            prompt_token_ids, num_mm_tokens_per_placeholder)
        return expanded

    # -- Public API ----------------------------------------------------------

    def get_prompt_token_ids(
        self,
        inputs: TextPrompt,
        mm_handles: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build input token IDs with multimodal placeholders expanded."""
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")
        if not isinstance(mm_handles, list):
            raise ValueError("mm_handles must be a list")

        expected_hidden_size = self.config.text_config.hidden_size
        for i, mm_handle in enumerate(mm_handles):
            hidden_size = mm_handle['tensor_size'][1]
            if hidden_size != expected_hidden_size:
                raise RuntimeError(
                    f"Multimodal embedding {i} hidden size {hidden_size} "
                    f"must match model hidden size {expected_hidden_size}")

        input_ids = self.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0]
        num_mm_tokens = [
            mm_handle["tensor_size"][0] for mm_handle in mm_handles
        ]
        expanded_ids, mm_token_length, mm_token_offsets = (
            self._expand_image_placeholders_in_token_ids(
                input_ids.tolist(), num_mm_tokens))
        return expanded_ids, mm_token_length, mm_token_offsets

    def attach_multimodal_embeddings(
        self,
        inputs: TextPrompt,
        multimodal_embedding: Dict[str, List[torch.Tensor]],
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """Attach pre-processed multimodal embeddings into the text token stream."""
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")
        if not isinstance(multimodal_embedding, dict):
            raise ValueError("multimodal_embedding must be a dictionary")
        if 'image' not in multimodal_embedding:
            raise ValueError(
                "Only image modality is supported for external multimodal embedding")

        input_ids = self.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0]
        mm_features = multimodal_embedding['image']
        fused_input_ids, mm_features = self._postprocess(
            input_ids, mm_features)
        multimodal_data: Dict[str, Any] = {}
        multimodal_data["multimodal_embedding"] = mm_features
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})

        images = mm_data.get('image', [])
        if not images:
            return (
                self.processor.tokenizer(
                    text_prompt,
                    return_tensors="pt",
                ).input_ids[0].to(torch.int32).tolist(),
                {},
            )

        processed_values = self.processor(
            messages=[
                {"role": "user", "content": [
                    *[{"type": "image_url", "image_url": img}
                      for img in images],
                    {"type": "text", "text": text_prompt},
                ]}
            ],
            return_tensors="pt",
        )

        raw_ids = processed_values['input_ids'][0]
        grid_thws = processed_values.get(
            'grid_thws', processed_values.get('image_grid_thw'))
        merge_k = getattr(
            self.config.vision_config, 'merge_kernel_size', [2, 2])
        num_mm_tokens_per_image = []
        for t, h, w in grid_thws.tolist():
            num_mm_tokens_per_image.append(
                (h // merge_k[0]) * (w // merge_k[1]))

        expanded_parts = []
        img_idx = 0
        for tok in raw_ids.tolist():
            if tok == self.image_token_index:
                n = num_mm_tokens_per_image[img_idx] \
                    if img_idx < len(num_mm_tokens_per_image) else 1
                expanded_parts.extend([self.vocab_size + 1] * n)
                img_idx += 1
            else:
                expanded_parts.append(tok)
        fused_input_ids = torch.tensor(expanded_parts, dtype=torch.int32)

        multimodal_data: Dict[str, Any] = {}
        multimodal_data["image"] = {
            "pixel_values": processed_values['pixel_values'],
        }
        # grid_thws is required by MoonViT; the HF processor returns it
        # under 'grid_thws' or 'image_grid_thw'
        for key in ('grid_thws', 'image_grid_thw', 'grid_thw'):
            if key in processed_values:
                multimodal_data["image"]["grid_thws"] = processed_values[key]
                break

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


# =============================================================================
# GPU Vision Encoder
# =============================================================================

class KimiK25VisionModel(nn.Module):
    """MoonViT vision tower + PatchMergerMLP projector for Kimi K2.5.

    ``forward()`` receives a list of ``MultimodalParams`` (one per in-flight
    request) and returns a list containing a single concatenated tensor of
    all image features projected to the LLM dimension.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.pretrained_config = model_config.pretrained_config
        vision_cfg = self.pretrained_config.vision_config
        text_cfg = self.pretrained_config.text_config

        self.dtype = (
            text_cfg.torch_dtype
            or self.pretrained_config.torch_dtype
        )

        # Determine attention implementation
        attn_impl = getattr(
            vision_cfg, '_attn_implementation', 'flash_attention_2')
        if attn_impl == 'flash_attention_2' and flash_attn_varlen_func is None:
            logger.warning(
                "flash_attn not available, falling back to eager attention "
                "for MoonViT vision encoder.")
            attn_impl = 'eager'

        # Vision tower: MoonViT-3D (from modeling_moonvit.py)
        self.vision_tower = MoonViT3dModel(
            hidden_size=getattr(vision_cfg, 'vt_hidden_size', 1152),
            intermediate_size=getattr(
                vision_cfg, 'vt_intermediate_size', 4304),
            num_hidden_layers=getattr(
                vision_cfg, 'vt_num_hidden_layers', 27),
            num_attention_heads=getattr(
                vision_cfg, 'vt_num_attention_heads', 16),
            patch_size=getattr(vision_cfg, 'patch_size', 14),
            init_pos_emb_height=getattr(
                vision_cfg, 'init_pos_emb_height', 64),
            init_pos_emb_width=getattr(
                vision_cfg, 'init_pos_emb_width', 64),
            init_pos_emb_time=getattr(vision_cfg, 'init_pos_emb_time', 4),
            pos_emb_type=getattr(vision_cfg, 'pos_emb_type', 'divided_fixed'),
            merge_kernel_size=tuple(getattr(
                vision_cfg, 'merge_kernel_size', (2, 2))),
            merge_type=getattr(vision_cfg, 'merge_type', 'sd2_tpool'),
            video_attn_type=getattr(
                vision_cfg, 'video_attn_type', 'spatial_temporal'),
            attn_implementation=attn_impl,
        ).to(self.dtype)

        # Multi-modal projector: PatchMergerMLP
        self.mm_projector = KimiK25PatchMergerMLP(
            mm_hidden_size=getattr(
                vision_cfg, 'mm_hidden_size',
                getattr(vision_cfg, 'vt_hidden_size', 1152)),
            text_hidden_size=getattr(
                vision_cfg, 'text_hidden_size',
                getattr(text_cfg, 'hidden_size', 7168)),
            merge_kernel_size=tuple(getattr(
                vision_cfg, 'merge_kernel_size', (2, 2))),
            projector_ln_eps=getattr(vision_cfg, 'projector_ln_eps', 1e-5),
        ).to(self.dtype)

        self.post_config()

    def post_config(self):
        self.config = self.pretrained_config.vision_config

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load vision tower and projector weights from a flat dict.

        Expects keys prefixed with ``vision_tower.`` and ``mm_projector.``.
        """

        def filter_weights(prefix: str, weights: Dict) -> Dict:
            return {
                key[len(prefix):]: weight
                for key, weight in weights.items()
                if key.startswith(prefix)
            }

        # Load vision tower weights (strict=False for non-persistent buffers
        # like time_weight and lazily-computed freqs_cis)
        vt_weights = filter_weights("vision_tower.", weights)
        if vt_weights:
            missing, unexpected = self.vision_tower.load_state_dict(
                vt_weights, strict=False)
            if unexpected:
                logger.warning(
                    f"Unexpected vision_tower keys: {unexpected}")

        # Load projector weights
        proj_weights = filter_weights("mm_projector.", weights)
        if proj_weights:
            self.mm_projector.load_state_dict(proj_weights, strict=True)

    @torch.inference_mode()
    def forward(
        self,
        multimodal_params: List[MultimodalParams],
    ) -> List[torch.Tensor]:
        """Run the MoonViT vision encoder on a batch of multimodal requests.

        Args:
            multimodal_params: One ``MultimodalParams`` per context request
                that carries ``multimodal_data["image"]["pixel_values"]``
                and ``multimodal_data["image"]["grid_thws"]``.

        Returns:
            A single-element list ``[image_features]`` where
            ``image_features`` has shape ``[total_mm_tokens, hidden_dim]``.
        """
        pixel_values_list = []
        grid_thws_list = []
        for mp in multimodal_params:
            img_data = mp.multimodal_data["image"]
            pixel_values_list.append(img_data["pixel_values"])
            grid_thws_list.append(img_data["grid_thws"])

        pixel_values = torch.cat(pixel_values_list, dim=0)
        grid_thws = torch.cat(grid_thws_list, dim=0)

        # Cast to vision tower dtype
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        # Run vision tower: pixel_values → features per image
        image_features = self.vision_tower(pixel_values, grid_thws)

        # Run projector: features → LLM-dimension embeddings
        image_features = self.mm_projector(image_features)

        # Concatenate all projected features into a single tensor
        image_features = torch.cat(image_features, dim=0)
        return [image_features]


# =============================================================================
# Composite VLM (Fusion + LLM Forward)
# =============================================================================

@support_multimodal_disaggregated
@register_vision_encoder(KimiK25VisionModel)
@register_auto_model("KimiK25ForConditionalGeneration")
@register_input_processor(
    KimiK25InputProcessor,
    model_type="kimi_k25",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<|media_placeholder|>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class KimiK25Model(PreTrainedModel):
    """Composite Kimi K2.5 model: MoonViT vision encoder + DeepSeek-V3 LLM.

    Registration decorators make this the entry point when HF ``config.json``
    contains ``"architectures": ["KimiK25ForConditionalGeneration"]``.

    NOTE: This supersedes the text-only ``KimiK25ForConditionalGeneration``
    that is defined in ``modeling_deepseekv3.py``.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        self._supports_sdpa = True
        super().__init__(config)

        if hasattr(self, "llm"):
            return

        # --- Vision encoder ---
        if not DISAGG:
            self.mm_encoder = KimiK25VisionModel(model_config)
        else:
            self.mm_encoder = None

        # --- LLM backbone (DeepSeek-V3) ---
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = (
            model_config.pretrained_config.text_config)
        # Share extra_attrs so MLA layers register into the same dict
        # the model engine reads from (deepcopy creates a separate dict).
        llm_model_config.extra_attrs = model_config.extra_attrs

        # Ensure torch_dtype propagates to text sub-config
        if llm_model_config.pretrained_config.torch_dtype is None:
            llm_model_config.pretrained_config.torch_dtype = (
                model_config.pretrained_config.torch_dtype)

        # Handle quant_config exclude_modules prefix stripping
        _LANG_PREFIX = "language_model."
        if model_config.quant_config.exclude_modules:
            llm_model_config.quant_config = copy.copy(
                model_config.quant_config)
            mapped = []
            for m in model_config.quant_config.exclude_modules:
                if m.startswith(_LANG_PREFIX):
                    rest = m[len(_LANG_PREFIX):]
                    if rest.startswith('layers.'):
                        rest = 'model.' + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            llm_model_config.quant_config.exclude_modules = mapped

        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.model_config = model_config
        self.post_config()

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        """Paths to multimodal data tensors that should be moved to GPU."""
        return [
            "image.pixel_values",
            "image.grid_thws",
            "multimodal_embedding",
        ]

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if isinstance(weight_mapper, KimiK25HfWeightMapper):
            weights = weight_mapper.preprocess_weights(weights)

        # Load vision encoder weights
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        # Filter and load LLM weights
        def filter_weights(weights: Dict) -> Dict:
            transformed_weights = {}
            for key, weight in weights.items():
                if key.startswith("language_model."):
                    new_key = key[len("language_model."):]
                    transformed_weights[new_key] = weight
                elif key.startswith("lm_head."):
                    transformed_weights[key] = weight
            return transformed_weights

        language_model_weights = filter_weights(weights)
        self.llm.load_weights(language_model_weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests = attn_metadata.num_contexts
        num_generation_requests = attn_metadata.num_generations
        logger.debug(
            f"{num_context_requests=}, {num_generation_requests=}")

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds: list = []
        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=multimodal_params[
                        :num_context_requests],
                )
            else:
                raise NotImplementedError(
                    "KimiK25Model does not support disaggregated inference "
                    "yet. Please unset the TLLM_MULTIMODAL_DISAGGREGATED "
                    "environment variable, or set it to '0'.")
            mm_embeds = find_input_mm_embeds(
                mm_embeds,
                multimodal_params[:num_context_requests],
            )

        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens, input_ids, mm_embeds, **kwargs)

        logits = self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
        )
        return logits
