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
"""Weight mapper for Kimi K2.5 (moonshotai/Kimi-K2.5).

Maps HF checkpoint weight names to TRT-LLM's expected names.
The HF checkpoint uses these top-level prefixes:
  * ``vision_tower.*``        — MoonViT encoder weights
  * ``mm_projector.*``        — PatchMergerMLP projector weights
  * ``language_model.*``      — DeepSeek-V3 LLM backbone weights
  * ``lm_head.*``             — LM head weights

``KimiK25Model.load_weights()`` handles prefix stripping for each
sub-component, so the mapper only needs to handle the optional
top-level ``model.`` wrapper that some checkpoint variants include.
"""

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import (
    HfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "KimiK25ForConditionalGeneration")
class KimiK25HfWeightMapper(HfWeightMapper):
    """Renames HF checkpoint keys to match the TRT-LLM model structure.

    The moonshotai/Kimi-K2.5 safetensors checkpoint keys are already at the
    correct level (``vision_tower.*``, ``mm_projector.*``,
    ``language_model.*``), so this is largely a pass-through.

    Some checkpoint variants may wrap everything under ``model.``; this
    mapper strips that prefix when present.
    """

    def preprocess_weights(self, weights: dict) -> dict:
        transformed_weights: dict = {}
        for key, value in weights.items():
            if key.startswith("model."):
                # Strip the top-level model. wrapper if present.
                # e.g. model.language_model.* → language_model.*
                #      model.vision_tower.*   → vision_tower.*
                #      model.mm_projector.*   → mm_projector.*
                new_key = key[len("model."):]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value
        return transformed_weights
