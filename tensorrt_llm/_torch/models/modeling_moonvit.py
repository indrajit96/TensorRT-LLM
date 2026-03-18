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
"""MoonViT-3D vision encoder (used by Kimi K2.5).

Standalone vision tower implementation, analogous to ``modeling_clip.py``
for CLIP.  The composite VLM in ``modeling_kimi_k25.py`` imports
``MoonViT3dModel`` from here.

Architecture:
  * 27-layer ViT with 1152 hidden_size, 16 heads, 4304 intermediate
  * 14×14 patch size, 2D RoPE, flash attention
  * 2×2 spatial patch merge with temporal pooling (sd2_tpool)
"""

import math
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash attention for MoonViT vision encoder
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


# =============================================================================
# Attention helpers
# =============================================================================

def _moonvit_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Multi-head attention using flash attention 2 (variable length)."""
    attn_out = flash_attn_varlen_func(
        q, k, v,
        q_cu_seqlens, k_cu_seqlens,
        max_seqlen_q, max_seqlen_k,
        causal=False,
        deterministic=deterministic,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return attn_out.flatten(start_dim=-2)


def _moonvit_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Fallback eager attention for MoonViT (no flash_attn dependency)."""
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1]:q_cu_seqlens[i],
            q_cu_seqlens[i - 1]:q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(
        attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    return attn_output.reshape(seq_length, -1)


_MOONVIT_ATTN_FNS = {
    "flash_attention_2": _moonvit_flash_attention,
    "eager": _moonvit_eager_attention,
}


# =============================================================================
# RoPE helpers
# =============================================================================

def _moonvit_apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary position embeddings to query and key tensors.

    Args:
        xq: (..., num_heads, head_dim)
        xk: (..., num_heads, head_dim)
        freqs_cis: (..., head_dim/2), dtype=complex64
    """
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# Positional embedding helpers
# =============================================================================

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    """Sincos positional embedding for 1D grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int):
    """Temporal sincos positional embedding."""
    grid_t = np.arange(t_size, dtype=np.float32)
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


def _get_rope_shape_decorate(func):
    """Decorator to warm up torch.compile for get_rope_shape."""
    _first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _first_call_flag:
            _first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@_get_rope_shape_decorate
@torch.compile(dynamic=True)
def _get_rope_shape(org, interpolation_mode, shape):
    """Interpolate 2D positional embedding to target shape."""
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        ).squeeze(0).permute((1, 2, 0)).flatten(end_dim=1)
    )


# =============================================================================
# MoonViT Modules
# =============================================================================

class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """Learnable 2D positional embedding with bicubic interpolation + temporal sincos."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = 'bicubic',
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            'time_weight',
            torch.from_numpy(
                _get_1d_sincos_pos_embed(self.dim, self.num_frames)
            ).float().unsqueeze(1),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(
        self, x: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, (
                f't:{t} > self.num_frames:{self.num_frames}')
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = _get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1)
                    + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))
        return x + torch.cat(pos_embs)


class MoonVision3dPatchEmbed(nn.Module):
    """Patch embedding: Conv2d(3→hidden) + positional embedding."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = 'divided_fixed',
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert isinstance(patch_size, Sequence) and len(patch_size) == 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        if pos_emb_type == 'divided_fixed':
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
            )
        else:
            raise NotImplementedError(
                f'Unsupported pos_emb_type: {pos_emb_type}')

    def forward(
        self, x: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        """Args: x (L, C_in, patch_h, patch_w), grid_thws (N, 3)."""
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support."""

    def __init__(
        self, dim: int, max_height: int, max_width: int,
        theta_base: float = 10000,
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, 'dim must be divisible by 4'
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(
            0, self.dim, 4)[:(self.dim // 4)].float().to(device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, 'freqs_cis'):
            self.register_buffer(
                'freqs_cis', self._precompute_freqs_cis(device),
                persistent=False)
        shapes = grid_thws.tolist()
        return torch.cat([
            self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
            for t, h, w in shapes
        ], dim=0)


class MoonViTMLP(nn.Module):
    """2-layer MLP for MoonViT encoder blocks."""

    def __init__(self, dims: List[int], activation, bias: bool = True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


class MoonViTEncoderLayer(nn.Module):
    """Single MoonViT transformer block: LN → Attn(+RoPE) → LN → MLP."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = 'flash_attention_2',
        activation=F.gelu,
        attn_bias: bool = False,
        use_deterministic_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = hidden_dim // num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MoonViTMLP([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ):
        xqkv = self.wqkv(x)
        qkv_shape = xqkv.size()[:-1] + (
            3, self.num_heads, self.hidden_size_per_attention_head)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = _moonvit_apply_rope(xq, xk, rope_freqs_cis)

        attn_func = _MOONVIT_ATTN_FNS.get(
            self.attn_implementation, _moonvit_eager_attention)
        attn_out = attn_func(
            xq, xk, xv,
            q_cu_seqlens=cu_seqlens,
            k_cu_seqlens=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            deterministic=self.use_deterministic_attn,
        )
        return self.wo(attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self.attention_qkvpacked(
            hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MoonViT3dEncoder(nn.Module):
    """Stack of MoonViT encoder layers with 2D RoPE."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = 'spatial_temporal',
    ) -> None:
        super().__init__()
        assert video_attn_type == 'spatial_temporal'
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg['hidden_dim'] // block_cfg['num_heads'], 512, 512)
        self.blocks = nn.ModuleList([
            MoonViTEncoderLayer(**block_cfg, use_deterministic_attn=False)
            for _ in range(num_layers)
        ])
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device)
        lengths = torch.cat((
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        ))
        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(
            dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens, max_seqlen,
                rope_freqs_cis=rope_freqs_cis)
        return self.final_layernorm(hidden_states)


def _tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> List[torch.Tensor]:
    """Spatial 2×2 downsampling with temporal pooling (sd2_tpool)."""
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum:pre_sum + t * h * w]
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(
            t, new_height, kernel_height, new_width, kernel_width, -1)
        # permute to (t, new_h, new_w, kh, kw, d) then temporal pool
        reshaped_seq = reshaped_seq.permute(
            0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        # reshape to (new_h*new_w, kh*kw, d) for the projector
        padded_seq = reshaped_seq.view(
            new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w
    return outputs


class MoonViT3dModel(nn.Module):
    """Complete MoonViT-3D vision tower: patch_embed → encoder → patch_merger.

    Attribute names match the HF checkpoint key hierarchy:
        ``vision_tower.patch_embed.*``, ``vision_tower.encoder.*``

    This is the standalone vision encoder, analogous to ``CLIPVisionModel``
    in ``modeling_clip.py``.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = 'divided_fixed',
        merge_kernel_size: Tuple[int, int] = (2, 2),
        merge_type: str = 'sd2_tpool',
        video_attn_type: str = 'spatial_temporal',
        attn_implementation: str = 'flash_attention_2',
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.merge_type = merge_type

        # Use PytorchGELUTanh as activation (matches HF reference)
        try:
            from transformers.activations import PytorchGELUTanh
        except ImportError:
            from transformers.activations import GELUTanh as PytorchGELUTanh

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=hidden_size,
            patch_size=patch_size,
            pos_emb_height=init_pos_emb_height,
            pos_emb_width=init_pos_emb_width,
            pos_emb_time=init_pos_emb_time,
            pos_emb_type=pos_emb_type,
        )
        self.encoder = MoonViT3dEncoder(
            hidden_dim=hidden_size,
            num_layers=num_hidden_layers,
            block_cfg={
                'num_heads': num_attention_heads,
                'hidden_dim': hidden_size,
                'mlp_dim': intermediate_size,
                'activation': PytorchGELUTanh(),
                'attn_bias': True,
                'attn_implementation': attn_implementation,
            },
            video_attn_type=video_attn_type,
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run vision tower: patch_embed → encoder → patch_merger.

        Args:
            pixel_values: (total_patches, 3, patch_h, patch_w)
            grid_thws: (num_images, 3) with (t, h, w) per image

        Returns:
            List of tensors, each (num_merged_patches, merge_k*merge_k,
            hidden_size)
        """
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == 'sd2_tpool':
            return _tpool_patch_merger(
                hidden_states, grid_thws,
                merge_kernel_size=self.merge_kernel_size)
        raise NotImplementedError(
            f'Unsupported merge_type: {self.merge_type}')
