# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.causal_wanvideo import CausalWanTransformerBlock
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.models.dits.wanvideo import WanTimeTextImageEmbedding
from fastvideo.platforms import current_platform

logger = init_logger(__name__)


def _resize_bcfhw_nearest(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    th, tw = int(target_hw[0]), int(target_hw[1])
    if x.shape[-2:] == (th, tw):
        return x
    b, c, f, h, w = x.shape
    x2d = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    x2d = F.interpolate(x2d, size=(th, tw), mode="nearest")
    return x2d.reshape(b, f, c, th, tw).permute(0, 2, 1, 3, 4).contiguous()


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class CausalWanControlnet3DModel(BaseDiT):
    """
    Causal Wan ControlNet for video latents.

    Expected inputs:
    - `hidden_states`: (B, 16, F, H, W)
    - `controlnet_states`: (B, 48, F, H, W) where 48 = 16(depth) + 16(masked_rgb) + 16(mask)

    Fusion follows Diff-Factory / diffusers-style WanControlnet:
      repeat video latents along channel (x3) then add control latents:
        (B,16,F,H,W) -> repeat -> (B,48,F,H,W); fused = repeat + control
    """

    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size

        # 1. Patch embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embedding
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks (same blocks as causal Wan transformer)
        self.blocks = nn.ModuleList([
            CausalWanTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.local_attn_size,
                config.sink_size,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.added_kv_proj_dim,
                self._supported_attention_backends,
                prefix=f"{config.prefix}.blocks.{i}",
            ) for i in range(config.num_layers)
        ])

        # 4. Zero-init projections to produce per-block residuals
        self.controlnet_blocks = nn.ModuleList([
            _zero_module(nn.Linear(inner_dim, inner_dim))
            for _ in range(len(self.blocks))
        ])

        # Causal-specific
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        self._block_mask_cache: dict[tuple[int, int], Any] = {}

        self.gradient_checkpointing = False
        self.__post_init__()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        *,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int,
        local_attn_size: int,
    ):
        from torch.nn.attention.flex_attention import create_block_mask

        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )
        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(_b, _h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return (((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen)))
                    | (q_idx == kv_idx))

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
        )

    def _get_block_mask(self, *, num_frames: int, frame_seqlen: int, device: torch.device) -> Any:
        key = (int(num_frames), int(frame_seqlen))
        mask = self._block_mask_cache.get(key)
        if mask is None:
            mask = self._prepare_blockwise_causal_attn_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )
            self._block_mask_cache[key] = mask
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        *,
        controlnet_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        start_frame: int = 0,
        kv_cache: list[dict[str, Any]] | None = None,
        crossattn_cache: list[dict[str, Any]] | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        use_cache = kv_cache is not None and crossattn_cache is not None
        if (kv_cache is None) ^ (crossattn_cache is None):
            raise ValueError("kv_cache and crossattn_cache must be both set or both None.")

        orig_dtype = hidden_states.dtype

        # Align device/dtype & spatial shape
        controlnet_states = controlnet_states.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if controlnet_states.shape[-2:] != hidden_states.shape[-2:]:
            controlnet_states = _resize_bcfhw_nearest(controlnet_states, hidden_states.shape[-2:])

        # Fuse: repeat latents (x3) then add control
        hidden_states = hidden_states.repeat(1, 3, 1, 1, 1) + controlnet_states

        batch_size, _c, num_frames, height, width = hidden_states.shape
        _, p_h, p_w = self.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Rotary embeddings (frame offset = start_frame)
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (num_frames * get_sp_world_size(), post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame,
        )
        freqs_cis = (freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device))

        # Block-wise causal attention mask (intra-chunk bidirectional)
        block_mask = self._get_block_mask(
            num_frames=num_frames,
            frame_seqlen=post_patch_height * post_patch_width,
            device=hidden_states.device,
        )

        # Patchify
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Pad text embeddings to fixed length
        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                batch_size,
                self.text_len - encoder_hidden_states.size(1),
                encoder_hidden_states.size(2),
            ),
        ], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(orig_dtype) if current_platform.is_mps() else encoder_hidden_states
        assert encoder_hidden_states.dtype == orig_dtype

        # Transformer blocks + residual projections
        residuals: list[torch.Tensor] = []
        for block_index, (block, proj) in enumerate(zip(self.blocks, self.controlnet_blocks, strict=True)):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                kv_cache=kv_cache[block_index] if use_cache else None,
                crossattn_cache=crossattn_cache[block_index] if use_cache else None,
                current_start=current_start if use_cache else 0,
                cache_start=cache_start if use_cache else None,
                block_mask=block_mask,
            )
            residuals.append(proj(hidden_states))

        return tuple(residuals)
