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

from fastvideo.models.dits.controlnet_union_components import (
    ResidualAttentionBlock,
    WanControlNetConditioningEmbedding,
    WanControlNetUnionInput,
    zero_module,
)

logger = init_logger(__name__)


def _resize_bcfhw_nearest(x: torch.Tensor,
                          target_hw: tuple[int, int]) -> torch.Tensor:
    th, tw = int(target_hw[0]), int(target_hw[1])
    if x.shape[-2:] == (th, tw):
        return x
    b, c, f, h, w = x.shape
    x2d = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    x2d = F.interpolate(x2d, size=(th, tw), mode="nearest")
    return x2d.reshape(b, f, c, th, tw).permute(0, 2, 1, 3,
                                                4).contiguous()


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class CausalWanControlnetUnion3DModel(BaseDiT):
    """
    Causal Union ControlNet for video latents.
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

        # Union config (fallback to safe defaults)
        self.num_control_type = int(hf_config.get("num_control_type", 2))
        self.union_dim = int(hf_config.get("num_trans_channel", inner_dim))
        if self.union_dim != inner_dim:
            logger.warning(
                "union_dim (%s) != inner_dim (%s); overriding to inner_dim to match checkpoint shapes.",
                self.union_dim,
                inner_dim,
            )
            self.union_dim = inner_dim
        self.num_trans_head = int(
            hf_config.get("num_trans_head", min(8, config.num_attention_heads)))
        self.num_trans_layer = int(hf_config.get("num_trans_layer", 1))
        # Diff-Factory Union expects *latent* control inputs (C_lat),
        # which is in_channels // 3 (since in_channels = 3 * C_lat for
        # [noisy_latent, mask, masked_latent] concat).
        # Some configs store control_input_channels=3 (raw RGB); override
        # to the latent channel count for consistency.
        control_input_channels = int(self.in_channels // 3)

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

        # 3. Union conditioning branch
        self.union_cond_embedding = WanControlNetConditioningEmbedding(
            conditioning_embedding_channels=inner_dim,
            conditioning_channels=control_input_channels,
        )
        task_scale_factor = self.union_dim**0.5
        self.task_embedding = nn.Parameter(
            task_scale_factor *
            torch.randn(self.num_control_type, self.union_dim))
        self.union_transformer = nn.ModuleList([
            ResidualAttentionBlock(self.union_dim, self.num_trans_head)
            for _ in range(self.num_trans_layer)
        ])
        self.spatial_ch_projs = zero_module(
            nn.Linear(self.union_dim, inner_dim))
        self.feat_to_union = nn.Identity()
        if self.union_dim != inner_dim:
            self.feat_to_union = nn.Linear(inner_dim, self.union_dim)
        self.control_type_proj = nn.Linear(self.num_control_type,
                                           inner_dim * 6)

        # 4. Transformer blocks (causal)
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

        # 5. Zero-init projections to produce per-block residuals
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

        ends = torch.zeros(total_length + padded_length,
                           device=device,
                           dtype=torch.long)
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
            return (((kv_idx < ends[q_idx]) &
                     (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen)))
                    | (q_idx == kv_idx))

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
        )

    def _get_block_mask(self, *, num_frames: int, frame_seqlen: int,
                        device: torch.device) -> Any:
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
        controlnet_cond: WanControlNetUnionInput,
        mask: torch.Tensor,
        masked_latent: torch.Tensor,
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
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        use_cache = kv_cache is not None and crossattn_cache is not None
        if (kv_cache is None) ^ (crossattn_cache is None):
            raise ValueError(
                "kv_cache and crossattn_cache must be both set or both None.")

        orig_dtype = hidden_states.dtype

        # Align device/dtype & spatial shape
        mask = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        masked_latent = masked_latent.to(device=hidden_states.device,
                                         dtype=hidden_states.dtype)
        if mask.shape[-2:] != hidden_states.shape[-2:]:
            mask = _resize_bcfhw_nearest(mask, hidden_states.shape[-2:])
        if masked_latent.shape[-2:] != hidden_states.shape[-2:]:
            masked_latent = _resize_bcfhw_nearest(masked_latent,
                                                  hidden_states.shape[-2:])

        # Union control inputs
        control_id_vec = torch.zeros(hidden_states.shape[0],
                                     self.num_control_type,
                                     device=hidden_states.device,
                                     dtype=hidden_states.dtype)
        for i, item in enumerate(list(controlnet_cond)):
            if controlnet_cond[item] is not None:
                control_id_vec[:, i] = 1.0

        # 1) Patch embedding from concatenated latents
        hidden_states_concat = torch.cat(
            [hidden_states, mask, masked_latent], dim=1)
        latents_emb = self.patch_embedding(hidden_states_concat)

        # 2) Union fusion
        inputs_seq: list[torch.Tensor] = []
        condition_maps: list[torch.Tensor] = []
        for idx, cond_type in enumerate(controlnet_cond):
            cond_input = controlnet_cond[cond_type]
            if cond_input is None:
                continue
            cond_input = cond_input.to(device=hidden_states.device,
                                       dtype=hidden_states.dtype)
            if cond_input.shape[-2:] != hidden_states.shape[-2:]:
                cond_input = _resize_bcfhw_nearest(
                    cond_input, hidden_states.shape[-2:])
            cond_feat = self.union_cond_embedding(cond_input)
            feat_vec = torch.mean(cond_feat, dim=(2, 3, 4))
            if self.union_dim != latents_emb.shape[1]:
                feat_vec = self.feat_to_union(feat_vec)
            feat_vec = feat_vec + self.task_embedding[idx]
            inputs_seq.append(feat_vec.unsqueeze(1))
            condition_maps.append(cond_feat)

        latent_vec = torch.mean(latents_emb, dim=(2, 3, 4))
        if self.union_dim != latents_emb.shape[1]:
            latent_vec = self.feat_to_union(latent_vec)
        inputs_seq.append(latent_vec.unsqueeze(1))

        if len(inputs_seq) > 1:
            x_union = torch.cat(inputs_seq, dim=1)
            for layer in self.union_transformer:
                x_union = layer(x_union)
            fused_condition = torch.zeros_like(latents_emb)
            for i, cond_map in enumerate(condition_maps):
                alpha = self.spatial_ch_projs(x_union[:, i]).view(
                    latents_emb.shape[0], latents_emb.shape[1], 1, 1, 1)
                fused_condition = fused_condition + (cond_map + alpha)
            latents_emb = latents_emb + fused_condition

        hidden_states = latents_emb
        batch_size, _c, num_frames, height, width = hidden_states.shape

        # Normalize timestep shape
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.as_tensor(timestep, device=hidden_states.device)
        else:
            timestep = timestep.to(device=hidden_states.device)
        timestep_2d = timestep
        if timestep_2d.dim() == 0:
            timestep_2d = timestep_2d.view(1, 1).expand(batch_size, num_frames)
        elif timestep_2d.dim() == 1:
            if timestep_2d.numel() == 1:
                timestep_2d = timestep_2d.view(1, 1).expand(
                    batch_size, num_frames)
            elif timestep_2d.numel() == batch_size:
                timestep_2d = timestep_2d.view(batch_size, 1).expand(
                    batch_size, num_frames)
            elif timestep_2d.numel() == batch_size * num_frames:
                timestep_2d = timestep_2d.view(batch_size, num_frames)
            elif batch_size == 1 and timestep_2d.numel() == num_frames:
                timestep_2d = timestep_2d.view(1, num_frames)
            elif timestep_2d.numel() % max(1, num_frames) == 0:
                if timestep_2d.numel() % batch_size == 0:
                    timestep_2d = timestep_2d.view(batch_size,
                                                   timestep_2d.numel() //
                                                   batch_size)
                else:
                    raise ValueError(
                        f"Unsupported timestep shape {tuple(timestep_2d.shape)} for batch_size={batch_size} num_frames={num_frames}"
                    )
            else:
                raise ValueError(
                    f"Unsupported timestep shape {tuple(timestep_2d.shape)} for batch_size={batch_size} num_frames={num_frames}"
                )
        elif timestep_2d.dim() == 2:
            if timestep_2d.shape == (batch_size, 1):
                timestep_2d = timestep_2d.expand(batch_size, num_frames)
            elif timestep_2d.shape == (batch_size, num_frames):
                pass
            elif timestep_2d.shape[0] == batch_size and timestep_2d.shape[
                    1] % max(1, num_frames) == 0:
                pass
            else:
                raise ValueError(
                    f"Unsupported timestep shape {tuple(timestep_2d.shape)} for batch_size={batch_size} num_frames={num_frames}"
                )
        else:
            raise ValueError(
                f"Unsupported timestep dim {timestep_2d.dim()} for shape {tuple(timestep_2d.shape)}"
            )

        # hidden_states already patchified by PatchEmbed; do not divide again.
        post_patch_height = height
        post_patch_width = width

        # Rotary embeddings (frame offset = start_frame)
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (num_frames * get_sp_world_size(), post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame,
        )
        freqs_cis = (freqs_cos.to(hidden_states.device),
                     freqs_sin.to(hidden_states.device))

        # Block-wise causal attention mask (intra-chunk bidirectional)
        block_mask = self._get_block_mask(
            num_frames=num_frames,
            frame_seqlen=post_patch_height * post_patch_width,
            device=hidden_states.device,
        )

        # Patchify
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Pad text embeddings to fixed length
        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                batch_size,
                self.text_len - encoder_hidden_states.size(1),
                encoder_hidden_states.size(2),
            ),
        ],
                                          dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep_2d.flatten(), encoder_hidden_states,
            encoder_hidden_states_image)

        control_global_emb = self.control_type_proj(control_id_vec)
        control_global_emb = control_global_emb.repeat_interleave(
            num_frames, dim=0)
        timestep_proj = timestep_proj + control_global_emb

        timestep_proj = timestep_proj.unflatten(1, (6,
                                                    self.hidden_size)).unflatten(
                                                        dim=0,
                                                        sizes=timestep_2d.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states
        assert encoder_hidden_states.dtype == orig_dtype

        residuals: list[torch.Tensor] = []
        for block_index, (block,
                          proj) in enumerate(zip(self.blocks,
                                                 self.controlnet_blocks,
                                                 strict=True)):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                kv_cache=kv_cache[block_index] if use_cache else None,
                crossattn_cache=crossattn_cache[block_index]
                if use_cache else None,
                current_start=current_start if use_cache else 0,
                cache_start=cache_start if use_cache else None,
                block_mask=block_mask,
            )
            residuals.append(proj(hidden_states))

        return tuple(residuals)
