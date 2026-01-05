# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanTimeTextImageEmbedding, WanTransformerBlock
from fastvideo.platforms import current_platform

logger = init_logger(__name__)


def _resize_bcfhw_nearest(x: torch.Tensor,
                          target_hw: tuple[int, int]) -> torch.Tensor:
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


class WanControlnet3DModel(BaseDiT):
    """
    Bidirectional Wan ControlNet for video latents.

    Expected inputs:
    - `hidden_states`: (B, C_lat, F, H, W) where `C_lat` is the Wan VAE latent channel size
    - `controlnet_states`: (B, 3*C_lat, F, H, W) corresponding to (depth, masked_rgb, mask), each VAE-encoded

    Fusion follows Diff-Factory / diffusers-style WanControlnet:
      repeat video latents along channel (x3) then add control latents:
        (B,C_lat,F,H,W) -> repeat -> (B,3*C_lat,F,H,W); fused = repeat + control
    """

    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig(
    )._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig,
                 hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

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

        # 3. Transformer blocks (bidirectional Wan transformer blocks)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
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

        self.gradient_checkpointing = False
        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        *,
        controlnet_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        start_frame: int = 0,
        kv_cache: list[dict[str, Any]] | None = None,
        crossattn_cache: list[dict[str, Any]] | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, ...]:
        if kv_cache is not None or crossattn_cache is not None:
            logger.warning(
                "WanControlnet3DModel does not use kv_cache/crossattn_cache; ignoring caches."
            )
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        orig_dtype = hidden_states.dtype

        # Align device/dtype & spatial shape
        controlnet_states = controlnet_states.to(device=hidden_states.device,
                                                 dtype=hidden_states.dtype)
        if controlnet_states.shape[-2:] != hidden_states.shape[-2:]:
            controlnet_states = _resize_bcfhw_nearest(
                controlnet_states, hidden_states.shape[-2:])

        # Fuse: repeat latents (x3) then add control
        hidden_states = hidden_states.repeat(1, 3, 1, 1, 1) + controlnet_states

        batch_size, _c, num_frames, height, width = hidden_states.shape

        # Normalize timestep shape to (B, num_frames) so that downstream
        # modulation tensors can be reshaped consistently.
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.as_tensor(timestep, device=hidden_states.device)
        else:
            timestep = timestep.to(device=hidden_states.device)
        timestep_2d = timestep
        if timestep_2d.dim() == 0:
            timestep_2d = timestep_2d.view(1, 1).expand(batch_size, num_frames)
        elif timestep_2d.dim() == 1:
            if timestep_2d.numel() == 1:
                timestep_2d = timestep_2d.view(
                    1, 1).expand(batch_size, num_frames)
            elif timestep_2d.numel() == batch_size:
                timestep_2d = timestep_2d.view(batch_size, 1).expand(
                    batch_size, num_frames)
            elif timestep_2d.numel() == batch_size * num_frames:
                timestep_2d = timestep_2d.view(batch_size, num_frames)
            elif batch_size == 1 and timestep_2d.numel() == num_frames:
                timestep_2d = timestep_2d.view(1, num_frames)
            else:
                raise ValueError(
                    f"Unsupported timestep shape {tuple(timestep_2d.shape)} for batch_size={batch_size} num_frames={num_frames}"
                )
        elif timestep_2d.dim() == 2:
            if timestep_2d.shape == (batch_size, 1):
                timestep_2d = timestep_2d.expand(batch_size, num_frames)
            elif timestep_2d.shape != (batch_size, num_frames):
                raise ValueError(
                    f"Unsupported timestep shape {tuple(timestep_2d.shape)} for batch_size={batch_size} num_frames={num_frames}"
                )
        else:
            raise ValueError(
                f"Unsupported timestep dim {timestep_2d.dim()} for shape {tuple(timestep_2d.shape)}"
            )

        _, p_h, p_w = self.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Rotary embeddings (bidirectional)
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
        ],
                                          dim=1)

        # Match WanTransformer3DModel behavior for TI2V:
        # If timestep is per-frame, expand to per-token (seq_len) so that
        # timestep_proj aligns with hidden_states length.
        frame_seq_len = post_patch_height * post_patch_width
        if timestep_2d.shape[1] == num_frames:
            timestep_seq = timestep_2d.repeat_interleave(frame_seq_len, dim=1)
        else:
            timestep_seq = timestep_2d

        ts_seq_len = int(timestep_seq.shape[1]) if timestep_seq.dim() == 2 else None
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep_seq.flatten(), encoder_hidden_states,
            encoder_hidden_states_image, timestep_seq_len=ts_seq_len)
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states
        assert encoder_hidden_states.dtype == orig_dtype

        # Transformer blocks + residual projections
        residuals: list[torch.Tensor] = []
        for block_index, (block,
                          proj) in enumerate(zip(self.blocks,
                                                 self.controlnet_blocks,
                                                 strict=True)):
            hidden_states = block(hidden_states, encoder_hidden_states,
                                  timestep_proj, freqs_cis, None)
            residuals.append(proj(hidden_states))

        return tuple(residuals)
