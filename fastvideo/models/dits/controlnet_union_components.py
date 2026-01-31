# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


@dataclass
class WanControlNetUnionInput:
    """
    Union control inputs (matching Diff-Factory semantics).

    - depth
    - normal
    """
    depth: Optional[torch.Tensor] = None
    normal: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(vars(self))

    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class QuickGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class ResidualAttentionMlp(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = ResidualAttentionMlp(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class WanControlNetConditioningEmbedding(nn.Module):
    """
    3D conditioning embedding used by Union ControlNet.
    Input:  [B, C_in, F, H, W]
    Output: [B, C_out, F, H/2, W/2]
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 512,
        conditioning_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(
            conditioning_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList([])
        self.blocks.append(
            nn.Conv3d(
                block_out_channels[0],
                block_out_channels[1],
                kernel_size=3,
                padding=1,
                stride=(1, 2, 2),
            )
        )
        self.blocks.append(nn.SiLU())

        self.conv_out = zero_module(
            nn.Conv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        embedding = self.conv_in(conditioning)
        embedding = torch.nn.functional.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
        embedding = self.conv_out(embedding)
        return embedding
