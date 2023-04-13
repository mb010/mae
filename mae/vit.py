# ------------------------------------------------------------------------
# Modified from TIMM (https://github.com/rwightman/pytorch-image-models)
# Copyright (c) 2020 Ross Wightman.
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the CC-BY-NC 4.0 license.
# ------------------------------------------------------------------------
# Modified from  Semi-Vit (https://github.com/amazon-science/semi-vit)

from functools import partial

import torch
import torch.nn as nn
import numpy as np
import timm.models.vision_transformer

from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class ViT_Encoder(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        global_pool=False,
        use_fixed_pos_emb=False,
        init_scale=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = self.patch_embed.patch_size[0]
        self.in_chans = self.patch_embed.proj.in_channels

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        if use_fixed_pos_emb:
            embed_dim = self.pos_embed.shape[-1]
            # remove the original pos_embed
            del self.pos_embed

            # fixed sin-cos embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False
            )

            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
            )

            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, reduce=True):
        """
        Args:
            x: input image
            reduce: whether to reduce the output to [B, D] or [B, L, D]

        Returns:
            outcome: class token or global pooled features
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :]
            if reduce:
                x = x.mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            # Extract CLS token
            outcome = x[:, 0]

        return outcome

    def transform(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x


class Transformer(nn.Module):
    """
    ViT decoder for masked autoencoder (MAE)

    Args:
        dim: dimension of the input
        depth: number of layers
        heads: number of attention heads
        mlp_ratio: ratio of mlp hidden dim to embedding dim
    """

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4):
        super().__init__()

        self.dim = self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x
