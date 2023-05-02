# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from util.embedding import TimeStepEmbedding, PoseEmbedding

import torch
import torch.nn as nn

from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    def __init__(
        self,
        TRANSFORMER: Dict,
        target_dim: int = 9,  # TODO: reduce fl dim from 2 to 1
        pivot_cam_onehot: bool = True,
        z_dim: int = 384,
        mlp_hidden_dim: bool = 128,
    ):
        super().__init__()

        self.pivot_cam_onehot = pivot_cam_onehot
        self.target_dim = target_dim

        self.time_embed = TimeStepEmbedding()
        self.pose_embed = PoseEmbedding(target_dim=self.target_dim)

        first_dim = (
            self.time_embed.out_dim
            + self.pose_embed.out_dim
            + z_dim
            + int(self.pivot_cam_onehot)
        )

        d_model = TRANSFORMER.d_model
        self._first = nn.Linear(first_dim, d_model)

        # slightly different from the paper that 
        # we use 2 encoder layers and 6 decoder layers
        # here we use a transformer with 8 encoder layers
        # call TransformerEncoderWrapper() to build a encoder-only transformer
        self._trunk = instantiate(TRANSFORMER, _recursive_=False)

        # TODO: change the implementation of MLP to a more mature one
        self._last = MLP(
            d_model, [mlp_hidden_dim, self.target_dim], norm_layer=nn.LayerNorm,
        )

    def forward(
        self,
        x: torch.Tensor,  # B x N x dim
        t: torch.Tensor,  # B
        z: torch.Tensor,  # B x N x dim_z
    ):
        B, N, _ = x.shape

        t_emb = self.time_embed(t)
        # expand t from B x C to B x N x C
        t_emb = t_emb.view(B, 1, t_emb.shape[-1]).expand(-1, N, -1)

        x_emb = self.pose_embed(x)

        if self.pivot_cam_onehot:
            # add the one hot vector identifying the first camera as pivot
            cam_pivot_id = torch.zeros_like(z[..., :1])
            cam_pivot_id[:, 0, ...] = 1.0
            z = torch.cat([z, cam_pivot_id], dim=-1)

        feed_feats = torch.cat([x_emb, t_emb, z], dim=-1)

        input_ = self._first(feed_feats)

        feats_ = self._trunk(input_)

        output = self._last(feats_)

        return output


def TransformerEncoderWrapper(
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    norm_first: bool = True,
    batch_first: bool = True,
):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=norm_first,
    )

    _trunk = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    return _trunk


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional):
            Norm layer that will be stacked on top of the convolution layer.
            If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional):
            Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the
            conv layer. If ``None`` this layer wont be used.
            Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can
            optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[
            Callable[..., torch.nn.Module]
        ] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        norm_first: bool = False,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from 
        # the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels

        for hidden_dim in hidden_channels[:-1]:
            if norm_first and norm_layer is not None:
                layers.append(norm_layer(in_dim))

            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            if not norm_first and norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            layers.append(activation_layer(**params))

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout, **params))

            in_dim = hidden_dim

        if norm_first and norm_layer is not None:
            layers.append(norm_layer(in_dim))

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
