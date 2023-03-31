# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


import torch
import torch.nn as nn
import torchvision

import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    def __init__(
        self,
        target_dim: int = 9,  # TODO: fl dim from 2 to 1
        pivot_cam: bool = True,
        append_t: bool = True,
        z_dim: int = 384,
        mlp_hidden_dim: bool = 128,
        time_multiplier: float = 0.1,
        nhead: int = 4,
        d_model: int = 512,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,  # TODO: necessary?
        norm_first: bool = True,
        proj_xt_first: bool = True,
        proj_dim: int = 96,
    ):
        super().__init__()

        self.pivot_cam = pivot_cam
        self.append_t = append_t
        self.target_dim = target_dim
        self.time_multiplier = time_multiplier
        self.proj_xt_first = proj_xt_first

        self.proj_xt = torch.nn.Linear(self.target_dim + 1, proj_dim)
        first_dim = (
            self.target_dim
            + z_dim
            + proj_dim
            + int(self.pivot_cam)
            + int(self.append_t)
        )

        # TODO: rename _first, _trunk, and _last
        self._first = nn.Linear(first_dim, d_model)
        self._trunk = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )

        # TODO: change the implementation of MLP to a more mature one
        self._last = MLP(
            d_model,
            [mlp_hidden_dim, self.target_dim],
            norm_layer=nn.LayerNorm,
        )

    def forward(
        self,
        x: torch.Tensor,  # B x N_x x dim
        t: torch.Tensor,  # B
        z: torch.Tensor,  # B x N_z x dim_z
    ):
        B, N, Xdim = x.shape

        # expand t from B to B x N_x x 1
        t_expand = (t * self.time_multiplier).view(B, 1, 1).expand(-1, N, -1)

        if self.pivot_cam:
            # add the one hot vector identifying the first camera as pivot
            # NOTE we treat the first camera as pivot during training
            cam_pivot_id = torch.zeros_like(z[..., :1])
            cam_pivot_id[:, 0, ...] = 1.0
            z = torch.cat([z, cam_pivot_id], dim=-1)

        if self.proj_xt_first:
            xt = self.proj_xt(torch.cat([x, t_expand], dim=-1))
            feed_feats = torch.cat([x, t_expand, xt, z], dim=-1)
        else:
            # TODO: add the variants here
            raise NotImplementedError(f"Variants to be implemented")

        input_ = self._first(feed_feats)

        feats_ = self._trunk(input_, input_)

        if isinstance(self._last, MLP):
            _, _, featdim = feats_.shape
            output = self._last(feats_.reshape(-1, featdim)).reshape(B, N, -1)
        else:
            output = self._last(feats_)

        return output


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        norm_first: bool = False,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        if norm_first:
            for hidden_dim in hidden_channels[:-1]:
                if norm_layer is not None:
                    layers.append(norm_layer(in_dim))
                layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
                layers.append(activation_layer(**params))
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout, **params))
                in_dim = hidden_dim
        else:
            for hidden_dim in hidden_channels[:-1]:
                layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
                if norm_layer is not None:
                    layers.append(norm_layer(hidden_dim))
                layers.append(activation_layer(**params))
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout, **params))
                in_dim = hidden_dim

        if norm_first:
            layers.append(norm_layer(in_dim))
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
