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
import numpy as np

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class MultiScaleImageFeatureExtractor(nn.Module):
    def __init__(self, modelname: str = "dino_vits16", freeze: bool = False, scale_factors: list = [1, 1 / 2, 1 / 3]):
        super().__init__()
        self.freeze = freeze
        self.scale_factors = scale_factors

        if "res" in modelname:
            self._net = getattr(torchvision.models, modelname)(pretrained=True)
            self._output_dim = self._net.fc.weight.shape[1]
            self._net.fc = nn.Identity()
        elif "dinov2" in modelname:
            self._net = torch.hub.load("facebookresearch/dinov2", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        elif "dino" in modelname:
            self._net = torch.hub.load("facebookresearch/dino:main", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        else:
            raise ValueError(f"Unknown model name {modelname}")

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def get_output_dim(self):
        return self._output_dim

    def forward(self, image_rgb: torch.Tensor) -> torch.Tensor:
        img_normed = self._resnet_normalize_image(image_rgb)
        features = self._compute_multiscale_features(img_normed)
        return features

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def _compute_multiscale_features(self, img_normed: torch.Tensor) -> torch.Tensor:
        multiscale_features = None

        if len(self.scale_factors) <= 0:
            raise ValueError(f"Wrong format of self.scale_factors: {self.scale_factors}")

        for scale_factor in self.scale_factors:
            if scale_factor == 1:
                inp = img_normed
            else:
                inp = self._resize_image(img_normed, scale_factor)

            if multiscale_features is None:
                multiscale_features = self._net(inp)
            else:
                multiscale_features += self._net(inp)

        averaged_features = multiscale_features / len(self.scale_factors)
        return averaged_features

    @staticmethod
    def _resize_image(image: torch.Tensor, scale_factor: float) -> torch.Tensor:
        return nn.functional.interpolate(image, scale_factor=scale_factor, mode="bilinear", align_corners=False)
