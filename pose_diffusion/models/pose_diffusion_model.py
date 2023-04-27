# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import (
    se3_exp_map,
    se3_log_map,
    Transform3d,
    so3_relative_angle,
)
from util.camera_transform import pose_encoding_to_camera

import models
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras


logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(
        self,
        pose_encoding_type: str,
        IMAGE_FEATURE_EXTRACTOR: Dict,
        DIFFUSER: Dict,
        DENOISER: Dict,
    ):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.pose_encoding_type = pose_encoding_type

        self.image_feature_extractor = instantiate(
            IMAGE_FEATURE_EXTRACTOR, _recursive_=False
        )
        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

    def forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """

        z = self.image_feature_extractor(image)

        z = z.unsqueeze(0)

        B, N, _ = z.shape
        target_shape = [B, N, self.target_dim]

        # sampling
        pose_encoding, pose_encoding_diffusion_samples = self.diffuser.sample(
            shape=target_shape,
            z=z,
            cond_fn=cond_fn,
            cond_start_step=cond_start_step,
        )

        # convert the encoded representation to PyTorch3D cameras
        pred_cameras = pose_encoding_to_camera(
            pose_encoding, pose_encoding_type=self.pose_encoding_type
        )

        return pred_cameras
