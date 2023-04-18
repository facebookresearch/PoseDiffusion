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

# from pytorch3d.ops import corresponding_cameras_alignment
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


logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(
        self,
        pose_encoding_type: str,
        IMAGE_FEATURE_EXTRACTOR: Dict,
        DIFFUSER: Dict,
        DENOISER: Dict,
    ):
        """
        Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type: defines the internal representation if extrinsics and intrinsics (i.e., the rotation `R`  and translation `t`, focal length).
            Currently, only `"absT_quaR_logFL"` is supported - comprising a concatenation of the translation vector, rotation quaternion, and logarithm of focal length.
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
        camera: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
    ) -> Dict[str, Any]:
        z = self.image_feature_extractor(image)

        z = z.unsqueeze(0)

        B, N, _ = z.shape
        target_shape = [B, N, self.target_dim]

        pose_encoding, pose_encoding_diffusion_samples = self.diffuser.sample(
            shape=target_shape,
            z=z,
            cond_fn=cond_fn,
            cond_start_step=cond_start_step,
        )

        pred_cameras = pose_encoding_to_camera(
            pose_encoding, pose_encoding_type=self.pose_encoding_type
        )

        return pred_cameras
