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
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import (
    se3_exp_map,
    se3_log_map,
    Transform3d,
    so3_relative_angle,
)
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


import models
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(
        self,
        IMG_MODEL: Dict,
        GAU_DIFFUSER: Dict,
        DENOISER: Dict,
        target_dim: int = 9,  # TODO: fl dim from 2 to 1
    ):
        super().__init__()
        self.target_dim = target_dim
        DENOISER.target_dim = target_dim

        self.img_model = instantiate(IMG_MODEL)
        self.gau_diffuser = instantiate(GAU_DIFFUSER)

        denoiser = instantiate(DENOISER)
        self.gau_diffuser.model = denoiser

    def forward(
        self,
        image: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        matches_dict=None,
    ) -> Dict[str, Any]:
        z = self.img_model(image)
        # TODO: unsqueeze to be consistent with our original implementation
        # remove this in the future
        z = z.unsqueeze(0)

        B, N, _ = z.shape
        target_shape = [B, N, self.target_dim]

        pose, pose_process = self.gau_diffuser.sample(shape=target_shape, z=z)

        return pose
