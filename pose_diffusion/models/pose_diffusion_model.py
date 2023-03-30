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
import pytorch3d.renderer.cameras as cameras
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


@dataclass
class PoseDiffusionModel(nn.Module):
    Img_model: Dict
    Denoiser: Dict

    # Img_model: Dict = field(default_factory=dict)
    # Denoiser: Dict = field(default_factory=dict)

    def __post_init__(self):
        super().__init__()
        self.img_model = instantiate(self.Img_model)
        self.denoiser = instantiate(self.Denoiser)
        
        print('done')
        # import pdb;pdb.set_trace()
        
    def forward(self):
        return None
