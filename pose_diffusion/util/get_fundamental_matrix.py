# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytorch3d
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.transforms.so3 import hat
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


def get_fundamental_matrices(
    camera: CamerasBase,
    height: int,
    width: int,
    index1: torch.LongTensor,
    index2: torch.LongTensor,
    l2_normalize_F=False,
):
    """Compute fundamental matrices for given camera parameters."""
    batch_size = camera.R.shape[0]

    # Convert to opencv / colmap / Hartley&Zisserman convention
    image_size_t = torch.LongTensor([height, width])[None].repeat(batch_size, 1).to(camera.device)
    R, t, K = opencv_from_cameras_projection(camera, image_size=image_size_t)

    F, E = get_fundamental_matrix(K[index1], R[index1], t[index1], K[index2], R[index2], t[index2])

    if l2_normalize_F:
        F_scale = torch.norm(F, dim=(1, 2))
        F_scale = F_scale.clamp(min=0.0001)
        F = F / F_scale[:, None, None]

    return F


def get_fundamental_matrix(K1, R1, t1, K2, R2, t2):
    E = get_essential_matrix(R1, t1, R2, t2)
    F = K2.inverse().permute(0, 2, 1).matmul(E).matmul(K1.inverse())
    return F, E  # p2^T F p1 = 0


def get_essential_matrix(R1, t1, R2, t2):
    R12 = R2.matmul(R1.permute(0, 2, 1))
    t12 = t2 - R12.matmul(t1[..., None])[..., 0]
    E_R = R12
    E_t = -E_R.permute(0, 2, 1).matmul(t12[..., None])[..., 0]
    E = E_R.matmul(hat(E_t))
    return E
