# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR_logFL",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=20,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
                        only "absT_quaR_logFL" is supported.
    """

    batch_size, num_poses, _ = pose_encoding.shape
    pose_encoding_reshaped = pose_encoding.reshape(
        -1, pose_encoding.shape[-1]
    )  # Reshape to BNxC

    if pose_encoding_type == "absT_quaR_logFL":
        # forced that 3 for absT, 4 for quaR, 2 logFL
        # TODO: converted to 1 dim for logFL, consistent with our paper
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)

        log_focal_length = pose_encoding_reshaped[:, 7:9]

        # log_focal_length_bias was the hyperparameter
        # to ensure the mean of logFL close to 0 during training
        # Now converted back
        focal_length = (log_focal_length + log_focal_length_bias).exp()

        # clamp to avoid weird fl values
        focal_length = torch.clamp(
            focal_length, min=min_focal_length, max=max_focal_length
        )
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    pred_cameras = PerspectiveCameras(
        focal_length=focal_length,
        R=R,
        T=abs_T,
        device=R.device,
    )

    return pred_cameras
