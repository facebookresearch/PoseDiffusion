import torch
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def pose_encoding_to_camera(pose_encoding, pose_encoding_type="basic", FL_val=1.8):
    # pose_encoding: BxNxC
    # process an optimization form camera pose to SE3 [R, T]

    batch_size, num_poses, _ = pose_encoding.shape
    pose_encoding_reshaped = pose_encoding.reshape(-1, pose_encoding.shape[-1])  # Reshape to BNxC

    # forced to be 4x4 here
    se3 = torch.zeros(
        batch_size * num_poses, 4, 4, dtype=pose_encoding.dtype, device=pose_encoding.device
    )

    if pose_encoding_type == "absT_quaR_logFL":
        # forced that 3 for absT, 4 for quaR, 2 logFL
        # TODO: converted to 1 dim for logFL, consistent with our paper
        absT = pose_encoding_reshaped[:, :3]
        quaR = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaR)
        se3[:, :3, :3] = R
        se3[:, 3, :3] = absT
        se3[:, 3, 3] = 1.0

        logFL = pose_encoding_reshaped[:, 7:9]
        # FL_val was the hyperparameter to ensure the mean of logFL
        # close to 0 during training
        # Now converted back
        flength = (logFL + FL_val).exp()

        # clamp to avoid weird fl values
        flength = torch.clamp(flength, min=0.1, max=20)
    else:
        raise NotImplementedError(f"More variants to be implemented")

    # Reshape se3 back to BxNx4x4
    se3 = se3.reshape(batch_size, num_poses, 4, 4)
    flength = flength.reshape(batch_size, num_poses, 2)
    return se3, flength
