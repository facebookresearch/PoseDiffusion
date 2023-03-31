import torch
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def optform_to_rt(optform, optform_type="basic"):
    # optform: BxNxC
    # process an optimization form camera pose to SE3 [R, T]
    # TODO: enable focal length 

    batch_size, num_poses, _ = optform.shape
    optform_reshaped = optform.reshape(-1, optform.shape[-1])  # Reshape to BNxC

    # forced to be 4x4 here
    se3 = torch.zeros(batch_size * num_poses, 4, 4, dtype=optform.dtype, device=optform.device)

    if optform_type == "absT_quaR_logFL":
        # force to 
        absT = optform_reshaped[:, :3]
        quaR = optform_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaR)
        se3[:, :3, :3] = R
        se3[:, 3, :3] = absT
        se3[:, 3, 3] = 1.0
    else:
        raise NotImplementedError(f"More variants to be implemented")
    
    # Reshape se3 back to BxNx4x4
    se3 = se3.reshape(batch_size, num_poses, 4, 4)
    return se3
