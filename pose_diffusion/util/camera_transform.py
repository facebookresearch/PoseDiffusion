import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
import numpy as np


def bbox_xyxy_to_xywh(xyxy):
    wh = xyxy[2:] - xyxy[:2]
    xywh = np.concatenate([xyxy[:2], wh])
    return xywh


def adjust_camera_to_bbox_crop_(fl, pp, image_size_wh: torch.Tensor, clamp_bbox_xywh: torch.Tensor):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, image_size_wh)
    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px, principal_point_px_cropped, clamp_bbox_xywh[2:]
    )

    return focal_length, principal_point_cropped


def adjust_camera_to_image_scale_(fl, pp, original_size_wh: torch.Tensor, new_size_wh: torch.LongTensor):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, original_size_wh)

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.float()
    scale = (image_size_wh_output / original_size_wh).min(dim=-1, keepdim=True).values
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled, principal_point_px_scaled, image_size_wh_output
    )
    return focal_length_scaled, principal_point_scaled


def _convert_ndc_to_pixels(focal_length: torch.Tensor, principal_point: torch.Tensor, image_size_wh: torch.Tensor):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: torch.Tensor, principal_point_px: torch.Tensor, image_size_wh: torch.Tensor
):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR_logFL",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=20,
    return_dict=False,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
                        only "absT_quaR_logFL" is supported.
    """

    pose_encoding_reshaped = pose_encoding.reshape(-1, pose_encoding.shape[-1])  # Reshape to BNxC

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
        focal_length = torch.clamp(focal_length, min=min_focal_length, max=max_focal_length)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if return_dict:
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    pred_cameras = PerspectiveCameras(focal_length=focal_length, R=R, T=abs_T, device=R.device)
    return pred_cameras


def camera_to_pose_encoding(
    camera, pose_encoding_type="absT_quaR_logFL", log_focal_length_bias=1.8, min_focal_length=0.1, max_focal_length=20
):
    """ """

    if pose_encoding_type == "absT_quaR_logFL":
        # Convert rotation matrix to quaternion
        quaternion_R = matrix_to_quaternion(camera.R)

        # Calculate log_focal_length
        log_focal_length = (
            torch.log(torch.clamp(camera.focal_length, min=min_focal_length, max=max_focal_length))
            - log_focal_length_bias
        )

        # Concatenate to form pose_encoding
        pose_encoding = torch.cat([camera.T, quaternion_R, log_focal_length], dim=-1)

    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding
