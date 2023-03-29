# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import torch
from typing import List, Optional

from collections import defaultdict
import numpy as np


from PIL import Image
import torchvision.transforms as transforms
from pytorch3d.transforms import (
    se3_exp_map,
    se3_log_map,
    Transform3d,
    so3_relative_angle,
)
import pickle
import os
from relpose_utils import (
    TRAINING_CATEGORIES,
    TEST_CATEGORIES,
    get_permutations,
    compute_angular_error_batch,
)
from copy import deepcopy
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.ops import corresponding_cameras_alignment


def align_7dof(rot_gt, tvec_gt, rot_pred, tvec_pred, align_to_gt_pose):
    use_pytorch3d = True

    if use_pytorch3d:
        camera_gt = PerspectiveCameras(
            R=torch.from_numpy(rot_gt).float(),
            T=torch.from_numpy(tvec_gt).float(),
        )

        camera_pred = PerspectiveCameras(
            R=torch.from_numpy(rot_pred).float(),
            T=torch.from_numpy(tvec_pred).float(),
        )

        cameras_pred_aligned = corresponding_cameras_alignment(
            cameras_src=camera_pred,
            cameras_tgt=camera_gt,
            estimate_scale=True,
            mode="centers",
            eps=1e-4,
        )
        rot_pred = cameras_pred_aligned.R.numpy()
        tvec_pred = cameras_pred_aligned.T.numpy()
    else:
        xyz_pred = tvec_pred.transpose(1, 0)
        xyz_gt = tvec_gt.transpose(1, 0)
        rr, tt, optimized_scale = umeyama_alignment(
            xyz_pred, xyz_gt, align_to_gt_pose != "6dof"
        )

        align_transformation = np.eye(4)
        align_transformation[:3:, :3] = rr
        align_transformation[:3, 3] = tt
        inter_poses = np.tile(np.eye(4), (len(tvec_pred), 1, 1))
        inter_poses[:, :3:, :3] = rot_pred
        inter_poses[:, :3, 3] = tvec_pred * optimized_scale

        align_transformation = np.tile(align_transformation, (len(tvec_pred), 1, 1))
        new_inter_poses = np.einsum("Bij,Bjk ->Bik", align_transformation, inter_poses)
        rot_pred = new_inter_poses[:, :3:, :3]
        tvec_pred = new_inter_poses[:, :3, 3]
    return rot_pred, tvec_pred


def compute_RT_metric(
    rotations_gt,
    tvec_gt,
    fl_gt,
    rotations_pred,
    tvec_pred,
    fl_pred,
    permutations,
    category,
    sequence_name,
    gt_camera,
    align_to_gt_pose,
):
    # TODO: clean this
    # rotations_gt, rotations_pred: (N, 3, 3)
    # tvec_gt, tvec_pred: (N, 3)
    # permutations: (N*N-1, 2), inherited from RelPose
    # align_to_gt_pose in "scale", "6dof" or "7dof"

    # Relative R for GT
    R_gt_batched = rotations_gt[permutations]
    R_gt_rel = np.einsum(
        "Bij,Bjk ->Bik",
        R_gt_batched[:, 0].transpose(0, 2, 1),
        R_gt_batched[:, 1],
    )

    # Relative T for GT
    T_gt_batched = tvec_gt[permutations]
    tmp = np.einsum(
        "Bij,Bjk ->Bik",
        R_gt_rel.transpose(0, 2, 1),
        T_gt_batched[:, 0][..., None],
    )
    T_gt_rel = -tmp[..., 0] + T_gt_batched[:, 1]

    if align_to_gt_pose == "scale":
        optimized_scale = scale_lse_solver(tvec_pred, tvec_gt)
        tvec_pred = tvec_pred * optimized_scale
    elif align_to_gt_pose == "7dof" or align_to_gt_pose == "6dof":
        rotations_pred, tvec_pred = align_7dof(
            rotations_gt, tvec_gt, rotations_pred, tvec_pred, align_to_gt_pose
        )
    else:
        raise NotImplementedError

    # rot_pred_aligned_first, _ = align_rt_to_first(rotations_pred, tvec_pred)

    # rot_gt_aligned_first, _ = align_rt_to_first(rotations_gt, tvec_gt)

    # rotations_gt.std()
    # 0.5723779
    # import pdb;pdb.set_trace()

    R_err_abs = compute_angular_error_batch(rotations_pred, rotations_gt)
    # avoid flip
    min_array = np.minimum(R_err_abs, np.abs(180 - R_err_abs))
    R_err_abs = min_array

    # R_err_abs = min(R_err_abs, 180)

    # import pdb;pdb.set_trace()

    center_dis = compute_camera_center_dis(
        rotations_gt, tvec_gt, rotations_pred, tvec_pred, gt_camera
    )

    ### ATE
    ate, ate_david = compute_ATE(
        rotations_gt, tvec_gt, rotations_pred, tvec_pred, align_to_gt_pose
    )

    # Relative R for Pred
    R_pred_batched = rotations_pred[permutations]
    R_pred_rel = np.einsum(
        "Bij,Bjk ->Bik",
        R_pred_batched[:, 0].transpose(0, 2, 1),
        R_pred_batched[:, 1],
    )

    # Relative T for Pred
    T_pred_batched = tvec_pred[permutations]
    T_pred_rel_part = np.einsum(
        "Bij,Bjk ->Bik",
        R_pred_rel.transpose(0, 2, 1),
        T_pred_batched[:, 0][..., None],
    )

    T_pred_rel = -T_pred_rel_part[..., 0] + T_pred_batched[:, 1]

    # Compute angular error for R and T
    R_errors_angular = compute_angular_error_batch(R_pred_rel, R_gt_rel)
    # T_error_angular, _ = evaluate_t_batch(T_gt_rel, T_pred_rel)

    rpe_trans, rpe_trans_david, rpe_rot, T_error_angular, mnorm_T_gt_rel = compute_RPE(
        rotations_gt,
        tvec_gt,
        rotations_pred,
        tvec_pred,
        permutations,
    )

    T_error_angular = T_error_angular * 180.0 / np.pi
    print(
        f"******************************     For category {category} scene {sequence_name}  ******************************"
    )

    metric_dict = {}

    rpe_trans_ratio = rpe_trans / mnorm_T_gt_rel * 100
    rpe_trans_ratio_david = rpe_trans_david / mnorm_T_gt_rel * 100

    metric_dict["rot_angular"] = R_errors_angular.tolist()
    metric_dict["rot_angular_abs"] = R_err_abs.tolist()
    metric_dict["tvec_angular"] = T_error_angular.tolist()

    metric_dict["rpe_trans"] = rpe_trans
    metric_dict["rpe_trans_david"] = rpe_trans_david

    metric_dict["rpe_tratio"] = rpe_trans_ratio
    metric_dict["rpe_tratio_david"] = rpe_trans_ratio_david
    metric_dict["rpe_rot"] = rpe_rot

    metric_dict["ate"] = ate
    metric_dict["ate_david"] = ate_david

    # metric_dict["T_gt_rel_norm"]  = T_gt_rel_norm.tolist()

    metric_dict["center_dis"] = center_dis

    if len(fl_pred.shape) < 2:
        try:
            fl_pred = fl_pred.cpu().numpy()
        except:
            fl_pred = fl_pred
        fl_dis = np.mean(np.abs(fl_pred[:, None] - fl_gt))
    else:
        fl_dis = np.mean(np.abs(fl_pred - fl_gt))

    metric_dict["fl_dis"] = fl_dis

    for key in metric_dict:
        mean_result = np.mean(metric_dict[key])
        print(f"Sequence mean {key.ljust(22)} is {mean_result:.6f}")

    return metric_dict


def compute_RPE(rot_gt, tvec_gt, rot_pred, tvec_pred, permutations):
    gt_poses = np.tile(np.eye(4), (len(tvec_gt), 1, 1))
    pred_poses = np.tile(np.eye(4), (len(tvec_gt), 1, 1))

    gt_poses[:, :3, :3] = rot_gt
    gt_poses[:, 3, :3] = tvec_gt

    pred_poses[:, :3, :3] = rot_pred
    pred_poses[:, 3, :3] = tvec_pred

    gt_poses_batch = gt_poses[permutations]
    pred_poses_batch = pred_poses[permutations]

    gt_poses_batch = torch.from_numpy(gt_poses_batch)
    pred_poses_batch = torch.from_numpy(pred_poses_batch)

    gt_poses_rel = closed_form_inverse_torch(gt_poses_batch[:, 0]).bmm(
        gt_poses_batch[:, 1]
    )

    pred_poses_rel = closed_form_inverse_torch(pred_poses_batch[:, 0]).bmm(
        pred_poses_batch[:, 1]
    )

    rel_err = closed_form_inverse_torch(gt_poses_rel).bmm(pred_poses_rel)

    rel_err_numpy = rel_err.numpy()
    rot_error_batch = rotation_error_batch(rel_err_numpy)
    rot_error_batch = rot_error_batch * 180 / np.pi
    rel_t_error = np.linalg.norm(rel_err[:, 3, :3], axis=1)

    rel_R = rel_err_numpy[:, :3, :3]
    gt_poses_rel_numpy = gt_poses_rel.numpy()
    pred_poses_rel_numpy = pred_poses_rel.numpy()

    part1 = np.einsum(
        "Bij,Bjk ->Bik",
        rel_R.transpose(0, 2, 1),
        gt_poses_rel_numpy[:, 3, :3][..., None],
    )
    part1 = part1[..., 0]
    part2 = pred_poses_rel_numpy[:, 3, :3]

    optimized_scale = scale_lse_solver(part2, part1)
    T_rel_David = -part1 + part2 * optimized_scale
    T_rel_David = np.linalg.norm(T_rel_David, axis=1)

    T_angular, _ = evaluate_t_batch(
        gt_poses_rel_numpy[:, 3, :3], pred_poses_rel_numpy[:, 3, :3]
    )
    T_gt_rel_norm = np.linalg.norm(gt_poses_rel_numpy[:, 3, :3], axis=1)
    T_gt_rel_norm = np.mean(T_gt_rel_norm)

    return rel_t_error, T_rel_David, rot_error_batch, T_angular, T_gt_rel_norm


def fetch_color_jit(color_aug_num):
    if color_aug_num == 0:
        color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.65,
                ),
                transforms.RandomGrayscale(p=0.15),
            ]
        )
    elif color_aug_num == 1:
        color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.65,
                ),
                transforms.RandomGrayscale(p=0.15),
            ]
        )
    elif color_aug_num == 2:
        color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                        )
                    ],
                    p=0.65,
                ),
                transforms.RandomGrayscale(p=0.15),
            ]
        )
    elif color_aug_num == 3:
        color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                        )
                    ],
                    p=0.65,
                ),
            ]
        )
    elif color_aug_num == 4:
        color_jitter = transforms.Compose(
            [
                transforms.RandomGrayscale(p=0.15),
            ]
        )
    elif color_aug_num == 5:
        color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.65,
                ),
            ]
        )
    elif color_aug_num == 6:
        color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
    else:
        raise NotImplementedError

    return color_jitter


def batch_by_sequence_name(
    sequence_name: List[str],
    x: torch.Tensor,
    seq_to_batch_idx=None,
    fetch_seq_name_by_batch=False,
):
    if (seq_to_batch_idx is None) or fetch_seq_name_by_batch:
        seq_to_batch_idx = defaultdict(list)
        batched_seq_name = []
        for i, seq_name in enumerate(sequence_name):
            seq_to_batch_idx[seq_name].append(i)
            if seq_name not in batched_seq_name:
                batched_seq_name.append(seq_name)

    mask_tensor = torch.ones_like(x[:, :1, ...])
    x_and_mask = torch.cat([x, mask_tensor], dim=1)
    x_and_mask_pad = torch.nn.utils.rnn.pad_sequence(
        [x_and_mask[idx] for idx in seq_to_batch_idx.values()],
        batch_first=True,
        padding_value=0.0,
    )

    x_pad, mask_pad = x_and_mask_pad.split([x.shape[1], 1], dim=2)
    if fetch_seq_name_by_batch:
        return x_pad, mask_pad, seq_to_batch_idx, batched_seq_name
    return x_pad, mask_pad, seq_to_batch_idx


def read_dict_by_list(target_dict, input_list, keyname=None):
    output_list = []
    for name in input_list:
        if name in target_dict:
            value = target_dict[name]
            if keyname is not None:
                value = value[keyname]
            output_list.append(value)
        else:
            output_list.append(None)
            print(f"{name} not found in target_dict")

    return output_list


def compute_camera_center_dis(
    rotations_gt, tvec_gt, rotations_pred, tvec_pred, gt_camera
):
    # align to the canonical system to help compute the camera centers
    # force to be measured in a canonical way

    # rotations_pred, tvec_pred = align_rt_to_first(rotations_pred, tvec_pred)
    # rotations_gt, tvec_gt= align_rt_to_first(rotations_gt, tvec_gt)

    # optimized_scale = scale_lse_solver(tvec_pred, tvec_gt)
    # tvec_pred = tvec_pred * optimized_scale

    gt_centers = -np.einsum(
        "Bij,Bjk ->Bik",
        rotations_gt,
        tvec_gt[..., None],
    )
    gt_centers = gt_centers[..., 0]

    pred_centers = -np.einsum(
        "Bij,Bjk ->Bik",
        rotations_pred,
        tvec_pred[..., None],
    )
    pred_centers = pred_centers[..., 0]

    center_dis = np.linalg.norm(gt_centers - pred_centers, axis=1)

    return center_dis


def repeat_batch(x, repeat_num):
    if x is None:
        return x
    tensor_shape = x.shape

    if len(tensor_shape) == 1:
        x = x.repeat(repeat_num)
    elif len(tensor_shape) == 2:
        x = x.repeat(repeat_num, 1)
    elif len(tensor_shape) == 3:
        x = x.repeat(repeat_num, 1, 1)
    elif len(tensor_shape) == 4:
        x = x.repeat(repeat_num, 1, 1, 1)
    else:
        import pdb

        pdb.set_trace()
    return x


def huber_loss(diff, delta=1.0, return_mean=True, use_sqrt=False):
    """
    Computes the Huber loss.
    See https://en.wikipedia.org/wiki/Huber_loss for more details.
    """
    assert not isinstance(delta, torch.Tensor)
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = (abs_diff - quadratic).abs()
    loss = 0.5 * quadratic**2 + delta * linear
    if use_sqrt:
        if len(loss) > 0:
            loss = loss.sqrt()

    if return_mean:
        return loss.mean()
    else:
        return loss


def Racc_func(rot_gt, rot_pred, masks_flat, batch_size=-1, per_scene_acc=True):
    # rot_gt, rot_pred (B, 3, 3)
    # masks_flat: B, 1
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-2)
    rel_angle_deg = rel_angle_cos * 180 / np.pi

    if per_scene_acc and batch_size > 0:
        mask_per_scene = masks_flat.reshape(batch_size, -1)
        mask_per_scene = mask_per_scene.mean(dim=-1) > 0

        R_acc_15 = rel_angle_deg.reshape(batch_size, -1) < 15.0
        R_acc_5 = rel_angle_deg.reshape(batch_size, -1) < 5.0

        R_acc_15 = R_acc_15.float().mean(dim=-1)[mask_per_scene]
        R_acc_5 = R_acc_5.float().mean(dim=-1)[mask_per_scene]
    else:
        R_acc_15 = (rel_angle_deg < 15.0)[masks_flat.squeeze(-1) == 1]
        R_acc_5 = (rel_angle_deg < 30.0)[masks_flat.squeeze(-1) == 1]

    return R_acc_15.float().mean(), R_acc_5.float().mean()


def Tacc_func(tvec_gt, tvec_pred, masks_flat, batch_size=-1, per_scene_acc=True):
    pick_idx = masks_flat.squeeze(-1) == 1

    tvec_dis = tvec_gt - tvec_pred
    tvec_dis_norm = torch.norm(tvec_dis, dim=-1)

    # Is translation vector also affected by rotation? Yes

    T_dis = (tvec_dis_norm[pick_idx]).mean()

    optimized_scale = scale_lse_solver(tvec_pred[pick_idx], tvec_gt[pick_idx])

    tvec_dis = tvec_gt - tvec_pred * optimized_scale
    tvec_dis_norm = torch.norm(tvec_dis, dim=-1)

    T_dis_align = (tvec_dis_norm[pick_idx]).mean()

    # T_acc_20 = (tvec_dis_norm<0.2)[pick_idx]
    T_angular = evaluate_t_batch_torch(tvec_gt, tvec_pred)
    T_angular = T_angular * 180.0 / np.pi

    T_acc_5 = (T_angular < 5)[pick_idx]
    return T_dis, T_dis_align, T_acc_5.float().mean()


def evaluate_t(t_gt, t):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(t_gt, t)
        import IPython

        IPython.embed()

    return err_t


def evaluate_t_batch_torch(t_gt, t):
    eps = 1e-15
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    # inf_check = t_norm[:,0] == 0
    # err_t[inf_check] = 1e6
    default_err = 1e6
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def evaluate_t_batch(t_gt, t):
    eps = 1e-15
    # normalize to 1
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / (t_norm + eps)

    t_gt_norm = np.linalg.norm(t_gt, axis=1, keepdims=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt, axis=1) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    inf_check = t_norm[:, 0] == 0

    # Give it a very large value if the predicted is inf
    # To real with ColMAP failure

    err_t[inf_check] = 1e6
    if len(err_t[inf_check]) > 0:
        print("Filtering " * 10)

    return err_t, t_gt_norm


def align_rt_to_first(rotations_pred, tvec_pred):
    assert len(rotations_pred) == len(tvec_pred)
    bs = len(rotations_pred)
    poses = np.tile(np.eye(4), (bs, 1, 1))
    poses[:, :3, :3] = rotations_pred
    poses[:, 3, :3] = tvec_pred

    first_pose_inv = np.tile(np.linalg.inv(poses[0]), (bs, 1, 1))

    rel_poses = np.einsum(
        "Bij,Bjk ->Bik",
        first_pose_inv,
        poses,
    )

    alignedR = rel_poses[:, :3, :3]
    alignedT = rel_poses[:, 3, :3]

    return alignedR, alignedT


def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    if torch.is_tensor(X):
        with torch.no_grad():
            scale = torch.sum(X * Y) / (torch.sum(X**2) + 1e-15)
    else:
        scale = np.sum(X * Y) / (np.sum(X**2) + 1e-15)

    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """

    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    if np.trace(np.diag(d).dot(s)) == 0:
        c = 1
    else:
        c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0

    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def dump_pickle(file_path, my_dict):
    with open(file_path, "wb") as f:
        pickle.dump(my_dict, f)
    return True


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def compute_ATE(rot_gt, tvec_gt, rot_pred, tvec_pred, align_to_gt_pose):
    """Compute RMSE of ATE
    Args:
        gt (4x4 array dict): ground-truth poses
        pred (4x4 array dict): predicted poses
    """
    # if align_to_gt_pose!="7dof":
    #     rot_pred, tvec_pred = align_7dof(rot_gt, tvec_gt, rot_pred, tvec_pred, align_to_gt_pose)

    ate_david = np.linalg.norm(tvec_gt - tvec_pred, axis=-1).mean()
    # ate_david = np.linalg.norm(tvec_gt-tvec_pred, axis=-1)

    scene_scale = np.linalg.norm(tvec_gt, axis=-1).mean()

    # KITTI ate
    errors = []

    for i in range(len(tvec_pred)):
        gt_xyz = tvec_gt[i]
        pred_xyz = tvec_pred[i]
        align_err = gt_xyz - pred_xyz
        errors.append(np.sqrt(np.sum(align_err**2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))

    return ate, ate_david


"""
def compute_RPE(rot_gt, tvec_gt, rot_pred, tvec_pred, permutations, david_style = False):
    # https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    
    trans_errors = []
    rot_errors = []
    
    pred_rel_list = []
    gt_rel_list = []
    
    trans_dis = []

            
            # rel_t = closed_form_inverse(gt_se3_batch[:,0,:,:])
            # rel_t = rel_t.unsqueeze(1).expand(-1, gt_se3_batch.shape[1], -1, -1)
            # rel_t = rel_t.reshape(-1, 4,4)

            # gt_se3_rel = torch.bmm(rel_t, gt_se3_batch.reshape(-1,4,4))
            # gt_se3_rel[..., :3, 3] = 0.0
            # gt_se3_rel[..., 3, 3] = 1.0
    import pdb;pdb.set_trace()
    # for i in range(len(permutations)):
    #     idx1, idx2 = permutations[i]
        
    #     pred_pose1 = rt_to_pose(rot_pred[idx1], tvec_pred[idx1])
    #     pred_pose2 = rt_to_pose(rot_pred[idx2], tvec_pred[idx2])
    #     pred_pose_rel = np.linalg.inv(pred_pose1) @ pred_pose2
    #     gt_pose1 = rt_to_pose(rot_gt[idx1], tvec_gt[idx1])
    #     gt_pose2 = rt_to_pose(rot_gt[idx2], tvec_gt[idx2])
    #     gt_pose_rel = np.linalg.inv(gt_pose1) @ gt_pose2

    #     # import pdb;pdb.set_trace()
                
    #     # trans_dis.append(np.linalg.norm(pred_pose_rel[:3, 3] - gt_pose_rel[:3, 3]))
        
        
    #     rel_err = np.linalg.inv(gt_pose_rel) @ pred_pose_rel
    #     trans_errors.append(translation_error(rel_err))
    #     rot_errors.append(rotation_error(rel_err))


    #     pred_rel_list.append(pred_pose_rel)
    #     gt_rel_list.append(gt_pose_rel)

    rpe_trans = np.asarray(trans_errors)
    rpe_rot = np.asarray(rot_errors)
    rpe_rot = rpe_rot * 180 / np.pi
    
    ################################################################## Compute mean?
    # rpe_trans = np.mean(np.asarray(trans_errors))
    # rpe_rot = np.mean(np.asarray(rot_errors))
    ##################################################################
    
    if david_style:
        pred_rel_list = np.array(pred_rel_list)
        gt_rel_list = np.array(gt_rel_list)
        
        pred_trans_all = pred_rel_list[...,:3, 3]
        gt_trans_all = gt_rel_list[...,:3, 3]
        optimized_scale = scale_lse_solver(pred_trans_all, gt_trans_all)
        
        scaled_pred_T = pred_rel_list[...,:3, 3] * optimized_scale
        new_diff = gt_rel_list[...,:3, 3] - scaled_pred_T
        new_rpe_trans = np.linalg.norm(new_diff, axis=1)
        
        # import pdb;pdb.set_trace()
        return new_rpe_trans, rpe_rot
    
    return rpe_trans, rpe_rot
"""


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def rotation_error_batch(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[:, 0, 0]
    b = pose_error[:, 1, 1]
    c = pose_error[:, 2, 2]
    d = 0.5 * (a + b + c - 1.0)
    d_clipped = np.clip(d, -1, 1)
    rot_error = np.arccos(d_clipped)
    return rot_error


def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def rt_to_pose(rr, tt):
    pose = np.eye(4)
    pose[:3, :3] = rr
    pose[:3, 3] = tt
    return pose


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        # set exist_ok as True in case multiprocess conflict
        os.makedirs(folder_path, exist_ok=True)


def get_DEFAULT_STATS():
    terms = [
        "sec/it",
        "objective",
        "epoch",
        "rpe_trans",
        "rpe_tratio",
        "rpe_rot",
        "rpe_trans_david",
        "rpe_tratio_david",
        "rpe_tratio15",
        "rpe_tratio15_david",
        "center_dis",
        "ate",
        "ate_david",
        "ate_david_acc1",
        "Rcc5",
        "Rcc15",
        "Rcc30",
        "Tcc5",
        "Tcc15",
        "Tcc30",
        "R_ang_mean",
        "T_ang_mean",
        "loss_diffusion",
        "loss_camera_se3",
        "loss_T_distance",
        "loss_R_distance",
        "loss_center",
        "loss_v3_T",
        "loss_v3_R",
        "loss_v3_inv_T",
        "loss_v3_inv_R",
        "R_acc_15_rel",
        "R_acc_5_rel",
        "T_acc_20_rel",
        "T_acc_5_rel",
        "dis_fl",
        "dis_pp",
        "T_dis",
        "T_dis_align",
        "T_mean",
        "T_std",
    ]

    DEFAULT_STATS = []
    for name in terms:
        DEFAULT_STATS.append("train/" + name)
        DEFAULT_STATS.append("val/" + name)
    return DEFAULT_STATS


DEFAULT_STATS = get_DEFAULT_STATS()


def nonzero_pad(tvec_pred):
    pred = deepcopy(tvec_pred)
    # Find indices of non-zero rows
    nonzero_rows = np.nonzero(np.any(pred != 0, axis=1))[0]

    # Create mask of zero rows
    mask = pred == 0

    # Find indices of zero rows
    zero_row_indices = np.argwhere(mask)

    # Loop over zero row indices
    for i in zero_row_indices:
        i = i[0]  # Get index from array

        # Find indices of first left and right non-zero rows
        left_idx = (
            nonzero_rows[nonzero_rows < i][-1]
            if (nonzero_rows[nonzero_rows < i].size > 0)
            else i
        )
        right_idx = (
            nonzero_rows[nonzero_rows > i][0]
            if (nonzero_rows[nonzero_rows > i].size > 0)
            else i
        )

        # Compute interpolated values
        x_left = left_idx + 1
        x_right = right_idx + 1
        y_left = pred[left_idx]
        y_right = pred[right_idx]
        x_interp = i + 1
        y_interp = (x_interp - x_left) / (x_right - x_left) * (
            y_right - y_left
        ) + y_left

        # Assign interpolated values to zero row
        pred[i] = y_interp

    return pred


def proces_kp_for_GGS(cond_dict):
    # kp1_hom_good, kp2_hom_good, i1, i2, he, wi, pick_idx = proces_kp_for_GGS(cond_dict)

    image_rgb_batch = cond_dict["image_rgb_batch"]

    _, ba, dim, he, wi = image_rgb_batch.shape

    i1, i2 = [i.reshape(-1) for i in torch.meshgrid(torch.arange(ba), torch.arange(ba))]

    matches_dict = cond_dict["matches_dict"]

    if not matches_dict["gt_matches"]:
        device = image_rgb_batch.device
        kp1 = matches_dict["kp1"]
        kp1 = torch.from_numpy(kp1).to(device)
        kp2 = matches_dict["kp2"]
        kp2 = torch.from_numpy(kp2).to(device)

        i12 = matches_dict["i12"]
        i12 = torch.from_numpy(i12).to(device)

        # i12_raw = torch.cat((i1[:,None],i2[:,None]),dim=-1)

        pick_idx = i12[:, 0] * ba + i12[:, 1]

        kp1_hom = torch.nn.functional.pad(kp1, [0, 1], value=1)
        kp2_hom = torch.nn.functional.pad(kp2, [0, 1], value=1)

        # now don't care about good matches
        kp1_hom_good = kp1_hom
        kp2_hom_good = kp2_hom

    else:
        print("Using GT matches")
        max_pairs = -1
        sel = torch.arange(len(i1))
        sel = sel[i1 != i2]
        if max_pairs > 0 and (len(sel) > max_pairs):
            sel = torch.randperm(len(sel))[:max_pairs]

        i1, i2 = i1[sel], i2[sel]
        kp1 = matches_dict["kp1"][sel]
        kp1_hom = torch.nn.functional.pad(kp1, [0, 1], value=1)

        kp2 = matches_dict["kp2"][sel]
        kp2_hom = torch.nn.functional.pad(kp2, [0, 1], value=1)

        i12 = matches_dict["i12"][sel]
        pts3d = matches_dict["pts3d"][sel]

        good_matches = matches_dict["good_matches"][sel]

        pick_idx = torch.arange(len(i12))[:, None]
        pick_idx = pick_idx.expand(-1, good_matches.shape[-1])
        pick_idx = pick_idx[good_matches]

        kp1_hom_good = kp1_hom[good_matches]
        kp2_hom_good = kp2_hom[good_matches]

    return kp1_hom_good, kp2_hom_good, i1, i2, he, wi, pick_idx


def closed_form_inverse_torch(se3):
    # se3:    Nx4x4
    # return: Nx4x4
    # inverse each 4x4 matrix
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]
    R_trans = R.transpose(1, 2)

    left_down = -T.bmm(R_trans)
    left = torch.cat((R_trans, left_down), dim=1)
    right = se3[:, :, 3:].detach().clone()
    inversed = torch.cat((left, right), dim=-1)
    return inversed


def pytorch3d_camera_to_canoical(camera):
    # No batch here
    camera_pred_canonical = camera.clone()
    se3 = camera_pred_canonical.get_world_to_view_transform().get_matrix()
    rel_t = closed_form_inverse_torch(se3[0][None])
    rel_t = rel_t.expand(len(se3), -1, -1)

    se3_rel = torch.bmm(rel_t, se3)
    se3_rel[..., :3, 3] = 0.0
    se3_rel[..., 3, 3] = 1.0

    # Cameras are moved to a canonical coordinate now
    camera_pred_canonical.R = se3_rel[:, :3, :3].clone()
    camera_pred_canonical.T = se3_rel[:, 3, :3].clone()

    return camera_pred_canonical


def opencv_np_to_pytorch3d(rot, tvec):
    rot = torch.from_numpy(rot)
    tvec = torch.from_numpy(tvec)
    # rot_raw = rot.clone()
    # tvec_raw = tvec.clone()

    R_pytorch3d = rot.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    # import pdb;pdb.set_trace()

    return R_pytorch3d.numpy(), T_pytorch3d.numpy()


def load_image_PIL(path) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im
