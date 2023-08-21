import logging

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    quaternion_to_matrix,
)


# A should be GT, B should be predicted
def compute_optimal_alignment(A, B):
    """
    Compute the optimal scale s, rotation R, and translation t that minimizes:
    || A - (s * B @ R + T) || ^ 2

    Reference: Umeyama (TPAMI 91)

    Args:
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        s (float): scale.
        R (torch.Tensor): rotation matrix (3, 3).
        t (torch.Tensor): translation (3,).
    """
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    # normally with R @ B, this would be A @ B.T
    H = (B - B_bar).T @ (A - A_bar)
    U, S, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    variance = torch.sum((B - B_bar) ** 2)
    scale = 1 / variance * torch.trace(torch.diag(S) @ S_prime)
    R = U @ S_prime @ Vh
    t = A_bar - scale * B_bar @ R

    A_hat = scale * B @ R + t
    return A_hat, scale, R, t


def compute_optimal_translation_alignment(T_A, T_B, R_B):
    """
    Assuming right-multiplied rotation matrices.

    E.g., for world2cam R and T, a world coordinate is transformed to camera coordinate
    system using X_cam = X_world.T @ R + T = R.T @ X_world + T

    Finds s, t that minimizes || T_A - (s * T_B + R_B.T @ t) ||^2

    Args:
        T_A (torch.Tensor): Target translation (N, 3).
        T_B (torch.Tensor): Initial translation (N, 3).
        R_B (torch.Tensor): Initial rotation (N, 3, 3).

    Returns:
        T_A_hat (torch.Tensor): s * T_B + t @ R_B (N, 3).
        scale s (torch.Tensor): (1,).
        translation t (torch.Tensor): (1, 3).
    """
    n = len(T_A)

    T_A = T_A.unsqueeze(2)
    T_B = T_B.unsqueeze(2)

    A = torch.sum(T_B * T_A)
    B = (T_B.transpose(1, 2) @ R_B.transpose(1, 2)).sum(0) @ (R_B @ T_A).sum(0) / n
    C = torch.sum(T_B * T_B)
    D = (T_B.transpose(1, 2) @ R_B.transpose(1, 2)).sum(0)
    E = (D * D).sum() / n

    s = (A - B.sum()) / (C - E.sum())

    t = (R_B @ (T_A - s * T_B)).sum(0) / n

    T_A_hat = s * T_B + R_B.transpose(1, 2) @ t

    return T_A_hat.squeeze(2), s, t.transpose(1, 0)


def compute_optimal_scaling(A, B):
    """
    Compute the optimal scale s and translation t that minimizes:
    || A - (s * B + T) || ^ 2

    Args:
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        s (float): scale.
        t (torch.Tensor): translation (3,).
    """
    raise DeprecationWarning("Use compute_optimal_translation_alignment instead.")
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    n = len(A)
    numer = (B * A).sum() - n * (A_bar * B_bar).sum()
    denom = (B**2).sum() - n * (B_bar**2).sum()
    s = numer / denom
    t = A_bar - s * B_bar
    return s, t


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def generate_random_rotations(n=1, device="cpu"):
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    return quaternion_to_matrix(quats)


def generate_noisy_rotation(n=1, eps=0, degrees=True, device="cpu"):
    """
    Generates a random rotation matrix with Gaussian-distributed magnitude.

    Args:
        n: Number of rotations to generate.
        eps: Standard deviation of the magnitude of the rotation.
        degrees: If True, eps is in degrees. If False, eps is in radians.

    Returns:
        tensor: rotation matrices (N, 3, 3).
    """
    if degrees:
        eps *= np.pi / 180
    magnitude = eps * torch.randn(n, 1, device=device)
    axis = torch.randn(n, 3, device=device)
    axis /= axis.norm(dim=1, keepdim=True)
    return axis_angle_to_matrix(magnitude * axis)


def generate_equivolumetric_grid(recursion_level=3):
    """
    Generates an equivolumetric grid on SO(3).

    Uses a Healpix grid on S2 and then tiles 6 * 2 ** recursion level over 2pi.

    Code adapted from https://github.com/google-research/google-research/blob/master/
        implicit_pdf/models.py

    Grid sizes:
        1: 576
        2: 4608
        3: 36864
        4: 294912
        5: 2359296
        n: 72 * 8 ** n

    Args:
        recursion_level: The recursion level of the Healpix grid.

    Returns:
        tensor: rotation matrices (N, 3, 3).
    """
    import healpy

    log = logging.getLogger("healpy")
    log.setLevel(logging.ERROR)  # Supress healpy linking warnings.

    number_per_side = 2**recursion_level
    number_pix = healpy.nside2npix(number_per_side)
    s2_points = healpy.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = torch.tensor(np.stack([*s2_points], 1))

    azimuths = torch.atan2(s2_points[:, 1], s2_points[:, 0])
    # torch doesn't have endpoint=False for linspace yet.
    tilts = torch.tensor(
        np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)
    )
    polars = torch.arccos(s2_points[:, 2])
    grid_rots_mats = []
    for tilt in tilts:
        rot_mats = euler_angles_to_matrix(
            torch.stack(
                [azimuths, torch.zeros(number_pix), torch.zeros(number_pix)], 1
            ),
            "XYZ",
        )
        rot_mats = rot_mats @ euler_angles_to_matrix(
            torch.stack([torch.zeros(number_pix), torch.zeros(number_pix), polars], 1),
            "XYZ",
        )
        rot_mats = rot_mats @ euler_angles_to_matrix(
            torch.tensor([[tilt, 0.0, 0.0]]), "XYZ"
        )
        grid_rots_mats.append(rot_mats)

    grid_rots_mats = torch.cat(grid_rots_mats, 0)
    return grid_rots_mats.float()
