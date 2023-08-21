import random
import numpy as np
import torch


def compute_Rerror(rotation1, rotation2):
    if isinstance(rotation1, torch.Tensor):
        rotation1 = rotation1.cpu().detach().numpy()
    if isinstance(rotation2, torch.Tensor):
        rotation2 = rotation2.cpu().detach().numpy() 
    
    num_frames = len(rotation1)
    permutations = get_permutations(num_frames)
    ARE = compute_ARE(rotation1, rotation2)
    RRE = compute_RRE(rotation1, rotation2, permutations)
    
    return ARE, RRE

# def compute_ARE(rotation1, rotation2):
#     R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
#     t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
#     theta = np.arccos(np.clip(t, -1, 1))
#     error = theta * 180 / np.pi
#     return np.minimum(error, np.abs(180 - error))


def compute_ARE(rotation1, rotation2):
    if isinstance(rotation1, torch.Tensor):
        rotation1 = rotation1.cpu().detach().numpy()
    if isinstance(rotation2, torch.Tensor):
        rotation2 = rotation2.cpu().detach().numpy()

    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    error = theta * 180 / np.pi
    return np.minimum(error, np.abs(180 - error))


def compute_RRE(rotation1, rotation2, permutations):
    R1_by_permutation = rotation1[permutations]
    R2_by_permutation = rotation2[permutations]

    R1_rel = np.einsum(
        "Bij,Bjk ->Bik",
        R1_by_permutation[:, 0].transpose(0, 2, 1),
        R1_by_permutation[:, 1],
    )
    R2_rel = np.einsum(
        "Bij,Bjk ->Bik",
        R2_by_permutation[:, 0].transpose(0, 2, 1),
        R2_by_permutation[:, 1],
    )
    
    R_rel = np.einsum("Bij,Bjk ->Bik", R1_rel.transpose(0, 2, 1), R2_rel)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


def get_permutations(num_images):
    indices = np.arange(num_images)
    i, j = np.meshgrid(indices, indices)
    
    mask = i != j
    permutations = np.vstack((i[mask], j[mask])).T
    return permutations