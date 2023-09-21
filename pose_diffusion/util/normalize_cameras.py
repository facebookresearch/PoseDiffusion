"""
Adapted from code originally written by David Novotny.
"""
import torch
from pytorch3d.transforms import Rotate, Translate


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    _, p_line_intersect = _point_line_distance(p, r, p_intersect[..., None, :].expand_as(p))
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(dim=-1)
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        raise ValueError(f"p_intersect is NaN")

    return p_intersect, r


def _point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def compute_optical_axis_intersection(cameras):
    centers = cameras.get_camera_center()
    principal_points = cameras.principal_point

    one_vec = torch.ones((len(cameras), 1))
    optical_axis = torch.cat((principal_points, one_vec), -1)

    pp = cameras.unproject_points(optical_axis, from_ndc=True, world_coordinates=True)

    # pp0 = torch.zeros((pp.shape[0], 3))
    # for i in range(0, pp.shape[0]): pp0[i] = pp[i][i]
    # pp2 = pp.diagonal(dim1=0, dim2=1)

    pp2 = pp[torch.arange(pp.shape[0]), torch.arange(pp.shape[0])]

    directions = pp2 - centers
    centers = centers.unsqueeze(0).unsqueeze(0)
    directions = directions.unsqueeze(0).unsqueeze(0)

    p_intersect, p_line_intersect, _, r = intersect_skew_line_groups(p=centers, r=directions, mask=None)

    p_intersect = p_intersect.squeeze().unsqueeze(0)
    dist = (p_intersect - centers).norm(dim=-1)

    return p_intersect, dist, p_line_intersect, pp2, r


def normalize_cameras(cameras, compute_optical=True, first_camera=True, scale=1.0):
    """
    Normalizes cameras such that the optical axes point to the origin and the average
    distance to the origin is 1.

    Args:
        cameras (List[camera]).
    """
    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()

    if compute_optical:
        new_transform = new_cameras.get_world_to_view_transform()

        (p_intersect, dist, p_line_intersect, pp, r) = compute_optical_axis_intersection(cameras)
        t = Translate(p_intersect)

        scale = dist.squeeze()[0]

        # Degenerate case
        if scale == 0:
            scale = torch.norm(new_cameras.T, dim=(0, 1))
            scale = torch.sqrt(scale)
            new_cameras.T = new_cameras.T / scale
        else:
            new_matrix = t.compose(new_transform).get_matrix()
            new_cameras.R = new_matrix[:, :3, :3]
            new_cameras.T = new_matrix[:, 3, :3] / scale
    else:
        scale = torch.norm(new_cameras.T, dim=(0, 1))
        scale = torch.sqrt(scale)
        new_cameras.T = new_cameras.T / scale

    if first_camera:
        new_cameras = first_camera_transform(new_cameras)

    return new_cameras


def first_camera_transform(cameras, rotation_only=False):
    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()
    tR = Rotate(new_cameras.R[0].unsqueeze(0))
    if rotation_only:
        t = tR.inverse()
    else:
        tT = Translate(new_cameras.T[0].unsqueeze(0))
        t = tR.compose(tT).inverse()

    new_matrix = t.compose(new_transform).get_matrix()

    new_cameras.R = new_matrix[:, :3, :3]
    new_cameras.T = new_matrix[:, 3, :3]

    return new_cameras
