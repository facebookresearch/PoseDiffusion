# https://math.stackexchange.com/a/3334866
# from numpy import array, cross
# from numpy.linalg import solve, norm

# # define lines A and B by two points
# XA0 = array([1, 0, 0])
# XA1 = array([1, 1, 1])
# XB0 = array([0, 0, 0])
# XB1 = array([0, 0, 1])

# # compute unit vectors of directions of lines A and B
# UA = (XA1 - XA0) / norm(XA1 - XA0)
# UB = (XB1 - XB0) / norm(XB1 - XB0)
# # find unit direction vector for line C, which is perpendicular to lines A and B
# UC = cross(UB, UA); UC /= norm(UC)

# # solve the system derived in user2255770's answer from StackExchange: https://math.stackexchange.com/q/1993990
# RHS = XB0 - XA0
# LHS = array([UA, -UB, UC]).T
# print(solve(LHS, RHS))
# # prints "[ 0. -0.  1.]"

# Also a useful resource:
# https://en.wikipedia.org/wiki/Skew_lines

import torch


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect = intersect_skew_lines_high_dim(p, r, mask=mask)
    _, p_line_intersect = _point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(
        dim=-1
    )
    return p_intersect, p_line_intersect, intersect_dist_squared


def intersect_skew_lines(p1, r1, p2, r2):
    # p1, r1: starting point and unit heading vector of the first line
    # p2, r2: starting point and unit heading vector of the second line
    # all of shape [B, 3]

    p_intersect = intersect_skew_lines_high_dim(
        torch.stack([p1, p2], dim=-2),
        torch.stack([r1, r2], dim=-2),
    )

    _, p1_intersect = _point_line_distance(p1, r1, p_intersect)
    _, p2_intersect = _point_line_distance(p2, r2, p_intersect)

    return p_intersect, p1_intersect, p2_intersect

    # specific solution for dim=3 ...
    r_orth = torch.nn.functional.normalize(torch.cross(r1, r2, dim=-1), dim=-1)
    rhs = p2 - p1
    lhs = torch.stack([r1, -r2, r_orth], dim=-2)
    t = _batch_lstsq(A=lhs, input=rhs)
    t1, t2, t3 = t.unbind(dim=-2)

    p1_intersect = p1 + t1 * r1
    p2_intersect = p2 + t2 * r2
    p_intersect = (p1_intersect + p2_intersect) * 0.5

    return p_intersect


def intersect_skew_lines_high_dim(p, r, eps=1e-6, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)
    I = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (I - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)
    I_eps = torch.eye(dim, device=p.device, dtype=p.dtype)[None] * eps
    p_intersect = torch.pinverse(I_min_cov.sum(dim=-3) + I_eps).matmul(sum_proj)[..., 0]
    return p_intersect


def _point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 + proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def _batch_lstsq(
    input: torch.Tensor,  # matrix B of shape (batch * m * k)
    A: torch.Tensor,  # matrix A of shape (batch * m * n)
):
    X = torch.bmm(torch.pinverse(A), input[..., None])
    return X