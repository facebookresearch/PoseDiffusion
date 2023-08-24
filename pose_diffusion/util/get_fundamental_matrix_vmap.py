import torch
import pytorch3d
# from pytorch3d.utils import opencv_from_cameras_projection
# from pytorch3d.transforms.so3 import hat
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


def get_fundamental_matrices_vmap(
    camera,
    index1: torch.LongTensor,
    index2: torch.LongTensor,
    height: int = 224,
    width: int = 224,
    l2_normalize_F=False,
):
    """Compute fundamental matrices for given camera parameters."""
    # batch_size = camera.R.shape[0]

    batch_size = camera["R"].shape[0]

    # Convert to opencv / colmap / Hartley&Zisserman convention
    image_size_t = (
        torch.LongTensor([height, width])[None]
        .repeat(batch_size, 1)
        .to(camera["R"].device)
    )
        
    R, t, K = _opencv_from_cameras_projection(camera, image_size=image_size_t)

    F, E = get_fundamental_matrix(
        K[index1], R[index1], t[index1], K[index2], R[index2], t[index2]
    )

    if l2_normalize_F:
        F_scale = torch.norm(F, dim=(1, 2))
        F_scale = F_scale.clamp(min=0.0001)
        F = F / F_scale[:, None, None]

    return F



def _opencv_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
):
    
    R_pytorch3d = cameras["R"].clone()  # pyre-ignore
    T_pytorch3d = cameras["T"].clone()  # pyre-ignore
    focal_pytorch3d = cameras["focal_length"].clone()
    # p0_pytorch3d = cameras.principal_point # all zeros[N, 2]
    p0_pytorch3d = 0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]

    return R, tvec, camera_matrix

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


def hat(v: torch.Tensor) -> torch.Tensor:
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    x, y, z = v.unbind(1)

    h_01 = -z.view(N, 1, 1)
    h_02 = y.view(N, 1, 1)
    h_10 = z.view(N, 1, 1)
    h_12 = -x.view(N, 1, 1)
    h_20 = -y.view(N, 1, 1)
    h_21 = x.view(N, 1, 1)

    zeros = torch.zeros((N, 1, 1), dtype=v.dtype, device=v.device)
    
    row1 = torch.cat((zeros, h_01, h_02), dim=2)
    row2 = torch.cat((h_10, zeros, h_12), dim=2)
    row3 = torch.cat((h_20, h_21, zeros), dim=2)

    h = torch.cat((row1, row2, row3), dim=1)

    return h

# loop_batch

# def hat(v: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the Hat operator [1] of a batch of 3D vectors.

#     Args:
#         v: Batch of vectors of shape `(minibatch , 3)`.

#     Returns:
#         Batch of skew-symmetric matrices of shape
#         `(minibatch, 3 , 3)` where each matrix is of the form:
#             `[    0  -v_z   v_y ]
#              [  v_z     0  -v_x ]
#              [ -v_y   v_x     0 ]`

#     Raises:
#         ValueError if `v` is of incorrect shape.

#     [1] https://en.wikipedia.org/wiki/Hat_operator
#     """

#     N, dim = v.shape
#     if dim != 3:
#         raise ValueError("Input vectors have to be 3-dimensional.")

#     h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

#     x, y, z = v.unbind(1)

#     h[:, 0, 1] = -z
#     h[:, 0, 2] = y
#     h[:, 1, 0] = z
#     h[:, 1, 2] = -x
#     h[:, 2, 0] = -y
#     h[:, 2, 1] = x

#     return h