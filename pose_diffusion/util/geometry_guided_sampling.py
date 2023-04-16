import torch
from typing import Dict, List, Optional, Union
from util.camera_transform import pose_encoding_to_camera
from util.get_fundamental_matrix import get_fundamental_matrices
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


def geometry_guided_sampling(
    model_mean: torch.Tensor,
    t: int,
    matches_dict=None,
):
    # pre-process matches
    b, c, h, w = matches_dict["img_shape"]
    device = model_mean.device

    def _to_device(tensor):
        return torch.from_numpy(tensor).to(device)

    kp1 = _to_device(matches_dict["kp1"])
    kp2 = _to_device(matches_dict["kp2"])
    i12 = _to_device(matches_dict["i12"])

    pair_idx = i12[:, 0] * b + i12[:, 1]

    def _to_homogeneous(tensor):
        return torch.nn.functional.pad(tensor, [0, 1], value=1)

    kp1_homo = _to_homogeneous(kp1)
    kp2_homo = _to_homogeneous(kp2)

    i1, i2 = [
        i.reshape(-1) for i in torch.meshgrid(torch.arange(b), torch.arange(b))
    ]

    processed_matches = {
        "kp1_homo": kp1_homo,
        "kp2_homo": kp2_homo,
        "i1": i1,
        "i2": i2,
        "h": h,
        "w": w,
        "pair_idx": pair_idx,
    }

    # GGS
    GGS_cfg = matches_dict["GGS_cfg"]

    model_mean = GGS_optimize(model_mean, t, processed_matches, GGS_cfg)

    model_mean = GGS_optimize(
        model_mean,
        t,
        processed_matches,
        GGS_cfg,
        update_T=False,
        update_R=False,
    )  # only optimize FL
    model_mean = GGS_optimize(
        model_mean,
        t,
        processed_matches,
        GGS_cfg,
        update_T=False,
        update_FL=False,
    )  # only optimize R
    model_mean = GGS_optimize(
        model_mean,
        t,
        processed_matches,
        GGS_cfg,
        update_R=False,
        update_FL=False,
    )  # only optimize T

    model_mean = GGS_optimize(model_mean, t, processed_matches, GGS_cfg)
    return model_mean


def GGS_optimize(
    model_mean: torch.Tensor,
    t: int,
    processed_matches: Dict,
    GGS_cfg: Optional[Dict] = None,
    update_R: bool = True,
    update_T: bool = True,
    update_FL: bool = True,
):
    with torch.enable_grad():
        model_mean.requires_grad_(True)

        learning_rate = GGS_cfg.lr
        iter_num = GGS_cfg.iter_num

        if update_R and update_T and update_FL:
            iter_num = iter_num * 2

        optimizer = torch.optim.SGD(
            [model_mean], lr=learning_rate, momentum=0.9
        )
        batch_size = model_mean.shape[1]

        for _ in range(iter_num):
            valid_sampson, sampson_to_print = compute_sampson_distance(
                model_mean,
                t,
                processed_matches,
                GGS_cfg=GGS_cfg,
                update_R=update_R,
                update_T=update_T,
                update_FL=update_FL,
            )

            if GGS_cfg.min_matches > 0:
                valid_match_per_frame = len(valid_sampson) / batch_size
                if valid_match_per_frame < GGS_cfg.min_matches:
                    print(
                        "Drop this pair because of insufficient valid matches"
                    )
                    break

            loss = valid_sampson.mean()
            optimizer.zero_grad()
            loss.backward()

            grads = model_mean.grad
            grad_norm = grads.norm()
            grad_mask = (grads.abs() > 0).detach()
            model_mean_norm = (model_mean * grad_mask).norm()

            max_norm = GGS_cfg.alpha * model_mean_norm / learning_rate

            total_norm = torch.nn.utils.clip_grad_norm_(model_mean, max_norm)
            optimizer.step()

        print(f"t={t:02d} | sampson={sampson_to_print:05f}")
        model_mean = model_mean.detach()
    return model_mean


def compute_sampson_distance(
    model_mean: torch.Tensor,
    t: int,
    processed_matches: Dict,
    GGS_cfg: Dict,
    update_R=True,
    update_T=True,
    update_FL=True,
):
    camera = pose_encoding_to_camera(model_mean, GGS_cfg.pose_encoding_type)

    # pick the mean of the predicted focal length
    camera.focal_length = camera.focal_length.mean(dim=0).repeat(
        len(camera.focal_length), 1
    )

    if not update_R:
        camera.R = camera.R.detach()

    if not update_T:
        camera.T = camera.T.detach()

    if not update_FL:
        camera.focal_length = camera.focal_length.detach()

    kp1_homo, kp2_homo, i1, i2, he, wi, pair_idx = processed_matches.values()
    F_2_to_1 = get_fundamental_matrices(
        camera, he, wi, i1, i2, normalize_to_one=False
    )
    F = F_2_to_1.permute(0, 2, 1)  # y1^T F y2 = 0

    def _sampson_distance(F, kp1_homo, kp2_homo, pair_idx):
        left = torch.bmm(kp1_homo[:, None], F[pair_idx])
        right = torch.bmm(F[pair_idx], kp2_homo[..., None])

        bottom = (
            left[:, :, 0].square()
            + left[:, :, 1].square()
            + right[:, 0, :].square()
            + right[:, 1, :].square()
        )
        top = torch.bmm(left, kp2_homo[..., None]).square()

        sampson = top[:, 0] / bottom
        return sampson

    sampson = _sampson_distance(
        F,
        kp1_homo.float(),
        kp2_homo.float(),
        pair_idx,
    )

    sampson_to_print = (
        sampson.detach().clone().clamp(max=GGS_cfg.sampson_max).mean()
    )
    sampson = sampson[sampson < GGS_cfg.sampson_max]

    return sampson, sampson_to_print
