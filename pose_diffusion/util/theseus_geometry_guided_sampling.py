import torch
from typing import Dict, List, Optional, Union
from util.camera_transform import pose_encoding_to_camera
from util.get_fundamental_matrix import get_fundamental_matrices
from util.get_fundamental_matrix_vmap import get_fundamental_matrices_vmap
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
import theseus as th
from functools import partial

def geometry_guided_sampling_theseus(
    model_mean: torch.Tensor,
    t: int,
    matches_dict: Dict,
    GGS_cfg: Dict,
):
    # pre-process matches
    b, c, h, w = matches_dict["img_shape"]
    device = model_mean.device
    model_mean_tensor_init = model_mean.detach().clone()

    # model_mean = normalize_quaternions(model_mean)

    print("******* Start Tensor *******")
    print(model_mean)

    def _to_device(tensor):
        return torch.from_numpy(tensor).to(device)

    kp1 = _to_device(matches_dict["kp1"])
    kp2 = _to_device(matches_dict["kp2"])
    i12 = _to_device(matches_dict["i12"])

    pair_idx = i12[:, 0] * b + i12[:, 1]
    pair_idx = pair_idx.long()
    
    def _to_homogeneous(tensor):
        return torch.nn.functional.pad(tensor, [0, 1], value=1)

    kp1_homo = _to_homogeneous(kp1)
    kp2_homo = _to_homogeneous(kp2)


    i1, i2 = [
        i.reshape(-1) for i in torch.meshgrid(torch.arange(b), torch.arange(b))
    ]

    # h = torch.full((1, 1), h)
    # w = torch.full((1, 1), w)
    
    processed_matches = {
        "kp1_homo": kp1_homo.unsqueeze(0),
        "kp2_homo": kp2_homo.unsqueeze(0),
        "i1": i1.unsqueeze(0),
        "i2": i2.unsqueeze(0),
        # "h": h,
        # "w": w,
        "pair_idx": pair_idx.unsqueeze(0),
    }
    
    # to fit theseus dtype
    for key in processed_matches: processed_matches[key] = processed_matches[key].double()
    
    # from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
    # model_mean[:,:,3:7].norm(dim=-1)
    
    # quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    model_mean = normalize_quaternions(model_mean)

    model_mean = model_mean.reshape(1, -1)
    model_mean = model_mean.double()
    model_mean_tensor = model_mean.detach().clone()
    
    # conduct theseus GGS
    theseus_dict = {}
    for key in processed_matches:
        theseus_dict[key] = th.Variable(processed_matches[key], key)
        
    model_mean_init = th.Vector(tensor=model_mean.detach().clone(),name= "model_mean_init")
    
    
    model_mean = th.Vector(tensor=model_mean, name = "model_mean")
    w1 = th.ScaleCostWeight(th.Variable(1 * torch.ones(1, 1).double().to(device), name="scale_w1"))

    
    theseus_dict["model_mean_init"] = model_mean_init
    
    optim_vars = [model_mean]
    aux_vars = theseus_dict.values()
    
    
    # cost_function = th.AutoDiffCostFunction(
    #     optim_vars, theseus_error_fn, 1, 
    #     aux_vars=aux_vars, cost_weight=w1, name="theseus_error_fn", autograd_mode="dense"
    # )
    
    # cost_function = th.AutoDiffCostFunction(
    #     optim_vars, theseus_error_fn, 1, 
    #     aux_vars=aux_vars, cost_weight=w1, name="theseus_error_fn", autograd_mode="loop_batch",
    # )
    

    partial_fn = partial(theseus_error_fn, update_FL=False)
    partial_fn_withFL = partial(theseus_error_fn, update_FL=True)
    
    cost_function = th.AutoDiffCostFunction(
        optim_vars, partial_fn, 1, 
        aux_vars=aux_vars, cost_weight=w1, name="theseus_error_fn", #autograd_mode="loop_batch"
    )
    
    cost_function_withFL = th.AutoDiffCostFunction(
        optim_vars, partial_fn_withFL, 1, 
        aux_vars=aux_vars, cost_weight=w1, name="theseus_error_fn", #autograd_mode="loop_batch"
    )
    
    
    
    ########### regularization term
    # reg_func = False
    
    # if reg_func:
    #     reg_w2 = th.ScaleCostWeight(th.Variable(0.001 * torch.ones(1, 1).double().to(device), name="reg_w2"))

    #     reg_function = th.AutoDiffCostFunction(
    #         [model_mean], reg_error_fn, 1, 
    #         aux_vars=[model_mean_init], cost_weight=reg_w2, name="reg_error_fn", #autograd_mode="loop_batch"
    #     )
    #     objective.add(reg_function)
    ########### regularization term
    iter_num = 1000

    with_FL = False
    if with_FL:
        objective_withFL = th.Objective(dtype=torch.float64)
        objective_withFL.add(cost_function_withFL)
        optimizer_withFL = th.LevenbergMarquardt(
            objective_withFL,
            max_iterations=iter_num,
            step_size=0.1,
        )
        theseus_optim_withFL = th.TheseusLayer(optimizer_withFL)

        theseus_inputs = {**processed_matches,
            "model_mean_init": model_mean_tensor, 
            "model_mean": model_mean_tensor,
            }
        
        with torch.no_grad():
            updated_inputs, info = theseus_optim_withFL.forward(
                theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})
        model_mean_tensor = info.best_solution['model_mean']
        model_mean_tensor = model_mean_tensor.to(device)

    objective = th.Objective(dtype=torch.float64)
    objective.add(cost_function)
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=iter_num,
        step_size=0.1,
    )
    theseus_optim = th.TheseusLayer(optimizer)

    theseus_inputs = {**processed_matches,
        "model_mean_init": model_mean_tensor, 
        "model_mean": model_mean_tensor,
        }
    
    with torch.no_grad():
        updated_inputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})
    
    
    
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Time taken: {:.4f} seconds".format(elapsed_time/1000))
    # if reg_func:
    #     print(f"Unweighted Reg error: {reg_function.error().mean()}\n")
    # print(f"Unweighted Cost error: {cost_function.error().mean()}\n")
    # print(f"Objective value: {objective.error_metric().mean()}")


    # print("Best solution:", info.best_solution)  
    model_mean = info.best_solution['model_mean']
    # model_mean = info.updated_inputs['model_mean']
        
    model_mean = model_mean.reshape(model_mean_tensor_init.shape).to(device).float()
    model_mean = normalize_quaternions(model_mean)
  
    print("******* End Tensor *******")
    print(model_mean)
    
    return model_mean
    # return updated_inputs['model_mean']

def theseus_error_fn(optim_vars, aux_vars, update_R=True, update_T=True, update_FL=True):
    # import pdb;pdb.set_trace()

    # kp1_homo, kp2_homo, i1, i2, h, w, pair_idx, model_mean_init = aux_vars
    kp1_homo, kp2_homo, i1, i2, pair_idx, model_mean_init = aux_vars
    model_mean = optim_vars[0]
    
    processed_matches = {
        "kp1_homo": kp1_homo.tensor,
        "kp2_homo": kp2_homo.tensor,
        "i1": i1.tensor.long()[0],
        "i2": i2.tensor.long()[0],
        # "h": h.tensor.long(),
        # "w": w.tensor.long(),
        "pair_idx": pair_idx.tensor.long()[0],
    }

    valid_sampson = compute_sampson_distance(
        model_mean.tensor,
        model_mean_init.tensor,
        processed_matches,
        update_R=update_R,
        update_T=update_T,
        update_FL=update_FL,
        sampson_max=10,
        # sampson_max=-1,
    )
    
    return valid_sampson
    
def reg_error_fn(optim_vars, aux_vars):
    loss = torch.abs(optim_vars[0].tensor - aux_vars[0].tensor)
    loss = loss.sum(dim=1,keepdim=True)
    # if is_tensor(loss): print("reg_term:", loss)
    return loss


def compute_sampson_distance(
    model_mean,
    model_mean_init,
    processed_matches: Dict,
    update_R=True,
    update_T=True,
    update_FL=True,
    pose_encoding_type: str = "absT_quaR_logFL",
    sampson_max: int = 10,
):
    if pose_encoding_type == 'absT_quaR_logFL':
        dims = 9
        

    model_mean = model_mean.reshape(-1, 9)

    
    camera = pose_encoding_to_camera(model_mean, pose_encoding_type, return_dict=True)
    
    camera_init = pose_encoding_to_camera(model_mean_init, pose_encoding_type, return_dict=True)
    
    camera_init["focal_length"] = camera_init["focal_length"].mean(dim=0).repeat(len(camera_init["focal_length"]), 1)

    camera["focal_length"] = camera["focal_length"].mean(dim=0).repeat(len(camera["focal_length"]), 1)

    if not update_FL:
        camera["focal_length"] = camera_init["focal_length"]

    # kp1_homo, kp2_homo, i1, i2, he, wi, pair_idx = processed_matches.values()
    kp1_homo, kp2_homo, i1, i2, pair_idx = processed_matches.values()
    kp1_homo = kp1_homo[0]
    kp2_homo = kp2_homo[0]
    
    F_2_to_1 = get_fundamental_matrices_vmap(
        camera, i1, i2, l2_normalize_F=False
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
        kp1_homo,
        kp2_homo,
        pair_idx,
    )
    # sampson_to_print = sampson.detach().clone().clamp(max=sampson_max).mean(dim=0,keepdim=True)
    # sampson = sampson[sampson < sampson_max]
    
    # indices = torch.where(sampson < sampson_max)
    # sampson_filtered = sampson[indices]
    
    # mask = sampson < sampson_max
    # sampson_filtered = torch.masked_select(sampson, mask)


    sampson = sampson.reshape(-1, 1)
    
    clamp_flag = True
    
    if clamp_flag:
        sampson = torch.clamp(sampson, max=sampson_max, min=0)
    else:
        if sampson_max>0:
            mask = (sampson <= sampson_max).float()
            sampson = sampson * mask
    
    
    sampson = sampson.mean(dim=0,keepdim=True)
    # if is_tensor(sampson): print("sampson:", sampson)
    # if is_tensor(sampson_to_print): print("sampson_print:", sampson_to_print)

    # mask = sampson < sampson_max
    # sampson = sampson
    # import pdb;pdb.set_trace()
    # print(sampson.detach().cpu().numpy().mean())
    # print(f"mean sampson:{sampson.mean()}")
    return sampson # , sampson_to_print


def is_tensor(obj):
    return isinstance(obj, torch.Tensor)

def normalize_quaternions(model_mean: torch.Tensor) -> torch.Tensor:
    quaternions = model_mean[:,:,3:7]
    normalized_quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    model_mean[:,:,3:7] = normalized_quaternions
    return model_mean
