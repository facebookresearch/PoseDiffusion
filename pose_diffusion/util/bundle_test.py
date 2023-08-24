import logging
import os
import pathlib
import random
import time
from typing import Dict, List, Type
import argparse

import hydra
import numpy as np
import omegaconf
import torch

import theseus as th
import theseus.utils.examples as theg
import pickle
from theseus.utils import Profiler, Timer
from ba_util import *
from utils import seed_all_random_engines
from torchlie.functional import SE3 as SE3_base

THRES = 2

log = logging.getLogger(__name__)


def proj_point_err_all(optim_vars, aux_vars):
    
    pose_all, point_all, fl_all = optim_vars 
    point2d_all, camera_idx_all, point_idx_all  =  aux_vars 
    

    point_all = point_all.tensor.reshape(-1,3)
    pose_all = pose_all.tensor.reshape(-1,3,4)
    fl_all = fl_all.tensor.reshape(-1,1)
    point2d_all = point2d_all.tensor
    camera_idx_all = camera_idx_all.tensor.long()
    point_idx_all = point_idx_all.tensor.long()





    cameras_compute = pose_all[camera_idx_all[:,0]]
    fl_compute = fl_all[camera_idx_all[:,0]]
    point_compute = point_all[point_idx_all[:,0]]

    point_cam = cameras_compute[...,-1:] + cameras_compute[...,:3] @ point_compute.unsqueeze(-1)
    
    # point_cam -  SE3_base.transform(cameras_compute, point_compute)
    import pdb;pdb.set_trace()
    proj = -point_cam[:, :2] / point_cam[:, 2:3]
    proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
    proj_factor = fl_compute.unsqueeze(1) * (1.0 + proj_sqn * (0 + proj_sqn * 0))

    point_projection = proj * proj_factor

    err = point_projection[...,0] - point2d_all
    return err


def get_batch(
    ba: theg.BundleAdjustmentDataset,
    orig_poses: Dict[str, torch.Tensor],
    orig_points: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    retv = {}
    for cam in ba.cameras:
        retv[cam[0].name] = orig_poses[cam[0].name].clone()
            
    for i, pt in ba.points.items():
        retv[pt.name] = orig_points[pt.name].clone()
    return retv

def run_ba(ba_path, cfg_path):
    with open(ba_path, 'rb') as file: ba = pickle.load(file)
    with open(cfg_path, 'rb') as file: cfg = pickle.load(file)
    sampled_obs =  random.sample(ba.observations, 10000)
    
    cfg.ba.linear_solver_cls = "LUCudaSparseSolver"
    
    device = ba.cameras[0][0].device
        
    onetensor=torch.ones(1,1).to(dtype=ba.cameras[0][0].dtype).to(device)
    log_loss_radius = th.Vector(tensor=(-cfg.ba.log_radius * onetensor), 
                                name="log_loss_radius", dtype=torch.float64)

    profiler = Profiler(cfg.ba_profile)

    objective = th.Objective(dtype=torch.float64)
    objective.to(device)

    weight = th.ScaleCostWeight(torch.tensor(1.0, device=device).to(dtype=ba.cameras[0][0].dtype))
    dtype = objective.dtype


    auto_diff = True
    
    
    if auto_diff:
        pose_tensors, fl_tensors, calib_k1_tensors, calib_k2_tensors = zip(*[(camera[0].tensor, camera[1].tensor, camera[2].tensor, camera[3].tensor) for camera in ba.cameras])
        pose_all = torch.cat(pose_tensors).to(device)
        fl_all = torch.cat(fl_tensors).to(device)
        calib_k1_all = torch.cat(calib_k1_tensors).to(device)
        calib_k2_all = torch.cat(calib_k2_tensors).to(device)
        
        # point_tensors = [pt.tensor for pt in ba.points]
        # for pt in ba.points: print(pt.tensor)
        # point_tensors = []
        # for track_idx in ba.points:
        #     point_tensors.append(ba.points[track_idx].tensor)

        index_mapping = {}
        current_index = 0

        point_tensors = []
        for key, tensor in ba.points.items():
            tensor_batch_size = tensor.shape[0]  # batch size is typically the first dimension
            point_tensors.append(tensor.tensor)
            index_mapping[key] = (current_index)
            current_index += tensor_batch_size
        point_all = torch.cat(point_tensors).to(device)
        
        camera_idx_all = []
        point_idx_all = []
        point2d_tensors = []
        for obs in sampled_obs:
            camera_idx_all.append(obs.camera_index)
            point_idx_all.append(index_mapping[obs.point_index])
            point2d_tensors.append(obs.image_feature_point.tensor)
            
        
        camera_idx_all = torch.from_numpy(np.array(camera_idx_all)).to(device)
        point_idx_all = torch.from_numpy(np.array(point_idx_all)).to(device)
        point2d_tensors = torch.cat(point2d_tensors)
        
        point_all_feed = point_all.reshape(1,-1)
        point_all = th.Vector(tensor=point_all_feed,name=f"point_all")
        fl_all_feed = fl_all.reshape(1, -1)
        fl_all = th.Vector(tensor=fl_all_feed,name=f"fl_all")
        pose_all_feed = pose_all.reshape(1, -1)
        pose_all = th.Vector(tensor=pose_all_feed,name=f"pose_all")
        
        # calib_k1_all = th.Vector(tensor=calib_k1_all.double(),name=f"calib_k1_all",)
        # calib_k2_all = th.Vector(tensor=calib_k2_all.double(),name=f"calib_k2_all",)
        camera_idx_all_feed = camera_idx_all.reshape(-1,1).double()
        point_idx_all_feed = point_idx_all.reshape(-1,1).double()
        camera_idx_all = th.Vector(tensor=camera_idx_all_feed,name=f"camera_idx_all",)
        point_idx_all = th.Vector(tensor=point_idx_all_feed,name=f"point_idx_all",)
        point2d_all = th.Vector(tensor=point2d_tensors.double(),name=f"point2d_all",)
        
        cost_function = ReprojectionBatch(camera_pose=pose_all,   world_point=point_all,
                            focal_length=fl_all,    image_feature_point=point2d_all,
                            camera_idx = camera_idx_all, point_idx = point_idx_all,
                            weight=weight)
        # cost_function.error()

        # optim_vars = [pose_all, point_all, fl_all]
        # aux_vars = [point2d_all, calib_k1_all, calib_k2_all, camera_idx_all, point_idx_all]
        # aux_vars = [point2d_all, camera_idx_all, point_idx_all]

        # cost_function = th.AutoDiffCostFunction(
        #         optim_vars, proj_point_err_all, 2, 
        #         aux_vars=aux_vars, cost_weight=weight,)
            
        # cost_function = th.RobustCostFunction(
        #         cost_function,
        #         th.HuberLoss,
        #         log_loss_radius,
        #         name=f"robust_{cost_function.name}",
        #     )
        # objective.cost_functions
        objective.add(cost_function)
    else:
        for obs in sampled_obs:
            pose, fl, calib_k1, calib_k2 = ba.cameras[obs.camera_index]
            point3d = ba.points[obs.point_index]
            point2d = obs.image_feature_point
            
            pose.to(device)
            fl.to(device)
            point3d.to(device)
            point2d.to(device)
            calib_k1.to(device)
            calib_k2.to(device)

            # cost_function = th.eb.Reprojection(camera_pose=pose,   world_point=ba.points[obs.point_index],
            #                                     focal_length=fl,    image_feature_point=obs.image_feature_point,
            #                                     calib_k1 = calib_k1, calib_k2 = calib_k2,
            #                                     weight=weight)

            cost_function = ReprojectionTH(camera_pose=pose,   world_point=ba.points[obs.point_index],
                                                focal_length=fl,    image_feature_point=obs.image_feature_point,
                                                calib_k1 = calib_k1, calib_k2 = calib_k2,
                                                weight=weight)

            # ReprojectionTH
            # ReprojectionBatch
            
            cost_function = th.RobustCostFunction(
                    cost_function,
                    th.HuberLoss,
                    log_loss_radius,
                    name=f"robust_{cost_function.name}",
                )
            objective.add(cost_function)


    # if cfg.ba.regularize:
    #     pose_prior_cost = th.Difference(
    #         var=ba.cameras[0][0],
    #         target=ba.cameras[0][0].copy(new_name=ba.cameras[0][0].name + "__PRIOR"),
    #         cost_weight=th.ScaleCostWeight(
    #             torch.tensor(cfg.ba.reg_w, dtype=dtype).to(device)
    #         ),
    #     )
    #     objective.add(pose_prior_cost)

    # Create optimizer
    optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
        th, cfg.ba.optimizer_cls
    )

    optimizer = optimizer_cls(
        objective,
        max_iterations=cfg.ba.max_iters,
        step_size=cfg.ba.step_size,
        linear_solver_cls=getattr(th, cfg.ba.linear_solver_cls),
    )

    # Set up Theseus layer
    theseus_optim = th.TheseusLayer(
        optimizer,
        vectorize=cfg.ba.vectorize,
        empty_cuda_cache=cfg.ba.empty_cuda_cache,
    )
    theseus_optim.to(device=device)

    orig_poses = {cam[0].name: cam[0].tensor.clone() for cam in ba.cameras}
    orig_points = {point.name: point.tensor.clone() for i, point in ba.points.items()}

    theseus_inputs = get_batch(ba, orig_poses, orig_points)
    
    timer = Timer(device)

    with timer:
        profiler.enable()
        
        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs={**cfg.ba.optimizer_kwargs},
        )
            
        profiler.disable()

    forward_time = 1000 * timer.elapsed_time
    print(f"Forward pass took {forward_time} ms.")

    profiler.print()

    return theseus_outputs, info


if __name__ == "__main__":
    seed_all_random_engines(0)
    run_ba('../ba.pkl', '../cfg.pkl')
    