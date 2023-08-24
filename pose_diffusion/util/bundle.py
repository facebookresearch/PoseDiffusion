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
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.vis.plotly_vis import plot_scene
from theseus.utils.examples.bundle_adjustment.data import Observation
from pytorch3d.structures import Pointclouds
from util.ba_util import *
from theseus.utils import Profiler, Timer

THRES = 2

log = logging.getLogger(__name__)

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

def run_ba(ba, cfg):
    # import pickle
    # with open('ba.pkl', 'wb') as file: pickle.dump(ba, file)
    # with open('cfg.pkl', 'wb') as file: pickle.dump(cfg, file)

    
    # import pdb;pdb.set_trace()
    
    device = ba.cameras[0][0].device
        
    onetensor=torch.ones(1,1).to(dtype=ba.cameras[0][0].dtype).to(device)
    log_loss_radius = th.Vector(tensor=(-cfg.ba.log_radius * onetensor), 
                                name="log_loss_radius", dtype=torch.float64)


    profiler = Profiler(cfg.ba_profile)

    objective = th.Objective(dtype=torch.float64)
    objective.to(device)

    weight = th.ScaleCostWeight(torch.tensor(1.0, device=device).to(dtype=ba.cameras[0][0].dtype))
    dtype = objective.dtype

    for obs in ba.observations:
        # assum calib as 0
        pose, fl, calib_k1, calib_k2 = ba.cameras[obs.camera_index]
        point3d = ba.points[obs.point_index]
        point2d = obs.image_feature_point
        
        node = obs.node
        if node.proj_pt is not None:
            dis = node.proj_pt - node.pt
            if np.linalg.norm(dis) > cfg.ba.max_2d_dis:
                continue
            
            if (node.proj_pt.max() > cfg.ba.max_2d_range) or (node.proj_pt.min() < -cfg.ba.max_2d_range):
                continue

        pose.to(device)
        fl.to(device)
        point3d.to(device)
        point2d.to(device)
        calib_k1.to(device)
        calib_k2.to(device)

        # cost_function = ReprojectionPT3D(camera_pose=pose,   world_point=ba.points[obs.point_index],
        cost_function = ReprojectionTH(camera_pose=pose,   world_point=ba.points[obs.point_index],
                                        focal_length=fl,    image_feature_point=obs.image_feature_point,
                                        calib_k1 = calib_k1, calib_k2 = calib_k2,
                                        weight=weight, device = device)

        cost_function = th.RobustCostFunction(
                cost_function,
                th.HuberLoss,
                log_loss_radius,
                name=f"robust_{cost_function.name}",
            )
        objective.add(cost_function)


    if cfg.ba.regularize:
        pose_prior_cost = th.Difference(
            var=ba.cameras[0][0],
            target=ba.cameras[0][0].copy(new_name=ba.cameras[0][0].name + "__PRIOR"),
            cost_weight=th.ScaleCostWeight(
                torch.tensor(cfg.ba.reg_w, dtype=dtype).to(device)
            ),
        )
        objective.add(pose_prior_cost)

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
        _maybe_reset_cuda_peak_mem()
        profiler.enable()
        
        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs={**cfg.ba.optimizer_kwargs},
        )
            
        profiler.disable()

    forward_time = 1000 * timer.elapsed_time
    forward_mem = _maybe_get_cuda_max_mem_alloc()
    print(f"Forward pass took {forward_time} ms.")
    print(f"Forward pass used {forward_mem} GPU MBs.")

    profiler.print()

    return theseus_outputs, info



def base_proj_points(pose, point3d, fl):
    # P * R + t
    transfer = point3d.tensor.unsqueeze(1) @ (pose.tensor[...,:3])
    transfer = transfer.squeeze(1)  # N, 1, 3
    transfer = transfer + pose.tensor[...,3]
    
    # to 2d point
    tmp = transfer[...,:2] / transfer[...,2:] 
    point_projection = tmp * fl.tensor
    # point_projection = tmp * fl.tensor + principal_point.tensor
    return point_projection

def proj_point_err(optim_vars, aux_vars):
    # pose, point3d = optim_vars
    # point2d, fl = aux_vars
    # 1x3x4
    
    pose, point3d, fl = optim_vars
    point2d = aux_vars[0]
    
    point_projection = base_proj_points(pose, point3d, fl)
    err = point_projection - point2d.tensor
    err = torch.clamp(err, max=THRES, min=-THRES)

    # print("compting")
    if err.shape[0]>1:
        print(torch.norm(err, dim=1).mean())
    return err

def proj_point_wo_pose_opt_err(optim_vars, aux_vars):
    point3d = optim_vars[0]
    pose, point2d, fl = aux_vars
    point_projection = base_proj_points(pose, point3d, fl)
    err = point_projection - point2d.tensor
    err = torch.clamp(err, max=THRES, min=-THRES)
    
    if err.shape[0]>1:
        print(torch.norm(err, dim=1).mean())
    return err



def _maybe_reset_cuda_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _maybe_get_cuda_max_mem_alloc():
    return (
        torch.cuda.max_memory_allocated() / 1048576
        if torch.cuda.is_available()
        else torch.nan
    )