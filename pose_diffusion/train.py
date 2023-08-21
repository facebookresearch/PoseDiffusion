# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_original_cwd
import models
import time
from functools import partial
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.ops import corresponding_cameras_alignment
import json
# from accelerate.utils import set_seed as accelerate_set_seed, PrecisionType

# from util.utils import seed_all_random_engines
from util.match_extraction import extract_match
from util.load_img_folder import load_and_preprocess_images
from util.geometry_guided_sampling import geometry_guided_sampling
from util.metric import compute_ARE
from util.theseus_geometry_guided_sampling import geometry_guided_sampling_theseus
from util.metric import compute_Rerror
from util.triangulation import intersect_skew_line_groups
from multiprocessing import Pool
from functools import partial
import tqdm
import visdom
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.points_alignment import iterative_closest_point, _apply_similarity_transform
import pickle
import theseus as th
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs

from datasets.co3d_v2 import Co3dDataset, TRAINING_CATEGORIES
from util.train_util import *
import psutil

import io
import pstats


import cProfile  
  
# Wrapper for cProfile.Profile for easily make optional, turn on/off and printing
class Profiler:
    def __init__(self, active: bool):
        self.c_profiler = cProfile.Profile()
        self.active = active

    def enable(self):
        if self.active:
            self.c_profiler.enable()

    def disable(self):
        if self.active:
            self.c_profiler.disable()

    def print(self):
        if self.active:
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(self.c_profiler, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
    

def get_thread_count(var_name):
    return os.environ.get(var_name)



def train_fn(cfg: DictConfig):    
    # NOTE carefully double check the instruction from huggingface!
    
    OmegaConf.set_struct(cfg, False)
    
    # Initialize the accelerator
    if "dinov2" in cfg.MODEL.IMAGE_FEATURE_EXTRACTOR.modelname:
        ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(even_batches=False, device_placement=False,kwargs_handlers=[ddp_scaler])
    else:
        accelerator = Accelerator(even_batches=False, device_placement=False)
        
    accelerator.print("Model Config:")
    accelerator.print(OmegaConf.to_yaml(cfg))

    accelerator.print(accelerator.state)

    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark
        
    set_seed_and_print(cfg.seed)
    
    if accelerator.is_main_process:
        viz = vis_utils.get_visdom_connection(
            server="http://10.201.16.195",
            port=int(os.environ.get("VISDOM_PORT", 10088)),
        )


    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  OMP_NUM_THREADS: {get_thread_count('OMP_NUM_THREADS')}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  MKL_NUM_THREADS: {get_thread_count('MKL_NUM_THREADS')}")

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_CPU_BIND: {get_thread_count('SLURM_CPU_BIND')}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_JOB_CPUS_PER_NODE: {get_thread_count('SLURM_JOB_CPUS_PER_NODE')}")
    

    cpu_num = psutil.cpu_count()
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  CPU AVAI: {cpu_num}")


    if cfg.train.pt3d_co3d:
        alldatasets, dataloadermaps = get_datasource(dataset_root="/fsx-repligen/shared/datasets/co3d/", category=TRAINING_CATEGORIES, subset_name="fewview_dev", cfg=cfg)
        dataloader = dataloadermaps.train
        eval_dataloader = dataloadermaps.val
        dataset = alldatasets.train
        eval_dataset = alldatasets.val
    else:
        dataset, eval_dataset = get_co3d_dataset(cfg)

        if cfg.train.num_workers>0:
            persistent_workers = cfg.train.persistent_workers
        else:
            persistent_workers = False

        if cfg.train.dynamic_batch:
            batch_sampler = DynamicBatchSampler(len(dataset), dataset_len=cfg.train.len_train, max_images=cfg.train.max_images)
        else:
            batch_sampler = FixBatchSampler(dataset, dataset_len=cfg.train.len_train, max_images=cfg.train.max_images)
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, 
                                                 num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
                                                 persistent_workers=persistent_workers) # collate_fn


        if cfg.train.dynamic_batch:
            eval_batch_sampler = DynamicBatchSampler(len(eval_dataset), dataset_len=cfg.train.len_eval, max_images=cfg.train.max_images // 2)
        else:
            eval_batch_sampler = FixBatchSampler(eval_dataset, dataset_len=cfg.train.len_eval, max_images=cfg.train.max_images // 2)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_sampler=eval_batch_sampler, 
                                                      num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
                                                      persistent_workers=persistent_workers) # collate_fn

    ########################################################
    # eval_dataset.__getitem__((10, 8))
    
    # def profile_function(dataset):
    #     images_per_seq = list(range(3,20))
    #     for idx in range(100):  # You will need to provide some sample indices
    #         n_per_seq = np.random.choice(images_per_seq)
    #         print(idx)
    #         dataset.__getitem__((idx, n_per_seq))

    # def profile_function(dataset):
    #     for idx in range(10000):  
    #         dataset.__getitem__((idx))

    # profiler = Profiler(active=True)
    # profiler.enable()
    # # Call the function you want to profile
    # profile_function(dataset)
    # # Stop profiling
    # profiler.disable()
    # # Print the profiling results
    # profiler.print()


    # import pdb;pdb.set_trace()
    
    ########################################################

    # to make accelerator happy
    dataloader.batch_sampler.drop_last = True
    dataloader.batch_sampler.sampler = dataloader.batch_sampler

    # to make accelerator happy
    eval_dataloader.batch_sampler.drop_last = True
    eval_dataloader.batch_sampler.sampler = eval_dataloader.batch_sampler
    


    accelerator.print("length of train dataloader is: ", len(dataloader))
    accelerator.print("length of eval dataloader is: ", len(eval_dataloader))
    
    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)
    model = model.to(accelerator.device)


    num_epochs = cfg.train.epochs

    # Define the optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr/10)
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=cfg.train.lr, 
                                                    epochs=num_epochs, steps_per_epoch=len(dataloader))


    # for data in dataloader: 
    #     # dataloader.batch_size
    #     import pdb;pdb.set_trace()
    #     print("m")
    

    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)  
    
    print(f"xxxxxxxxxxxxxxxxxx dataloader has {dataloader.num_workers} num_workers")

    start_epoch = 0

    if cfg.train.resume_ckpt:
        checkpoint = torch.load(cfg.train.resume_ckpt)  # Adjust this path as necessary
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"], strict=True)
            
            if "lr_scheduler_state_dict" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                
            # Restore epoch number (if present)
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
        except:
            model.load_state_dict(checkpoint, strict=True)
        
        accelerator.print(f"Successfully resumed from checkpoint at epoch {start_epoch}")

    
    to_plot = ("loss", "lr", "Racc_5", "Racc_15", "Racc_30","Tacc_5", "Tacc_15", "Tacc_30",)
    
    stats = VizStats(to_plot)
    
    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()

        set_seed_and_print(cfg.seed + epoch)

        # Evaluation        
        # if (epoch!=0) and (epoch%cfg.train.eval_interval ==0):
        if (epoch%cfg.train.eval_interval ==0):
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            _train_or_eval_fn(model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training = False, visualize = False)
            accelerator.print(f"----------Finish the eval at epoch {epoch}----------")
        else:
            accelerator.print(f"----------Skip the eval at epoch {epoch}----------")
            

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        _train_or_eval_fn(model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training = True, visualize = False)
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")
        
        if accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.exp_name}----------")
            stats.update({"lr":lr}, stat_set="train")
            stats.plot_stats(viz=viz, visdom_env=cfg.exp_name)
            accelerator.print(f"----------Done----------")
        
        

        if epoch%cfg.train.ckpt_interval==0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")
            accelerator.save_state(output_dir=ckpt_path)
            # accelerator.load_state(os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"))
        
            if accelerator.is_main_process:
                stats.save(cfg.exp_dir+"stats")
            
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"))
    return True

def _train_or_eval_fn(model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training = True, visualize = False):
    if training:
        model.train() 
    else:
        model.eval()

    # print(f"Start the loop for process {accelerator.process_index}")
    
    for step, batch in enumerate(dataloader):
        # print(f"Start the data processing for process {accelerator.process_index} at step {step}")
        images = batch["image"].to(accelerator.device)
        crop_params = batch["crop_params"].to(accelerator.device)
        translation = batch["T"].to(accelerator.device)
        rotation = batch["R"].to(accelerator.device)
        fl = batch["fl"].to(accelerator.device)
        pp = batch['pp'].to(accelerator.device)
        
        batch_size = len(images)
        frame_size = images.shape[1]
        
        # NOTE Do we really need batch repeat?
        
        gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1,2),
                R=rotation.reshape(-1,3,3),
                T=translation.reshape(-1,3),
                device=accelerator.device)


        # print(f"Start the model running for process {accelerator.process_index} at step {step}")
        if training:
            predictions = model(images, gt_cameras=gt_cameras, training=True)
            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
        else:
            with torch.no_grad():
                predictions = model(images, training=False)

            
        pred_cameras = predictions["pred_cameras"]
        # print(f"Start the metric computation for process {accelerator.process_index} at step {step}")

        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras, gt_cameras, accelerator, batch_size)

        # metrics to report
        Racc_5  = (rel_rangle_deg < 5).float().mean()
        Racc_15 = (rel_rangle_deg < 15).float().mean()
        Racc_30 = (rel_rangle_deg < 30).float().mean()
        
        Tacc_5  = (rel_tangle_deg<5).float().mean()
        Tacc_15 = (rel_tangle_deg<15).float().mean()
        Tacc_30 = (rel_tangle_deg<30).float().mean()

        predictions["Racc_5"] = Racc_5
        predictions["Racc_15"] = Racc_15
        predictions["Racc_30"] = Racc_30
        predictions["Tacc_5"] = Tacc_5
        predictions["Tacc_15"] = Tacc_15
        predictions["Tacc_30"] = Tacc_30


        if visualize:
            # print(f"Start the visualization for process {accelerator.process_index} at step {step}")

            camera_dict = {"pred_cameras": {},"gt_cameras": {},}
            
            for visidx in range(frame_size):
                camera_dict["pred_cameras"][visidx] = pred_cameras[visidx]
                camera_dict["gt_cameras"][visidx] = gt_cameras[visidx]

            fig = plotly_scene_visualization(camera_dict, frame_size)
            viz.plotlyplot(fig, env=cfg.exp_name, win="cams")

            show_img = view_color_coded_images_for_visdom(images[0])
            viz.images(show_img, env=cfg.exp_name, win="imgs")

        # print(f"Start the printing for process {accelerator.process_index} at step {step}")
        if training:
            stats.update(predictions, stat_set="train")
            if step%cfg.train.print_interval==0:
                accelerator.print(stats.get_status_string(stat_set="train"))
        else:
            stats.update(predictions, stat_set="eval")
            if step%cfg.train.print_interval==0:
                accelerator.print(stats.get_status_string(stat_set="eval"))
                
        if training:   
            # print(f"Start the backpropagation for process {accelerator.process_index} at step {step}")
            # Backward pass and optimization
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    return True



def camera_to_rel_deg(pred_cameras, gt_cameras, accelerator, batch_size):
    with torch.no_grad():
        gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0]//batch_size)
        pair_idx_i1 = pair_idx_i1.to(accelerator.device)

        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

        rel_rangle_deg = rotation_angle(relative_pose_gt[:,:3,:3], 
                                        relative_pose_pred[:,:3,:3])

        rel_tangle_deg = translation_angle(relative_pose_gt[:, 3, :3],
                                        relative_pose_pred[:, 3, :3])

    return rel_rangle_deg, rel_tangle_deg

