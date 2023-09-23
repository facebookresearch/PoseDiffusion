# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union
from multiprocessing import Pool


import hydra
import torch
import numpy as np
from hydra.utils import instantiate, get_original_cwd
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from datasets.co3d_v2 import TRAINING_CATEGORIES, TEST_CATEGORIES, DEBUG_CATEGORIES
from util.match_extraction import extract_match
from util.geometry_guided_sampling import geometry_guided_sampling
from util.metric import camera_to_rel_deg, calculate_auc_np
from util.load_img_folder import load_and_preprocess_images
from util.train_util import (
    get_co3d_dataset_test,
    set_seed_and_print,
)




@hydra.main(config_path="../cfgs/", config_name="default_test")
def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False)

    # Print configuration and accelerator state
    accelerator.print("Model Config:", OmegaConf.to_yaml(cfg), accelerator.state)

    torch.backends.cudnn.benchmark = cfg.test.cudnnbenchmark if not cfg.debug else False
    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True

    set_seed_and_print(cfg.seed)

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False)
    model = model.to(accelerator.device)

    # Accelerator setup
    model = accelerator.prepare(model)

    if cfg.test.resume_ckpt:
        checkpoint = torch.load(cfg.test.resume_ckpt)
        try:
            model.load_state_dict(prefix_with_module(checkpoint), strict=True)
        except:
            model.load_state_dict(checkpoint, strict=True)

        accelerator.print(f"Successfully resumed from {cfg.test.resume_ckpt}")


    categories = cfg.test.category

    if "seen" in categories:
        categories = TRAINING_CATEGORIES
            
    if "unseen" in categories:
        categories = TEST_CATEGORIES
    
    if "debug" in categories:
        categories = DEBUG_CATEGORIES
    
    if "all" in categories:
        categories = TRAINING_CATEGORIES + TEST_CATEGORIES
    
    categories = sorted(categories)

    print("-"*100)
    print(f"Testing on {categories}")
    print("-"*100)
    
    category_dict = {}
    metric_name = ["Auc_30", "Racc_5", "Racc_15", "Racc_30", "Tacc_5", "Tacc_15", "Tacc_30"]
    
    for m_name in metric_name:
        category_dict[m_name] = {}
    

    for category in categories:
        print("-"*100)
        print(f"Category {category} Start")

        error_dict = _test_one_category(
            model = model,
            category = category,
            cfg = cfg,
            num_frames = cfg.test.num_frames,
            random_order = cfg.test.random_order, 
            accelerator = accelerator,
        )
        
        rError = np.array(error_dict['rError'])
        tError = np.array(error_dict['tError'])
        
        category_dict["Racc_5"][category] = np.mean(rError < 5) * 100
        category_dict["Racc_15"][category] = np.mean(rError < 15) * 100
        category_dict["Racc_30"][category] = np.mean(rError < 30) * 100
        
        category_dict["Tacc_5"][category] = np.mean(tError < 5) * 100
        category_dict["Tacc_15"][category] = np.mean(tError < 15) * 100
        category_dict["Tacc_30"][category] = np.mean(tError < 30) * 100
        
        Auc_30 = calculate_auc_np(rError, tError, max_threshold=30)
        category_dict["Auc_30"][category]  = Auc_30 * 100
        
        print("-"*100)
        print(f"Category {category} Done")
    
    for m_name in metric_name:
        category_dict[m_name]["mean"] = np.mean(list(category_dict[m_name].values()))     

    for c_name in (categories + ["mean"]): 
        print_str = f"{c_name.ljust(20)}: "
        for m_name in metric_name:  
            print_num = np.mean(category_dict[m_name][c_name])
            print_str = print_str + f"{m_name} is {print_num:.3f} | " 
            
        if c_name == "mean":
            print("-"*100)
        print(print_str)
        

    return True

def _test_one_category(model, category, cfg, num_frames, random_order, accelerator):
    model.eval()
    
    print(f"******************************** test on {category} ********************************")

    # Data loading
    test_dataset = get_co3d_dataset_test(cfg, category = category)
    
    category_error_dict = {"rError":[], "tError":[]}
    
    for seq_name in test_dataset.sequence_list: 
        print(f"Testing the sequence {seq_name.ljust(15, ' ')} of category {category.ljust(15, ' ')}")
        metadata = test_dataset.rotations[seq_name]
        
        if len(metadata)<num_frames:
            print(f"Skip sequence {seq_name}")
            continue
        
        ids = np.random.choice(len(metadata), num_frames, replace=False)
        batch, image_paths = test_dataset.get_data(sequence_name=seq_name, ids=ids, return_path = True)

        # Use load_and_preprocess_images() here instead of using batch["image"] as
        #       images = batch["image"].to(accelerator.device)
        # because we need bboxes_xyxy and resized_scales for GGS
        # TODO combine this into Co3D V2 dataset
        images, image_info = load_and_preprocess_images(image_paths = image_paths, image_size = cfg.test.img_size)
        images = images.to(accelerator.device)
        
        if cfg.GGS.enable:
            kp1, kp2, i12 = extract_match(image_paths = image_paths, image_info = image_info)
            
            if kp1 is not None:
                keys = ["kp1", "kp2", "i12", "img_shape"]
                values = [kp1, kp2, i12, images.shape]
                matches_dict = dict(zip(keys, values))

                cfg.GGS.pose_encoding_type = cfg.MODEL.pose_encoding_type
                GGS_cfg = OmegaConf.to_container(cfg.GGS)

                cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)
                print("[92m=====> Sampling with GGS <=====[0m")
            else:
                cond_fn = None
        else:
            cond_fn = None
            print("[92m=====> Sampling without GGS <=====[0m")

            
        translation = batch["T"].to(accelerator.device)
        rotation = batch["R"].to(accelerator.device)
        fl = batch["fl"].to(accelerator.device)
        pp = batch["pp"].to(accelerator.device)

        gt_cameras = PerspectiveCameras(
            focal_length=fl.reshape(-1, 2),
            R=rotation.reshape(-1, 3, 3),
            T=translation.reshape(-1, 3),
            device=accelerator.device,
            )

        # expand to 1 x N x 3 x H x W
        images = images.unsqueeze(0)
        batch_size = len(images)

        with torch.no_grad():
            predictions = model(images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step, training=False)

        pred_cameras = predictions["pred_cameras"]
        
        # compute metrics
        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras, gt_cameras, accelerator.device, batch_size)

        print(f"    --  Pair Rot   Error (Deg): {rel_rangle_deg.mean():10.2f}")
        print(f"    --  Pair Trans Error (Deg): {rel_tangle_deg.mean():10.2f}")

        category_error_dict["rError"].extend(rel_rangle_deg.cpu().numpy())
        category_error_dict["tError"].extend(rel_tangle_deg.cpu().numpy())
    
    return category_error_dict


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


if __name__ == "__main__":
    test_fn()
