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
from util.utils import seed_all_random_engines
from util.match_extraction import extract_match
from util.load_img_folder import load_and_preprocess_images
import models
import time
from util.geometry_guided_sampling import geometry_guided_sampling
from functools import partial


@hydra.main(config_path="../cfgs/", config_name="default")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Construction
    model = instantiate(cfg.MODEL, _recursive_=False)

    # Loading Image
    original_cwd = (
        get_original_cwd()
    )  # hydra changes the default path, goes back
    folder_path = os.path.join(original_cwd, cfg.image_folder)
    images, image_info = load_and_preprocess_images(folder_path, cfg.image_size)

    # Load the pre-set checkpoint
    ckpt_path = os.path.join(original_cwd, cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")

    # Move to the GPU
    model = model.to(device)
    images = images.to(device)

    # Evaluation Mode
    model.eval()

    seed_all_random_engines(cfg.seed)

    # Start the timer
    start_time = time.time()

    # Match extraction
    if cfg.GGS.enable:
        # TODO Optional: remove the keypoints outside the cropped region?

        kp1, kp2, i12 = extract_match(folder_path, image_info)

        keys = ["kp1", "kp2", "i12", "img_shape"]
        values = [kp1, kp2, i12, images.shape]
        matches_dict = dict(zip(keys, values))

        cfg.GGS.pose_encoding_type = cfg.MODEL.pose_encoding_type        
        GGS_cfg = OmegaConf.to_container(cfg.GGS)
        
        # I am not really sure it is best to introduce partial() here
        cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)        
    else:
        cond_fn = None

    # Forward
    with torch.no_grad():
        # pred_pose: (B,N,4,4)
        # pred_fl:   (B,N,2)

        # The poses and focal length are defined as
        # NDC coordinate system in
        # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        pred_cameras = model(image=images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step)

    print(
        f"For samples/apple: the std of pred_cameras.R is {pred_cameras.R.std():.6f}, which should be close to 0.564747"
    )

    print(
        f"For samples/apple: the std of pred_cameras.T is {pred_cameras.T.mean():.6f}, which should be close to 0.395809"
    )

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken: {:.4f} seconds".format(elapsed_time))


if __name__ == "__main__":
    main()
