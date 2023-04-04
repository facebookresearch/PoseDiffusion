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
from util.load_img_folder import load_and_preprocess_images
import models


@hydra.main(config_path="../cfgs/", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Construction
    model = instantiate(cfg.MODEL, _recursive_=False)

    # Loading Image
    original_cwd = get_original_cwd()  # hydra changes the default path, goes back
    folder_path = os.path.join(original_cwd, cfg.image_folder)
    images = load_and_preprocess_images(folder_path, cfg.image_size)

    # Or randomly generated image, ranging from 0 to 1
    # images = torch.rand(10, 3, 224, 224)

    # Load the pre-set checkpoint
    ckpt_path = os.path.join(original_cwd, cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")
        return

    # Move to the GPU
    model = model.to(device)
    images = images.to(device)

    # Evaluation Mode
    model.eval()

    seed_all_random_engines(0)
    # Forward
    with torch.no_grad():
        # pred_pose: (B,N,4,4)
        # pred_fl:   (B,N,2)

        # The poses and focal length are defined as
        # NDC coordinate system in
        # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        pred_pose, pred_fl = model(image=images)

    print(
        f"For samples/apple: the std of pred_pose is {pred_pose.std():.2f}, which should be close to 0.67"
    )
    print(
        f"For samples/apple: the mean of pred_pose is {pred_pose.mean():.2f}, which should be close to 0.21"
    )

    print("done")


if __name__ == "__main__":
    main()
