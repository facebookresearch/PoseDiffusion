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
from util.load_img_folder import load_and_preprocess_images
import models


@hydra.main(config_path="cfgs/", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Model Construction
    model = instantiate(cfg.MODEL, _recursive_=False)

    # Evaluation Mode
    model.eval()

    # Loading Image
    original_cwd = get_original_cwd()  # hydra changes the default path, goes back
    folder_path = os.path.join(original_cwd, cfg.TEST.image_path)
    image_size = cfg.TEST.image_size
    images_tensor = load_and_preprocess_images(folder_path, image_size)

    # Or randomly generated image, range from 0 to 1
    # images_tensor = torch.rand(10, 3, 224, 224)

    # Forward
    with torch.no_grad():
        pred_pose = model(image=images_tensor)

    print("done")


if __name__ == "__main__":
    main()
