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
from hydra.utils import instantiate


import models


@hydra.main(config_path="cfgs/", config_name="default")
def main(cfg: DictConfig) -> None:
    print("*"*20)
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))
    print("*"*20)
    
    # MODEL
    model = instantiate(cfg.MODEL, _recursive_=False)
    
    # Evaluation
    model.eval()
    
    # randomly generated image, range from 0 to 1
    input_image = torch.rand(10, 3, 224, 224) 
    
    # forward 
    with torch.no_grad():
        pred_pose = model(image=input_image)
    print("done")


if __name__ == "__main__":
    main()
