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
    print("done")


if __name__ == "__main__":
    main()
