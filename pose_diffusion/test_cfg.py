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

from config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])
    np.random.seed(1024)
    return args, cfg


# pseudo code for build_transformer
# please imagine this is in a python file called build_transformer.py
__all__ = {
    "SimpleTransformer": transformer,
    "AAA": AAATransformer,
    "BBB": BBBTransformer,
}


def build_transformer(name, num_encoders, num_decoders, **kwargs):
    # num_encoders, num_decoders are common args
    # for all the transformer variants
    # kwargs: passing the customized args of different variants

    model = __all__[name](
        num_encoders=num_encoders, num_decoders=num_decoders, **kwargs
    )
    return model


class dummy_model:
    def __init__(
        self,
        name: str,
        backbone: Dict,
        transformer: Dict,
    ):
        self.name = name
        # it is easy to eliminate model_cfg.AAA from within the modules
        # the nodes of the config system are just dicts
        # the leaf can be any type, e.g., str, list, int, and so on
        # so we only pass various dicts

        # when we want to reuse codes across codebases,
        # just need to ensure the dicts (e.g., transformer) are also taken away
        # we can also call them backbone_cfg, transformer_cfg if beneficial

        # the content of dicts transformer and backbone 
        # are defined in the downsteam functions/modules

        # but to be honest, in this way, 
        # we are somehow back close to implicitron...
        self.transformer = build_transformer(**transformer)
        self.backbone = build_backbone(**backbone)


def main():
    args, cfg = parse_config()
    model = dummy_model(**cfg.MODEL)
    optimizer = dummy_optimizer(**cfg.OPTIMIZER)


if __name__ == "__main__":
    main()
