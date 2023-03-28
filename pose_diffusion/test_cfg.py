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
from omegaconf import OmegaConf



# pseudo code for build_transformer
# please imagine this is in a python file called build_transformer.py
# __all__ = {
#     "SimpleTransformer": Transformer,
#     "AAA": AAATransformer,
#     "BBB": BBBTransformer,
# }


# def build_transformer(name, num_encoders, num_decoders, **kwargs):
#     # num_encoders, num_decoders are common args
#     # for all the transformer variants
#     # kwargs: passing the customized args of different variants

#     model = __all__[name](
#         num_encoders=num_encoders, num_decoders=num_decoders, **kwargs
#     )
    
#     return model


class DummyModel:
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
        
        
        # import pdb;pdb.set_trace()
        self.transformer = build_transformer(**transformer)
        self.backbone = build_backbone(**backbone)


def main():
    cli_cfg = OmegaConf.from_cli()
    cfg_file = cli_cfg.get("cfg_file")
    yaml_cfg = OmegaConf.load(cfg_file)

    cfg = OmegaConf.merge(yaml_cfg, cli_cfg)

    model = DummyModel(**cfg.MODEL)
    optimizer = DummyOptimizer(**cfg.OPTIMIZER)


if __name__ == "__main__":
    main()
