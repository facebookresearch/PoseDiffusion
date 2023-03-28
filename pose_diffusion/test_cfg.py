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
        import pdb;pdb.set_trace()

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
        # self.transformer = build_transformer(**transformer)
        # self.backbone = build_backbone(**backbone)


class DummyModel:
    def __init__(
        self,
        name: str,
        backbone: Dict,
        transformer: Dict,
    ):
        self.name = name


@hydra.main(config_path="cfgs/", config_name="default")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = instantiate(cfg.MODEL)
    
    
    # instantiate
    # import pdb;pdb.set_trace()
    # TODO: The following may be needed for hydra/submitit it to work

    # dump_cfg(cfg)
    # experiment.run()



if __name__ == "__main__":
    main()
