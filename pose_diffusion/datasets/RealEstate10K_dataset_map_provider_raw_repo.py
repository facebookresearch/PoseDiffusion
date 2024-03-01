# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import json
import logging
import multiprocessing
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from iopath.common.file_io import PathManager

from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.dataset_map_provider import (
    DatasetMap,
    DatasetMapProviderBase,
    PathManagerFactory,
)
# from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
# from pytorch3d.implicitron.dataset.json_index_dataset_v2 import JsonIndexDatasetV2
from pytorch3d.implicitron.dataset.RealEstate10K_dataset import RealEstate10KDataset

from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    registry,
    run_auto_creation,
)

from pytorch3d.renderer.cameras import CamerasBase
from tqdm import tqdm
import pickle
import boto3

_CO3DV2_DATASET_ROOT: str = os.getenv("CO3DV2_DATASET_ROOT", "")

# _NEED_CONTROL is a list of those elements of JsonIndexDataset which
# are not directly specified for it in the config but come from the
# DatasetMapProvider.
_NEED_CONTROL: Tuple[str, ...] = (
    "dataset_root",
    "eval_batches",
    "eval_batch_index",
    "path_manager",
    "subsets",
    "frame_annotations_file",
    "sequence_annotations_file",
    "subset_lists_file",
)

logger = logging.getLogger(__name__)


@registry.register
class RealEstate10KMapProvider(DatasetMapProviderBase):  # pyre-ignore [13]
    """
    Generates the training, validation, and testing dataset objects for
    a dataset laid out on disk like CO3Dv2, with annotations in gzipped json files.

    The dataset is organized in the filesystem as follows::

    """

    subset_name: str
    dataset_root: str 

    test_on_train: bool = False
    only_test_set: bool = False
    load_eval_batches: bool = True
    num_load_workers: int = 4

    n_known_frames_for_test: int = 0

    dataset_class_type: str = "RealEstate10KDataset"
    dataset: RealEstate10KDataset

    # path_manager_factory: PathManagerFactory
    # path_manager_factory_class_type: str = "PathManagerFactory"
    video_loc: str = "/private/home/jianyuan/src/pixar_replay/pixar_replay/experimental/models/jaypose/video_loc.txt"

    def __post_init__(self):
        super().__init__()
        run_auto_creation(self)

        if self.only_test_set and self.test_on_train:
            raise ValueError("Cannot have only_test_set and test_on_train")

        self.imageset = np.loadtxt(self.video_loc, dtype=np.str,)
        self.imageset = sorted(self.imageset)

        seq_num = len(self.imageset)

        self.imageset_train = self.imageset[0 : int(0.8 * seq_num)]
        self.imageset_test = self.imageset[int(0.8 * seq_num) :]        

                
        common_dataset_kwargs = getattr(self, f"dataset_{self.dataset_class_type}_args")
        common_dataset_kwargs = {
            **common_dataset_kwargs,
            "dataset_root": self.dataset_root,
            "subsets": None,
            "subset_lists_file": "",
        }

        dataset_type: Type[RealEstate10KDataset] = registry.get(
            RealEstate10KDataset, self.dataset_class_type
        )
        expand_args_fields(dataset_type)
        
        all_dataset_kwargs = {
            **common_dataset_kwargs,
            "seq_list": self.imageset,
            # "split": "train",
            "is_train": False,
        }
        
        all_dataset = dataset_type(**all_dataset_kwargs)
        
        train_dataset_kwargs = {
            **common_dataset_kwargs,
            "seq_list": self.imageset_train,
            # "split": "train",
            "is_train": True,
        }
        train_dataset = dataset_type(**train_dataset_kwargs)
        
        test_dataset_kwargs = {
            **common_dataset_kwargs,
            "seq_list": self.imageset_test,
            # "split": "test",
            "is_train": False,
        }
        test_dataset = dataset_type(**test_dataset_kwargs)

        self.dataset_map = DatasetMap(train=train_dataset, val=test_dataset, test=test_dataset)

    def create_dataset(self):
        # The dataset object is created inside `self.get_dataset_map`
        pass

    def get_dataset_map(self) -> DatasetMap:
        return self.dataset_map  # pyre-ignore [16]
