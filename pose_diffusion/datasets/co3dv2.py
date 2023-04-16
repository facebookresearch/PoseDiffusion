# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import gzip
import hashlib
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures.pointclouds import Pointclouds
from tqdm import tqdm
import torchvision.transforms as transforms
import pickle
from copy import deepcopy

from . import types
from .dataset_base import DatasetBase, FrameData
from .utils import is_known_frame_scalar
from multiprocessing import Manager


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import TypedDict

    class FrameAnnotsEntry(TypedDict):
        subset: Optional[str]
        frame_annotation: types.FrameAnnotation

else:
    FrameAnnotsEntry = dict


@registry.register
class JsonIndexDatasetV2(DatasetBase, ReplaceableBase):
    """
    A dataset with annotations in json files like the Common Objects in 3D
    (CO3D) dataset.

    Args:
        frame_annotations_file: A zipped json file containing metadata of the
            frames in the dataset, serialized List[types.FrameAnnotation].
        sequence_annotations_file: A zipped json file containing metadata of the
            sequences in the dataset, serialized List[types.SequenceAnnotation].
        subset_lists_file: A json file containing the lists of frames corresponding
            corresponding to different subsets (e.g. train/val/test) of the dataset;
            format: {subset: (sequence_name, frame_id, file_path)}.
        subsets: Restrict frames/sequences only to the given list of subsets
            as defined in subset_lists_file (see above).
        limit_to: Limit the dataset to the first #limit_to frames (after other
            filters have been applied).
        limit_sequences_to: Limit the dataset to the first
            #limit_sequences_to sequences (after other sequence filters have been
            applied but before frame-based filters).
        pick_sequence: A list of sequence names to restrict the dataset to.
        exclude_sequence: A list of the names of the sequences to exclude.
        limit_category_to: Restrict the dataset to the given list of categories.
        dataset_root: The root folder of the dataset; all the paths in jsons are
            specified relative to this root (but not json paths themselves).
        load_images: Enable loading the frame RGB data.
        load_depths: Enable loading the frame depth maps.
        load_depth_masks: Enable loading the frame depth map masks denoting the
            depth values used for evaluation (the points consistent across views).
        load_masks: Enable loading frame foreground masks.
        load_point_clouds: Enable loading sequence-level point clouds.
        max_points: Cap on the number of loaded points in the point cloud;
            if reached, they are randomly sampled without replacement.
        mask_images: Whether to mask the images with the loaded foreground masks;
            0 value is used for background.
        mask_depths: Whether to mask the depth maps with the loaded foreground
            masks; 0 value is used for background.
        image_height: The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        image_width: The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        box_crop: Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected.
        box_crop_mask_thr: The threshold used to separate pixels into foreground
            and background based on the foreground_probability mask; if no value
            is greater than this threshold, the loader lowers it and repeats.
        box_crop_context: The amount of additional padding added to each
            dimension of the cropping bounding box, relative to box size.
        remove_empty_masks: Removes the frames with no active foreground pixels
            in the segmentation mask after thresholding (see box_crop_mask_thr).
        n_frames_per_sequence: If > 0, randomly samples #n_frames_per_sequence
            frames in each sequences uniformly without replacement if it has
            more frames than that; applied before other frame-level filters.
        seed: The seed of the random generator sampling #n_frames_per_sequence
            random frames per sequence.
        sort_frames: Enable frame annotations sorting to group frames from the
            same sequences together and order them by timestamps
        eval_batches: A list of batches that form the evaluation set;
            list of batch-sized lists of indices corresponding to __getitem__
            of this class, thus it can be used directly as a batch sampler.
        eval_batch_index:
            ( Optional[List[List[Union[Tuple[str, int, str], Tuple[str, int]]]] )
            A list of batches of frames described as (sequence_name, frame_idx)
            that can form the evaluation set, `eval_batches` will be set from this.

    """

    frame_annotations_type: ClassVar[
        Type[types.FrameAnnotation]
    ] = types.FrameAnnotation

    path_manager: Any = None
    frame_annotations_file: str = ""
    sequence_annotations_file: str = ""
    subset_lists_file: str = ""
    subsets: Optional[List[str]] = None
    limit_to: int = 0
    limit_sequences_to: int = 0
    pick_sequence: Tuple[str, ...] = ()
    exclude_sequence: Tuple[str, ...] = ()
    limit_category_to: Tuple[int, ...] = ()
    dataset_root: str = ""

    crop_wo_mask: bool = False
    load_images: bool = True
    load_depths: bool = False
    load_depth_masks: bool = False
    load_masks: bool = True
    force_square: bool = True
    load_point_clouds: bool = False
    max_points: int = 0
    mask_images: bool = False
    mask_depths: bool = False
    image_height: Optional[int] = 224
    image_width: Optional[int] = 224
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    remove_empty_masks: bool = True
    n_frames_per_sequence: int = -1
    seed: int = 0
    sort_frames: bool = False
    eval_batches: Any = None
    eval_batch_index: Any = None

    sample_ratio: float = -1
    ### Augmentation
    is_train: bool = False
    box_random_aug: bool = False

    def __post_init__(self) -> None:
        # pyre-fixme[16]: `JsonIndexDataset` has no attribute `subset_to_image_path`.

        self.subset_to_image_path = None
        self._load_frames()
        self._load_sequences()
        if self.sort_frames:
            self._sort_frames()
        self._load_subset_lists()
        self._filter_db()  # also computes sequence indices
        self._extract_and_set_eval_batches()

        logger.info(str(self))

    def _extract_and_set_eval_batches(self):
        """
        Sets eval_batches based on input eval_batch_index.
        """
        if self.eval_batch_index is not None:
            if self.eval_batches is not None:
                raise ValueError(
                    "Cannot define both eval_batch_index and eval_batches."
                )
            self.eval_batches = self.seq_frame_index_to_dataset_index(
                self.eval_batch_index
            )

    def join(self, other_datasets: Iterable[DatasetBase]) -> None:
        """
        Join the dataset with other JsonIndexDataset objects.

        Args:
            other_datasets: A list of JsonIndexDataset objects to be joined
                into the current dataset.
        """
        # if not all(isinstance(d, JsonIndexDatasetV2) for d in other_datasets):
        #     raise ValueError("This function can only join a list of JsonIndexDataset")
        # pyre-ignore[16]
        self.frame_annots.extend(
            [fa for d in other_datasets for fa in d.frame_annots]
        )
        # pyre-ignore[16]
        self.seq_annots.update(
            # https://gist.github.com/treyhunner/f35292e676efa0be1728
            functools.reduce(
                lambda a, b: {**a, **b},
                [d.seq_annots for d in other_datasets],  # pyre-ignore[16]
            )
        )

        all_eval_batches = [
            self.eval_batches,
            # pyre-ignore
            *[d.eval_batches for d in other_datasets],
        ]
        if not (
            all(ba is None for ba in all_eval_batches)
            or all(ba is not None for ba in all_eval_batches)
        ):
            raise ValueError(
                "When joining datasets, either all joined datasets have to have their"
                " eval_batches defined, or all should have their eval batches undefined."
            )
        if self.eval_batches is not None:
            self.eval_batches = sum(all_eval_batches, [])
        self._invalidate_indexes(filter_seq_annots=True)

    def is_filtered(self) -> bool:
        """
        Returns `True` in case the dataset has been filtered and thus some frame annotations
        stored on the disk might be missing in the dataset object.

        Returns:
            is_filtered: `True` if the dataset has been filtered, else `False`.
        """
        return (
            self.remove_empty_masks
            or self.limit_to > 0
            or self.limit_sequences_to > 0
            or len(self.pick_sequence) > 0
            or len(self.exclude_sequence) > 0
            or len(self.limit_category_to) > 0
            or self.n_frames_per_sequence > 0
        )

    def seq_frame_index_to_dataset_index(
        self,
        seq_frame_index: List[
            List[Union[Tuple[str, int, str], Tuple[str, int]]]
        ],
        allow_missing_indices: bool = False,
        remove_missing_indices: bool = False,
        suppress_missing_index_warning: bool = True,
    ) -> List[List[Union[Optional[int], int]]]:
        """
        Obtain indices into the dataset object given a list of frame ids.

        Args:
            seq_frame_index: The list of frame ids specified as
                `List[List[Tuple[sequence_name:str, frame_number:int]]]`. Optionally,
                Image paths relative to the dataset_root can be stored specified as well:
                `List[List[Tuple[sequence_name:str, frame_number:int, image_path:str]]]`
            allow_missing_indices: If `False`, throws an IndexError upon reaching the first
                entry from `seq_frame_index` which is missing in the dataset.
                Otherwise, depending on `remove_missing_indices`, either returns `None`
                in place of missing entries or removes the indices of missing entries.
            remove_missing_indices: Active when `allow_missing_indices=True`.
                If `False`, returns `None` in place of `seq_frame_index` entries that
                are not present in the dataset.
                If `True` removes missing indices from the returned indices.
            suppress_missing_index_warning:
                Active if `allow_missing_indices==True`. Suppressess a warning message
                in case an entry from `seq_frame_index` is missing in the dataset
                (expected in certain cases - e.g. when setting
                `self.remove_empty_masks=True`).

        Returns:
            dataset_idx: Indices of dataset entries corresponding to`seq_frame_index`.
        """
        _dataset_seq_frame_n_index = {
            seq: {
                # pyre-ignore[16]
                self.frame_annots[idx]["frame_annotation"].frame_number: idx
                for idx in seq_idx
            }
            # pyre-ignore[16]
            for seq, seq_idx in self._seq_to_idx.items()
        }

        def _get_dataset_idx(
            seq_name: str, frame_no: int, path: Optional[str] = None
        ) -> Optional[int]:
            idx_seq = _dataset_seq_frame_n_index.get(seq_name, None)
            idx = idx_seq.get(frame_no, None) if idx_seq is not None else None

            if idx is None:
                msg = (
                    f"sequence_name={seq_name} / frame_number={frame_no}"
                    " not in the dataset!"
                )
                if not allow_missing_indices:
                    raise IndexError(msg)
                if not suppress_missing_index_warning:
                    warnings.warn(msg)
                return idx
            if path is not None:
                # Check that the loaded frame path is consistent
                # with the one stored in self.frame_annots.
                assert os.path.normpath(
                    # pyre-ignore[16]
                    self.frame_annots[idx]["frame_annotation"].image.path
                ) == os.path.normpath(
                    path
                ), f"Inconsistent frame indices {seq_name, frame_no, path}."
            return idx

        dataset_idx = [
            [_get_dataset_idx(*b) for b in batch]  # pyre-ignore [6]
            for batch in seq_frame_index
        ]

        if allow_missing_indices and remove_missing_indices:
            # remove all None indices, and also batches with only None entries
            valid_dataset_idx = [
                [b for b in batch if b is not None] for batch in dataset_idx
            ]
            return [  # pyre-ignore[7]
                batch for batch in valid_dataset_idx if len(batch) > 0
            ]

        return dataset_idx

    def subset_from_frame_index(
        self,
        frame_index: List[Union[Tuple[str, int], Tuple[str, int, str]]],
        istrain: bool = False,
        allow_missing_indices: bool = True,
    ) -> "JsonIndexDataset":
        """
        Generate a dataset subset given the list of frames specified in `frame_index`.

        Args:
            frame_index: The list of frame indentifiers (as stored in the metadata)
                specified as `List[Tuple[sequence_name:str, frame_number:int]]`. Optionally,
                Image paths relative to the dataset_root can be stored specified as well:
                `List[Tuple[sequence_name:str, frame_number:int, image_path:str]]`,
                in the latter case, if imaga_path do not match the stored paths, an error
                is raised.
            allow_missing_indices: If `False`, throws an IndexError upon reaching the first
                entry from `frame_index` which is missing in the dataset.
                Otherwise, generates a subset consisting of frames entries that actually
                exist in the dataset.
        """
        # Get the indices into the frame annots.
        dataset_indices = self.seq_frame_index_to_dataset_index(
            [frame_index],
            allow_missing_indices=self.is_filtered() and allow_missing_indices,
        )[0]
        valid_dataset_indices = [i for i in dataset_indices if i is not None]

        # Deep copy the whole dataset except frame_annots, which are large so we
        # deep copy only the requested subset of frame_annots.
        memo = {id(self.frame_annots): None}  # pyre-ignore[16]
        dataset_new = copy.deepcopy(self, memo)

        dataset_new.is_train = istrain
        dataset_new.frame_annots = copy.deepcopy(
            [self.frame_annots[i] for i in valid_dataset_indices]
        )

        # This will kill all unneeded sequence annotations.
        dataset_new._invalidate_indexes(filter_seq_annots=True)

        # Finally annotate the frame annotations with the name of the subset
        # stored in meta.
        for frame_annot in dataset_new.frame_annots:
            frame_annotation = frame_annot["frame_annotation"]
            if frame_annotation.meta is not None:
                frame_annot["subset"] = frame_annotation.meta.get(
                    "frame_type", None
                )

        # A sanity check - this will crash in case some entries from frame_index are missing
        # in dataset_new.
        valid_frame_index = [
            fi for fi, di in zip(frame_index, dataset_indices) if di is not None
        ]
        dataset_new.seq_frame_index_to_dataset_index(
            [valid_frame_index], allow_missing_indices=False
        )

        return dataset_new

    def __str__(self) -> str:
        # pyre-ignore[16]
        return f"JsonIndexDataset #frames={len(self.frame_annots)}"

    def _get_frame_type(self, entry: FrameAnnotsEntry) -> Optional[str]:
        return entry["subset"]

    def get_all_train_cameras(self) -> CamerasBase:
        """
        Returns the cameras corresponding to all the known frames.
        """
        logger.info("Loading all train cameras.")
        cameras = []
        # pyre-ignore[16]
        for frame_idx, frame_annot in enumerate(tqdm(self.frame_annots)):
            frame_type = self._get_frame_type(frame_annot)
            if frame_type is None:
                raise ValueError("subsets not loaded")
            if is_known_frame_scalar(frame_type):
                cameras.append(self[frame_idx].camera)
        return join_cameras_as_batch(cameras)

    def __getitem__(self, index) -> FrameData:
        # pyre-ignore[16]
        # WHY WOULD WE CHECK THIS? JUST LET IT OUT OF THE RANGE AND WE WILL SEE THE ERROR
        if index >= len(self.frame_annots):
            raise IndexError(
                f"index {index} out of range {len(self.frame_annots)}"
            )

        raw_entry = self.frame_annots[index]
        entry = raw_entry["frame_annotation"]
        seq_annotation = self.seq_annots[entry.sequence_name]
        point_cloud = seq_annotation.point_cloud

        frame_data = FrameData(
            frame_number=_safe_as_tensor(entry.frame_number, torch.long),
            frame_timestamp=_safe_as_tensor(entry.frame_timestamp, torch.float),
            sequence_name=entry.sequence_name,
            sequence_category=seq_annotation.category,
            camera_quality_score=_safe_as_tensor(
                seq_annotation.viewpoint_quality_score,
                torch.float,
            ),
            point_cloud_quality_score=_safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
        )

        # The rest of the fields are optional
        frame_data.frame_type = self._get_frame_type(raw_entry)

        (
            frame_data.fg_probability,
            frame_data.mask_path,
            frame_data.bbox_xywh,
            clamp_bbox_xyxy,
            frame_data.crop_bbox_xywh,
        ) = self._load_crop_fg_probability(entry)

        scale = 1.0
        if self.load_images and entry.image is not None:
            # original image size
            frame_data.image_size_hw = _safe_as_tensor(
                entry.image.size, torch.long
            )

            (
                frame_data.image_rgb,
                frame_data.image_path,
                frame_data.mask_crop,
                scale,
            ) = self._load_crop_images(
                entry, frame_data.fg_probability, clamp_bbox_xyxy
            )

        frame_data.clamp_bbox_xyxy = clamp_bbox_xyxy
        frame_data.scale = torch.FloatTensor([scale])

        if self.load_depths and entry.depth is not None:
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(
                entry, clamp_bbox_xyxy, frame_data.fg_probability
            )

        if entry.viewpoint is not None:
            (
                frame_data.camera,
                frame_data.half_image_size_output,
            ) = self._get_pytorch3d_camera(
                entry,
                scale,
                clamp_bbox_xyxy,
            )

        if self.load_point_clouds and point_cloud is not None:
            pcl_path = self._fix_point_cloud_path(point_cloud.path)
            frame_data.sequence_point_cloud = _load_pointcloud(
                self._local_path(pcl_path), max_points=self.max_points
            )
            frame_data.sequence_point_cloud_path = pcl_path

        return frame_data

    def __len__(self) -> int:
        # pyre-ignore[16]
        return len(self.frame_annots)

    def _fix_point_cloud_path(self, path: str) -> str:
        """
        Fix up a point cloud path from the dataset.
        Some files in Co3Dv2 have an accidental absolute path stored.
        """
        unwanted_prefix = "/large_experiments/p3/replay/datasets/co3d/co3d45k_220512/export_v23/"
        if path.startswith(unwanted_prefix):
            path = path[len(unwanted_prefix) :]
        return os.path.join(self.dataset_root, path)

    def _load_crop_fg_probability(
        self, entry: types.FrameAnnotation
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[str],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        fg_probability = None
        full_path = None
        bbox_xywh = None
        clamp_bbox_xyxy = None
        crop_box_xywh = None

        if (self.load_masks or self.box_crop) and entry.mask is not None:
            if self.crop_wo_mask:
                assert self.box_crop
                full_path = os.path.join(self.dataset_root, entry.mask.path)
                h, w = entry.image.size

                dis = abs(h - w) / 2

                if h > w:
                    bbox_xywh = torch.tensor([0, dis, w, h])
                else:
                    bbox_xywh = torch.tensor([dis, 0, w, h])

                bbox_xywh[2:] = bbox_xywh[2:].min()

                if self.is_train and self.box_random_aug:
                    # random scale
                    x_scaling, y_scaling = np.random.uniform(0.85, 1.15, 2)
                    bbox_xywh[2] = bbox_xywh[2] * x_scaling
                    bbox_xywh[3] = bbox_xywh[3] * y_scaling

                    # random corner
                    offset_x = np.random.uniform(-0.1, 0.1) * bbox_xywh[2]
                    offset_y = np.random.uniform(-0.1, 0.1) * bbox_xywh[3]
                    bbox_xywh[0] = bbox_xywh[0] + offset_x
                    bbox_xywh[1] = bbox_xywh[1] + offset_y

                clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(
                    _get_clamp_bbox(
                        bbox_xywh,
                        image_path=entry.image.path,
                        box_crop_context=0.0,
                    ),
                    image_size_hw=(h, w),
                )
                crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)
            else:
                full_path = os.path.join(self.dataset_root, entry.mask.path)
                mask = _load_mask(self._local_path(full_path))

                if mask.shape[-2:] != entry.image.size:
                    raise ValueError(
                        f"bad mask size: {mask.shape[-2:]} vs {entry.image.size}!"
                    )

                bbox_xywh = torch.tensor(
                    _get_bbox_from_mask(mask, self.box_crop_mask_thr)
                )

                if self.is_train and self.box_random_aug:
                    # random scale
                    x_scaling, y_scaling = np.random.uniform(0.8, 1.2, 2)
                    bbox_xywh[2] = bbox_xywh[2] * x_scaling
                    bbox_xywh[3] = bbox_xywh[3] * y_scaling
                    # random corner
                    offset_x = np.random.uniform(-0.1, 0.1) * bbox_xywh[2]
                    offset_y = np.random.uniform(-0.1, 0.1) * bbox_xywh[3]

                    bbox_xywh[0] = bbox_xywh[0] + offset_x
                    bbox_xywh[1] = bbox_xywh[1] + offset_y

                if self.force_square:
                    mean_hw = bbox_xywh[2:].float().mean().long()
                    bbox_xywh[2:] = mean_hw

                if self.box_crop:
                    clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(
                        _get_clamp_bbox(
                            bbox_xywh,
                            image_path=entry.image.path,
                            box_crop_context=self.box_crop_context,
                        ),
                        image_size_hw=tuple(mask.shape[-2:]),
                    )

                    crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)
                    mask = _crop_around_box(mask, clamp_bbox_xyxy, full_path)

                fg_probability, _, _ = self._resize_image(mask, mode="nearest")

        return (
            fg_probability,
            full_path,
            bbox_xywh,
            clamp_bbox_xyxy,
            crop_box_xywh,
        )

    def _load_crop_images(
        self,
        entry: types.FrameAnnotation,
        fg_probability: Optional[torch.Tensor],
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor, float]:
        assert self.dataset_root is not None and entry.image is not None
        path = os.path.join(self.dataset_root, entry.image.path)
        image_rgb = _load_image(self._local_path(path))

        if image_rgb.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!"
            )

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            image_rgb = _crop_around_box(image_rgb, clamp_bbox_xyxy, path)

        image_rgb, scale, mask_crop = self._resize_image(image_rgb)

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb, path, mask_crop, scale

    def _load_mask_depth(
        self,
        entry: types.FrameAnnotation,
        clamp_bbox_xyxy: Optional[torch.Tensor],
        fg_probability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = entry.depth
        assert entry_depth is not None
        path = os.path.join(self.dataset_root, entry_depth.path)
        depth_map = _load_depth(
            self._local_path(path), entry_depth.scale_adjustment
        )

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            depth_bbox_xyxy = _rescale_bbox(
                clamp_bbox_xyxy, entry.image.size, depth_map.shape[-2:]
            )
            depth_map = _crop_around_box(depth_map, depth_bbox_xyxy, path)

        depth_map, _, _ = self._resize_image(depth_map, mode="nearest")

        if self.mask_depths:
            assert fg_probability is not None
            depth_map *= fg_probability

        if self.load_depth_masks:
            assert entry_depth.mask_path is not None
            mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
            depth_mask = _load_depth_mask(self._local_path(mask_path))

            if self.box_crop:
                assert clamp_bbox_xyxy is not None
                depth_mask_bbox_xyxy = _rescale_bbox(
                    clamp_bbox_xyxy, entry.image.size, depth_mask.shape[-2:]
                )
                depth_mask = _crop_around_box(
                    depth_mask, depth_mask_bbox_xyxy, mask_path
                )

            depth_mask, _, _ = self._resize_image(depth_mask, mode="nearest")
        else:
            depth_mask = torch.ones_like(depth_map)

        return depth_map, path, depth_mask

    def _get_pytorch3d_camera(
        self,
        entry: types.FrameAnnotation,
        scale: float,
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> PerspectiveCameras:
        entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(
            entry_viewpoint.focal_length, dtype=torch.float
        )

        center_point_px = torch.tensor(
            [entry.image.size[1] / 2, entry.image.size[0] / 2],
            dtype=torch.float,
        )

        half_image_size_wh_orig = (
            torch.tensor(list(reversed(entry.image.size)), dtype=torch.float)
            / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale

        # assert (principal_point_px-center_point_px).abs().mean() < 1

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            principal_point_px -= clamp_bbox_xyxy[:2]
            center_point_px -= clamp_bbox_xyxy[:2]

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        if self.image_height is None or self.image_width is None:
            out_size = list(reversed(entry.image.size))
        else:
            out_size = [self.image_width, self.image_height]

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        fl_scale = rescale * scale / half_min_image_size_output
        fl_scale = torch.ones_like(principal_point) * fl_scale

        center_point = (
            half_image_size_output - center_point_px * scale
        ) / half_min_image_size_output

        return (
            PerspectiveCameras(
                focal_length=focal_length[None],
                principal_point=principal_point[None],
                center_point=center_point[None],
                fl_scale=fl_scale[None],
                R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
                T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
            ),
            half_image_size_output,
        )

    def _load_frames(self) -> None:
        logger.info(f"Loading Co3D frames from {self.frame_annotations_file}.")
        local_file = self._local_path(self.frame_annotations_file)
        with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
            frame_annots_list = types.load_dataclass(
                zipfile, List[self.frame_annotations_type]
            )
        if not frame_annots_list:
            raise ValueError("Empty dataset!")
        # pyre-ignore[16]
        self.frame_annots = [
            FrameAnnotsEntry(frame_annotation=a, subset=None)
            for a in frame_annots_list
        ]

    def _load_sequences(self) -> None:
        logger.info(
            f"Loading Co3D sequences from {self.sequence_annotations_file}."
        )
        local_file = self._local_path(self.sequence_annotations_file)
        with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
            seq_annots = types.load_dataclass(
                zipfile, List[types.SequenceAnnotation]
            )
        if not seq_annots:
            raise ValueError("Empty sequences file!")
        if self.sample_ratio > 0:
            seq_annots = random.sample(
                seq_annots, int(len(seq_annots) * self.sample_ratio)
            )

        # pyre-ignore[16]
        self.seq_annots = {entry.sequence_name: entry for entry in seq_annots}

    def _load_subset_lists(self) -> None:
        logger.info(f"Loading Co3D subset lists from {self.subset_lists_file}.")
        if not self.subset_lists_file:
            return

        with open(self._local_path(self.subset_lists_file), "r") as f:
            subset_to_seq_frame = json.load(f)

        frame_path_to_subset = {
            path: subset
            for subset, frames in subset_to_seq_frame.items()
            for _, _, path in frames
        }
        # pyre-ignore[16]
        for frame in self.frame_annots:
            frame["subset"] = frame_path_to_subset.get(
                frame["frame_annotation"].image.path, None
            )
            if frame["subset"] is None:
                warnings.warn(
                    "Subset lists are given but don't include "
                    + frame["frame_annotation"].image.path
                )

    def _sort_frames(self) -> None:
        # Sort frames to have them grouped by sequence, ordered by timestamp
        # pyre-ignore[16]
        self.frame_annots = sorted(
            self.frame_annots,
            key=lambda f: (
                f["frame_annotation"].sequence_name,
                f["frame_annotation"].frame_timestamp or 0,
            ),
        )

    def _filter_db(self) -> None:
        if self.remove_empty_masks:
            logger.info("Removing images with empty masks.")
            # pyre-ignore[16]
            old_len = len(self.frame_annots)

            msg = (
                "remove_empty_masks needs every MaskAnnotation.mass to be set."
            )

            def positive_mass(frame_annot: types.FrameAnnotation) -> bool:
                mask = frame_annot.mask
                if mask is None:
                    return False
                if mask.mass is None:
                    raise ValueError(msg)
                return mask.mass > 1

            self.frame_annots = [
                frame
                for frame in self.frame_annots
                if positive_mass(frame["frame_annotation"])
            ]
            logger.info(
                "... filtered %d -> %d" % (old_len, len(self.frame_annots))
            )

        # this has to be called after joining with categories!!
        subsets = self.subsets
        if subsets:
            if not self.subset_lists_file:
                raise ValueError(
                    "Subset filter is on but subset_lists_file was not given"
                )

            logger.info(f"Limiting Co3D dataset to the '{subsets}' subsets.")

            # truncate the list of subsets to the valid one
            self.frame_annots = [
                entry
                for entry in self.frame_annots
                if entry["subset"] in subsets
            ]
            if len(self.frame_annots) == 0:
                raise ValueError(
                    f"There are no frames in the '{subsets}' subsets!"
                )

            self._invalidate_indexes(filter_seq_annots=True)

        if len(self.limit_category_to) > 0:
            logger.info(
                f"Limiting dataset to categories: {self.limit_category_to}"
            )
            # pyre-ignore[16]
            self.seq_annots = {
                name: entry
                for name, entry in self.seq_annots.items()
                if entry.category in self.limit_category_to
            }

        # sequence filters
        for prefix in ("pick", "exclude"):
            orig_len = len(self.seq_annots)
            attr = f"{prefix}_sequence"
            arr = getattr(self, attr)
            if len(arr) > 0:
                logger.info(f"{attr}: {str(arr)}")
                self.seq_annots = {
                    name: entry
                    for name, entry in self.seq_annots.items()
                    if (name in arr) == (prefix == "pick")
                }
                logger.info(
                    "... filtered %d -> %d" % (orig_len, len(self.seq_annots))
                )

        if self.limit_sequences_to > 0:
            self.seq_annots = dict(
                islice(self.seq_annots.items(), self.limit_sequences_to)
            )

        # retain only frames from retained sequences
        self.frame_annots = [
            f
            for f in self.frame_annots
            if f["frame_annotation"].sequence_name in self.seq_annots
        ]

        self._invalidate_indexes()

        if self.n_frames_per_sequence > 0:
            logger.info(
                f"Taking max {self.n_frames_per_sequence} per sequence."
            )
            keep_idx = []
            # pyre-ignore[16]
            for seq, seq_indices in self._seq_to_idx.items():
                # infer the seed from the sequence name, this is reproducible
                # and makes the selection differ for different sequences
                seed = _seq_name_to_seed(seq) + self.seed
                seq_idx_shuffled = random.Random(seed).sample(
                    sorted(seq_indices), len(seq_indices)
                )
                keep_idx.extend(seq_idx_shuffled[: self.n_frames_per_sequence])

            logger.info(
                "... filtered %d -> %d"
                % (len(self.frame_annots), len(keep_idx))
            )
            self.frame_annots = [self.frame_annots[i] for i in keep_idx]
            self._invalidate_indexes(filter_seq_annots=False)
            # sequences are not decimated, so self.seq_annots is valid

        if self.limit_to > 0 and self.limit_to < len(self.frame_annots):
            logger.info(
                "limit_to: filtered %d -> %d"
                % (len(self.frame_annots), self.limit_to)
            )
            self.frame_annots = self.frame_annots[: self.limit_to]
            self._invalidate_indexes(filter_seq_annots=True)

    def _invalidate_indexes(self, filter_seq_annots: bool = False) -> None:
        # update _seq_to_idx and filter seq_meta according to frame_annots change
        # if filter_seq_annots, also uldates seq_annots based on the changed _seq_to_idx
        self._invalidate_seq_to_idx()

        if filter_seq_annots:
            # pyre-ignore[16]
            self.seq_annots = {
                k: v
                for k, v in self.seq_annots.items()
                # pyre-ignore[16]
                if k in self._seq_to_idx
            }

    def _invalidate_seq_to_idx(self) -> None:
        seq_to_idx = defaultdict(list)
        # pyre-ignore[16]
        for idx, entry in enumerate(self.frame_annots):
            seq_to_idx[entry["frame_annotation"].sequence_name].append(idx)
        # pyre-ignore[16]
        self._seq_to_idx = seq_to_idx

    def _resize_image(
        self, image, mode="bilinear"
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        image_height, image_width = self.image_height, self.image_width
        if image_height is None or image_width is None:
            # skip the resizing
            imre_ = torch.from_numpy(image)
            return imre_, 1.0, torch.ones_like(imre_[:1])
        # takes numpy array, returns pytorch tensor
        minscale = min(
            image_height / image.shape[-2],
            image_width / image.shape[-1],
        )
        imre = torch.nn.functional.interpolate(
            torch.from_numpy(image)[None],
            scale_factor=minscale,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
            recompute_scale_factor=True,
        )[0]
        # pyre-fixme[19]: Expected 1 positional argument.
        imre_ = torch.zeros(image.shape[0], self.image_height, self.image_width)
        imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
        # pyre-fixme[6]: For 2nd param expected `int` but got `Optional[int]`.
        # pyre-fixme[6]: For 3rd param expected `int` but got `Optional[int]`.
        mask = torch.zeros(1, self.image_height, self.image_width)
        mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
        return imre_, minscale, mask

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def get_frame_numbers_and_timestamps(
        self, idxs: Sequence[int]
    ) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for idx in idxs:
            # pyre-ignore[16]
            frame_annotation = self.frame_annots[idx]["frame_annotation"]
            out.append(
                (
                    frame_annotation.frame_number,
                    frame_annotation.frame_timestamp,
                )
            )
        return out

    def category_to_sequence_names(self) -> Dict[str, List[str]]:
        c2seq = defaultdict(list)
        # pyre-ignore
        for sequence_name, sa in self.seq_annots.items():
            c2seq[sa.category].append(sequence_name)
        return dict(c2seq)

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return self.eval_batches


def _seq_name_to_seed(seq_name) -> int:
    return int(hashlib.sha1(seq_name.encode("utf-8")).hexdigest(), 16)


def _load_image(path) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im


def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(
                np.array(depth_pil, dtype=np.uint16), dtype=np.float16
            )
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def _load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def _load_depth_mask(path: str) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = _load_1bit_png_mask(path)
    return m[None]  # fake feature channel


def _load_depth(path, scale_adjustment) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth file name "%s"' % path)

    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


def _load_mask(path) -> np.ndarray:
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)
    mask = mask.astype(np.float32) / 255.0
    return mask[None]  # fake feature channel


def _get_1d_bounds(arr) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1] + 1


def _get_bbox_from_mask(
    mask, thr, decrease_quant: float = 0.05
) -> Tuple[int, int, int, int]:
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(f"Empty masks_for_bbox (thr={thr}) => using full image.")

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def _get_clamp_bbox(
    bbox: torch.Tensor,
    box_crop_context: float = 0.0,
    image_path: str = "",
) -> torch.Tensor:
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    bbox = bbox.clone()  # do not edit bbox in place

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        raise ValueError(
            f"squashed image {image_path}!! The bounding box contains no pixels."
        )

    bbox[2:] = torch.clamp(
        bbox[2:], 2
    )  # set min height, width to 2 along both axes
    bbox_xyxy = _bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def _crop_around_box(tensor, bbox, impath: str = ""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox = _clamp_box_to_image_bounds_and_round(
        bbox,
        image_size_hw=tensor.shape[-2:],
    )
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"
    return tensor


# _clamp_box_to_image_bounds_and_round(clamp_bbox_xyxy,image_size_hw=tensor.shape[-2:],)


def _clamp_box_to_image_bounds_and_round(
    bbox_xyxy: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> torch.LongTensor:
    bbox_xyxy = bbox_xyxy.clone()
    bbox_xyxy[[0, 2]] = torch.clamp(bbox_xyxy[[0, 2]], 0, image_size_hw[-1])
    bbox_xyxy[[1, 3]] = torch.clamp(bbox_xyxy[[1, 3]], 0, image_size_hw[-2])
    if not isinstance(bbox_xyxy, torch.LongTensor):
        bbox_xyxy = bbox_xyxy.round().long()
    return bbox_xyxy  # pyre-ignore [7]


def _rescale_bbox(bbox: torch.Tensor, orig_res, new_res) -> torch.Tensor:
    assert bbox is not None
    assert np.prod(orig_res) > 1e-8
    # average ratio of dimensions
    rel_size = (new_res[0] / orig_res[0] + new_res[1] / orig_res[1]) / 2.0
    return bbox * rel_size


def _bbox_xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    wh = xyxy[2:] - xyxy[:2]
    xywh = torch.cat([xyxy[:2], wh])
    return xywh


def _bbox_xywh_to_xyxy(
    xywh: torch.Tensor, clamp_size: Optional[int] = None
) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


def _safe_as_tensor(data, dtype):
    if data is None:
        return None
    return torch.tensor(data, dtype=dtype)


# NOTE this cache is per-worker; they are implemented as processes.
# each batch is loaded and collated by a single worker;
# since sequences tend to co-occur within batches, this is useful.
@functools.lru_cache(maxsize=256)
def _load_pointcloud(
    pcl_path: Union[str, Path], max_points: int = 0
) -> Pointclouds:
    pcl = IO().load_pointcloud(pcl_path)
    if max_points > 0:
        pcl = pcl.subsample(max_points)

    return pcl
