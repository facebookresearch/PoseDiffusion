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
import io
import numpy as np
import torch
from PIL import Image
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
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
import multiprocessing
from multiprocessing import Manager
import boto3
import glob 
from pytorch3d.implicitron.dataset.types import dump_dataclass_jgzip

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import TypedDict

    class FrameAnnotsEntry(TypedDict):
        subset: Optional[str]
        frame_annotation: types.FrameAnnotation
else:
    FrameAnnotsEntry = dict


@registry.register
class RealEstate10KDataset(DatasetBase, ReplaceableBase):
    """
    """

    frame_annotations_type: ClassVar[
        Type[types.FrameAnnotation]
    ] = types.FrameAnnotation

    ####################

    seq_list: Any = None
    # split: str = "train"
    bucket:  str = 'fairusersglobal'
    
    ####################
    
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
    is_train: bool = False
    random_aug: bool = False
    crop_wo_mask: bool = False
    load_images: bool = True
    load_depths: bool = True
    load_depth_masks: bool = True
    load_masks: bool = True
    force_square: bool = True
    load_point_clouds: bool = False
    max_points: int = 0
    mask_images: bool = False
    mask_depths: bool = False
    image_height: Optional[int] = 256
    image_width: Optional[int] = 256
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    remove_empty_masks: bool = True
    n_frames_per_sequence: int = -1
    seed: int = 0
    sort_frames: bool = False
    eval_batches: Any = None
    eval_batch_index: Any = None
    
    rand_color: bool = False
    rand_era: bool = False
    reverse: bool = False
    
    process_sub: int = 0
    
    def __post_init__(self) -> None:
        # pyre-fixme[16]: `JsonIndexDataset` has no attribute `subset_to_image_path`.
        session = boto3.Session(profile_name='saml')

        self.save_dir = "/checkpoint/jianyuan/dataset/RealEstate10K_with_img"
        create_folder_if_not_exists(self.save_dir)
        # check_list = self.seq_list[:30]
        # split_num = 4       

        check_list = (self.seq_list)
        split_num = 20
        cpu_num = 16
        
        print("*"*100)
        print("Processing %d Sequences"%len(check_list))
        print("Using %d Slices"%split_num)
        print("Using %d CPUs"%cpu_num)

        self.check_list_slices = np.array_split(check_list, split_num)
        
        to_process_idx = np.arange(split_num)
        to_process_idx = to_process_idx[self.process_sub:(self.process_sub+1)]
        
        
        # for i in range(10): print(to_process_idx[100*i:100*(i+1)])
        # self.process_sub
        # import pdb;pdb.set_trace()
        
        to_process_list = np.array_split(to_process_idx, cpu_num)
        
        self._load_frame_and_seq(to_process_list[0])
        
        with multiprocessing.Pool(processes=cpu_num) as pool:
            frame_seq_maps = list(
                tqdm(
                    pool.imap(self._load_frame_and_seq, to_process_list),
                    total=cpu_num,
                )
            )
            
        print("All finished")
        import pdb;pdb.set_trace()

        frame_annots_list = []
        seq_annots_list = []
        import itertools
        for idx in range(len(frame_seq_maps)):
            frame_annots_list.append(frame_seq_maps[idx][0])
            seq_annots_list.append(frame_seq_maps[idx][1])
        
        combined_frame_annots_list = list(itertools.chain(*frame_annots_list))
        combined_seq_annots_list = list(itertools.chain(*seq_annots_list))
        
        

        # dump_dataclass_jgzip("/checkpoint/jianyuan/RealEstate10Kjgz"+str(split_num)+"/frames_anno.jgz", combined_frame_annots_list)
        # dump_dataclass_jgzip("/checkpoint/jianyuan/RealEstate10Kjgz"+str(split_num)+"/seq_anno.jgz", combined_seq_annots_list)
        # dump_dataclass_jgzip("/checkpoint/jianyuan/RealEstate10Kjgz/frames_anno.jgz", combined_frame_annots_list)
        # dump_dataclass_jgzip("/checkpoint/jianyuan/RealEstate10Kjgz/seq_anno.jgz", combined_seq_annots_list)


        import pdb;pdb.set_trace()
 
        self.frame_annots = [
            FrameAnnotsEntry(frame_annotation=a, subset=None) for a in frame_annots_list
        ]
        self.seq_annots = {entry.sequence_name: entry for entry in seq_annots_list}

        s3 = session.client('s3')
        self.s3 = s3

        # self._load_frames()
        # self._load_sequences()
        # if self.sort_frames:
        #     self._sort_frames()
        # self._load_subset_lists()
        # self._filter_db()  # also computes sequence indices
        # self._extract_and_set_eval_batches()
                
        logger.info(str(self))


        self.color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                p=0.65
            ),
            transforms.RandomGrayscale(p=0.15),
        ])
        
        self.rand_erase = transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

        self._invalidate_indexes(filter_seq_annots=True)

    def _load_frame_and_seq(self, process_idx):


        frame_annots_list = []
        seq_annots_list = []
        session = boto3.Session(profile_name='saml')
        s3session = session.client('s3')

        
        for pro_idx in process_idx:
            seqs_list = self.check_list_slices[pro_idx]
            print("-"*100)
            print("Processing Slice %d"%pro_idx)
            print(str(len(seqs_list))+ " Sequences")
            anno_dir = os.path.join(self.save_dir,"annos")
            
            create_folder_if_not_exists(anno_dir)
            frame_anno_name = os.path.join(anno_dir, "frames_anno_%08d.jgz"%pro_idx)
            seq_anno_name = os.path.join(anno_dir, "seq_anno_%08d.jgz"%pro_idx)
            
            print(frame_anno_name)
            print(seq_anno_name)
            
            if (not os.path.exists(frame_anno_name)) and (not os.path.exists(seq_anno_name)):
                for seq_name in seqs_list:
                    print(seq_name)
                    # try:
                    
                    objname = os.path.join(self.dataset_root, seq_name+'.txt')

                    response = s3session.get_object(Bucket=self.bucket, Key=objname)
                    file_content = response['Body'].read().decode('utf-8')
                    data = np.genfromtxt(io.StringIO(file_content), delimiter=' ', dtype=np.float64)
                    seq_anno = types.SequenceAnnotation(sequence_name = seq_name, category = "all",)
                    seq_annots_list.append(seq_anno)
                    
                    for frame_idx in range(len(data)):
                        print(frame_idx)
                        raw_line = data[frame_idx]
                        timestamp = raw_line[0]
                        intrinsics = raw_line[1:7]
                        extrinsics = raw_line[7:]
                        
                        # str(int(timestamp))
                        imgpath = objname.replace('.txt','/%s'%int(timestamp)+'.png')
                        imgresponse = s3session.get_object(Bucket=self.bucket, Key=imgpath)
                        image_array = np.array(Image.open(io.BytesIO(imgresponse['Body'].read())))

                        frame_name = os.path.basename(imgpath)
                        rel_name = extract_last_three_levels(imgpath)
                        output_folder = os.path.join(self.save_dir, rel_name)
                        
                        create_folder_if_not_exists(output_folder)
                        output_image_path = os.path.join(output_folder, frame_name)
                        
                        scale = 0.5
                        resize_image_scale_and_save(image_array, output_image_path, scale)
                        

                        cur_image = types.ImageAnnotation(
                                    path = output_image_path,
                                    size = image_array.shape[:2])
                        

                        Pmat = extrinsics.reshape(3, 4).astype('float64')
                        cur_viewpoint = types.ViewpointAnnotation(
                                    R = tuple(map(tuple, Pmat[:3,:3])),
                                    T = tuple(Pmat[:3,-1]),
                                    focal_length = tuple(intrinsics[:2]),
                                    principal_point = tuple(intrinsics[2:4]),)
                        cur_frame_anno = types.FrameAnnotation(
                                        sequence_name = seq_name,
                                        frame_number = frame_idx,
                                        frame_timestamp = timestamp,
                                        image = cur_image,
                                        viewpoint = cur_viewpoint,)
                        frame_annots_list.append(cur_frame_anno)

                dump_dataclass_jgzip(frame_anno_name, frame_annots_list)
                dump_dataclass_jgzip(seq_anno_name, seq_annots_list)
                print("Finished Slice %d"%pro_idx)
            else:
                print("Existing Slice %d"%pro_idx)
                
        print("This sub process is successful")
        return None

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
            FrameAnnotsEntry(frame_annotation=a, subset=None) for a in frame_annots_list
        ]
        

    def join(self, other_datasets: Iterable[DatasetBase]) -> None:
        """
        Join the dataset with other JsonIndexDataset objects.

        Args:
            other_datasets: A list of JsonIndexDataset objects to be joined
                into the current dataset.
        """
        if not all(isinstance(d, JsonIndexDatasetV2) for d in other_datasets):
            raise ValueError("This function can only join a list of JsonIndexDataset")
        # pyre-ignore[16]
        self.frame_annots.extend([fa for d in other_datasets for fa in d.frame_annots])
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
        # self._invalidate_indexes(filter_seq_annots=True)


    def __str__(self) -> str:
        # pyre-ignore[16]
        return f"JsonIndexDataset #frames={len(self.frame_annots)}"

    def __len__(self) -> int:
        # pyre-ignore[16]
        return len(self.frame_annots)

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

        raw_entry = self.frame_annots[index]
        entry = raw_entry["frame_annotation"]
        seq_annotation = self.seq_annots[entry.sequence_name]
        point_cloud = seq_annotation.point_cloud
                
        frame_data = FrameData(
            frame_number    =   _safe_as_tensor(entry.frame_number, torch.long),
            frame_timestamp =   _safe_as_tensor(entry.frame_timestamp, torch.float),
            sequence_name   =   entry.sequence_name,
            sequence_category       =  seq_annotation.category,
            camera_quality_score    =   _safe_as_tensor(
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

        scale = 1.0
        
        (
            frame_data.fg_probability,
            frame_data.mask_path,
            frame_data.bbox_xywh,
            clamp_bbox_xyxy,
            frame_data.crop_bbox_xywh,
            frame_data.image_rgb,
            frame_data.image_path,
            frame_data.mask_crop,
            scale,
        ) = self._load_crop_and_image(entry)
        
        if entry.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera_RealEstate10K(
                entry,
                scale,
                clamp_bbox_xyxy,
            )

        return frame_data


    def _load_crop_and_image(
        self, entry: types.FrameAnnotation):
        fg_probability = None
        full_path = None
        bbox_xywh = None
        clamp_bbox_xyxy = None
        crop_box_xywh = None
        
        img = self.s3_load_image(entry.image.path)
        
        if img.shape[-2:] != entry.image.size:
                    raise ValueError(
                        f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!")


        if self.box_crop:
            _, h, w = img.shape
            
            dis= abs(h-w) / 2
            
            if h>w:
                bbox_xywh = torch.tensor([0,dis,w,h])    
            else:
                bbox_xywh = torch.tensor([dis,0,w,h])  
                    
            bbox_xywh[2:] = bbox_xywh[2:].min()

            if self.is_train and self.random_aug:
                # random scale
                x_scaling, y_scaling = np.random.uniform(0.85,1.15, 2)
                bbox_xywh[2] = bbox_xywh[2] * x_scaling
                bbox_xywh[3] = bbox_xywh[3] * y_scaling
                # random corner
                offset_x = np.random.uniform(-0.1,0.1) * bbox_xywh[2]
                offset_y = np.random.uniform(-0.1,0.1) * bbox_xywh[3]
                bbox_xywh[0] = bbox_xywh[0] + offset_x
                bbox_xywh[1] = bbox_xywh[1] + offset_y

            clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(
                _get_clamp_bbox(
                    bbox_xywh,
                    image_path=entry.image.path,
                    box_crop_context=0.0,
                ),
                image_size_hw=tuple(img.shape[-2:]),
            )
            
            crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)
            img = _crop_around_box(img, clamp_bbox_xyxy, entry.image.path)
            
        # Resize Image
        img, scale, mask_crop = self._resize_image(img)

        return fg_probability, full_path, bbox_xywh, clamp_bbox_xyxy, crop_box_xywh, img, entry.image.path, mask_crop, scale


    def s3_load_image(self, path) -> np.ndarray:
        imgresponse = self.s3.get_object(Bucket=self.bucket, Key=path) 
        im = np.array(Image.open(io.BytesIO(imgresponse['Body'].read())))
        im = im.transpose((2, 0, 1))
        im = im.astype(np.float32) / 255.0
        return im



    def _get_pytorch3d_camera_RealEstate10K(
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
        principal_point -=0.5
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        # center_point_px = torch.tensor([entry.image.size[1]//2,entry.image.size[0]//2], dtype=torch.float)   
        # center_point_px = torch.tensor([entry.image.size[1]/2,entry.image.size[0]/2], dtype=torch.float)   
                
                
        half_image_size_wh_orig = (
            torch.tensor(list(reversed(entry.image.size)), dtype=torch.float) / 2.0
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
        
        

        focal_length_px = focal_length.clone() 
        focal_length_px[0] = focal_length_px[0] * entry.image.size[1]
        focal_length_px[1] = focal_length_px[1] * entry.image.size[0]
        
        
        
        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            principal_point_px -= clamp_bbox_xyxy[:2]
            # center_point_px -= clamp_bbox_xyxy[:2]

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
        
        # center_point = (
        #     half_image_size_output - center_point_px * scale
        # ) / half_min_image_size_output
        
        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            center_point=principal_point[None],
            fl_scale = fl_scale[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )


    # def _load_sequences(self) -> None:
    #     logger.info(f"Loading Co3D sequences from {self.sequence_annotations_file}.")
    #     local_file = self._local_path(self.sequence_annotations_file)
    #     with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
    #         seq_annots = types.load_dataclass(zipfile, List[types.SequenceAnnotation])
    #     if not seq_annots:
    #         raise ValueError("Empty sequences file!")
    #     # pyre-ignore[16]
    #     self.seq_annots = {entry.sequence_name: entry for entry in seq_annots}


    # def _sort_frames(self) -> None:
    #     # Sort frames to have them grouped by sequence, ordered by timestamp
    #     # pyre-ignore[16]
    #     self.frame_annots = sorted(
    #         self.frame_annots,
    #         key=lambda f: (
    #             f["frame_annotation"].sequence_name,
    #             f["frame_annotation"].frame_timestamp or 0,
    #         ),
    #     )

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
                (frame_annotation.frame_number, frame_annotation.frame_timestamp)
            )
        return out

    # def category_to_sequence_names(self) -> Dict[str, List[str]]:
    #     c2seq = defaultdict(list)
    #     # pyre-ignore
    #     for sequence_name, sa in self.seq_annots.items():
    #         c2seq[sa.category].append(sequence_name)
    #     return dict(c2seq)

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
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
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

    bbox[2:] = torch.clamp(bbox[2:], 2)  # set min height, width to 2 along both axes
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
def _load_pointcloud(pcl_path: Union[str, Path], max_points: int = 0) -> Pointclouds:
    pcl = IO().load_pointcloud(pcl_path)
    if max_points > 0:
        pcl = pcl.subsample(max_points)

    return pcl



def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        


def extract_last_three_levels(file_path):
    head, tail = os.path.split(file_path)
    head, tail = os.path.split(head)
    head, penultimate = os.path.split(head)
    head, third_last = os.path.split(head)
    return os.path.join(third_last, penultimate, tail)


def resize_image_scale_and_save(image_array, output_image_path, scale):
    image = Image.fromarray(np.uint8(image_array))
    width, height = image.size
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized_image = image.resize((resized_width, resized_height))
    resized_image.save(output_image_path)
