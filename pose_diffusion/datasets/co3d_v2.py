import gzip
import json
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset
from torchvision import transforms

from util.relpose_utils.bbox import square_bbox
from util.relpose_utils.misc import get_permutations
from util.relpose_utils.normalize_cameras import first_camera_transform, normalize_cameras

from multiprocessing import Pool
import tqdm
from util.camera_transform import adjust_camera_to_bbox_crop_, adjust_camera_to_image_scale_, bbox_xyxy_to_xywh #, adjust_camera_to_bbox_crop_np, adjust_camera_to_image_scale_np

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

DEBUG_CATEGORIES = ["apple",
    # "chair",
    "teddybear",]

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Co3dDataset(Dataset):
    def __init__(
        self,
        category=("all",),
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        jitter_scale=[1.1, 1.2],
        jitter_trans=[-0.07, 0.07],
        # num_images=2,
        min_num_images = 20,
        img_size=224,
        # random_num_images=True,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=False,
        first_camera_rotation_only=False,
        mask_images=False,
        CO3D_DIR = None,
        CO3D_ANNOTATION_DIR = None,
        foreground_crop = True,
        preload_image = False,
        center_box = True,
        sort_by_filename = True,
    ):
        """
        Args:
            category (iterable): List of categories to use. If "all" is in the list,
                all training categories are used.
            num_images (int): Default number of images in each batch.
            normalize_cameras (bool): If True, normalizes cameras so that the
                intersection of the optical axes is placed at the origin and the norm
                of the first camera translation is 1.
            first_camera_transform (bool): If True, tranforms the cameras such that
                camera 1 has extrinsics [I | 0].
            first_camera_rotation_only (bool): If True, transforms the cameras such that
                camera 1 has identity rotation.
            mask_images (bool): If True, masks out the background of the images.
        """
        if "all" in category:
            category = TRAINING_CATEGORIES
        if "debug" in category:
            category = DEBUG_CATEGORIES
            
        category = sorted(category)

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"

        # if eval_time:
        #     torch.manual_seed(0)
        #     random.seed(0)
        #     np.random.seed(0)

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        
        if CO3D_DIR == None:
            raise NotImplementedError
            
        print(f"CO3D_DIR is {CO3D_DIR}")
        
        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR
        self.center_box = center_box
        self.split_name = split_name
        self.min_num_images = min_num_images
        self.foreground_crop = foreground_crop
        
        for c in category:
            annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < min_num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous
                    if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    # filtered_data.append(data)
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )
                    
                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

            print(annotation_file)
            print(counter)
        
        # aggregated_rotations, aggregated_category_map, aggregated_low_quality_translations = self.load_data(category, 2)

        self.sequence_list = list(self.rotations.keys())
        self.split = split
        self.debug = debug
        self.sort_by_filename = sort_by_filename

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size, antialias=True),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # ),
                ]
            )
        else:
            self.transform = transform
        
        if random_aug and not eval_time:
            self.jitter_scale = jitter_scale
            self.jitter_trans = jitter_trans
        else:
            self.jitter_scale = [1.15, 1.15]
            self.jitter_trans = [0, 0]

        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.first_camera_rotation_only = first_camera_rotation_only
        self.mask_images = mask_images

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)


    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))

        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop


    def __getitem__(self, idx_N):
        # print(idx_N)
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)


    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        if no_images:
            annos = [metadata[i] for i in ids]
            rotations = [torch.tensor(anno["R"]) for anno in annos]
            translations = [torch.tensor(anno["T"]) for anno in annos]
            batch = {}
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            return batch
        
        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x['filepath'])        

        images = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        # filepaths = []
        for anno in annos:
            filepath = anno["filepath"]

            image = Image.open(osp.join(self.CO3D_DIR, filepath)).convert("RGB")
            
            #############################################################################################
            # Copy image for debugging
            # import shutil
            # import os
            # os.makedirs(osp.join("/data/home/jianyuan/src/ReconstructionJ/pose/pose_diffusion/tmp", sequence_name), exist_ok=True)
            # shutil.copy(osp.join(self.CO3D_DIR, filepath), osp.join("/data/home/jianyuan/src/ReconstructionJ/pose/pose_diffusion/tmp",sequence_name))
            #############################################################################################

            if self.mask_images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                mask_path = osp.join(
                    self.CO3D_DIR, category, sequence_name, "masks", mask_name
                )
                mask = Image.open(mask_path).convert("L")

                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)
                
            # filepaths.append(filepath)
            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            focal_lengths.append(torch.tensor(anno["focal_length"]))
            principal_points.append(torch.tensor(anno["principal_point"]))

        crop_parameters = []
        images_transformed = []
        
        new_fls = []
        new_pps = []
        
        for i, (anno, image) in enumerate(zip(annos, images)):
                
            w, h = image.width, image.height
            
            if self.center_box:                
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                bbox = np.array([left, top, left+min_dim, top+min_dim])
            else:
                bbox = np.array(anno["bbox"])


            if not self.eval_time:
                bbox_jitter = self._jitter_bbox(bbox) 
            else:
                bbox_jitter = bbox
                
            bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox_jitter))
            focal_length_cropped, principal_point_cropped = adjust_camera_to_bbox_crop_(focal_lengths[i], principal_points[i], 
                                                                                        torch.FloatTensor(image.size), bbox_xywh)

            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)        

            new_focal_length, new_principal_point = adjust_camera_to_image_scale_(focal_length_cropped, 
                                                                            principal_point_cropped, 
                                                                            torch.FloatTensor(image.size), 
                                                                            torch.FloatTensor([self.img_size, self.img_size]))
            
            new_fls.append(new_focal_length); new_pps.append(new_principal_point)

            images_transformed.append(self.transform(image))
            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            cc = (2 * crop_center / min(h, w)) - 1
            crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

            crop_parameters.append(
                torch.tensor([-cc[0], -cc[1], crop_width]).float()
            )
            
        images = images_transformed

        # print(filepaths)
        # print(len(filepaths))
        batch = {
            # "filepath": filepaths,
            "seq_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": torch.tensor(ids),
        }
         
        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        if self.normalize_cameras:
            cameras = PerspectiveCameras(
                focal_length=new_fls.numpy(),
                principal_point=new_pps.numpy(),
                R=[data["R"] for data in annos],
                T=[data["T"] for data in annos],
            )

            normalized_cameras, _, _, _, _ = normalize_cameras(cameras)

            if self.first_camera_transform or self.first_camera_rotation_only:
                normalized_cameras = first_camera_transform(
                    normalized_cameras,
                    rotation_only=self.first_camera_rotation_only,
                )

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                raise RuntimeError

            batch["R"] = normalized_cameras.R
            batch["T"] = normalized_cameras.T
            batch["crop_params"] = torch.stack(crop_parameters)
            batch["R_original"] = torch.stack(
                [torch.tensor(anno["R"]) for anno in annos]
            )
            batch["T_original"] = torch.stack(
                [torch.tensor(anno["T"]) for anno in annos]
            )

            batch["fl"] = normalized_cameras.focal_length
            batch["pp"] = normalized_cameras.principal_point
            
            if torch.any(torch.isnan(batch["T"])):
                print(ids)
                print(category)
                print(sequence_name)
                raise RuntimeError

        else:
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            batch["crop_params"] = torch.stack(crop_parameters)
            
            batch["fl"] = new_fls
            batch["pp"] = new_pps
            
        # Add images
        if self.transform is None:
            batch["image"] = images
        else:
            batch["image"] = torch.stack(images)

        return batch









    ###############
    
    # TO BE REMOVED



    def load_data(self, category, num_load_workers):
        with Pool(processes=num_load_workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(self._load_annotation, category),
                    total=len(category),
                )
            )
            
        # Aggregating results
        aggregated_rotations = {}
        aggregated_category_map = {}
        aggregated_low_quality_translations = []
        
        for result in results:
            annotation_file, counter, rotations, category_map, low_quality_translations = result

            aggregated_rotations.update(rotations)
            aggregated_category_map.update(category_map)
            aggregated_low_quality_translations.extend(low_quality_translations)
            
        return aggregated_rotations, aggregated_category_map, aggregated_low_quality_translations

    def _load_annotation(self, c):
        annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{self.split_name}.jgz")
        with gzip.open(annotation_file, "r") as fin:
            annotation = json.loads(fin.read())

        counter = 0
        rotations = {}
        category_map = {}
        low_quality_translations = []
        
        for seq_name, seq_data in annotation.items():
            counter += 1
            if len(seq_data) < self.min_num_images:
                continue

            filtered_data = []
            category_map[seq_name] = c
            bad_seq = False
            for data in seq_data:
                # Make sure translations are not ridiculous
                if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                    bad_seq = True
                    low_quality_translations.append(seq_name)
                    break

                # Ignore all unnecessary information.
                filtered_data.append(
                    {
                        "filepath": data["filepath"],
                        "bbox": data["bbox"],
                        "R": data["R"],
                        "T": data["T"],
                        "focal_length": data["focal_length"],
                        "principal_point": data["principal_point"],
                    },
                )

            if not bad_seq:
                rotations[seq_name] = filtered_data

        print(annotation_file)
        print(counter)
            
        
        return annotation_file, counter, rotations, category_map, low_quality_translations
