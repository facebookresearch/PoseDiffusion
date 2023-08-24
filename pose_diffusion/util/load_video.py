import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
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
import skvideo.io

def load_videos(
    folder_path: str, image_size: int = 224, mode: str = "bilinear"
) -> torch.Tensor:
    
    videodata = skvideo.io.vread(folder_path)
    videodata = videodata.transpose((1, 0, 2, 3))
    videodata = downsample_video(videodata)


    trarget_folder = folder_path.replace(".mp4", "")
    
    if not os.path.exists(trarget_folder):
        os.makedirs(trarget_folder)

    for i in range(videodata.shape[0]):
        img = Image.fromarray(videodata[i].astype('uint8'))
        # create a filename for each image
        filename = os.path.join(trarget_folder, f"image_{i}.png")
        # save the image
        img.save(filename)

    
    images = []
    bboxes_xyxy = []
    scales = []
    

    for num in range(len(videodata)):
        image = videodata[num]
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0

        image, bbox_xyxy, min_hw = _center_crop_square(image)
        minscale = image_size / min_hw

        imre = F.interpolate(
            torch.from_numpy(image)[None],
            size=(image_size, image_size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )[0]

        images.append(imre.numpy())
        bboxes_xyxy.append(bbox_xyxy.numpy())
        scales.append(minscale)

    images_tensor = torch.from_numpy(np.stack(images))

    # assume all the images have the same shape for GGS
    image_info = {
        "size": (min_hw, min_hw),
        "bboxes_xyxy": np.stack(bboxes_xyxy),
        "resized_scales": np.stack(scales),
    }
    return images_tensor, image_info


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[1:]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped_image = image[:, top : top + min_dim, left : left + min_dim]

    # bbox_xywh: the cropped region
    bbox_xywh = torch.tensor([left, top, min_dim, min_dim])

    # the format from xywh to xyxy
    bbox_xyxy = _clamp_box_to_image_bounds_and_round(
        _get_clamp_bbox(
            bbox_xywh,
            box_crop_context=0.0,
        ),
        image_size_hw=(h, w),
    )
    return cropped_image, bbox_xyxy, min_dim


def _get_clamp_bbox(
    bbox: torch.Tensor,
    box_crop_context: float = 0.0,
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
            f"squashed image!! The bounding box contains no pixels."
        )

    bbox[2:] = torch.clamp(
        bbox[2:], 2
    )  # set min height, width to 2 along both axes
    bbox_xyxy = _bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def _bbox_xywh_to_xyxy(
    xywh: torch.Tensor, clamp_size: Optional[int] = None
) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


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



def downsample_video(video, ratio=1/3):
    """
    Downsample a video represented as a numpy array.

    Parameters:
    video: numpy array
        The original video. Expected shape is (num_frames, height, width, num_channels)
    
    ratio: float
        The ratio to downsample the video by.

    Returns:
    downsampled_video: numpy array
        The downsampled video. 
    """
    # Get the original dimensions
    num_frames, height, width, num_channels = video.shape

    # Calculate the new dimensions
    new_num_frames = int(num_frames * ratio)
    
    # Initialize an empty array for the downsampled video
    downsampled_video = np.empty((new_num_frames, height, width, num_channels))

    # For each frame in the original video
    for i in range(new_num_frames):
        # Calculate the corresponding frame in the original video
        original_frame = int(i / ratio)

        # Downsample the frame and store it in the new video array
        downsampled_video[i] = video[original_frame]

    return downsampled_video


if __name__ == "__main__":
    # Example usage:
    folder_path = "path/to/your/folder"
    image_size = 224
    images_tensor = load_and_preprocess_images(folder_path, image_size)
    print(images_tensor.shape)
