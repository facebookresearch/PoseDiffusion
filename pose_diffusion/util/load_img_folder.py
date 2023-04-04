import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _load_image(path) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[1:]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped_image = image[:, top : top + min_dim, left : left + min_dim]
    return cropped_image


def load_and_preprocess_images(
    folder_path: str, image_size: int = 224, mode: str = "bilinear"
) -> torch.Tensor:
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_paths.sort()

    images = []
    for path in image_paths:
        image = _load_image(path)
        image = _center_crop_square(image)

        imre = F.interpolate(
            torch.from_numpy(image)[None],
            size=(image_size, image_size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )[0]

        images.append(imre.numpy())

    images_tensor = torch.from_numpy(np.stack(images))
    return images_tensor


if __name__ == "__main__":
    # Example usage:
    folder_path = "path/to/your/folder"
    image_size = 224
    images_tensor = load_and_preprocess_images(folder_path, image_size)
    print(images_tensor.shape)
