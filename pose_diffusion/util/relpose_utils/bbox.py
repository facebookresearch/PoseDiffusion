"""
Adapted from code originally written by David Novotny.
"""
import numpy as np
import pytorch3d
import torch


def mask_to_bbox(mask, thresh=0.4):
    """
    xyxy format
    """
    mask = mask > thresh
    if not np.any(mask):
        return []
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1]


def pixel_to_ndc(coords, image_size):
    """
    Converts pixel coordinates to normalized device coordinates (Pytorch3D convention
    with upper left = (1, 1)) for a square image.

    Args:
        coords: Pixel coordinates UL=(0, 0), LR=(image_size, image_size).
        image_size (int): Image size.

    Returns:
        NDC coordinates UL=(1, 1) LR=(-1, -1).
    """
    coords = np.array(coords)
    return 1 - coords / image_size * 2


def ndc_to_pixel(coords, image_size):
    """
    Converts normalized device coordinates to pixel coordinates for a square image.
    """
    coords = np.array(coords)
    return (1 - coords) * image_size / 2


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox


def compute_relative_rotation(crop_center, principal_point, focal_length):
    """
    Computes the relative rotation between the original image and the cropped image.

    Args:
        crop_center (np.ndarray): Center of the crop in the original image NDC (2,).
        principal_point (np.ndarray): Principal point of the original image NDC (2,).
        focal_length (float): Focal length of the original image.

    Returns:
        Rotation matrix (3, 3).
    """
    optical_axis = np.array([0, 0, focal_length])
    optical_axis_new = np.array(
        [
            crop_center[0] - principal_point[0],
            crop_center[1] - principal_point[1],
            focal_length,
        ]
    )
    optical_axis = optical_axis / np.linalg.norm(optical_axis)
    optical_axis_new = optical_axis_new / np.linalg.norm(optical_axis_new)
    rotation_axis = np.cross(optical_axis, optical_axis_new)
    if np.linalg.norm(rotation_axis) < 1e-6:
        rotation_axis = np.zeros(3)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_magnitude = np.arccos(np.dot(optical_axis, optical_axis_new).clip(-1, 1))
    axis_angle = torch.from_numpy(rotation_axis * rotation_magnitude)
    return pytorch3d.transforms.axis_angle_to_matrix(axis_angle).float().numpy()


def compute_projection(crop_center, query_point, principal_point, focal_length):
    """
    Maps coordinates from original image to a new image with the optical axis placed
    through the crop center.

    Args:
        crop_center (np.ndarray): Center of the crop in the original image NDC (2,).
        query_point (np.ndarray): Point in the original image NDC (2,).
        principal_point (np.ndarray): Principal point of the original image NDC (2,).
        focal_length (float): Focal length of the original image.

    Returns:
        Point in the new image NDC (2,).
    """
    # P3D convention, top left is (1, 1)
    # Assumption here is that top left is (-1, 1)
    crop_center = np.array(crop_center).copy()
    query_point = np.array(query_point).copy()
    principal_point = np.array(principal_point).copy()

    crop_center[0] = crop_center[0] * -1
    query_point[0] = query_point[0] * -1
    principal_point[0] = principal_point[0] * -1

    camera_center = np.array([principal_point[0], principal_point[1], 0])
    principal_point = np.array([principal_point[0], principal_point[1], focal_length])
    optical_axis_og = np.array([0, 0, focal_length])
    optical_axis_og = optical_axis_og / np.linalg.norm(optical_axis_og)
    optical_axis_new = (
        np.array([crop_center[0], crop_center[1], focal_length]) - camera_center
    )
    optical_axis_new = optical_axis_new / np.linalg.norm(optical_axis_new)
    principal_point_new = focal_length * optical_axis_new + camera_center

    a = np.array([query_point[0], query_point[1], focal_length]) - camera_center
    a = a / np.linalg.norm(a)
    d = optical_axis_new.dot(
        principal_point_new - camera_center
    ) / optical_axis_new.dot(a)
    a_projected = d * a + camera_center

    a_x, a_y, a_z = a_projected
    p_x, p_y, p_z = principal_point_new

    x = np.sign(a_x - p_x) * np.sqrt((a_x - p_x) ** 2 + (a_z - p_z) ** 2)
    y = np.sign(a_y - p_y) * np.sqrt((a_y - p_y) ** 2 + (a_z - p_z) ** 2)

    x = x * -1
    return (x, y)


def compute_projection_inv(crop_center, query_point, principal_point, focal_length):
    crop_center = np.array(crop_center).copy()
    query_point = np.array(query_point).copy()
    principal_point = np.array(principal_point).copy()

    crop_center[0] = crop_center[0] * -1
    query_point[0] = query_point[0] * -1
    principal_point[0] = principal_point[0] * -1

    camera_center = np.array([principal_point[0], principal_point[1], 0])
    principal_point = np.array([principal_point[0], principal_point[1], focal_length])
    optical_axis_og = np.array([0, 0, focal_length])
    optical_axis_og = optical_axis_og / np.linalg.norm(optical_axis_og)
    optical_axis_new = (
        np.array([crop_center[0], crop_center[1], focal_length]) - camera_center
    )
    optical_axis_new = optical_axis_new / np.linalg.norm(optical_axis_new)
    principal_point_new = focal_length * optical_axis_new + camera_center

    x_axis = np.cross([0, 1, 0], optical_axis_new)
    y_axis = np.cross(optical_axis_new, x_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    a = principal_point_new + query_point[0] * x_axis + query_point[1] * y_axis
    d = optical_axis_og.dot(principal_point - camera_center) / optical_axis_og.dot(
        a - camera_center
    )
    x, y, _ = camera_center + d * (a - camera_center)

    x *= -1
    return (x, y)


def find_coeffs(pa, pb):
    """
    Find coefficients of homography from points pa to pb.
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)
