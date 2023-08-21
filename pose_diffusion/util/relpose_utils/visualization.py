"""
Adapted from https://github.com/google-research/google-research/blob/6e69c1d72617a0b98aa865901ea0249a62bfe6b1/implicit_pdf/evaluation.py
"""
import io
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch3d import transforms

from .misc import get_permutations


def visualize_so3_probabilities(
    rotations,
    probabilities,
    rotations_gt=None,
    ax=None,
    fig=None,
    display_threshold_probability=0,
    to_image=True,
    show_color_wheel=True,
    canonical_rotation=np.eye(3),
):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
        rotations: [N, 3, 3] tensor of rotation matrices
        probabilities: [N] tensor of probabilities
        rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
        ax: The matplotlib.pyplot.axis object to paint
        fig: The matplotlib.pyplot.figure object to paint
        display_threshold_probability: The probability threshold below which to omit
        the marker
        to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
        show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
        canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
        A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        eulers = transforms.matrix_to_euler_angles(torch.tensor(rotation), "ZXY")
        eulers = eulers.numpy()

        tilt_angle = eulers[0]
        latitude = eulers[1]
        longitude = eulers[2]

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=2500,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=4,
        )

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=112)
        ax = fig.add_subplot(111, projection="mollweide")
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e3
    eulers_queries = transforms.matrix_to_euler_angles(
        torch.tensor(display_rotations), "XYZ"
    )

    eulers_queries = transforms.matrix_to_euler_angles(
        torch.tensor(display_rotations), "ZXY"
    )
    eulers_queries = eulers_queries.numpy()

    tilt_angles = eulers_queries[:, 0]
    longitudes = eulers_queries[:, 2]
    latitudes = eulers_queries[:, 1]

    which_to_display = probabilities > display_threshold_probability

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, "o")
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(
                ax, rotation, "o", edgecolors=False, facecolors="#ffffff"
            )

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2.0 / np.pi),
    )

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.0
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap, shading="auto")
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
        ax.set_xticklabels(
            [
                r"90$\degree$",
                r"180$\degree$",
                r"270$\degree$",
                r"0$\degree$",
            ],
            fontsize=14,
        )
        ax.spines["polar"].set_visible(False)
        plt.text(
            0.5,
            0.5,
            "Tilt",
            fontsize=14,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    if to_image:
        return plot_to_image(fig)
    else:
        return fig


def visualize_two_rotations(
    rotations,
    rotations_gt,
    to_image=True,
):
    rotations_gt = rotations_gt.squeeze()
    rotations = rotations.squeeze()

    cmap = plt.cm.hsv

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        eulers = transforms.matrix_to_euler_angles(torch.tensor(rotation), "ZXY")
        eulers = eulers.numpy()

        tilt_angle = eulers[0]
        latitude = eulers[1]
        longitude = eulers[2]

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=500,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=4,
        )

    fig = plt.figure(figsize=(8, 4), dpi=112)
    ax = fig.add_subplot(111, projection="mollweide")

    # GT
    _show_single_marker(ax, rotations, "*")

    # Predicted
    _show_single_marker(ax, rotations_gt, "o")

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if to_image:
        return plot_to_image(fig)
    else:
        return fig


def plot_to_image(figure):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format="raw", dpi=112)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def make_visualization_pair(
    images1,
    images2,
    rotations,
    probabilities=None,
    gt_rotations=None,
    num_vis=16,
    model_id=None,
    category=None,
    ind1=None,
    ind2=None,
    regress=False,
):
    images1 = images1[:num_vis].detach().cpu().numpy().transpose(0, 2, 3, 1)
    images2 = images2[:num_vis].detach().cpu().numpy().transpose(0, 2, 3, 1)
    rotations = rotations[:num_vis].detach().cpu().numpy()
    probabilities = probabilities[:num_vis].detach().cpu().numpy()
    gt_rotations = gt_rotations[:num_vis].detach().cpu().numpy()

    visuals = []
    for i in range(len(images1)):
        image1 = unnormalize_image(cv2.resize(images1[i], (448, 448)))
        image2 = unnormalize_image(cv2.resize(images2[i], (448, 448)))

        if regress:
            so3_vis = visualize_two_rotations(rotations, gt_rotations)
        else:
            so3_vis = visualize_so3_probabilities(
                rotations=rotations[i],
                probabilities=probabilities[i],
                rotations_gt=gt_rotations[i],
                to_image=True,
                display_threshold_probability=(1 / len(probabilities[i])),
            )

        full_image = np.vstack((np.hstack((image1, image2)), so3_vis))
        visuals.append(full_image)
    return visuals


def make_visualization_n(
    images,
    rotations,
    probabilities,
    gt_rotations,
    num_vis=16,
    model_id=None,
    category=None,
    ind=None,
    regress=False,
):
    # Images:   (B, n_t, 3, 224, 224)
    # Queries:  (B, n_q, n_t - 1, 3, 3)
    # Logits:   (B, n_t - 1, n_q)

    num_tokens = images.shape[1]
    permutations = get_permutations(num_tokens)

    visuals2d = []
    for k, (i, j) in enumerate(permutations):
        visuals = make_visualization_pair(
            images1=images[:, i, :, :, :],
            images2=images[:, j, :, :, :],
            rotations=rotations[:, :, k, :, :],
            probabilities=probabilities[:, k, :],
            gt_rotations=gt_rotations[:, k],
            num_vis=num_vis,
            model_id=model_id,
            category=category,
            ind1=ind[:, i],
            ind2=ind[:, j],
            regress=regress,
        )
        visuals2d.append(visuals)

    return visuals2d


def view_color_coded_images_from_path(image_dir):
    cmap = plt.get_cmap("hsv")
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()

    def hidden(x):
        return not x.startswith(".")

    image_paths = sorted(os.listdir(image_dir))
    image_paths = list(filter(hidden, image_paths))
    image_paths = image_paths[0 : (min(len(image_paths), 8))]
    num_frames = len(image_paths)

    for i in range(num_rows * num_cols):
        if i < num_frames:
            img = np.asarray(Image.open(osp.join(image_dir, image_paths[i])))
            print(img.shape)
            axs[i].imshow(img)
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    return fig, num_frames


def view_color_coded_images_from_tensor(images):
    num_frames = images.shape[0]
    cmap = plt.get_cmap("hsv")
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            axs[i].imshow(unnormalize_image(images[i]))
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
