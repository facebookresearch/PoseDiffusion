# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import numpy as np
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm
import torch.optim
from torch.utils.data import BatchSampler
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from pytorch3d.implicitron.tools.stats import Stats
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.implicitron.tools.vis_utils import get_visdom_connection
from datasets.co3d_v2 import Co3dDataset


logger = logging.getLogger(__name__)


class DynamicBatchSampler(BatchSampler):
    def __init__(self, num_sequences, dataset_len=1024, max_images=128, images_per_seq=(3, 20)):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence

        self.max_images = max_images
        self.images_per_seq = list(range(images_per_seq[0], images_per_seq[1]))
        self.num_sequences = num_sequences
        self.dataset_len = dataset_len

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            n_per_seq = np.random.choice(self.images_per_seq)
            # number of sequences
            n_seqs = self.max_images // n_per_seq

            # randomly select sequences
            chosen_seq = self._capped_random_choice(self.num_sequences, n_seqs)

            # get item
            batches = [(bidx, n_per_seq) for bidx in chosen_seq]
            yield batches

    def _capped_random_choice(self, x, size, replace: bool = True):
        len_x = x if isinstance(x, int) else len(x)
        if replace:
            return np.random.choice(x, size=size, replace=len_x < size)
        else:
            return np.random.choice(x, size=min(size, len_x), replace=False)

    def __len__(self):
        return self.dataset_len


class WarmupCosineRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, T_0, iters_per_epoch, T_mult=1, eta_min=0, warmup_ratio=0.1, warmup_lr_init=1e-7, last_epoch=-1
    ):
        # Similar to torch.optim.lr_scheduler.OneCycleLR()
        # But allow multiple cycles and a warmup
        self.T_0 = T_0 * iters_per_epoch
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_iters = int(T_0 * warmup_ratio * iters_per_epoch)
        self.warmup_lr_init = warmup_lr_init
        super(WarmupCosineRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_mult == 1:
            i_restart = self.last_epoch // self.T_0
            T_cur = self.last_epoch - i_restart * self.T_0
        else:
            n = int(math.log((self.last_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            T_cur = self.last_epoch - self.T_0 * (self.T_mult**n - 1) // (self.T_mult - 1)

        if T_cur < self.warmup_iters:
            warmup_ratio = T_cur / self.warmup_iters
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * warmup_ratio for base_lr in self.base_lrs]
        else:
            T_cur_adjusted = T_cur - self.warmup_iters
            T_i = self.T_0 - self.warmup_iters
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur_adjusted / T_i)) / 2
                for base_lr in self.base_lrs
            ]


def get_co3d_dataset(cfg):
    # Common dataset parameters
    common_params = {
        "category": (cfg.train.category,),
        "debug": False,
        "mask_images": False,
        "img_size": cfg.train.img_size,
        "normalize_cameras": cfg.train.normalize_cameras,
        "min_num_images": cfg.train.min_num_images,
        "CO3D_DIR": cfg.train.CO3D_DIR,
        "CO3D_ANNOTATION_DIR": cfg.train.CO3D_ANNOTATION_DIR,
        "first_camera_transform": cfg.train.first_camera_transform,
        "compute_optical": cfg.train.compute_optical,
        "color_aug": cfg.train.color_aug,
        "erase_aug": cfg.train.erase_aug,
    }

    # Create the train dataset
    dataset = Co3dDataset(**common_params, split="train")

    # Create the eval dataset
    eval_dataset = Co3dDataset(**common_params, split="test", eval_time=True)

    return dataset, eval_dataset


def get_co3d_dataset_test(cfg, category = None):
    # Common dataset parameters
    if category is None:
        category = cfg.test.category
        
    common_params = {
        "category": (category,),
        "debug": False,
        "mask_images": False,
        "img_size": cfg.test.img_size,
        "normalize_cameras": cfg.test.normalize_cameras,
        "min_num_images": cfg.test.min_num_images,
        "CO3D_DIR": cfg.test.CO3D_DIR,
        "CO3D_ANNOTATION_DIR": cfg.test.CO3D_ANNOTATION_DIR,
        "first_camera_transform": cfg.test.first_camera_transform,
        "compute_optical": cfg.test.compute_optical,
        "sort_by_filename": True,       # to ensure images are aligned with extracted matches
    }

    # Create the test dataset
    test_dataset = Co3dDataset(**common_params, split="test", eval_time=True)

    return test_dataset


def set_seed_and_print(seed):
    accelerate_set_seed(seed, device_specific=True)
    print(f"----------Seed is set to {np.random.get_state()[1][0]} now----------")


class VizStats(Stats):
    def plot_stats(self, viz=None, visdom_env=None, plot_file=None, visdom_server=None, visdom_port=None):
        # use the cached visdom env if none supplied
        if visdom_env is None:
            visdom_env = self.visdom_env
        if visdom_server is None:
            visdom_server = self.visdom_server
        if visdom_port is None:
            visdom_port = self.visdom_port
        if plot_file is None:
            plot_file = self.plot_file

        stat_sets = list(self.stats.keys())

        logger.debug(f"printing charts to visdom env '{visdom_env}' ({visdom_server}:{visdom_port})")

        novisdom = False

        if viz is None:
            viz = get_visdom_connection(server=visdom_server, port=visdom_port)

        if viz is None or not viz.check_connection():
            logger.info("no visdom server! -> skipping visdom plots")
            novisdom = True

        lines = []

        # plot metrics
        if not novisdom:
            viz.close(env=visdom_env, win=None)

        for stat in self.log_vars:
            vals = []
            stat_sets_now = []
            for stat_set in stat_sets:
                val = self.stats[stat_set][stat].get_epoch_averages()
                if val is None:
                    continue
                else:
                    val = np.array(val).reshape(-1)
                    stat_sets_now.append(stat_set)
                vals.append(val)

            if len(vals) == 0:
                continue

            lines.append((stat_sets_now, stat, vals))

        if not novisdom:
            for tmodes, stat, vals in lines:
                title = "%s" % stat
                opts = {"title": title, "legend": list(tmodes)}
                for i, (tmode, val) in enumerate(zip(tmodes, vals)):
                    update = "append" if i > 0 else None
                    valid = np.where(np.isfinite(val))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(val))
                    viz.line(
                        Y=val[valid],
                        X=x[valid],
                        env=visdom_env,
                        opts=opts,
                        win=f"stat_plot_{title}",
                        name=tmode,
                        update=update,
                    )

        if plot_file:
            logger.info(f"plotting stats to {plot_file}")
            ncol = 3
            nrow = int(np.ceil(float(len(lines)) / ncol))
            matplotlib.rcParams.update({"font.size": 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx + 1)
                plt.gca()
                for vali, vals_ in enumerate(vals):
                    c_ = c * (1.0 - float(vali) * 0.3)
                    valid = np.where(np.isfinite(vals_))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(vals_))
                    plt.plot(x[valid], vals_[valid], c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3] * 0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba("lightgray"))
                grid_params = {"visible": True, "color": gcolor}
                plt.grid(**grid_params, which="major", linestyle="-", linewidth=0.4)
                plt.grid(**grid_params, which="minor", linestyle="--", linewidth=0.2)
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            try:
                fig.savefig(plot_file)
            except PermissionError:
                warnings.warn("Cant dump stats due to insufficient permissions!")


def view_color_coded_images_for_visdom(images):
    num_frames, _, height, width = images.shape
    cmap = cm.get_cmap("hsv")
    bordered_images = []

    for i in range(num_frames):
        img = images[i]
        color = torch.tensor(np.array(cmap(i / num_frames))[:3], dtype=img.dtype, device=img.device)
        # Create colored borders
        thickness = 5  # Border thickness
        # Left border
        img[:, :, :thickness] = color[:, None, None]

        # Right border
        img[:, :, -thickness:] = color[:, None, None]

        # Top border
        img[:, :thickness, :] = color[:, None, None]

        # Bottom border
        img[:, -thickness:, :] = color[:, None, None]

        bordered_images.append(img)

    return torch.stack(bordered_images)


def plotly_scene_visualization(camera_dict, batch_size):
    fig = plot_scene(camera_dict, camera_scale=0.03, ncols=2)
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")

    for i in range(batch_size):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (batch_size)))
        fig.data[i + batch_size].line.color = matplotlib.colors.to_hex(cmap(i / (batch_size)))

    return fig
