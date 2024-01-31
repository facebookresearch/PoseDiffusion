# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union
from multiprocessing import Pool

import hydra
import torch
from hydra.utils import instantiate, get_original_cwd
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
from util.metric import camera_to_rel_deg, calculate_auc
from util.train_util import (
    DynamicBatchSampler,
    VizStats,
    WarmupCosineRestarts,
    get_co3d_dataset,
    plotly_scene_visualization,
    set_seed_and_print,
    view_color_coded_images_for_visdom,
)


@hydra.main(config_path="../cfgs/", config_name="default_train")
def train_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False)

    # Print configuration and accelerator state
    accelerator.print("Model Config:", OmegaConf.to_yaml(cfg), accelerator.state)

    torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark if not cfg.debug else False
    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True

    set_seed_and_print(cfg.seed)

    # Visualization setup
    if accelerator.is_main_process:
        try:
            from visdom import Visdom

            viz = Visdom()
            # cams_show = {"ours_pred": pred_cameras, "ours_pred_aligned": pred_cameras_aligned, "gt_cameras": gt_cameras}
            # fig = plot_scene({f"{folder_path}": cams_show})
            # viz.plotlyplot(fig, env="visual", win="cams")
        except:
            print("Warning: please check your visdom connection for visualization")

    # Data loading
    dataset, eval_dataset = get_co3d_dataset(cfg)
    dataloader = get_dataloader(cfg, dataset)
    eval_dataloader = get_dataloader(cfg, eval_dataset, is_eval=True)

    accelerator.print("length of train dataloader is: ", len(dataloader))
    accelerator.print("length of eval dataloader is: ", len(eval_dataloader))

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False)
    model = model.to(accelerator.device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)

    lr_scheduler = WarmupCosineRestarts(
        optimizer=optimizer, T_0=cfg.train.restart_num, iters_per_epoch=len(dataloader), warmup_ratio=0.1
    )
    # torch.optim.lr_scheduler.OneCycleLR() can achieve similar performance

    # Accelerator setup
    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    start_epoch = 0
    if cfg.train.resume_ckpt:
        checkpoint = torch.load(cfg.train.resume_ckpt)
        try:
            model.load_state_dict(prefix_with_module(checkpoint), strict=True)
        except:
            model.load_state_dict(checkpoint, strict=True)

        accelerator.print(f"Successfully resumed from {cfg.train.resume_ckpt}")

    # metrics to record
    stats = VizStats(("loss", "lr", "sec/it", "Auc_30", "Racc_5", "Racc_15", "Racc_30", "Tacc_5", "Tacc_15", "Tacc_30"))
    num_epochs = cfg.train.epochs

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()

        set_seed_and_print(cfg.seed + epoch)

        # Evaluation
        if (epoch != 0) and (epoch % cfg.train.eval_interval == 0):
            # if (epoch%cfg.train.eval_interval ==0):
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            _train_or_eval_fn(
                model,
                eval_dataloader,
                cfg,
                optimizer,
                stats,
                accelerator,
                lr_scheduler,
                training=False,
                visualize=False,
            )
            accelerator.print(f"----------Finish the eval at epoch {epoch}----------")
        else:
            accelerator.print(f"----------Skip the eval at epoch {epoch}----------")

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        _train_or_eval_fn(
            model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, visualize=False
        )
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.exp_name}----------")
            stats.update({"lr": lr}, stat_set="train")
            stats.plot_stats(viz=viz, visdom_env=cfg.exp_name)
            accelerator.print(f"----------Done----------")

        if epoch % cfg.train.ckpt_interval == 0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")
            accelerator.save_state(output_dir=ckpt_path, safe_serialization=False)

            if accelerator.is_main_process:
                stats.save(cfg.exp_dir + "stats")

    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"), safe_serialization=False)

    return True


def _train_or_eval_fn(
    model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, visualize=False
):
    if training:
        model.train()
    else:
        model.eval()

    time_start = time.time()
    max_it = len(dataloader)

    stat_set = "train" if training else "eval"

    for step, batch in enumerate(dataloader):
        # data preparation
        images = batch["image"].to(accelerator.device)
        translation = batch["T"].to(accelerator.device)
        rotation = batch["R"].to(accelerator.device)
        fl = batch["fl"].to(accelerator.device)
        pp = batch["pp"].to(accelerator.device)

        if training and cfg.train.batch_repeat > 0:
            # repeat samples by several times
            # to accelerate training
            br = cfg.train.batch_repeat
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2).repeat(br, 1),
                R=rotation.reshape(-1, 3, 3).repeat(br, 1, 1),
                T=translation.reshape(-1, 3).repeat(br, 1),
                device=accelerator.device,
            )
            batch_size = len(images) * br
        else:
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2),
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=accelerator.device,
            )
            batch_size = len(images)

        if training:
            predictions = model(images, gt_cameras=gt_cameras, training=True, batch_repeat=cfg.train.batch_repeat)
            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
        else:
            with torch.no_grad():
                predictions = model(images, training=False)

        pred_cameras = predictions["pred_cameras"]

        # compute metrics
        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras, gt_cameras, accelerator.device, batch_size)

        # metrics to report
        Racc_5 = (rel_rangle_deg < 5).float().mean()
        Racc_15 = (rel_rangle_deg < 15).float().mean()
        Racc_30 = (rel_rangle_deg < 30).float().mean()

        Tacc_5 = (rel_tangle_deg < 5).float().mean()
        Tacc_15 = (rel_tangle_deg < 15).float().mean()
        Tacc_30 = (rel_tangle_deg < 30).float().mean()

        # also called mAA in some literature
        Auc_30 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30)

        predictions["Racc_5"] = Racc_5
        predictions["Racc_15"] = Racc_15
        predictions["Racc_30"] = Racc_30
        predictions["Tacc_5"] = Tacc_5
        predictions["Tacc_15"] = Tacc_15
        predictions["Tacc_30"] = Tacc_30
        predictions["Auc_30"] = Auc_30

        if visualize:
            # an example if trying to conduct visualization by visdom
            frame_num = images.shape[1]

            camera_dict = {"pred_cameras": {}, "gt_cameras": {}}

            for visidx in range(frame_num):
                camera_dict["pred_cameras"][visidx] = pred_cameras[visidx]
                camera_dict["gt_cameras"][visidx] = gt_cameras[visidx]

            fig = plotly_scene_visualization(camera_dict, frame_num)
            viz.plotlyplot(fig, env=cfg.exp_name, win="cams")

            show_img = view_color_coded_images_for_visdom(images[0])
            viz.images(show_img, env=cfg.exp_name, win="imgs")

        stats.update(predictions, time_start=time_start, stat_set=stat_set)
        if step % cfg.train.print_interval == 0:
            accelerator.print(stats.get_status_string(stat_set=stat_set, max_it=max_it))

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.train.clip_grad > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            lr_scheduler.step()

    return True


def get_dataloader(cfg, dataset, is_eval=False):
    """Utility function to get DataLoader."""
    prefix = "eval" if is_eval else "train"
    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=getattr(cfg.train, f"len_{prefix}"),
        max_images=cfg.train.max_images // (2 if is_eval else 1),
        images_per_seq=cfg.train.images_per_seq,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )
    dataloader.batch_sampler.drop_last = True
    dataloader.batch_sampler.sampler = dataloader.batch_sampler
    return dataloader


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


if __name__ == "__main__":
    train_fn()
