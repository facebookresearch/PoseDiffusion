# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps=100,
        sampling_timesteps=None,
        beta_1=0.0001,
        beta_T=0.1,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="custom",
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
    ):
        super().__init__()

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start)"

        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.loss_type = loss_type
        self.objective = objective
        self.beta_schedule = beta_schedule
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k

        self.init_diff_hyper(
            self.timesteps,
            self.sampling_timesteps,
            self.beta_1,
            self.beta_T,
            self.loss_type,
            self.objective,
            self.beta_schedule,
            self.p2_loss_weight_gamma,
            self.p2_loss_weight_k,
        )

    def init_diff_hyper(
        self,
        timesteps,
        sampling_timesteps,
        beta_1,
        beta_T,
        loss_type,
        objective,
        beta_schedule,
        p2_loss_weight_gamma,
        p2_loss_weight_k,
    ):
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "custom":
            betas = torch.linspace(beta_1, beta_T, timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    # helper functions
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(self, x, t, z, x_self_cond=None):
        model_output = self.model(x, t, z)

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == "pred_x0":
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        x: torch.Tensor,  # B x N_x x dim
        t: int,
        z: torch.Tensor,
        x_self_cond=None,
        clip_denoised=False,
    ):
        preds = self.model_predictions(x, t, z)

        x_start = preds.pred_x_start

        if clip_denoised:
            raise NotImplementedError(
                "We don't clip the output because pose does not have a clear bound."
            )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,  # B x N_x x dim
        t: int,
        z: torch.Tensor,
        x_self_cond=None,
        clip_denoised=False,
    ):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            z=z,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )

        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        z: torch.Tensor,
    ):
        batch, device = shape[0], self.betas.device

        # Init here
        pose = torch.randn(shape, device=device)

        x_start = None

        pose_process = []
        pose_process.append(pose.unsqueeze(0))

        for t in reversed(range(0, self.num_timesteps)):
            pose, _ = self.p_sample(
                x=pose,
                t=t,
                z=z,
            )
            pose_process.append(pose.unsqueeze(0))

        return pose, torch.cat(pose_process)

    @torch.no_grad()
    def sample(self, shape, z):
        # TODO: add more variants
        sample_fn = self.p_sample_loop
        return sample_fn(shape, z=z)

    def p_losses(
        self,
        x_start,
        t,
        z=None,
        noise=None,
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.model(x, t, z)

        if self.objective == "pred_noise":
            target = noise
            x_0_pred = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == "pred_x0":
            target = x_start
            x_0_pred = model_out
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.loss_fn(model_out, target, reduction="none")

        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        return {
            "loss": loss,
            "noise": noise,
            "x_0_pred": x_0_pred,
            "x_t": x,
            "t": t,
        }

    def forward(self, pose, z=None, *args, **kwargs):
        b = len(pose)
        t = torch.randint(0, self.num_timesteps, (b,), device=pose.device).long()
        return self.p_losses(pose, t, z=z, *args, **kwargs)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")
