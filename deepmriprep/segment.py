import torch
import fill_voids
import numpy as np
import pandas as pd
import torch.nn.functional as F
from deepbet.utils import load_model
from torchreg.utils import INTERP_KWARGS, smooth_kernel
from spline_resize import resize

from .utils import DATA_PATH, DEVICE


class BrainSegmentation:
    def __init__(self, no_gpu=False, model_path=None, patch_model_path=None, shape=(336, 384, 336),
                 patch_shape=(128, 128, 128), sigma=30., fill_holes=True):
        self.device = torch.device('cpu' if no_gpu else DEVICE)
        model_path = f'{DATA_PATH}/models/segmentation_model.pt' if model_path is None else model_path
        patch_model_path = f'{DATA_PATH}/models/segmentation_patch_model.pt' if patch_model_path is None else patch_model_path
        self.model = load_model(model_path, no_gpu)
        self.patch_models = [load_model(patch_model_path.replace('patch', f'patch_{i}'), no_gpu) for i in range(18)]
        bounds = pd.read_csv(f'{DATA_PATH}/patches.csv')
        self.patch_slices = get_patch_slices(bounds.values, patch_shape)
        self.patch_weights = get_patch_weights(self.patch_slices, shape, patch_shape, sigma * torch.ones(3))
        self.fill_holes = fill_holes

    def __call__(self, x, mask):
        x = x[:, :, 1:-2, 15:-12, :-3]
        x = scale_intensity(x)
        p0 = self.run_model(x)
        p0 = self.run_patch_models(x, p0)
        if self.fill_holes:
            mask = p0[0, 0].cpu().numpy() > .9
            mask_filled = fill_voids.fill(mask)
            filled = (mask == 0) & (mask_filled == 1)
            p0[0, 0][filled] = 1.
        return F.pad(p0, (0, 3, 15, 12, 1, 2))

    def run_patch_models(self, x, p0):
        patch_p0 = torch.zeros(x.shape, dtype=x.dtype)
        for i, (patch, weight) in enumerate(zip(self.patch_slices, self.patch_weights)):
            patch_inp = torch.cat([x[patch].to(self.device), p0[patch]], dim=1)
            patch_inp = patch_inp.flip(2) if i >= 18 else patch_inp
            with torch.no_grad():
                p0_patch = self.patch_models[i % 18](patch_inp).cpu()
            p0_patch = p0_patch.flip(2) if i >= 18 else p0_patch
            patch_p0[patch] += p0_patch * weight
        return patch_p0

    def run_model(self, x, scale_factor=1.5):
        with torch.no_grad():
            p0 = self.model(resize(x, scale_factor=1 / scale_factor, align_corners=INTERP_KWARGS['align_corners'], mask_value=0))
        return resize(p0, scale_factor=scale_factor, align_corners=INTERP_KWARGS['align_corners'], mask_value=0)


def scale_intensity(x, low=.5, high=99.5):
    x_nonzero = x[x > 0].cpu()
    low = np.percentile(x_nonzero, low)
    high = np.percentile(x_nonzero, high)
    x = (x - low) / (high - low)
    x[x > 1] = 1 + torch.log10(x[x > 1])
    return x


def get_patch_slices(bounds, shape):
    return [[slice(None), slice(None)] + [slice(b, b + s) for b, s in zip(bound, shape)] for bound in bounds]


def get_patch_weights(patch_slices, img_shape, patch_shape, sigma):
    kernel = smooth_kernel(patch_shape, sigma)[None, None]
    weight_sum = torch.zeros(img_shape, device=kernel.device)[None, None]
    for s_patch in patch_slices:
        weight_sum[s_patch] += kernel
    return [kernel / weight_sum[patch] for patch in patch_slices]


class NoGMSegmentation:
    def __init__(self, no_gpu=False, model_path=None, shape=(336, 384, 336),
                 patch_shape=(128, 288, 256), sigma=30, bounds=((56, 28, 0), (152, 28, 0))):
        self.device = torch.device('cpu' if no_gpu else DEVICE)
        model_path = f'{DATA_PATH}/models/segmentation_nogm_model.pt' if model_path is None else model_path
        self.model = load_model(model_path, no_gpu)
        self.patch_slices = get_patch_slices(bounds, patch_shape)
        self.patch_weights = get_patch_weights(self.patch_slices, shape, patch_shape, sigma * torch.ones(3))

    def __call__(self, p0):
        nogm = self.run_model(p0[:, :, 1:-2, 15:-12, :-3])
        nogm = F.pad(nogm, (0, 3, 15, 12, 1, 2)).to(self.device)
        p = one_hot(p0)
        p = torch.stack([p[:, 2], p[:, 3], p[:, 1]], dim=1)
        p = self.apply_nogm(p, nogm)
        return p, nogm

    def run_model(self, p0):
        nogm = torch.zeros(p0.shape, dtype=p0.dtype)
        for i, (patch, weight) in enumerate(zip(self.patch_slices, self.patch_weights)):
            patch_inp = p0[patch]
            patch_inp = patch_inp.flip(2) if i > 0 else patch_inp
            with torch.no_grad():
                nogm_patch = F.softmax(self.model(patch_inp), dim=1)[:, 1:].cpu()
            nogm_patch = nogm_patch.flip(2) if i > 0 else nogm_patch
            nogm[patch] += nogm_patch * weight
        return nogm

    @staticmethod
    def apply_nogm(p, nogm):
        mask = (nogm > .5)[:, 0]
        gm = p[:, 0][mask]
        p[:, 1][mask] += gm / 2
        p[:, 2][mask] += gm / 2
        p[:, 0][mask] = 0
        return p


def one_hot(p0, n_cls=4):
    p0_clipped = p0.clip(max=n_cls - 1)[:, 0]
    p = torch.zeros((n_cls, *p0_clipped.shape), dtype=torch.float32, device=p0.device)
    for c in range(n_cls):
        mask = p0_clipped.gt(c - 1) & p0_clipped.le(c + 1)
        p[c, mask] = 1 - (p0_clipped[mask] - c).abs()
    return p.transpose(0, 1)
