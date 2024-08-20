import torch
import torch.nn.functional as F
from nibabel.processing import SIGMA2FWHM
from torchreg.utils import smooth_kernel

from .utils import DEVICE


class Smoothing:
    def __init__(self, no_gpu=False, fwhms=(6, 8), resolution=1.5):
        self.device = torch.device('cpu' if no_gpu else DEVICE)
        self.fwhms = fwhms
        self.kernels = []
        ones = torch.ones(3, device=self.device)
        for fwhm in fwhms:
            kernel_size = round(fwhm * resolution / 2) * 2 + 1
            kernel = smooth_kernel(kernel_size=kernel_size * ones, sigma=fwhm / (resolution * SIGMA2FWHM) * ones)
            self.kernels.append(kernel)

    def __call__(self, x):
        xs = []
        for kernel in self.kernels:
            xs.append(F.conv3d(x, kernel[None, None], padding=kernel.shape[-1] // 2))
        return torch.cat(xs, 1)
