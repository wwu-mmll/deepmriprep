import glob
import torch
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from torchreg.utils import smooth_kernel
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = f'{Path(__file__).parents[1].resolve()}/data'


def nifti_to_tensor(nifti):
    return torch.from_numpy(nib.as_closest_canonical(nifti).get_fdata(dtype=np.float32).copy())


def nifti_volume(nifti):
    oriented_nifti = nib.as_closest_canonical(nifti)
    voxel_volume = np.prod(oriented_nifti.header.get_zooms()[:3])
    return voxel_volume * np.prod(np.array(oriented_nifti.shape[:3]))


def unsmooth_kernel(factor=3., sigma=.6, device='cpu'):
    # Hand-optimized factor and sigma for compensation of smoothing caused by affine transformation (inspired by CAT12)
    kernel = -factor * smooth_kernel(kernel_size=3 * [3], sigma=torch.tensor(3 * [sigma], device=device))
    kernel[1, 1, 1] = 0
    kernel[1, 1, 1] = 1 - kernel.sum()
    return kernel


def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_bids_t1w_files(bids_dir, liberal=False):
    filepaths = []
    for pattern in ['*/*/anat/*_T1w.nii.gz', '*/anat/*_T1w.nii.gz', '*/*/anat/*_T1w.nii', '*/anat/*_T1w.nii']:
        pattern = pattern.replace('w', '*') if liberal else pattern
        filepaths += glob.glob(f'{bids_dir}/{pattern}')
    filepaths = [fp for fp in filepaths if not '/derivatives/' in fp]
    return sorted(filepaths)
