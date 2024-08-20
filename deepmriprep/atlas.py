import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torchreg.utils import INTERP_KWARGS

from deepmriprep.utils import DEVICE, DATA_PATH, nifti_to_tensor
WARP_TEMPLATE = nib.as_closest_canonical(nib.load(f'{DATA_PATH}/templates/Template_4_GS.nii.gz'))
ATLASES = ('aal3', 'anatomy3', 'cobra', 'hammers', 'ibsr', 'julichbrain', 'lpba40',
           'mori', 'neuromorphometrics', 'suit', 'thalamic_nuclei', 'thalamus',
           'Schaefer2018_100Parcels_17Networks_order', 'Schaefer2018_200Parcels_17Networks_order',
           'Schaefer2018_400Parcels_17Networks_order', 'Schaefer2018_600Parcels_17Networks_order')


class AtlasRegistration:
    def __init__(self, no_gpu=False):
        self.device = torch.device('cpu' if no_gpu else DEVICE)

    def __call__(self, affine, warp_yx, atlas, t1_shape):
        atlas = pad_from_to(atlas, WARP_TEMPLATE)
        affine, header = atlas.affine, atlas.header
        atlas = nifti_to_tensor(atlas)[None, None].to(self.device)
        atlas = F.grid_sample(atlas, warp_yx, mode='nearest', align_corners=INTERP_KWARGS['align_corners'])[0, 0]
        return nib.Nifti1Image(atlas.cpu().numpy(), affine, header)


def get_volumes(atlas, p1_large, p2_large, p3_large):
    atlas = F.interpolate(atlas[None, None], p1_large.shape, mode='nearest')[0, 0]
    atlas = atlas.type(torch.uint8 if atlas.max() < 256 else torch.int16)
    atlas = atlas.flatten()
    mask = atlas > 0
    atlas = atlas[mask]
    idxs = atlas.argsort()
    atlas = atlas[idxs]
    splits = torch.argwhere(torch.gradient(atlas.type(torch.int16))[0] > 0)[1::2]
    regions = np.split(idxs, splits[:, 0])
    p_large = torch.stack([p1_large, p2_large, p3_large], dim=-1)
    p_large = p_large.view(-1, p_large.shape[-1])[mask]
    return np.array([[*p_large[region].sum(0).tolist(), len(region)] for region in regions])


def pad_from_to(from_img, to_vox_map):
    shape = shape_from_to(from_img, to_vox_map)
    from_x = from_img.get_fdata(dtype=np.float32)
    x = np.zeros(shape, dtype=np.float32)
    offset = from_img.affine[:3, 3] - to_vox_map.affine[:3, 3]
    offset /= from_img.affine[np.diag_indices(3)]
    crop = -offset.clip(max=0).astype(int)
    pad = offset.clip(min=0).astype(int)
    from_x = from_x[crop[0]:, crop[1]:, crop[2]:]
    x[pad[0]:pad[0] + from_x.shape[0], pad[1]:pad[1] + from_x.shape[1], pad[2]:pad[2] + from_x.shape[2]] = from_x
    affine = np.concatenate([from_img.affine[:, :3], to_vox_map.affine[:, 3:]], 1)
    return nib.Nifti1Image(x, affine, from_img.header)


def shape_from_to(from_img, to_vox_map):
    scale = to_vox_map.affine[np.diag_indices(3)] / from_img.affine[np.diag_indices(3)]
    shape = scale * (np.array(to_vox_map.shape[:3]) - 1) + 1
    return shape.astype(int)
