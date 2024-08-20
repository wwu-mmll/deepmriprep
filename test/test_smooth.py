import torch
import numpy as np
import nibabel as nib
import unittest

from deepmriprep.preprocess import WARP_TEMPLATE
from deepmriprep.smooth import Smoothing
from deepmriprep.utils import DEVICE, nifti_to_tensor


class TestSmoothing(unittest.TestCase):
    def setUp(self, no_gpu=False):
        self.no_gpu = no_gpu
        self.template_nifti = nib.as_closest_canonical(WARP_TEMPLATE)
        self.template_nifti.header.set_data_dtype(np.float32)
        self.template = nifti_to_tensor(self.template_nifti).permute(3, 0, 1, 2)[:, None]
        self.template = self.template.to(torch.device('cpu' if no_gpu else DEVICE))

    def test_shape(self):
        for i in range(1, len(self.template)):
            for fwhms in ((4,), (4, 2)):
                smooth = Smoothing(no_gpu=self.no_gpu, fwhms=fwhms)
                s = smooth(self.template[:i])
                self.assertTrue([*s.shape] == [i, len(fwhms), *self.template.shape[2:]])

    def test_values(self):
        for fwhm in [6, 8]:
            smooth = Smoothing(no_gpu=self.no_gpu, fwhms=(fwhm,))
            s = smooth(self.template)
            s = s[:, 0].permute(1, 2, 3, 0)
            s_nib = nib.processing.smooth_image(self.template_nifti, fwhm, mode='constant')
            np.testing.assert_allclose(s.cpu().numpy(), s_nib.get_fdata(dtype=np.float32), rtol=10, atol=.02)

    def test_channel_independence(self):
        smooth = Smoothing(no_gpu=self.no_gpu, fwhms=(4,))
        s4 = smooth(self.template)
        smooth = Smoothing(no_gpu=self.no_gpu, fwhms=(4, 2))
        s42 = smooth(self.template)
        self.assertTrue(torch.equal(s4, s42[:, :1]))


class TestSmoothingGPU(TestSmoothing):
    def setUp(self, use_gpu=True):
        super().setUp(use_gpu)


if __name__ == '__main__':
    unittest.main()
