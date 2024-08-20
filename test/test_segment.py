import torch
import unittest
import nibabel as nib

from deepmriprep.segment import one_hot, scale_intensity, NoGMSegmentation, BrainSegmentation
from deepmriprep.utils import DEVICE, DATA_PATH, nifti_to_tensor


class TestBrainSegmentation(unittest.TestCase):
    def setUp(self, no_gpu=True):
        self.segment = BrainSegmentation(no_gpu)
        self.template = nib.load(f'{DATA_PATH}/templates/Template_05mm_bet.nii.gz')
        self.template = nifti_to_tensor(self.template)[None, None]
        self.template = self.template.to(torch.device(DEVICE if no_gpu else 'cpu'))

    def test_run(self, mse_threshold=5e-2):
        p0 = self.segment(self.template)
        mse = ((p0 - 3 * self.template) ** 2).mean().item()
        print(f'Brain segment total run MSE(NN(p0_template), p0_template): {mse:.3f} (< {mse_threshold:.3f})')
        self.assertTrue(mse < mse_threshold)

    def test_run_model(self, mse_threshold=5e-2):
        p0_template = self.template[:, :, 1:-2, 15:-12, :-3]
        p0 = scale_intensity(p0_template)
        p0 = self.segment.run_model(p0)
        mse = ((p0 - 3 * p0_template) ** 2).mean().item()
        print(f'Brain segment model run MSE(NN(p0_template), p0_template): {mse:.3f} (< {mse_threshold:.3f})')
        self.assertTrue(mse < mse_threshold)

    def test_run_patch_models(self, mse_threshold=5e-2):
        p0_template = self.template[:, :, 1:-2, 15:-12, :-3]
        p0 = scale_intensity(p0_template)
        p0 = self.segment.run_patch_models(x=p0, p0=p0)
        mse = ((p0 - 3 * p0_template) ** 2).mean().item()
        print(f'Brain segment patch models run MSE(NN(p0_template), p0_template): {mse:.3f} (< {mse_threshold:.3f})')
        self.assertTrue(mse < mse_threshold)


class TestBrainSegmentationGPU(TestBrainSegmentation):
    def setUp(self, no_gpu=False):
        super().setUp(no_gpu)


class TestNoGMSegmentation(unittest.TestCase):
    def setUp(self, no_gpu=True):
        self.segment = NoGMSegmentation(no_gpu)
        self.template = nib.load(f'{DATA_PATH}/templates/Template_05mm_bet.nii.gz')
        self.template = nifti_to_tensor(self.template)[None, None]
        self.template = 3 * self.template.to(torch.device('cpu' if no_gpu else DEVICE))

    def test_run(self, mse_threshold=1e-3):
        p, nogm = self.segment(self.template)
        p_template = one_hot(self.template, n_cls=4)
        p_template = torch.stack([p_template[:, 2], p_template[:, 3], p_template[:, 1]], dim=1)
        mse = ((p - p_template) ** 2).mean()
        print(f'NOGM total run MSE(one_hot(NN(template)), one_hot(template)): {mse:.4f} (< {mse_threshold:.4f})')
        self.assertTrue(mse < mse_threshold)

    def test_run_model(self, nogm_mean=3e-3, tol=5e-4):
        cropped_template = self.template[:, :, 1:-2, 15:-12, :-3]
        nogm = self.segment.run_model(cropped_template)
        self.assertTrue(nogm.shape == cropped_template.shape)
        nogm_mean_pred = nogm.mean().item()
        print(f'NOGM model run mean voxel value: (< {nogm_mean - tol:.4f}) {nogm_mean_pred:.4f} (< {nogm_mean + tol:.4f})')
        self.assertTrue(nogm_mean - tol < nogm_mean_pred < nogm_mean + tol)


class TestNoGMSegmentationGPU(TestNoGMSegmentation):
    def setUp(self, no_gpu=False):
        super().setUp(no_gpu)


if __name__ == '__main__':
    unittest.main()
