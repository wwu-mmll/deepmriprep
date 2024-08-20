import torch
import torch.nn.functional as F
import unittest
from torchreg.utils import INTERP_KWARGS

from deepmriprep.preprocess import WARP_TEMPLATE
from deepmriprep.register import WarpRegistration
from deepmriprep.utils import nifti_to_tensor
ALIGN_CORNERS = INTERP_KWARGS['align_corners']


class TestWarpRegistration(unittest.TestCase):
    def setUp(self):
        self.warp_register = WarpRegistration(no_gpu=True)
        self.template = nifti_to_tensor(WARP_TEMPLATE).permute(3, 0, 1, 2)[None, :2]
        self.id_grid = F.affine_grid(torch.eye(4)[None, :3], [1, 3, *self.template.shape[2:]], ALIGN_CORNERS)

    def test_warp(self, flow_std=5e-3, xx_mse=1e-3, xx_movedback_mse=5e-3):
        # Register Template -> Template
        _, flows, _ = self.warp_register(x=self.template, y=self.template, return_displacement=True)
        flow_xy_std = flows['xy_full'].std().item()
        flow_yx_std = flows['yx_full'].std().item()
        print(f'Template -> Template Flow STD XY: {flow_xy_std:.4f} (< {flow_std:.4f})')
        print(f'Template -> Template Flow STD YX: {flow_xy_std:.4f} (< {flow_std:.4f})')
        # For "real" registration (real tissue map -> template) flow STD is ~1e-2
        # NN should recognize that less (ideally 0) flow is needed: STD should be < 5e-3
        self.assertTrue(flow_xy_std < flow_std)
        self.assertTrue(flow_yx_std < flow_std)

        xy = F.grid_sample(self.template.clone(), flows['xy_full'] + self.id_grid, align_corners=ALIGN_CORNERS)
        yx = F.grid_sample(self.template.clone(), flows['yx_full'] + self.id_grid, align_corners=ALIGN_CORNERS)
        xy_mse = ((xy - self.template) ** 2).mean().item()
        yx_mse = ((yx - self.template) ** 2).mean().item()
        print(f'Template -> Template MSE XY: {xy_mse:.4f} (< {xx_mse:.4f})')
        print(f'Template -> Template MSE YX: {yx_mse:.4f} (< {xx_mse:.4f})')
        # For "real" registration (real tissue map -> template) MSE is ~1e-2
        # Template -> Template registration is easier: MSE should be < 1e-3
        self.assertTrue(xy_mse < xx_mse)
        self.assertTrue(yx_mse < xx_mse)

        x_moved = F.grid_sample(self.template.clone(), 5 * flows['xy_full'] + self.id_grid, align_corners=ALIGN_CORNERS)
        images, _, _ = self.warp_register(x=x_moved, y=self.template.clone(), return_displacement=False)
        xx_moved_back_mse = ((images['xy_full'] - self.template) ** 2).mean().item()
        print(f'Moved Template -> Template MSE: {xx_moved_back_mse:.4f} (< {flow_std:.4f})')
        # For "real" registration (real tissue map -> template) MSE is ~1e-2
        # Moved Template -> Template registration is easier: MSE should be < 5e-3
        self.assertTrue(xx_moved_back_mse < xx_movedback_mse)


class TestWarpRegistrationGPU(TestWarpRegistration):
    def setUp(self):
        self.warp_register = WarpRegistration(no_gpu=False)
        device = self.warp_register.device
        self.template = nifti_to_tensor(WARP_TEMPLATE).permute(3, 0, 1, 2)[None, :2].to(device)
        self.id_grid = F.affine_grid(torch.eye(4).to(device)[None, :3], [1, 3, *self.template.shape[2:]], ALIGN_CORNERS)


if __name__ == '__main__':
    unittest.main()
