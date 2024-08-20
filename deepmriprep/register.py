import torch
from torchreg.syn import SyNBase
from torchreg.metrics import dice_score, LinearElasticity
from deepbet.utils import load_model

from .utils import DATA_PATH, DEVICE
LIN_ELAST_FUNC = lambda x: LinearElasticity(mu=2., lam=1.)(x)


class WarpRegistration:
    def __init__(self, no_gpu=False, model_path=None):
        self.device = torch.device('cpu' if no_gpu else DEVICE)
        model_path = f'{DATA_PATH}/models/warp_model.pt' or model_path
        self.model = load_model(model_path, no_gpu)
        self.syn = SyNBase(time_steps=7)

    def __call__(self, x, y, x_cat=None, return_displacement=False):
        with torch.no_grad():
            v_xy, v_yx = self.model(torch.cat([x, y], dim=1))
        x_ = x.clone() if x_cat is None else torch.cat([x, x_cat], dim=1)
        y_ = y.clone() if x_cat is None else torch.cat([y, x_cat], dim=1)
        images, flows = self.syn.apply_flows(x_, y_, v_xy, v_yx)
        if not return_displacement:
            flows = {k: flow + self.syn._grid for k, flow in flows.items()}
        return images, flows, {'xy_velocity': v_xy, 'yx_velocity': v_yx}


class MSEAndDice:
    def __init__(self, alpha=.5):
        self.alpha = alpha

    def __call__(self, x, y):
        return (1 - self.alpha) * ((x[:, 1:] - y[:, 1:]) ** 2).mean() - self.alpha * dice_score(x[:, :1], y[:, :1])
