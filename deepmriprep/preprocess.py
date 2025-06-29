import torch
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import spline_resize as sr
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from deepbet import BrainExtraction
from torchreg.affine import AffineRegistration
from deepbet.utils import is_file_broken, reoriented_nifti
from torchreg.utils import INTERP_KWARGS, jacobi_determinant

from .segment import BrainSegmentation, NoGMSegmentation
from .register import WarpRegistration, MSEAndDice
from .smooth import Smoothing
from .atlas import ATLASES, get_volumes, shape_from_to, AtlasRegistration
from .utils import (DEVICE, DATA_PATH, seed_all, nifti_volume, nifti_to_tensor,
                    find_bids_t1w_files, download_missing_models)
AFFINE_TEMPLATE = nib.load(f'{DATA_PATH}/templates/Template_05mm_bet.nii.gz')
WARP_TEMPLATE = nib.load(f'{DATA_PATH}/templates/Template_4_GS.nii.gz')
BET_MODEL_PATHS = {'model_path': f'{DATA_PATH}/models/brain_extraction_model.pt',
                   'bbox_model_path': f'{DATA_PATH}/models/brain_extraction_bbox_model.pt'}
BET_KWARGS = {'threshold': .5, 'n_dilate': 0}
AFFINE_KWARGS = {'scales': (24, 12), 'iterations': (500, 100), 'with_shear': True, 'verbose': False,
                 'dissimilarity_function': MSEAndDice(alpha=.5), 'learning_rate': 1e-3, 'padding_mode': 'zeros'}
SEGMENT_BRAIN_KWARGS = {'shape': (336, 384, 336), 'patch_shape': (128, 128, 128), 'sigma': 30.}
SEGMENT_NOGM_KWARGS = {'shape': (336, 384, 336), 'patch_shape': (128, 288, 256), 'sigma': 30.,
                       'bounds': ((56, 28, 0), (152, 28, 0))}
SMOOTH_KWARGS = {'fwhms': (6, 8), 'resolution': 1.5}
ATLASES_AFFINE = tuple([f'{atlas}_affine' for atlas in ATLASES])
ATLAS_VOLUMES = tuple([f'{atlas}_volumes' for atlas in ATLASES])
IO = {'bet': {'input': ('t1',),
              'output': ('brain', 'mask', 'bet_tiv')},
      'affine': {'input': ('brain', 'mask'),
                 'output': ('brain_large', 'mask_large', 'affine',
                            'zoom', 'translation', 'shear', 'rotation', 'affine_loss')},
      'segment_brain': {'input': ('brain_large', 'mask', 'affine', 'mask_large'),
                        'output': ('p0_large', 'p0')},
      'segment_nogm': {'input': ('p0_large', 'affine', 't1'),
                       'output': ('nogm', 'p1', 'p2', 'p3', 'p1_large', 'p2_large', 'p3_large',
                                  'p1_affine', 'p2_affine', 'p3_affine',
                                  'gmv', 'wmv', 'csfv', 'tiv', 'rel_gmv', 'rel_wmv', 'rel_csfv', 'wj_affine')},
      'warp': {'input': ('p0_large', 'p1_affine', 'p2_affine', 'wj_affine'),
               'output': ('warp_xy', 'warp_yx', 'warp_mse', 'iy_', 'wj_', 'wp0', 'wp1', 'wp2', 'mwp1', 'mwp2',
                          'v_xy', 'v_yx')},  # 'y_', 'iy_'
      'smooth': {'input': ('mwp1', 'mwp2'),
                 'output': ('s6mwp1', 's6mwp2', 's8mwp1', 's8mwp2')},
      'atlas': {'input': ('t1', 'affine', 'warp_yx', 'p1_large', 'p2_large', 'p3_large'),
                'output': ATLASES + ATLASES_AFFINE + ATLAS_VOLUMES}}
OUTPUTS = {'all': sum([list(v['output']) for v in IO.values()], []),
           'vbm': ['mwp1', 'mwp2', 's6mwp1', 's8mwp1', 'tiv', 'affine_loss', 'warp_mse'],
           'rbm': ATLASES + ATLAS_VOLUMES + ('affine_loss', 'warp_mse'),
           'scalar': ['bet_tiv', 'gmv', 'wmv', 'csfv', 'tiv', 'rel_gmv', 'rel_wmv', 'rel_csfv', 'wj_affine',
                      'affine_loss', 'warp_mse'],
           'tensor': ['affine', 'zoom', 'translation', 'shear', 'rotation'] + list(ATLAS_VOLUMES)}
OUTPUTS['csv'] = OUTPUTS['scalar'] + OUTPUTS['tensor']
DIR_FORMATS = ['sub', 'mod', 'cat', 'flat']


def run_preprocess(input_paths=None, bids_dir=None, output_paths=None, output_dir=None, outputs='vbm', dir_format='sub',
                    no_gpu=False, progress_bar_func=None, skip_broken=True, skip_unprocessed=True, **kwargs):
    table_path = output_dir if bids_dir is None else f'{bids_dir}/derivatives/deepmriprep'
    table_path = str(Path.cwd()) if table_path is None else table_path
    table_path += '/deepmriprep_outputs.csv'
    df = get_path_dataframe(input_paths, bids_dir, output_paths, output_dir, outputs, dir_format)
    value_columns = [f'{c}_value' for c in df if c in OUTPUTS['scalar']]
    table = df.assign(**{c: None for c in value_columns})
    prep = Preprocess(no_gpu=no_gpu, **kwargs)
    progress_bar = tqdm(enumerate(df.iterrows()), disable=len(df) == 1, total=len(df))
    for i, (in_path, out_paths) in progress_bar:
        if progress_bar_func is not None and i > 0:
            progress_bar_func(progress_bar)
        out_paths = out_paths.to_dict()
        if skip_broken and is_file_broken(in_path):
            warnings.warn(f'Skipped file {in_path} (for error messages, use skip_broken=False)', Warning)
        else:
            for o_path in out_paths.values():
                Path(o_path).parent.mkdir(exist_ok=True, parents=True)
            output = prep.run(in_path, out_paths, run_all=False, skip_unprocessed=skip_unprocessed)
            if table.shape[1] > df.shape[1] and output is not None:
                table.loc[in_path, value_columns] = [output[c[:-6]].values.item() for c in table.columns[df.shape[1]:]]
            table.to_csv(table_path)
    return table


class Preprocess:
    def __init__(self, no_gpu=False, affine_template=AFFINE_TEMPLATE, warp_template=WARP_TEMPLATE,
                 bet_model_paths=None, bet_kwargs=None, affine_kwargs=None, segment_brain_kwargs=None,
                 segment_nogm_kwargs=None, warp_model_path=None, smooth_kwargs=None):
        download_missing_models()
        self.device = torch.device('cpu' if no_gpu else DEVICE)
        self.affine_template = affine_template
        self.affine_template_metadata = {'affine': affine_template.affine, 'header': affine_template.header}
        self.warp_template = warp_template
        self.warp_template_metadata = {'affine': warp_template.affine, 'header': warp_template.header}
        bet_model_paths = BET_MODEL_PATHS if bet_model_paths is None else bet_model_paths
        self.bet_kwargs = BET_KWARGS if bet_kwargs is None else bet_kwargs
        self.affine_kwargs = AFFINE_KWARGS if affine_kwargs is None else affine_kwargs
        segment_brain_kwargs = SEGMENT_BRAIN_KWARGS if segment_brain_kwargs is None else segment_brain_kwargs
        segment_nogm_kwargs = SEGMENT_NOGM_KWARGS if segment_nogm_kwargs is None else segment_nogm_kwargs
        smooth_kwargs = SMOOTH_KWARGS if smooth_kwargs is None else smooth_kwargs
        self.brain_extract = BrainExtraction(no_gpu, **bet_model_paths)
        self.brain_segment = BrainSegmentation(no_gpu, **segment_brain_kwargs)
        self.nogm_segment = NoGMSegmentation(no_gpu, **segment_nogm_kwargs)
        self.warp_register = WarpRegistration(no_gpu, warp_model_path)
        self.smoothing = Smoothing(no_gpu, **smooth_kwargs)
        self.atlas_register = AtlasRegistration(no_gpu)
        self._outputs = {}

    def run(self, input_path, output_paths=None, run_all=True, seed=0, skip_unprocessed=True):
        if output_paths is None:
            output_paths = {}
            run_all = True
        self._outputs = {}
        functions = {'bet': self.run_bet,
                     'affine': self.run_affine_register,
                     'segment_brain': self.run_segment_brain,
                     'segment_nogm': self.run_segment_nogm,
                     'warp': self.run_warp_register,
                     'smooth': self.run_smooth,
                     'atlas': self.run_atlas_register}
        atlas_kwargs = {'atlas_list': [o for o in IO['atlas']['output'] if o in output_paths or run_all]}
        try:
            t1 = nib.load(input_path)
            t1_array = t1.get_fdata()[..., 0] if len(t1.shape) == 4 else t1.get_fdata()
        except:
            warnings.warn(f'File {input_path} could not be properly loaded (might be broken)', Warning)
            return None
        self._outputs = {'t1': nib.Nifti1Image(t1_array, t1.affine, t1.header)}
        steps = needed_steps(output_paths)
        for step, io_dict in IO.items():
            if seed is not None:
                seed_all(seed)
            if step in steps or run_all:
                imgs = tuple(self._outputs[inp] for inp in io_dict['input'])
                kw = atlas_kwargs if step == 'atlas' else {}
                if skip_unprocessed:
                    try:
                        outputs = functions[step](*imgs, **kw)
                    except:
                        warnings.warn(f'Processing of {input_path} skipped at step "{step}" (for error messages, use skip_unprocessed=False)', Warning)
                        self._outputs = {k: self._outputs[k] if k in self._outputs else pd.DataFrame([None])
                                         for k, v in output_paths.items()}
                        break
                else:
                    outputs = functions[step](*imgs, **kw)
                self._outputs.update(**outputs)
                save_output(outputs, output_paths)
        return self._outputs

    def run_bet(self, t1):
        brain, mask, tiv = self.brain_extract.run(t1, **self.bet_kwargs)
        return {'brain': brain, 'mask': mask, 'bet_tiv': pd.Series(tiv, index=['tiv_bet_cm3'])}

    def run_affine_register(self, brain_nifti, mask_nifti):
        brain = nifti_to_tensor(brain_nifti)[None, None]
        mask = nifti_to_tensor(mask_nifti)[None, None]
        brain = torch.nan_to_num(brain, nan=0)
        max_value = np.quantile(brain[mask.bool()], .95)
        x = torch.cat([(brain / max_value).clip(max=1.), mask], dim=1)
        template = nifti_to_tensor(self.affine_template)[None, None]
        y = torch.cat([template, (template > .0).float()], dim=1)
        aff_reg = AffineRegistration(**self.affine_kwargs)
        aff_reg(x.to(self.device), y.to(self.device), return_moved=False)
        affine = aff_reg.get_affine().detach()
        grid = F.affine_grid(affine, [1, 3, *template.shape[-3:]], align_corners=INTERP_KWARGS['align_corners'])
        mask_large = F.grid_sample(mask.to(self.device), grid, align_corners=INTERP_KWARGS['align_corners'], mode='bilinear')[0, 0]
        brain_large = sr.grid_sample(brain.to(self.device), grid, align_corners=INTERP_KWARGS['align_corners'], prefilter=True, mask_value=0)[0, 0]
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        affine = affine[0].cpu().numpy()
        translation, rotation, zoom, shear = [param[0].detach().cpu().numpy() for param in aff_reg._parameters]
        header_fp32 = self.affine_template.header.copy()
        header_fp32.set_data_dtype(np.float32)
        return {'brain_large': reoriented_nifti(brain_large.cpu().numpy(), self.affine_template.affine, header_fp32),
                'mask_large': reoriented_nifti(mask_large.cpu().numpy(), **self.affine_template_metadata),
                'affine': pd.DataFrame(np.vstack((affine, [[0, 0, 0, 1]]))),
                'translation': pd.Series(translation),
                'rotation': pd.DataFrame(rotation),
                'zoom': pd.Series(zoom),
                'shear': pd.Series(shear),
                'affine_loss': pd.Series([aff_reg._loss])}

    def run_segment_brain(self, brain_large, mask, affine, mask_large):
        brain_large = nifti_to_tensor(brain_large)
        mask_large = nifti_to_tensor(mask_large)
        p0_large = self.brain_segment(brain_large[None, None].to(self.device), mask_large[None, None].to(self.device))[0, 0]
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        p0_large[mask_large == 0.] = 0.
        inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float().to(self.device))
        shape = nib.as_closest_canonical(mask).shape
        grid = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
        p0 = sr.grid_sample(p0_large[None, None].to(self.device), grid, align_corners=INTERP_KWARGS['align_corners'], mask_value=0)[0, 0]
        p0 = p0.clip(min=0, max=3).cpu()
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        return {'p0_large': reoriented_nifti(p0_large.cpu().numpy(), **self.affine_template_metadata),
                'p0': reoriented_nifti(p0.cpu().numpy(), mask.affine, mask.header)}

    def run_segment_nogm(self, p0_large, affine, t1):
        p0_large = nifti_to_tensor(p0_large)[None, None]
        p_large, nogm = self.nogm_segment(p0_large.to(self.device))
        p_affine = F.interpolate(p_large.clone(), scale_factor=1 / 3, mode='area')
        inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float().to(self.device))
        t1_shape = nib.as_closest_canonical(t1).get_fdata().shape
        grid = F.affine_grid(inv_affine[None, :3], [1, 3, *t1_shape], align_corners=INTERP_KWARGS['align_corners'])
        p = sr.grid_sample(p_large, grid, align_corners=INTERP_KWARGS['align_corners'], mask_value=0)[0]
        p = p.clip(min=0, max=1).cpu()
        if self.device.type == 'cuda': torch.cuda.empty_cache()
        vol = 1e-3 * nifti_volume(t1) * p[None].mean((2, 3, 4)).cpu().numpy()
        abs_vol = pd.DataFrame(vol, columns=['gmv_cm3', 'wmv_cm3', 'csfv_cm3'])
        rel_vol = pd.DataFrame(vol / vol.sum(), columns=['gmv/tiv', 'wmv/tiv', 'csfv/tiv'])
        tiv = pd.Series([vol.sum()], name='tiv_cm3')
        wj_affine = np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(self.warp_template)
        header_uint8 = t1.header.copy()
        header_uint8.set_data_dtype(np.uint8)
        return {'p1': reoriented_nifti(p[0].cpu().numpy(), t1.affine, header_uint8),
                'p2': reoriented_nifti(p[1].cpu().numpy(), t1.affine, header_uint8),
                'p3': reoriented_nifti(p[2].cpu().numpy(), t1.affine, header_uint8),
                'p1_large': reoriented_nifti(p_large[0, 0].cpu().numpy(), **self.affine_template_metadata),
                'p2_large': reoriented_nifti(p_large[0, 1].cpu().numpy(), **self.affine_template_metadata),
                'p3_large': reoriented_nifti(p_large[0, 2].cpu().numpy(), **self.affine_template_metadata),
                'p1_affine': reoriented_nifti(p_affine[0, 0].cpu().numpy(), **self.warp_template_metadata),
                'p2_affine': reoriented_nifti(p_affine[0, 1].cpu().numpy(), **self.warp_template_metadata),
                'p3_affine': reoriented_nifti(p_affine[0, 2].cpu().numpy(), **self.warp_template_metadata),
                'nogm': reoriented_nifti(nogm[0, 0].cpu().numpy(), **self.affine_template_metadata),
                'gmv': abs_vol['gmv_cm3'], 'wmv': abs_vol['wmv_cm3'], 'csfv': abs_vol['csfv_cm3'],
                'rel_gmv': rel_vol['gmv/tiv'], 'rel_wmv': rel_vol['wmv/tiv'], 'rel_csfv': rel_vol['csfv/tiv'],
                'tiv': tiv, 'wj_affine': pd.Series([wj_affine])}

    def run_warp_register(self, p0_large, p1_affine, p2_affine, wj_affine):
        p0 = nifti_to_tensor(p0_large)[None, None].to(self.device)
        p0 = F.interpolate(p0, scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p = torch.stack([nifti_to_tensor(p1_affine).to(self.device), nifti_to_tensor(p2_affine).to(self.device), p0])
        template = nifti_to_tensor(self.warp_template).permute(3, 0, 1, 2).to(self.device)
        images, flows, v = self.warp_register(p[None, :2], template[None, :2], x_cat=p[None, -1:])
        wp = images['xy_full'][0]
        mse = ((wp[:2] - template[:2])**2).mean().item()
        wj = jacobi_determinant(flows['xy_full'] - self.warp_register.syn._grid)
        mwp = wp * wj[None] * wj_affine[0]
        v_xy = v['xy_velocity'][0].permute(1, 2, 3, 0)
        v_yx = v['yx_velocity'][0].permute(1, 2, 3, 0)
        header_fp32 = self.warp_template.header.copy()
        header_fp32.set_data_dtype(np.float32)
        header_uint8 = self.warp_template.header.copy()
        header_uint8.set_data_dtype(np.uint8)
        return {'warp_xy': reoriented_nifti(flows['xy_full'][0].cpu().numpy(), self.warp_template.affine, header_fp32),
                'warp_yx': reoriented_nifti(flows['yx_full'][0].cpu().numpy(), self.warp_template.affine, header_fp32),
                'wp0': reoriented_nifti(wp[-1].cpu().numpy(), self.warp_template.affine, header_uint8),
                'wp1': reoriented_nifti(wp[0].cpu().numpy(), self.warp_template.affine, header_uint8),
                'wp2': reoriented_nifti(wp[1].cpu().numpy(), self.warp_template.affine, header_uint8),
                'wj_': reoriented_nifti(wj.cpu().numpy(), self.warp_template.affine, header_fp32),
                'mwp1': reoriented_nifti(mwp[0].cpu().numpy(), self.warp_template.affine, header_fp32),
                'mwp2': reoriented_nifti(mwp[1].cpu().numpy(), self.warp_template.affine, header_fp32),
                'v_xy': reoriented_nifti(v_xy.detach().cpu().numpy(), self.warp_template.affine, header_fp32),
                'v_yx': reoriented_nifti(v_yx.detach().cpu().numpy(), self.warp_template.affine, header_fp32),
                'warp_mse': pd.Series([mse])}

    def run_smooth(self, mwp1, mwp2):
        smwp1 = self.smoothing(nifti_to_tensor(mwp1)[None, None].to(self.device))[0]
        smwp2 = self.smoothing(nifti_to_tensor(mwp2)[None, None].to(self.device))[0]
        header_fp32 = self.warp_template.header.copy()
        header_fp32.set_data_dtype(np.float32)
        output = {}
        for i, fwhm in enumerate(self.smoothing.fwhms):
            output.update(
                {f's{fwhm}mwp1': reoriented_nifti(smwp1[i].cpu().numpy(), self.warp_template.affine, header_fp32),
                 f's{fwhm}mwp2': reoriented_nifti(smwp2[i].cpu().numpy(), self.warp_template.affine, header_fp32)}
            )
        return output

    def run_atlas_register(self, t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list):
        voxel_vol = np.prod(p1_large.affine[np.diag_indices(3)])
        p1_large, p2_large, p3_large = [nifti_to_tensor(p).to(self.device) for p in [p1_large, p2_large, p3_large]]
        inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float().to(self.device))
        grid = F.affine_grid(inv_affine[None, :3], [1, 3, *t1.shape[:3]], align_corners=INTERP_KWARGS['align_corners'])
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(self.device)
        atlases, warps = {}, {}
        atl_list = ['_'.join(a.split('_')[:-1]) if a.endswith(('_affine', '_volumes')) else a for a in atlas_list]
        for atl in atl_list:
            atlas = nib.as_closest_canonical(nib.load(f'{DATA_PATH}/templates/{atl}.nii.gz'))
            header = atlas.header
            shape = tuple(shape_from_to(atlas, warp_yx))
            if shape not in warps:
                scaled_yx = F.interpolate(yx.permute(0, 4, 1, 2, 3), shape, mode='trilinear', align_corners=False)
                warps.update({shape: scaled_yx.permute(0, 2, 3, 4, 1)})
            atlas = self.atlas_register(affine, warps[shape], atlas, t1.shape)
            if f'{atl}_affine' in atlas_list:
                atlases.update({atl + '_affine': atlas})
            atlas = nifti_to_tensor(atlas).to(self.device)
            if f'{atl}_volumes' in atlas_list:
                rois = pd.read_csv(f'{DATA_PATH}/templates/{atl}.csv', sep=';')[['ROIid', 'ROIname']]
                volumes = voxel_vol * get_volumes(atlas, p1_large, p2_large, p3_large)
                volumes = pd.DataFrame(volumes, columns=['gmv_mm3', 'wmv_mm3', 'csfv_mm3', 'region_mm3'])
                atlases.update({atl + '_volumes': pd.concat([rois, volumes], axis=1)})
            if atl in atlas_list:
                sample_kwargs = {'mode': 'nearest', 'align_corners': INTERP_KWARGS['align_corners']}
                atlas = F.grid_sample(atlas[None, None], grid, **sample_kwargs)[0, 0]
                atlases.update({atl: reoriented_nifti(atlas.cpu().numpy(), t1.affine, header)})
        return atlases


def save_output(outputs, paths):
    for k, output in outputs.items():
        if k in paths:
            if isinstance(output, nib.Nifti1Image):
                output.to_filename(paths[k])
            elif isinstance(output, (pd.Series, pd.DataFrame)):
                output.to_csv(paths[k], index=False)
            else:
                raise TypeError(f'Can only save output of type Nifti1Image, Series or DataFrame not {type(output)}')


def needed_steps(output_paths):
    steps_bool = []
    for step, io_dict in IO.items():
        steps_bool.append(bool(set(io_dict['output']) & set(output_paths)))
        if steps_bool[-1]:
            steps_bool = [True for _ in steps_bool]
    return [step for sb, step in zip(steps_bool, IO) if sb]


def get_path_dataframe(input_paths=None, bids_dir=None, output_paths=None, output_dir=None, outputs='minimal',
                       dir_format='cat12'):
    assert not (input_paths is None and bids_dir is None), 'No input filepaths given'
    assert not (bids_dir is None and output_dir is None and output_paths is None), 'No output filepaths given'
    input_paths = find_bids_t1w_files(bids_dir) if input_paths is None else input_paths
    outputs = OUTPUTS[outputs] if isinstance(outputs, str) else outputs
    output_paths = output_paths if bids_dir is None else create_bids_output_paths(input_paths, bids_dir, outputs)
    if output_paths is None:
        output_paths = [get_output_paths(in_path, output_dir, outputs, dir_format) for in_path in input_paths]
    return pd.DataFrame(output_paths, index=pd.Series(input_paths, name='t1'))


def get_output_paths(filepath, output_dir, outputs=None, dir_format='sub'):
    assert dir_format in DIR_FORMATS, f'Given dir_format={dir_format} not supported. Must be in {DIR_FORMATS})'
    outputs = outputs if isinstance(outputs, list) else OUTPUTS[outputs]
    filename = str(Path(filepath).name).split('.')[0]
    output_paths = {}
    for o in outputs:
        file_format = 'csv' if o in OUTPUTS['csv'] else 'nii.gz'
        if dir_format == 'sub':
            fpath = f'{output_dir}/{filename}/{o}{filename}.{file_format}'
        elif dir_format == 'mod':
            fpath = f'{output_dir}/{o}/{filename}.{file_format}'
        elif dir_format == 'cat':
            mode_dir = 'label' if o in OUTPUTS['csv'] else 'mri'
            fpath = f'{output_dir}/{filename}/{mode_dir}/{o}{filename}.{file_format}'
        else:
            fpath = f'{output_dir}/{o}{filename}.{file_format}'
        output_paths.update({o: fpath})
    return output_paths


def create_bids_output_paths(input_paths, bids_dir, outputs):
    output_paths = []
    for in_path in input_paths:
        filename = str(Path(in_path).name).split('.')[0]
        output_dir = str(Path(in_path).parent).replace(bids_dir, f'{bids_dir}/derivatives/deepmriprep')
        output_dict = {o: f'{output_dir}/mri/{o}{filename}.nii.gz' for o in outputs}
        output_dict.update({o: f'{output_dir}/label/{o}{filename}.csv' for o in OUTPUTS['csv'] if o in outputs})
        output_paths.append(output_dict)
    return output_paths
