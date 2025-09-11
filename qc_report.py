import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from deepmriprep.preprocess import OUTPUTS
from niftiview import save_gif, NiftiImageGrid
AFFINE_LOSS_PERCENTILES = [-.3777, -.3743, -.3719, -.3701, -.3683, -.3665, -.3652, -.3642, -.3634, -.3628, -.3623,
                           -.3618, -.3614, -.3610, -.3606, -.3603, -.3600, -.3596, -.3593, -.3589, -.3586, -.3583,
                           -.3580, -.3577, -.3574, -.3570, -.3566, -.3563, -.3560, -.3556, -.3553, -.3549, -.3545,
                           -.3542, -.3538, -.3534, -.3530, -.3526, -.3523, -.3519, -.3515, -.3511, -.3507, -.3504,
                           -.3500, -.3496, -.3492, -.3489, -.3485, -.3481, -.3478, -.3474, -.3470, -.3466, -.3462,
                           -.3458, -.3454, -.3450, -.3446, -.3442, -.3438, -.3434, -.3429, -.3425, -.3420, -.3416,
                           -.3411, -.3406, -.3401, -.3395, -.3389, -.3383, -.3378, -.3372, -.3365, -.3358, -.3350,
                           -.3342, -.3334, -.3326, -.3318, -.3311, -.3302, -.3294, -.3286, -.3278, -.3269, -.3258,
                           -.3247, -.3233, -.3215, -.3193, -.3166, -.3132, -.3090, -.3044, -.2994, -.2932, -.2685]
AFFINE_LOSS_PERCENTILES = np.array(AFFINE_LOSS_PERCENTILES)
WARP_MSE_PERCENTILES = [.00919, .00936, .00948, .00955, .00963, .00969, .00975, .00980, .00985, .00990, .00994,
                        .00998, .01002, .01005, .01009, .01012, .01016, .01019, .01023, .01026, .01029, .01032,
                        .01035, .01038, .01041, .01044, .01047, .01049, .01052, .01055, .01057, .01060, .01063,
                        .01066, .01069, .01071, .01074, .01077, .01079, .01082, .01085, .01088, .01090, .01093,
                        .01096, .01099, .01102, .01104, .01108, .01111, .01114, .01116, .01120, .01123, .01126,
                        .01129, .01133, .01135, .01139, .01142, .01145, .01149, .01152, .01156, .01160, .01164,
                        .01168, .01172, .01176, .01180, .01185, .01190, .01195, .01201, .01207, .01213, .01219,
                        .01226, .01233, .01240, .01248, .01257, .01266, .01276, .01289, .01303, .01318, .01337,
                        .01360, .01386, .01418, .01463, .01521, .01591, .01779, .02101, .04433, .05938, .06271]
WARP_MSE_PERCENTILES = np.array(WARP_MSE_PERCENTILES)

bool_opt = argparse.BooleanOptionalAction
parser = argparse.ArgumentParser()
parser.add_argument('-csv', '--csv_path', help='Filepath of deepmriprep_outputs.csv', type=str, required=True)
parser.add_argument('-o', '--output_dir', help='Output folder', required=False, type=str, default='.')
parser.add_argument('-n', '--num_reports', help='Number of output reports', type=int, default=4)
parser.add_argument('-c', '--columns', help='List of columns (e.g. "mwp1 mwp2")',
                    required=False, nargs='+', default=['mwp1'])
parser.add_argument('-g', '--gif', help='If this flag is set save GIFs otherwise PNGs', action=bool_opt)
parser.add_argument('-l', '--layout', help='Layout', type=str, default='sagittal++')
parser.add_argument('--height', help='Image height (in pixels)', type=int, default=600)
parser.add_argument('--cmap', help='Colormap for image', type=str, default='gray')
parser.add_argument('--crosshair', help='Show crosshair', action=bool_opt, default=False)
parser.add_argument('--fpath', help='Show filepath', type=int, default=0)
parser.add_argument('--coordinates', help='Show coordinates', action=bool_opt, default=False)
parser.add_argument('--nrows', help='Number of rows', type=int, default=None)
args = parser.parse_args()

df = pd.read_csv(args.csv_path)
out_dir, num_reports, columns  = args.output_dir, args.num_reports, args.columns
nii_outputs = [o for o in columns if o in OUTPUTS['all'] and o not in OUTPUTS['csv']]
for c in columns:
    assert c in df.columns, f'Column {c} not in columns of CSV: {df.columns}'
for arg in ['csv_path', 'output_dir', 'num_reports', 'columns']: delattr(args, arg)
if 'affine_loss_value' in df.columns or 'warp_mse_value' in df.columns:
    df = df.sort_values('warp_mse_value' if 'warp_mse_value' in df.columns else 'affine_loss_value', ascending=False)
else:
    warnings.warn('Neither "warp_mse_value" nor "affine_loss_value" column found in CSV: Creating unsorted reports!')
df = df.reset_index(drop=True)
for i in range(min(num_reports, len(df))):
    filename = str(Path(df.t1[i]).name).split('.')[0]
    niigrid = NiftiImageGrid([df[o].iloc[i] for o in nii_outputs])
    title = ''
    if 'affine_loss_value' in df.columns:
        affine_loss = df.affine_loss_value[i]
        affine_perc = (affine_loss > AFFINE_LOSS_PERCENTILES).sum()
        title += f'Affine Loss: {affine_loss:.4f} ({affine_perc}th percentile)\n'
    if 'warp_mse_value' in df.columns:
        warp_mse = df.warp_mse_value[i]
        warp_perc = (warp_mse > WARP_MSE_PERCENTILES).sum()
        title += f'Warp MSE: {warp_mse:.5f} ({warp_perc}th percentile)'
    title = [title] + (len(nii_outputs) - 1) * ['']
    if args.gif:
        save_gif(niigrid, f'{out_dir}/{filename}.gif', title=title, **vars(args))
    else:
        niigrid.get_image(title=title, **vars(args)).save(f'{out_dir}/{filename}.png')
