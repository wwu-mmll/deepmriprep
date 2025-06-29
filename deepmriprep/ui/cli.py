import os
import glob
import argparse
import pandas as pd
from pathlib import Path

from deepmriprep.preprocess import OUTPUTS, run_preprocess


def get_paths_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    in_paths = df.index.tolist()
    cols = [col for col in df.columns if col in OUTPUTS['all'] and col != 't1']
    out_paths = list(df[cols].T.to_dict().values())
    return in_paths, out_paths


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file or folder', required=False, type=str, default=None)
    parser.add_argument('-bids', '--bids_dir', help='BIDS folder', required=False, type=str, default=None)
    parser.add_argument('-o', '--output_dir', help='Output folder', required=False, type=str, default=None)
    parser.add_argument('-out', '--outputs', help='all, vbm, rbm or list of output strings (e.g. [p0, mwp1])',
                        required=False, nargs='+', default=['vbm'])
    parser.add_argument('-f', '--dir_format', help='sub, mod, cat or flat', required=False, type=str, default='sub')
    parser.add_argument('-ng', '--no_gpu', help='If GPU should be avoided', required=False,
                        type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-ns', '--no_skip', help='Do not skip files that could not be processed', required=False,
                        type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-csv', '--csv_path', required=False, type=str, default=None,
                        help='Filepath of csv which contains input filepaths in first column and output filepaths in '
                             'subsequent columns. Column headers should be respective output strings (e.g. p0 or mwp1).'
                             'CAUTION: Specify --table_path otherwise no table will be written!')
    args = parser.parse_args()

    outputs = args.outputs[0] if args.outputs[0] in ['all', 'vbm', 'rbm'] else args.outputs
    assert not (args.input is None and args.bids_dir is None and args.csv_path is None), 'No input filepaths given'
    if args.csv_path is None:
        if args.bids_dir is None:
            input_paths = sorted(glob.glob(f'{args.input}/*.ni*')) if os.path.isdir(args.input) else [args.input]
        else:
            input_paths = None
        output_paths = None
    else:
        input_paths, output_paths = get_paths_from_csv(args.csv_path)
        args.output_dir = Path(args.csv_path).stem

    run_preprocess(input_paths, args.bids_dir, output_paths, args.output_dir, outputs, args.dir_format, args.no_gpu,
                   skip_unprocessed=not args.no_skip)


if __name__ == '__main__':
    run_cli()
