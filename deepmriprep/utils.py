import glob
import torch
import random
import requests
import numpy as np
import nibabel as nib
from pathlib import Path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = f'{Path(__file__).parent.resolve()}/data'
MODEL_FILES = (['brain_extraction_bbox_model.pt', 'brain_extraction_model.pt', 'segmentation_nogm_model.pt'] +
               [f'segmentation_patch_{i}_model.pt' for i in range(18)] + ['segmentation_model.pt', 'warp_model.pt'])


def nifti_to_tensor(nifti):
    return torch.from_numpy(nib.as_closest_canonical(nifti).get_fdata(dtype=np.float32).copy())


def nifti_volume(nifti):
    oriented_nifti = nib.as_closest_canonical(nifti)
    voxel_volume = np.prod(oriented_nifti.header.get_zooms()[:3])
    return voxel_volume * np.prod(np.array(oriented_nifti.shape[:3]))


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


def download_missing_models(api_url='https://api.github.com/repos/wwu-mmll/deepmriprep/contents'):
    Path(f'{DATA_PATH}/models').mkdir(exist_ok=True)
    for file in MODEL_FILES:
        if not Path(f'{DATA_PATH}/models/{file}').exists():
            download_file(f'{api_url}/deepmriprep/data/models/{file}', f'{DATA_PATH}/models/{file}')


def download_file(api_url, dest_path):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and 'download_url' in data:
            download_url = data['download_url']
            file_response = requests.get(download_url, stream=True)
            if file_response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive new chunks
                            f.write(chunk)
                print(f'Downloaded {download_url} to {dest_path}')
            else:
                raise Exception(f'Failed to download file from {download_url}')
        else:
            raise Exception(f'No download URL found in response from {api_url}')
    else:
        raise Exception(f'Failed to get metadata from {api_url}')
