![logo](https://github.com/user-attachments/assets/bbd01efd-ba71-4504-a085-909b28366de4)

[deepmriprep](https://arxiv.org/) runs **all preprocessing steps** needed for **Voxel-based Morphometry** (VBM) of T1w MR images:

- brain extraction (via [deepbet](https://github.com/wwu-mmll/deepbet))
- affine registration (via [torchreg](https://github.com/codingfisch/torchreg))
- tissue segmentation
- nonlinear registration
- smoothing

Using neural networks, deepmriprep only takes **~10 seconds** on GPU (~100 seconds without GPU) per image.

![main_fig](https://github.com/user-attachments/assets/f6dfd6a7-63c4-48d0-9477-af5c30b607cd)

Additionally, deepmriprep can also run **atlas registration** needed for **Region-based Morphometry** (RBM).

![atlases_small](https://github.com/user-attachments/assets/fc26fd66-b074-4900-9035-c8bc49f16346)

## Usage 💡
After [installation](https://github.com/codingfisch/deepmriprep_beta?tab=readme-ov-file#installation), there are three ways to use deepmriprep
1. ```deepmriprep-gui``` runs the **Graphical User Interface (GUI)**



2. ```deepmriprep-cli``` runs the **Command Line Interface (CLI)** (use `deepmriprep-cli --help` for usage instructions)



3. Run deepmriprep directly in Python

```python
from deepmriprep import run_preprocess

# Apply to BIDS dir. (saves to 'path/to/bids/derivatives')
run_preprocess(bids_dir='path/to/bids')
# Apply to list of files and save to output directory
input_paths = ['path/to/input/sub-1_t1w.nii.gz', 'path/to/input/sub-2_t1w.nii.gz']
run_preprocess(input_paths, output_dir='path/to/output')
# Apply to list of files and save to custom output filepaths
output_paths = [{'tiv': 'outpath1/tivsub-1_t1w.csv', 'p0': 'outpath2/p0sub-1_t1w.nii'},
                {'tiv': 'outpath3/tivsub-2_t1w.csv', 'p0': 'outpath4/p0sub-2_t1w.nii'}]
run_preprocess(input_paths, output_paths=output_paths)
```

Besides the three shown options to specify input and output paths, `run_preprocess` has the arguments

- `outputs`: Output modalities which can be either `'all'`, `'vbm'`, `'rbm'` or a custom list of [output strings](https://github.com/codingfisch/deepmriprep_alpha/tree/main#complete-list-of-output-strings)
- `dir_format`: Output directory structure which can be either `'sub'`, `'mod'`, `'cat'` or `'flat'`
- `no_gpu`: Avoids GPU utilization if set to `True` (set to `False` per default for GPU-acceleration 🔥)

`outputs` set to
- `'all'` results in [all output modalities](https://github.com/codingfisch/deepmriprep_alpha/tree/main#complete-list-of-output-strings) being saved
- `'vbm'` results in the outputs `tiv`, `mwp1`, `mwp2`, `s6mwp1` and `s6mwp2` being saved
- `'rbm'` results in all available atlases (including regions tissue volumes) being saved

`dir_format` set to (if `output_dir` is set)
- `'sub'` results in e.g. `'outpath/sub-1/tivsub-1.csv'` and `'outpath/sub-1/p0sub-1.nii.gz'`
- `'mod'` results in e.g. `'outpath/tiv/tivsub-1.csv'` and `'outpath/p0/p0sub-1.nii.gz'`
- `'cat'` results in e.g. `'outpath/sub-1/label/tivsub-1.csv'` and `'outpath/sub-1/mri/p0sub-1.nii.gz'`
- `'flat'` results in e.g. `'outpath/tivsub-1.csv'` and `'outpath/p0sub_1.nii.gz'`

## Tutorial 🧑‍🏫
In short, deepmriprep (containing only ~500 lines of code) internally calls the `.run` method of the `Preprocess` class, which sequentially calls the methods `.run_bet` to `.run_atlas_register` (see [deepmriprep/preprocess.py](https://github.com/codingfisch/deepmriprep_beta/blob/main/deepmriprep/preprocess.py#L136)).

A more detailed Tutorial-Notebook will be soon be published on Google Colab!

## Installation 🛠️
For GPU-acceleration of deepmriprep 🔥, PyTorch should be installed first via the [proper installation command](https://pytorch.org/get-started/locally) for your system (currently only possible for systems with a NVIDIA GPU).

After that, deepmriprep can be easily installed via
```bash
pip install deepmriprep
```

## Citation ©️
If you find this code useful in your research, please consider citing:

    @inproceedings{deepmriprep,
    Author = {Lukas Fisch, Nils R. Winter, Janik Goltermann, Carlotta Barkhau, Daniel Emden, Jan Ernsting, Maximilian Konowski, Ramona Leenings, Tiana Borgers, Kira Flinkenflügel, Dominik Grotegerd, Anna Kraus, Elisabeth J. Leehr, Susanne Meinert, Frederike Stein, Lea Teutenberg, Florian Thomas-Odenthal, Paula Usemann, Marco Hermesdorf, Hamidreza Jamalabadi, Andreas Jansen, Igor Nenadic, Benjamin Straube, Tilo Kircher, Klaus Berger, Benjamin Risse, Udo Dannlowski, Tim Hahn},
    Title = {deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks},
    Year = {2024}
    }
    
## Outputs 📋

Most output strings follow the [CAT12 naming convention of output files](https://neuro-jena.github.io/cat12-help/#naming). Here is the full list:

**Input**
- `'t1'`: T1 weighted MR image

**Brain Extraction**
- `'mask'`: Brain mask
- `'brain'`: T1 after brain mask is applied
- `'tiv_bet'`: Total intracranial volume in cm³ based on brain extraction

**Affine Registration**
- `'affine'`: Affine matrix moving 'brain' into template space (compatible with [`F.affine_grid`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'translate'`, `'rotation'`, `'zoom'`, `'shear'`: Parameters of the respective operation which the affine is [composed of](https://github.com/codingfisch/torchreg/blob/main/torchreg/affine.py#L83)
- `'mask_large'`: Affine applied to `'mask'` with 0.5mm resolution
- `'brain_large'`: Affine applied to `'brain'` with 0.5mm resolution

**Tissue Segmentation**
- `'p0_large'`: Tissue segmentation map of `'brain_large'`
- `'p0'`: Tissue segmentation map in `'t1'` image space (moved `'p0_large'`)

**Tissue Probabilities**
- `'nogm'`: Small area around the brain stem which is masked in subsequent GM output
- `'gmv'`: Gray matter (GM) volume in cm³
- `'wmv'`: White matter (WM) ""
- `'csfv'`: Cerebrospinal fluid (CSF) ""
- `'tiv'`: Total intracranial volume in cm³ based on tissue segmentation (gmv + wmv + csfv)
- `'rel_gmv'`: Proportion of gray matter (GM) volume relative to the total intracranial volume
- `'rel_wmv'`: Proportion of white matter (WM) ""
- `'rel_csfv'`: Proportion of cerebrospinal fluid (CSF) ""
- `'p1'`: Gray matter (GM) tissue probability in `'t1'` image space
- `'p2'`: White matter (WM) ""
- `'p3'`: Cerebrospinal fluid (CSF) ""
- `'p1_large'`: Gray matter (GM) tissue probability based on `'p0_large'`
- `'p2_large'`: White matter (WM) ""
- `'p3_large'`: Cerebrospinal fluid (CSF) ""
- `'p1_affine'`: Gray matter (GM) tissue probability based on `'p0_large'` in template resolution
- `'p2_affine'`: White matter (WM) ""
- `'p3_affine'`: Cerebrospinal fluid (CSF) ""

**Nonlinear Registration**
- `'wj_affine'`: Jacobian determinant of the affine matrix
- `'warp_xy'`: Forward warping field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'warp_yx'`: Backward warping field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'wj_'`: Jacobian determinant of forward warping field
- `'v_xy'`: Forward velocity field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'v_yx'`: Backward velocity field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'wp1'`: Forward **w**arped `'p1'`
- `'mwp1'`: **M**odulated (multiplied with `'wj_'`), forward **w**arped `'p1'`
- `'s6mwp1'`: **S**moothed with **6**mm kernel, **m**odulated (multiplied with `'wj_'`), forward **w**arped `'p1'`
- ...analogous for 'wp2', 'mwp2', 's6mwp2', 's8mwp1', ...

**Atlases**
- `'aal3'`: Registered AAL3 atlas in `'t1'` image space
- `'anatomy3'`: Registered Aanatomy3 ""
- `'cobra'`: Registered Cobra atlas ""
- `'hammers'`: Registered Hammers atlas ""
- `'ibsr'`: Registered IBSR atlas ""
- `'julichbrain'`: Registered Julichbrain atlas ""
- `'lpba40'`: Registered LPBA40 atlas ""
- `'mori'`: Registered Mori atlas ""
- `'neuromorphometrics'`: Registered Neuromorphometrics ""
- `'suit'`: Registered SUIT ""
- `'thalamic_nuclei'`: Registered Thalamic Nuclei ""
- `'thalamus'`: Registered Thalamus ""
- `'Schaefer2018_100Parcels_17Networks_order'`: Registered Schaefer 100 ""
- `'Schaefer2018_200Parcels_17Networks_order'`: Registered Schaefer 200 ""
- `'Schaefer2018_400Parcels_17Networks_order'`: Registered Schaefer 400 ""
- `'Schaefer2018_600Parcels_17Networks_order'`: Registered Schaefer 600 ""

For each atlas, there also can be outputted two more modalities:
- `'aal3_affine'`: Registered AAL3 atlas in template space
- `'aal3_volumes'`: Gray matter (GM), White matter (WM) and Cerebrospinal fluid (CSF) volume in mm³ per region of the AAL3 atlas
- ...analogous 'anatomy3', 'cobra', ...
