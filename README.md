**Disclaimer**: deepmriprep is not related to fMRIPrep or sMRIPrep and is not part of the NiPreps framework
 
![logo](https://github.com/user-attachments/assets/bbd01efd-ba71-4504-a085-909b28366de4)

[deepmriprep](https://arxiv.org/abs/2408.10656) runs **all preprocessing steps** needed for [**Voxel-based Morphometry** (VBM)](https://www.sciencedirect.com/science/article/pii/S1053811900905822) of T1w MR images:

- brain extraction (via [deepbet](https://github.com/wwu-mmll/deepbet))
- affine registration (via [torchreg](https://github.com/codingfisch/torchreg))
- tissue segmentation
- nonlinear registration
- smoothing

Using neural networks, it only takes **~10 seconds** on a GPU (~100 seconds without a GPU) per image 🔥

![main_fig](https://github.com/user-attachments/assets/f6dfd6a7-63c4-48d0-9477-af5c30b607cd)

Additionally, deepmriprep can also run **atlas registration** needed for **Region-based Morphometry** (RBM).

![atlases_small](https://github.com/user-attachments/assets/fc26fd66-b074-4900-9035-c8bc49f16346)

## Installation 🛠️
For GPU-acceleration (NVIDIA needed), PyTorch should be installed first via the [proper install command](https://pytorch.org/get-started/locally) 🔥

deepmriprep can be easily installed via
```bash
pip install deepmriprep
```

The `deepmriprep-gui` can look grainy in conda environments. To fix that run 
```bash
conda install -c conda-forge tk=*=xft_*
```

## Usage 💡
After installation, there are three ways to use deepmriprep
1. `deepmriprep-gui` runs the **Graphical User Interface (GUI)**

![gui](https://github.com/user-attachments/assets/def9fce5-b1b4-4dbf-9fb3-41dcccb03144)

2. `deepmriprep-cli` runs the **Command Line Interface (CLI)**

```bash
deepmriprep-cli -bids path/to/bids
```

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

- `outputs`: Output modalities which can be either `'all'`, `'vbm'`, `'rbm'` or a custom list of [output strings](https://github.com/codingfisch/deepmriprep_beta?tab=readme-ov-file#outputs-)
- `dir_format`: Output directory structure which can be either `'sub'`, `'mod'`, `'cat'` or `'flat'`
- `no_gpu`: Avoids GPU utilization if set to `True` (set to `False` per default for GPU-acceleration 🔥)

`outputs` set to
- `'all'` [all output modalities](https://github.com/codingfisch/deepmriprep_beta?tab=readme-ov-file#outputs-) are saved
- `'vbm'` the outputs `tiv`, `mwp1`, `mwp2`, `s6mwp1` and `s6mwp2` (+`affine_loss`&`warp_mse`, for QC) are saved
- `'rbm'` all available atlases (including regions tissue volumes +`affine_loss`&`warp_mse`, for QC) are saved

If `output_dir` is set, `dir_format` set to
- `'sub'` results in e.g. `'outpath/sub-1/tivsub-1.csv'` and `'outpath/sub-1/p0sub-1.nii.gz'`
- `'mod'` results in e.g. `'outpath/tiv/tivsub-1.csv'` and `'outpath/p0/p0sub-1.nii.gz'`
- `'cat'` results in e.g. `'outpath/sub-1/label/tivsub-1.csv'` and `'outpath/sub-1/mri/p0sub-1.nii.gz'`
- `'flat'` results in e.g. `'outpath/tivsub-1.csv'` and `'outpath/p0sub_1.nii.gz'`

## Quality Control 🔍
Use [NiftiView](https://github.com/codingfisch/niftiview-app) to quickly quality check outputs via patterns like
```bash
/path/to/bids/derivatives/deepmriprep/sub*/anat/mri/p0sub*.ni*
```
in the "Image files" field (all `p0` files in this example).

Sort `deepmriprep_outputs.csv` by `affine_loss` and `warp_mse`, since high values can indicate low quality! 

## Tutorial 🧑‍🏫
In short, deepmriprep (consisting of only ~500 lines of code) internally calls the `.run` method of the `Preprocess` class, which sequentially calls the methods `.run_bet` to `.run_atlas_register` (see [deepmriprep/preprocess.py](https://github.com/wwu-mmll/deepmriprep/blob/main/deepmriprep/preprocess.py#L132)).

[Click here to view the tutorial](https://colab.research.google.com/drive/1zwIaFob2Ri-JjDVhcCjD1_Ef7G62wfJN?usp=sharing) and get a more detailed look behind the scenes 👀

## Citation ©️
If you use deepmriprep in your research, please cite:

    @inproceedings{deepmriprep,
    Author = {Lukas Fisch, Nils R. Winter, Janik Goltermann, Carlotta Barkhau, Daniel Emden, Jan Ernsting, Maximilian Konowski, Ramona Leenings, Tiana Borgers, Kira Flinkenflügel, Dominik Grotegerd, Anna Kraus, Elisabeth J. Leehr, Susanne Meinert, Frederike Stein, Lea Teutenberg, Florian Thomas-Odenthal, Paula Usemann, Marco Hermesdorf, Hamidreza Jamalabadi, Andreas Jansen, Igor Nenadic, Benjamin Straube, Tilo Kircher, Klaus Berger, Benjamin Risse, Udo Dannlowski, Tim Hahn},
    Title = {deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks},
    Year = {2024},
    eprint = {arXiv:2408.10656}
    }
    
## Outputs 📋
When "Output Modalities" is set to "custom" in the [`deepmriprep-gui`](https://github.com/wwu-mmll/deepmriprep?tab=readme-ov-file#usage-), all output strings are shown:

![gui_custom](https://github.com/user-attachments/assets/b0cc4991-30a5-427a-9aa7-79f6449436fb)

Most output strings follow the [CAT12 naming convention of output files](https://neuro-jena.github.io/cat12-help/#naming). 

Here are descriptions for all output strings:

**Input**
- `'t1'`: T1-weighted MR image

**Brain Extraction**
- `'mask'`: Brain mask
- `'brain'`: `'t1'` after brain mask is applied
- `'tiv_bet'`: Total intracranial volume in cm³ based on brain extraction

**Affine Registration**
- `'affine'`: Affine matrix moving 'brain' into template space (compatible with [`F.affine_grid`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'translate'`, `'rotation'`, `'zoom'`, `'shear'`: Tranformation parameters the affine is [composed of](https://github.com/codingfisch/torchreg/blob/main/torchreg/affine.py#L83)
- `'mask_large'`: Affine applied to `'mask'` with 0.5mm resolution
- `'brain_large'`: Affine applied to `'brain'` with 0.5mm resolution
- `'affine_loss'`: Loss value of the affine registration

**Tissue Segmentation**
- `'p0_large'`: Tissue segmentation map of `'brain_large'`
- `'p0'`: Tissue segmentation map in `'t1'` image space (moved `'p0_large'`)

**Tissue Probabilities**
- `'nogm'`: Small area around the brain stem that is masked in subsequent GM output
- `'gmv'`: **Gray matter (GM)** volume in cm³
- `'wmv'`: **White matter (WM)** ""
- `'csfv'`: **Cerebrospinal fluid (CSF)** ""
- `'tiv'`: Total intracranial volume in cm³ based on tissue segmentation (gmv + wmv + csfv)
- `'rel_gmv'`: Proportion of GM volume relative to the total intracranial volume
- `'rel_wmv'`: Proportion of WM ""
- `'rel_csfv'`: Proportion of CSF ""
- `'p1'`: GM tissue probability in `'t1'` image space
- `'p2'`: WM ""
- `'p3'`: CSF ""
- `'p1_large'`: GM tissue probability based on `'p0_large'`
- `'p2_large'`: WM ""
- `'p3_large'`: CSF ""
- `'p1_affine'`: GM tissue probability based on `'p0_large'` in template resolution
- `'p2_affine'`: WM ""
- `'p3_affine'`: CSF ""

**Nonlinear Registration**
- `'wj_affine'`: Jacobian determinant of the affine matrix
- `'warp_xy'`: Forward warping field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'warp_yx'`: Backward warping field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'warp_mse'`: Mean squared error between warped `p` and warp template
- `'wj_'`: Jacobian determinant of forward warping field
- `'v_xy'`: Forward velocity field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'v_yx'`: Backward velocity field (compatible with [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html))
- `'wp1'`: Forward **w**arped `'p1'`
- `'mwp1'`: **M**odulated (multiplied with `'wj_'`), forward **w**arped `'p1'`
- `'s6mwp1'`: **S**moothed with **6**mm kernel, **m**odulated (multiplied with `'wj_'`), forward **w**arped `'p1'`
- ...analogous for 'wp2', 'mwp2', 's6mwp2', 's8mwp1', ...

**Atlases**
- `'aal3'`: Registered [AAL3](https://www.sciencedirect.com/science/article/pii/S1053811919307803) atlas in `'t1'` image space
- `'anatomy3'`: Registered [Anatomy3](https://www.sciencedirect.com/science/article/pii/S105381190400792X?via%3Dihub) ""
- `'cobra'`: Registered [Cobra](https://www.sciencedirect.com/science/article/pii/S1053811913001237) atlas ""
- `'hammers'`: Registered [Hammers](https://www.sciencedirect.com/science/article/pii/S1053811907010634?via%3Dihub) atlas ""
- `'ibsr'`: Registered [IBSR](https://ieeexplore.ieee.org/abstract/document/5977031) atlas ""
- `'julichbrain'`: Registered [Julichbrain](https://www.science.org/doi/10.1126/science.abb4588) atlas ""
- `'lpba40'`: Registered [LPBA40](https://www.sciencedirect.com/science/article/pii/S1053811907008099?via%3Dihub) atlas ""
- `'mori'`: Registered [Mori](https://www.sciencedirect.com/science/article/pii/S1053811909000093?via%3Dihub) atlas ""
- `'neuromorphometrics'`: Registered [Neuromorphometrics](http://www.neuromorphometrics.com/) ""
- `'suit'`: Registered [SUIT](https://www.sciencedirect.com/science/article/pii/S1053811909000809) ""
- `'thalamic_nuclei'`: Registered [Thalamic Nuclei](https://www.nature.com/articles/s41597-021-01062-y) ""
- `'thalamus'`: Registered [Thalamus](https://www.nature.com/articles/sdata2018270) ""
- `'Schaefer2018_100Parcels_17Networks_order'`: Registered [Schaefer](https://academic.oup.com/cercor/article/28/9/3095/3978804?login=true) 100 ""
- `'Schaefer2018_200Parcels_17Networks_order'`: Registered [Schaefer](https://academic.oup.com/cercor/article/28/9/3095/3978804?login=true) 200 ""
- `'Schaefer2018_400Parcels_17Networks_order'`: Registered [Schaefer](https://academic.oup.com/cercor/article/28/9/3095/3978804?login=true) 400 ""
- `'Schaefer2018_600Parcels_17Networks_order'`: Registered [Schaefer](https://academic.oup.com/cercor/article/28/9/3095/3978804?login=true) 600 ""

For each atlas, there also can be outputted two more modalities:
- `'aal3_affine'`: Registered AAL3 atlas in template space
- `'aal3_volumes'`: GM, WM and CSF volume in mm³ per region of the AAL3 atlas
- ...analogous `'anatomy3_affine'`, `'anatomy3_volumes'`, `'cobra_affine'`, `'cobra_volumes'` ...
