# Face-Scene 7T-MRI

Preprocessing and analysis pipeline for a **high-resolution (7 Tesla) laminar fMRI study of visual perception and mental imagery of faces and scenes**.

The study asks whether *seeing* and *imagining* faces and places engage different **cortical depths** in category-selective regions of ventral visual cortex: the fusiform face area (**FFA**) and the parahippocampal place area (**PPA**). The idea is that feedforward (perception) and feedback (imagery) signals are expected to arrive in different cortical layers.

---

## Experimental design

| Task | Purpose | Conditions |
|---|---|---|
| **Localizer** (`task-loc`) | Define FFA and PPA in each subject and hemisphere | `Faces`, `Places`, `Objects`, `Scrambled`, `Baseline` |
| **Main / imagery** (`task-img`) | Compare perception and imagery | `seen_*` / `img_*` × {`merkel`, `obama`, `berlin`, `paris`}, plus `visual_cue`, `auditory_cue`, `rvsp` and `baseline` |

The stimuli are two famous faces (Angela Merkel, Barack Obama) and two famous places (Berlin, Paris). Each one is either shown (*seen*) or imagined after a cue (*img*). Note that `data2glm_task_img.py` spells the seen-Berlin label `seen_berin`.

- **Subjects:** `01`–`18` (the default set is the `#SBATCH --array` line in `main`)
- **Runs per subject:** 9–12 (the `N_RUNS` table in `analysis/glm_analysis.py` and `analysis/glm_analysis_task_img.py`)
- **Acquisition:** 7T GE-BOLD, with VASO variants for some sessions, and an MP2RAGE anatomical scan

---

## Repository structure

```
.
├── main                     # SLURM driver: runs one pipeline step for one subject
├── check_path.py            # Resolves and validates the directories for each step (from paths.json)
├── paths.json               # Step number → [input_dir, output_dir, working_dir, script_dir, extra...]
├── bids/                    # DICOM → NIfTI conversion and BIDS organisation
├── preprocessing/
│   ├── anatomical_data/     # MP2RAGE denoising, bias correction, FreeSurfer, cortical layering
│   └── functional_data/     # Realignment, distortion correction, registration, deveining, ...
├── analysis/                # GLMs, ROI masks, laminar profiles, mean betas, ROI distances
├── quality_control/         # tSNR, motion, between-run spatial correlation
└── meta_learning.ipynb      # Unrelated (see Notes)
```

### `bids/`
- `bids_dcm2niix`, `dicom2nii_vaso`: DICOM → NIfTI with `dcm2niix`
- `folder2bids.py`, `folder2bids_vaso.py`: arrange the converted files in BIDS format
- `bids_conversion.sh`, `set_protocol.py`: helper scripts for the conversion

### `preprocessing/anatomical_data/`
- `mp2rage_denoise/`: remove MP2RAGE background noise (MATLAB)
- `bias_field_correction/`: SPM-based intensity normalisation (MATLAB)
- `brainmask_spm.py`: SPM brain mask that replaces the FreeSurfer one
- `autorecon1`, `autorecon2_3`, `autorecon2_3_wm`, `autorecon3`, `autorecon3_pial`, `expert.opts`: FreeSurfer surface reconstruction at high resolution
- `layerification` (nighres), `layer_laynii` / `laynii_layer` (LayNII): divide the cortex into layers and columns
- `mask_ribbon.py`, `ribbon_conv.py`, `get_layer.py`, `edge_detector.py`: utilities for the cortical ribbon and layer masks

### `preprocessing/functional_data/`
- `spm_realignment.py` / `run_spm_realignment`: motion correction (SPM through nipype)
- `topup.py`, `topup`, `b02b0.cnf`: distortion correction with FSL topup
- `run_bold_correction`: VASO BOLD correction
- `n4biascorr_timeseries.py`, `n4_bias_field_correction`: ANTs N4 bias correction
- `brainmask_epi.py`, `epi_mask`, `mean_epi.py`, `mean_epimask.py`, `mean4d.py`: mean EPI images and brain masks
- `ALF_melmac`: amplitude of low-frequency fluctuations, used as input for deveining
- `devein.py`, `devein`, `devein_laynii`, `multi_devein`, `run_multi_devein`: LayNII deveining, which corrects the draining-vein bias across depth
- `ants_registration`, `run_multi_registration`, `ants_apply_transform(.py)`, `run_ants_apply_transform.py`, `combine_deformation`: register EPI to the anatomy with ANTs
- `normalize(.py)`: SPM normalisation
- `smooth_img.py`, `run_smooth_img`: isotropic smoothing (nilearn)
- `z-score_img(.py)`, `run_multi_z-score`, `standarize_img.py`: standardise the time series
- `volume_float16(.py)`, `volume_combine.py`, `get_volume.py`, `first_volume.py`, `replicate_volume`: volume utilities

### `analysis/`
- `data2glm_task_loc.py`, `data2glm_task_img.py`: turn behavioural logs into BIDS-style `events.tsv` files
- `glm_analysis(.py)`, `glm_analysis_task_loc.py`, `glm_analysis_task_img.py`: first-level GLMs in SPM12 through nipype
- `ribbon_mask.py`: restrict ROIs to the cortical ribbon
- `laminar_profile`, `laminar_profile_control.py`, `image_stats.py`: layer-wise response profiles within FFA and PPA
- `mean_beta(.py)`: average beta estimates per ROI and condition
- `euclidean_distance.py`: distance between the FFA and PPA centres of mass in each hemisphere

### `quality_control/`
- `tsnr_analysis.py`, `tsnr_comp.py`, `tsnr_control`: temporal SNR, including per layer
- `motion_check.py`: summaries of the realignment parameters
- `spatial_corr.py`: correlation of mean intensity between runs, to check the realignment

---

## How the pipeline runs

Every step goes through the `main` SLURM script:

```bash
sbatch main <STEP>
```

The script reads the subject from `$SLURM_ARRAY_TASK_ID` (zero-padded to `01`, `02`, …) and:

1. calls `check_path.py <pipeline_dir> <STEP> <SUBJECT>`, which evaluates the path expressions for that step in `paths.json` (they are Python strings that use `subject`, e.g. `'/.../sub-{0}/anat'.format(subject)`). `'EXCLUDE'` marks a slot that isn't used;
2. stops unless every directory exists (exit code `0`);
3. dispatches the script for that step (`sbatch` or `python`), passing the resolved directories.

To submit for specific subjects only, override the array on the command line, e.g. `sbatch --array=5 main 29`. Use `--array=1` for steps that don't depend on the subject (e.g. 1, 2, 26); otherwise every array task repeats the same work.

### Steps

Steps are numbered by stage. The comment above each `if [ "$STEP" == ... ]` block in `main` describes it.

| Steps | Stage |
|---|---|
| 1–2 | File organisation (DICOM → NIfTI, BIDS) |
| 3–13 | Anatomical preprocessing (MP2RAGE, FreeSurfer, nighres/LayNII layering) |
| 14–27, 250 | Functional preprocessing (VASO split, topup, realignment, N4, deveining, ANTs registration, smoothing, z-scoring) |
| 28–41, 54 | Analysis (GLM events and first level, decoding, searchlight, mean betas, laminar profiles; 54 = time series → dataframe) |
| 42–43 | ROI extraction |
| 44–53 | Statistics and control analyses |

### Known gaps

- Some steps call scripts that are **not in this repository**: `anatomy`, `upsample`, `run_split_nulled_not_nulled`, `data2glm_vaso2.py`, `univariate_avg`, `nifti2dframe`, `svc_*`, `beta_decoding`, `multi_beta_decoding`, `contrast_profile`, `tmap2roi`, `roi2mask`, `prevalence`, `control`, `nilearn_first_level`, `trial_avg_pilot.py` and `layer_timeseries.py`.
- Steps 16, 22, 38, 42, 47, 51 and 250 have no entry in `paths.json`; `check_path.py` reports this and the step does not run.
- Some block comments in `main` describe an older command: step 6 says "freesurfer part 1" but runs `upsample`, and step 8 says "freesurfer part 2 & 3" but runs `edge_detector.py`.

---

## Requirements

**Software**
- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) and MATLAB (R2018b was used)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [ANTs](http://stnava.github.io/ANTs/)
- [FSL](https://fsl.fmrib.ox.ac.uk/) (topup)
- [LayNII](https://github.com/layerfMRI/LAYNII) and [nighres](https://nighres.readthedocs.io/)
- [dcm2niix](https://github.com/rordenlab/dcm2niix)
- [MP2RAGE-related-scripts](https://github.com/JosePMarques/MP2RAGE-related-scripts)
- SLURM (the `main` driver and most wrappers are `sbatch` scripts)

**Python packages**
`nipype`, `nilearn`, `nibabel`, `numpy`, `scipy`, `pandas`

---

## Adapting the pipeline to another environment

All paths are hardcoded for the original HPC cluster (`/home/carricarte/...`, `/scratch/carricarte/...`). To run it somewhere else:

1. Edit `PIPELINEDIR`, `SPMDIR` and `MATLABDIR` at the top of `main`.
2. Update the path expressions in `paths.json`. They are evaluated by `check_path.py` with `subject` and `main_dir` (the pipeline directory) in scope, so script paths can be written as `os.path.join(main_dir, 'bids')` and data roots changed with a single find-and-replace.
3. Update the `N_RUNS` table in **both** `analysis/glm_analysis.py` and `analysis/glm_analysis_task_img.py` when adding subjects or dropping runs, and check and the hardcoded paths in some analysis scripts (e.g. `analysis/data2glm_task_img.py`).
4. Adjust the `#SBATCH` directives (QOS, time, memory, array) to fit your scheduler.

---

## Notes

- `meta_learning.ipynb` is a PyTorch/BERT text-classification template. It has nothing to do with the MRI pipeline and was probably committed by mistake.
- The repository holds code only. No imaging or behavioural data are included.
