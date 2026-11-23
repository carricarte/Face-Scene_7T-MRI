# Face-Scene 7T-MRI

Preprocessing and analysis pipeline for a **high-resolution (7 Tesla) laminar fMRI study of visual perception and mental imagery of faces and scenes**.

The study asks whether *seeing* and *imagining* faces and places engage different **cortical depths** in category-selective regions of ventral visual cortex: the fusiform face area (**FFA**) and the parahippocampal place area (**PPA**). The idea is that feedforward (perception) and feedback (imagery) signals are expected to arrive in different cortical layers.

---

## Experimental design

| Task | Purpose | Conditions |
|---|---|---|
| **Localizer** (`task-loc`) | Define FFA and PPA in each subject and hemisphere | `Faces`, `Places`, `Objects`, `Scrambled`, `Baseline` |
| **Main / imagery** (`task-img`) | Compare perception and imagery | `seen_*` / `img_*` × {`merkel`, `obama`, `berlin`, `paris`}, plus `visual_cue`, `auditory_cue`, `rvsp` and `baseline` |

The stimuli are two famous faces (Angela Merkel, Barack Obama) and two famous places (Berlin, Paris). Each one is either shown (*seen*) or imagined after a cue (*img*). Note that `analysis/glm/data2glm_task_img.py` spells the seen-Berlin label `seen_berin`.

- **Subjects:** `01`–`18` (the default set is the `#SBATCH --array` line in `main`)
- **Runs per subject:** 9–12 (the `N_RUNS` table in `analysis/glm/glm_analysis.py` and `analysis/glm/glm_analysis_task_img.py`)
- **Acquisition:** 7T GE-BOLD, with VASO variants for some sessions, and an MP2RAGE anatomical scan

---

## Repository structure

```
.
├── main                         # SLURM driver: runs one pipeline step for one subject
├── check_path.py                # Resolves and validates the directories for each step (from paths.json)
├── paths.json                   # Step number → [input_dir, output_dir, working_dir, script_dir, extra...]
├── utils/                       # Shared Python helpers (image_stats.py)
├── bids/                        # DICOM → NIfTI conversion and BIDS organisation
├── preprocessing/
│   ├── anatomical_data/
│   │   ├── mp2rage_denoise/     # MP2RAGE background-noise removal (MATLAB)
│   │   ├── bias_field_correction/  # SPM intensity normalisation (MATLAB)
│   │   ├── freesurfer/          # Brain mask, edge detection, high-res recon-all
│   │   └── layering/            # nighres / LayNII cortical layers and columns
│   └── functional_data/
│       ├── vaso/                # VASO BOLD correction
│       ├── distortion_correction/  # FSL topup
│       ├── realignment/         # SPM motion correction
│       ├── bias_correction/     # ANTs N4
│       ├── brain_mask/          # Mean EPI images and brain masks
│       ├── deveining/           # ALF + LayNII deveining
│       ├── registration/        # ANTs registration, transforms, SPM normalisation
│       ├── smoothing/           # Isotropic smoothing
│       ├── zscore/              # Time-series standardisation
│       └── utils/               # Volume utilities
├── analysis/
│   ├── glm/                     # events.tsv generation and first-level GLMs
│   ├── laminar/                 # Layer-wise response profiles in FFA/PPA
│   └── roi/                     # Ribbon-restricted ROIs, mean betas, ROI distances
└── quality_control/             # tSNR, motion, between-run spatial correlation
```

Each pipeline step runs its scripts from a single `script_dir` (the 4th entry of its `paths.json` list), and wrapper scripts call their helpers through that same `$SCRIPTDIR`. Keep a step's wrapper and the scripts it calls in the same folder. Scripts that need `image_stats` find `utils/` relative to their own location.

### `bids/`
- `bids_dcm2niix`, `dicom2nii_vaso`: DICOM → NIfTI with `dcm2niix`
- `folder2bids.py`, `folder2bids_vaso.py`: arrange the converted files in BIDS format
- `bids_conversion.sh`, `set_protocol.py`: helper scripts for the conversion

### `preprocessing/anatomical_data/`
- `mp2rage_denoise/`: remove MP2RAGE background noise (MATLAB)
- `bias_field_correction/`: SPM-based intensity normalisation (MATLAB)
- `freesurfer/`: `brainmask_spm.py` (SPM brain mask that replaces the FreeSurfer one), `edge_detector.py`, and `autorecon1`, `autorecon2_3`, `autorecon2_3_wm`, `autorecon3`, `autorecon3_pial`, `expert.opts` for high-resolution surface reconstruction
- `layering/`: `layerification` + `get_layer.py` (nighres), `layer_laynii` / `laynii_layer` (LayNII), and the ribbon utilities `mask_ribbon.py`, `ribbon_conv.py`

### `preprocessing/functional_data/`
- `vaso/`: `run_bold_correction` (VASO BOLD correction)
- `distortion_correction/`: `topup`, `topup.py`, `b02b0.cnf` (FSL topup)
- `realignment/`: `run_spm_realignment`, `spm_realignment.py` (SPM through nipype)
- `bias_correction/`: `n4_bias_field_correction`, `n4biascorr_timeseries.py` (ANTs N4)
- `brain_mask/`: `brainmask_epi.py`, `brainmask.py`, `epi_mask`, `mean_epi.py`, `mean_epimask.py`, `mean4d.py`
- `deveining/`: `ALF_melmac` (amplitude of low-frequency fluctuations), `devein(.py)`, `devein_laynii`, `multi_devein`, `run_multi_devein`, `volume_combine.py`; corrects the draining-vein bias across depth
- `registration/`: `ants_registration`, `run_multi_registration`, `ants_apply_transform(.py)`, `run_ants_apply_transform.py`, `combine_deformation`, `normalize(.py)`
- `smoothing/`: `run_smooth_img`, `smooth_img.py` (nilearn)
- `zscore/`: `run_multi_z-score`, `z-score_img(.py)`, `standarize_img.py`
- `utils/`: `volume_float16(.py)`, `get_volume.py`, `first_volume.py`, `replicate_volume`

### `analysis/`
- `glm/`: `data2glm_task_loc.py`, `data2glm_task_img.py` (behavioural logs → BIDS `events.tsv`) and `glm_analysis(.py)`, `glm_analysis_task_loc.py`, `glm_analysis_task_img.py` (first-level GLMs in SPM12 through nipype)
- `laminar/`: `laminar_profile`, `laminar_profile_control.py` (layer-wise response profiles within FFA and PPA)
- `roi/`: `ribbon_mask.py` (restrict ROIs to the cortical ribbon), `mean_beta(.py)` (average betas per ROI and condition), `euclidean_distance.py` (distance between FFA and PPA centres of mass)

### `utils/`
- `image_stats.py`: shared image helpers used by analysis, deveining, z-scoring, ribbon and tSNR scripts

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
- For some steps, the `script_dir` in `paths.json` doesn't contain the script that `main` runs, so the step can't work as configured: 17 (`run_bold_correction` → `quality_control/`), 18 (`spatial_corr.py` → `functional_data/`, with registration inputs), 36 (`volume_float16` → `analysis/multivariate/`) and 49 (`tsnr_control` → `analysis/`). These `paths.json` entries look like they come from an older step numbering.
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
3. Update the `N_RUNS` table in **both** `analysis/glm/glm_analysis.py` and `analysis/glm/glm_analysis_task_img.py` when adding subjects or dropping runs, and check the hardcoded paths in some analysis scripts (e.g. `analysis/glm/data2glm_task_img.py`).
4. Adjust the `#SBATCH` directives (QOS, time, memory, array) to fit your scheduler.

---

## Notes

- The repository holds code only. No imaging or behavioural data are included.
