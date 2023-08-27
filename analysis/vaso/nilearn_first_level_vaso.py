from nilearn.image import concat_imgs, mean_img
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from os.path import join, exists
from os import listdir, mkdir
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from nilearn.datasets import func
from nilearn.glm.first_level import make_first_level_design_matrix

input_dir = sys.argv[1]
output_dir = sys.argv[2]
tsv_dir = sys.argv[3]

# input_dir = '/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-04/func'
# output_dir = '/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-04/analysis/vaso'
# tsv_dir = '/Users/carricarte/scratch/projects/imagery/main/vaso/behdata/sub-04/'

# tr = 2.413  # repetition time is 1 second
tr = 4.826  # repetition time is 1 second
n_scans = 72  # the acquisition comprises 152 scans
frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

seq = "bold"  # if BOLD change the last condition to "cr_2vaso_rsub"; if VASO change to "corr_"
fmri_img = []
[fmri_img.append(join(input_dir, f)) for f in listdir(input_dir) if "sm_" not in f and seq in f and
 f.endswith(".nii") and "._" not in f and "mean" not in f and "loc" not in f
 and "cr_" not in f and "std_" not in f and "rsub" in f and "bc" not in f and "tr_" not in f
 and "corr" in f and "intemp" not in f and "motion" not in f and "correlated" not in f]
fmri_img.sort()

motion_files = []
[motion_files.append(join(input_dir, f)) for f in listdir(input_dir) if "rp_sub" in f and seq in f and
 f.endswith(".txt") and "._" not in f and "mean" not in f and "loc" not in f and "motion" in f]
motion_files.sort()

tsvfiles = []
[tsvfiles.append(join(tsv_dir, t)) for t in listdir(tsv_dir) if "glm" in t
 and "._" not in t and "block" in t and t.endswith(".tsv")]
tsvfiles.sort()

events = [pd.read_table(df) for df in tsvfiles]

design_matrices = []
for i, e in enumerate(events):
    design_matrices.append(make_first_level_design_matrix(frame_times, e,
                                                          add_regs=pd.read_table(motion_files[i], sep='  ', header=None),
                                                          hrf_model='spm'))  # add_regs=pd.read_table(motion_files[i], sep='  ', header=None)

fmri_glm = FirstLevelModel(standardize=False, hrf_model='spm', signal_scaling=0, drift_model='cosine', high_pass=.01)
fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

design_matrix = fmri_glm.design_matrices_[0]
contrast_matrix = np.eye(design_matrix.shape[1])
basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])

# contrasts = {
#     'seen_places-baseline': basic_contrasts['seen_berlin'] + basic_contrasts['seen_paris'] +
#                             basic_contrasts['seen_pisa'] - basic_contrasts['baseline'],
#     'img_places-baseline': basic_contrasts['img_berlin'] + basic_contrasts['img_paris'] +
#                            basic_contrasts['img_pisa'] - basic_contrasts['baseline'],
#     'stimulation-baseline': basic_contrasts['seen_berlin'] + basic_contrasts['seen_paris'] +
#                             basic_contrasts['seen_pisa'] + basic_contrasts['img_berlin'] +
#                             basic_contrasts['img_paris'] + basic_contrasts['img_pisa'] -
#                             basic_contrasts['baseline']
#     }

# category_contrasts = {
#     'seen_places': basic_contrasts['seen_berlin'] + basic_contrasts['seen_paris'] +
#                             basic_contrasts['seen_pisa'],
#     'img_places': basic_contrasts['img_berlin'] + basic_contrasts['img_paris'] +
#                            basic_contrasts['img_pisa'],
#     'stimulation-baseline': basic_contrasts['seen_berlin'] + basic_contrasts['seen_paris'] +
#                             basic_contrasts['seen_pisa'] + basic_contrasts['img_berlin'] +
#                             basic_contrasts['img_paris'] + basic_contrasts['img_pisa'] -
#                             basic_contrasts['baseline']
#     }

print('Computing contrasts')
result_path = join(output_dir, seq)
if not exists(result_path):
    mkdir(result_path)
# Iterate on contrasts
for contrast_id, contrast_val in basic_contrasts.items():
    if contrast_id == "baseline" or contrast_id == "img_place" or contrast_id == "seen_place":
        b_map = fmri_glm.compute_contrast(contrast_val, output_type='all')
        for key, value in b_map.items():
            value.to_filename(join(result_path, key + "_" + str(contrast_id) + ".nii"))
print("Done")
