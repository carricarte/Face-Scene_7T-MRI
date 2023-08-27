import sys
import numpy as np
from nilearn.image import new_img_like, load_img
import pandas as pd
import image_stats
from os.path import join

nifti_file = sys.argv[1]
output_dir = sys.argv[2]
tsv_file = sys.argv[3]
mask_file = sys.argv[4]

# nifti_file = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/func/mean4d_bold.nii"
# output_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/func/"
# tsv_file = "/Users/carricarte/scratch/projects/imagery/main/vaso/behdata/sub-04/glm_block_sub04_run01.tsv"
# mask_file = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/mask/loc_mask.nii"

trigger = pd.read_table(tsv_file)
tr = np.tile(np.array([1, 2, 3]), len(trigger['trial_type']))
trigger = np.repeat(np.array(trigger['trial_type']), 3)

time_series = pd.DataFrame(image_stats.time_series_to_mat(nifti_file).T)
time_series['condition'] = trigger
time_series['order'] = tr
ori_shape = load_img(mask_file).get_fdata().shape

groups = time_series.groupby(['order', 'condition'])
cond = ['seen_place', 'img_place']
for tr in [1, 2, 3]:
    for c in cond:
        volume = np.mean(np.array(groups.get_group((tr, c)))[:, :-2], 0) - \
                 np.mean(np.array(groups.get_group((tr, 'baseline')))[:, :-2], 0)
        volume = np.array(volume).reshape(ori_shape).astype(np.float32)
        mean_file = new_img_like(mask_file, volume)
        mean_file.to_filename(join(output_dir, "mean-{0}_tr-{1}_vaso.nii".format(c, tr)))
