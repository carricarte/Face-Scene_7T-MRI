from os import listdir
from os.path import join
from nilearn.image import new_img_like, load_img
import numpy as np
import sys

# MEMORY ALLOCATION 3500
input_dir = sys.argv[1]

seq = 'vaso'

epi_list = []
[epi_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "_correlated" not in _ and "mean" not in _ and "corr" in _ and "motion" not in _]
epi_list.sort()

# Voxel- and volume-wise mean across runs (all runs must have the same 4D shape)
sum_epi = load_img(epi_list[0]).get_fdata()
for epi in epi_list[1:]:
    print(epi)
    sum_epi += load_img(epi).get_fdata()

mean_epi = (sum_epi / len(epi_list)).astype(np.float32)
# if seq == "vaso":
#     mean_epi = -1*mean_epi
new_img_like(epi_list[0], mean_epi).to_filename(join(input_dir, "mean4d_{}.nii".format(seq)))
