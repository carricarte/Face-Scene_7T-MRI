from nilearn.image import mean_img, new_img_like
from nilearn.masking import compute_brain_mask
from os import listdir
from os.path import join, split
import numpy as np
import sys

mean_epi = sys.argv[1]
input_dir, filename = split(mean_epi)
# input_dir = "/Users/carricarte/scratch/projects/imagery/pilot_07/derivatives/sub-01/func"

# niftifiles = []
# [niftifiles.append(join(input_dir,t)) for t in listdir(input_dir)
#  if "wcr_rsub" in t and t.endswith(".nii") and "._" not in t]
# niftifiles.sort()
#
# mean_epi = mean_img(niftifiles)

epi_mask = compute_brain_mask(mean_epi)
# mean_epi_data = mean_epi.get_fdata()
# mean_epi_data[mean_epi_data > 90] = 1
# mean_epi_data[mean_epi_data <= 90] = 0
# mean_epi_mask = np.array(bool(mean_epi)).astype(int)

# mean_epimask = new_img_like(mean_epi, mean_epi_data)
# mean_epi.to_filename(join(input_dir, 'mean_epi.nii'))
epi_mask.to_filename(join(input_dir, 'mask_mean_epi.nii'))
