import numpy as np
from os.path import join, split
import sys
import image_stats
from nilearn.image import new_img_like

mask_file = sys.argv[1]
output_dir = sys.argv[2]
mask_epi_file = sys.argv[3]
mask_manual_file = sys.argv[4]

path_file, name_file = split(mask_manual_file)

mask = image_stats.load_nifti(mask_file)
mask_manual = image_stats.load_nifti(mask_manual_file)
mask_epi = image_stats.load_nifti(mask_epi_file)

mask = np.logical_and(np.logical_and(mask != 0, mask_manual != 0), mask_epi != 0)
mask = new_img_like(mask_file, mask)
ribbon_epi_mask = join(path_file, "ribbon_mask.nii")
mask.to_filename(ribbon_epi_mask)
