import nibabel as nb
from nilearn import image
from nilearn.image import new_img_like, index_img
from os.path import split, join, exists
from os import listdir, mkdir
import sys
import numpy as np

max_vol = 88

nifti_dir = sys.argv[1]
epi_list = []

epi_file = join(nifti_dir, "cr_rsub-17_task-img_run-03_bold.nii")


volume_nifti = image.index_img(epi_file, 39)
volume_nifti.to_filename(join(nifti_dir, "one_epi_vol"))
