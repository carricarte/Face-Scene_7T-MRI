from nilearn import image
from os.path import join
import sys

max_vol = 88

nifti_dir = sys.argv[1]
epi_list = []

epi_file = join(nifti_dir, "cr_rsub-17_task-img_run-03_bold.nii")


volume_nifti = image.index_img(epi_file, 39)
volume_nifti.to_filename(join(nifti_dir, "one_epi_vol"))
