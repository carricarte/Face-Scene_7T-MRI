from nipype.interfaces.spm import Realign
from os.path import join
from nilearn.image import index_img, new_img_like
import nibabel as nb
from os.path import splitext, split
from shutil import copyfile
from os import rename, listdir
import numpy as np
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

epi_list = []
[epi_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "sub" in _ and "sm_cr_rsub" not in _ and "wcr_rsub" not in _
 and "cr_rsub" not in _ and "rsub" not in _ and "mean" not in _
 and "gz" not in _]
epi_list.sort()


for epi_file in epi_list:

    filepath, filename = split(epi_file)

    if "loc" in epi_file:

        copyfile(epi_file, join(output_dir, filename))

    else:

        # I am loading the files twice. Write script more efficeint please
        epi = nb.load(epi_file)
        max_vol = epi.shape[3]

        # remove the first three non-staedy state volumes WE NEED ALL VOLUMES IN TOPUP RUNS
        # not_nulled_indexes = slice(3, max_vol - 6, 2)  # for subjects 1-7 discard just the first volume
        # nulled_indexes = slice(2, max_vol - 6, 2)  # for subjects 1-7 discard just the first volume

        not_nulled_indexes = slice(7, max_vol, 2)  # for subjects 1-7 discard just the first volume
        nulled_indexes = slice(6, max_vol, 2)  # for subjects 1-7 discard just the first volume

        nulled_epi = np.array(index_img(epi, nulled_indexes).get_fdata()).astype(np.float32)
        not_nulled_epi = np.array(index_img(epi, not_nulled_indexes).get_fdata()).astype(np.float32)
        nulled_nifti = new_img_like(epi, nulled_epi)
        not_nulled_nifti = new_img_like(epi, not_nulled_epi)

        nulled_nifti.to_filename(join(output_dir, filename))
        not_nulled_nifti.to_filename(join(output_dir, filename[:-9] + "_bold.nii"))
        print(epi_file)
