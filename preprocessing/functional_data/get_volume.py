from os import listdir
from os.path import join
from nilearn.image import index_img
import sys

input_dir = sys.argv[1]

epi = join(input_dir, "")
# epi_list = []
# [epi_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
#  and "cr_rsub" in _ and "sm_cr_rsub" not in _ and "wcr_rsub" not in _]
# epi_list.sort()

vol = 30
mean_epi = index_img(epi, vol)
mean_epi.to_filename(join(input_dir, "epi_{}.nii".format(vol)))