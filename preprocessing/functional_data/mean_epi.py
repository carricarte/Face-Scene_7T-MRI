from os import listdir
from os.path import join
from nilearn.image import mean_img
import sys

input_dir = sys.argv[1]

epi_list = []
# [epi_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
#  and "cr_rsub" in _ and "sm_cr_rsub" not in _ and "wcr_rsub" not in _]

[epi_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "_correlated" in _ and "mean" not in _]

epi_list.sort()

mean_epi = mean_img(epi_list)
mean_epi.to_filename(join(input_dir, "mean_correlated.nii"))


from os.path import split

for epi in epi_list:
    mean_epi = mean_img(epi)
    path_epi, name_epi = split(epi)
    mean_epi.to_filename(join(input_dir, "mean_epi_" + name_epi))