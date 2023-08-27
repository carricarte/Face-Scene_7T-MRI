from os import listdir
from os.path import join
from nilearn.image import math_img, new_img_like, load_img
import numpy as np
from sklearn import preprocessing
import sys

# MEMORY ALLOCATION 3500
input_dir = "/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-08/func"
# input_dir = sys.argv[1]

seq = 'vaso'


def norm_epi(niimg, ori_shape):
    nifti = np.array(load_img(niimg).get_fdata())
    features = nifti.reshape(-1, nifti.shape[-1]).T
    mean_features = np.mean(features, 0)
    std_features = np.std(features, 0)
    psc_features = (features - mean_features)/std_features
    psc_features = psc_features.T
    psc_nifti = psc_features.reshape(ori_shape)
    return new_img_like(all_epi, psc_nifti)


def psc_epi(niimg, ori_shape):
    nifti = np.array(load_img(niimg).get_fdata())
    features = nifti.reshape(-1, nifti.shape[-1]).T
    mean_features = np.mean(features, 0)
    psc_features = (features/mean_features)*100
    psc_features = psc_features.T
    psc_nifti = psc_features.reshape(ori_shape)
    return new_img_like(all_epi, psc_nifti)


epi_list = []
[epi_list.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "_correlated" not in _ and "mean" not in _ and "corr" in _ and "motion" not in _]
epi_list.sort()

all_epi = epi_list[0]
nifti = np.array(load_img(all_epi).get_fdata())
orishape = nifti.shape
# nm_all_epi = norm_epi(all_epi, orishape)
# nm_all_epi = psc_epi(all_epi, orishape)
nm_all_epi = epi_list[0]

for epi in epi_list[1:]:
    print(epi)
    # nm_epi = norm_epi(epi, orishape)
    # nm_epi = psc_epi(epi, orishape)
    nm_epi = epi
    all_epi = math_img("img1 + img2", img1=nm_all_epi, img2=nm_epi)

mean_epi = np.array(all_epi.get_fdata() / len(epi_list)).astype(np.float32)
# if seq == "vaso":
#     mean_epi = -1*mean_epi
nm_all_epi = new_img_like(all_epi, mean_epi)
nm_all_epi.to_filename(join(input_dir, "mean4d_{}.nii".format(seq)))
