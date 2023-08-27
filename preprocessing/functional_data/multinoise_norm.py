from os import listdir
from os.path import split
from os.path import join
from nilearn.image import mean_img
import numpy as np
import scipy as sp
from nilearn.image import concat_imgs, new_img_like
from nilearn.masking import apply_mask
from analysis import image_stats
from sklearn.discriminant_analysis import _cov
import sys

input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-01/analysis/imagery"
mask = "/Users/carricarte/PhD/Debugging/bold/sub-01/mask/mask.nii"
# input_dir = sys.argv[1]
# input_dir = sys.argv[2]

residual_files = []
[residual_files.append(join(input_dir, _)) for _ in listdir(input_dir) if "Res" in _ and "._" not in _]
residual_files.sort()

residual_file = concat_imgs(residual_files)
residual_matrix = image_stats.time_series_to_matrix(residual_file, mask)
n_time = residual_matrix.shape[1]

residual_matrix = residual_matrix[~np.isnan(residual_matrix).any(axis=1)]
residual_matrix = residual_matrix[range(10), :].T

# compute sigma given a matrix mxn with m corresponding to the observations and n the variables
sigma = _cov(residual_matrix, shrinkage='auto')
sigma_inv = sp.linalg.fractional_matrix_power(sigma, -0.5)
print("done")

# apply sigma_inv
