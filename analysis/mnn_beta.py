import sys
import numpy as np
import image_stats
import scipy as sp
import random as rd
import pandas as pd
from os import listdir
from os.path import join
import rsatoolbox as rsa
from sklearn.svm import LinearSVC
from nilearn.masking import apply_mask
from sklearn.model_selection import GridSearchCV
from nilearn.image import new_img_like, concat_imgs, load_img


def beta_x_cond(inputdir, ids, run):

    all_betas = []
    id1 = ids[0]
    id2 = ids[1]
    for x in listdir(inputdir):
        if ("._" not in x and "_deveinDeconv" in x and "beta_" in x and "run-{0}".format(run) in x) \
                and (id1 in x or id2 in x):
            all_betas.append(join(inputdir, x))
    all_betas.sort()
    return all_betas


input_dir = sys.argv[1]
mask_dir = sys.argv[2]
tmap_dir = sys.argv[3]
layer_mask_file = sys.argv[4]
residuals_dir = sys.argv[5]
voxel = sys.argv[6]
condition = sys.argv[7]
r = sys.argv[8]


# input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/no_smoothing/beta/decoding"
# residuals_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/beta"
# mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask"
# tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/localizer"
# layer_mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask/rim_layers_equivol.nii"
# voxel = "50"
# condition = "FFA_per"
# r = "01"

run = join(residuals_dir, "run-{0}".format(r))

residual_files = []
[residual_files.append(join(run, _)) for _ in listdir(run) if "Res" in _ and "._" not in _]
residual_files.sort()
# _residuals = np.array(concat_imgs(residual_files).get_fdata()).astype(np.float32)
# nii_residuals = new_img_like(residual_files[0], _residuals)
# del _residuals

nan_mask = ~np.isnan(np.array(load_img(residual_files[0]).get_fdata()))

# Initialize variables
mask_file = join(mask_dir, "mask.nii")
left_mask_file = join(mask_dir, "left_mask.nii")
right_mask_file = join(mask_dir, "right_mask.nii")

conditions = {"FFA_per": ['0009', '0010'],  # perception faces
              "FFA_img": ['0004', '0005'],  # imagery faces
              "PPA_per": ['0008', '0011'],  # perception scenes
              "PPA_img": ['0003', '0006']}  # imagery scenes

item = conditions[condition]
# get betas per condition
betas = beta_x_cond(input_dir, item, r)
y = np.tile([0, 1], int(len(betas) / 2))
mask_layer_list = image_stats.get_layer(layer_mask_file)

# order = np.arange(0, int(len(y) / 2))
# svc = LinearSVC(penalty='l2', loss='hinge', C=1, multi_class='ovr', fit_intercept=True, max_iter=10000)
# c = [1]
# param_grid = dict(C=c)
# nperm = 1000
# num_subsamples = 5
# cv = 5

pre_mask_right = np.logical_and(np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(right_mask_file) != 0), nan_mask)
pre_mask_left = np.logical_and(np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(left_mask_file) != 0), nan_mask)

for file in betas:
    print(file)

df = pd.DataFrame()

if "PPA" in condition:
    tmap = "spmT_0010.nii"
elif "FFA" in condition:
    tmap = "spmT_0006.nii"

tmap_file = load_img(join(tmap_dir, tmap))

for k, layer_mask in enumerate(mask_layer_list):

    # compute mask
    mask_left = np.logical_and(pre_mask_left != 0, layer_mask)
    mask_right = np.logical_and(pre_mask_right != 0, layer_mask)

    tval_right = image_stats.img_mask(tmap_file, mask_right)
    tval_right.sort()
    tval_right = tval_right[::-1]
    t_right = tval_right[int(voxel)]
    mask_right = image_stats.threshold_mask(tmap_file, t_right, mask_right)

    tval_left = image_stats.img_mask(tmap_file, mask_left)
    tval_left.sort()
    tval_left = tval_left[::-1]
    t_left = tval_left[int(voxel)]
    mask_left = image_stats.threshold_mask(tmap_file, t_left, mask_left)

    mask = np.logical_or(mask_left, mask_right)
    mask = new_img_like(betas[0], mask)

    X = apply_mask(betas, mask)
    residual_matrix = apply_mask(residual_files, mask)

    # compute sigma given a matrix m x n = time points x voxels
    cov_ = rsa.data.noise._estimate_covariance(residual_matrix, dof=None, method='shrinkage_diag')
    sigma_inv = sp.linalg.fractional_matrix_power(cov_, -.5)

    # apply noise normalization to betas
    Xnorm = X @ sigma_inv
    df_xnorm = pd.DataFrame(data=Xnorm)
    df_xnorm.to_csv(join(input_dir, 'norm_beta_run-{0}_cond-{1}_vox-{2}_layer-{3}.tsv'.format(r, condition,  voxel, k + 1))
                       , sep='\t', index=True)
