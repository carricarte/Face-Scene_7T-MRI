import sys
import numpy as np
from os import listdir
from os.path import join
from nilearn.masking import apply_mask
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
import random as rd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import _cov
import image_stats
from nilearn.image import new_img_like, concat_imgs
import pandas as pd
import scipy as sp


def beta_x_cond(inputdir, conditions, run_list):
    beta = ['0009', '0010']  # perception faces
    # beta = ['0008', '0011']  # perception scenes
    # beta = ['0004', '0005']  # imagery faces
    # beta = ['0003', '0006']  # imagery scenes

    all_betas = []
    for run in run_list:
        for j, cond in enumerate(conditions):
            class_obj = beta[j]
            for x in listdir(join(inputdir, run)):
                if ("._" not in x and "_deveinDeconv" not in x and "beta_" in x) and (class_obj in x) \
                        and "rh" not in x and "lh" not in x:
                    # print(join(inputdir, run, x))
                    all_betas.append(join(inputdir, run, x))

    return all_betas


def main(input_dir, mask_dir, tmap_dir, layer_mask_file):
    run_list = []
    [run_list.append(join(input_dir, f)) for f in listdir(input_dir) if "run-" in f and '._' not in f
     and "00" not in f and "sm_" not in f]
    run_list.sort()

    # Initialize variables
    mask_file = join(mask_dir, "mask.nii")
    left_mask_file = join(mask_dir, "left_mask.nii")
    right_mask_file = join(mask_dir, "right_mask.nii")
    conditions = ['face', 'place']
    layers = ["deep", "middle", "superficial"]
    area = ["FFA"]
    voxel = np.arange(50, 300, 50)
    mask_layer_list = image_stats.get_layer(layer_mask_file)
    svc = SVC()
    c = [1]
    # c = [0.1, 1, 10]
    param_grid = dict(C=c)
    nperm = 1000
    num_subsamples = 3
    subsample_size = 0.3
    subsample_iterator = ShuffleSplit(n_splits=num_subsamples, test_size=subsample_size)

    # get betas per condition and labels
    betas = beta_x_cond(input_dir, conditions, run_list)
    y = np.tile([0, 1], int(len(betas) / 2))

    order = np.arange(0, int(len(y) / 2))

    for a in area:
        df = pd.DataFrame()

        if a == "PPA":
            tmap = "spmT_0010.nii"
        elif a == "FFA":
            tmap = "spmT_0006.nii"

        tmap_file = join(tmap_dir, tmap)

        for v in voxel:
            acc = []
            for k, layer_mask in enumerate(mask_layer_list):
                pre_mask_right = np.logical_and(
                    np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(right_mask_file) != 0)
                    , layer_mask)

                pre_mask_left = np.logical_and(
                    np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(left_mask_file) != 0)
                    , layer_mask)

                tval_right = image_stats.img_mask(tmap_file, pre_mask_right)
                tval_right.sort()
                tval_right = tval_right[::-1]
                t_right = tval_right[v]
                mask_right = image_stats.threshold_mask(tmap_file, t_right, pre_mask_right)

                tval_left = image_stats.img_mask(tmap_file, pre_mask_left)
                tval_left.sort()
                tval_left = tval_left[::-1]
                t_left = tval_left[v]
                mask_left = image_stats.threshold_mask(tmap_file, t_left, pre_mask_left)

                mask = np.logical_or(mask_left, mask_right)
                mask = new_img_like(betas[0], mask)

                sigmas = []
                for run in run_list:
                    residual_files = []
                    [residual_files.append(join(input_dir, run, _)) for _ in listdir(join(input_dir, run)) if
                     "Res" in _ and "._" not in _]
                    residual_files.sort()

                    residual_file = concat_imgs(residual_files)
                    residual_matrix = image_stats.time_series_to_matrix(residual_file, mask)
                    n_time = residual_matrix.shape[1]

                    residual_matrix = residual_matrix[~np.isnan(residual_matrix).any(axis=1)].T

                    sigma = _cov(residual_matrix, shrinkage='auto')
                    sigma_inv = sp.linalg.fractional_matrix_power(sigma, -0.5)
                    sigmas.append(sigma_inv)

                X = apply_mask(betas, mask)

                Xnorm = np.empty_like(X)

                for class_ in np.unique(y):
                    class_index = y == class_
                    class_X = X[class_index, :]
                    trial_index = np.array_split(range(class_X.shape[0]), len(run_list))

                    for e, itrial in enumerate(trial_index):
                        class_X[itrial, :] = np.dot(class_X[itrial, :], sigmas[e]).reshape(1, -1)

                    Xnorm[class_index, :] = class_X

                _acc = []
                for n in range(0, nperm):
                    pseudo_X = []
                    rd.shuffle(order)
                    sub_indexes = np.array_split(order, num_subsamples)

                    for c in conditions:
                        if c == "face":
                            s = np.arange(int(len(y) / 2))
                        elif c == "place":
                            s = np.arange(int(len(y) / 2), len(y))
                        sub_X = Xnorm[s, :]

                        for i, sub_index in enumerate(sub_indexes):
                            pseudo_X.append(np.mean(sub_X[sub_index], 0))

                    pseudo_X = np.array(pseudo_X)
                    pseudo_y = np.tile([0, 1], int(len(pseudo_X) / 2))

                    grid = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy')
                    grid.fit(pseudo_X, pseudo_y)
                    _acc.append(grid.best_score_)
                acc.append(np.mean(_acc))

            df[v] = acc
        df["layer"] = ["deep", "middle", "superficial"]
        df.to_csv(join(input_dir, '{0}_decoding_'.format(a) + '.csv'))


if __name__ == "__main__":

    input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/beta"
    mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask"
    tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/localizer"
    layer_mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask/rim_layers_equivol.nii"

    # input_dir = sys.argv[1]
    # mask_dir = sys.argv[2]
    # tmap_dir = sys.argv[3]
    # layer_mask_file = sys.argv[4]
    main(input_dir, mask_dir, tmap_dir, layer_mask_file)
