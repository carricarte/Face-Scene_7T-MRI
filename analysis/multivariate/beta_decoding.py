import sys
import numpy as np
from os import listdir
from os.path import join
from nilearn.masking import apply_mask
<<<<<<< HEAD
from sklearn.svm import LinearSVC
=======
from sklearn.svm import SVC
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold
import image_stats
from nilearn.image import new_img_like
import pandas as pd


def beta_x_cond(inputdir, conditions):

<<<<<<< HEAD
    beta = [['0004', '0005'], ['0003', '0006']]  # imagery
    # beta = [['0009', '0010'], ['0008', '0011']]  #perception
=======
    # beta = [['0004', '0005'], ['0003', '0006']]  # imagery
    beta = [['0009', '0010'], ['0008', '0011']]  #perception
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302

    # run_list = []
    # [run_list.append(join(inputdir, f)) for f in listdir(inputdir) if "run-" in f and '._' not in f
    #  and "00" not in f and "sm_" not in f]

    # run_list.sort()
    all_betas = []
    # for run in run_list:
    for j, cond in enumerate(conditions):
        sub_string_1, sub_string_2 = beta[j]
        for x in listdir(inputdir):
<<<<<<< HEAD
            if ("._" not in x and "dm_" in x and "beta_" in x) and (
                    sub_string_1 in x or sub_string_2 in x) \
                    and "rh" not in x and "lh" not in x:
                all_betas.append(join(inputdir, x))

    all_betas.sort()
    return all_betas



input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_dir = sys.argv[3]
tmap_dir = sys.argv[4]
layer_mask_file = sys.argv[5]

# input_dir = "/Users/carricarte/scratch/projects/object/simulation/sub-01/rsa/demean"
# # input_dir = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/derivatives/sub-01/analysis/no_smoothing/beta/decoding"
# output_dir = "/Users/carricarte/PhD/Debugging/bold/sub-01/analysis/beta"
# mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-01/mask"
# tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-01/analysis/localizer"
# layer_mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-01/anat/rim_layers_equivol.nii"
=======
            if ("._" not in x and "_deveinDeconv" not in x and "beta_" in x) and (sub_string_1 in x or sub_string_2 in x)\
                    and "rh" not in x and "lh" not in x:
                all_betas.append(join(inputdir, x))

    return all_betas


input_dir = sys.argv[1]
mask_dir = sys.argv[2]
tmap_dir = sys.argv[3]
layer_mask_file = sys.argv[4]

# input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/beta"
# mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask"
# tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/localizer"
# layer_mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask/rim_layers_equivol.nii"
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302

mask_file = join(mask_dir, "mask.nii")
left_mask_file = join(mask_dir, "left_mask.nii")
right_mask_file = join(mask_dir, "right_mask.nii")

conditions = ['face', 'place']
layers = ["deep", "middle", "superficial"]
area = ["FFA", "PPA"]

voxel = np.arange(50, 1550, 50)
mask_layer_list = image_stats.get_layer(layer_mask_file)

<<<<<<< HEAD
svc = LinearSVC(penalty='l2', loss='hinge', C=1, multi_class='ovr', fit_intercept=True, max_iter=10000)
=======
svc = SVC()
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
c = [1]
# c = [0.1, 1, 10]
param_grid = dict(C=c)

betas = beta_x_cond(input_dir, conditions)
<<<<<<< HEAD
y = np.tile([1, 0, 0, 1], int(len(betas)/4))
=======
y = np.tile([0, 0, 1, 1], int(len(betas)/4))
# df = pd.DataFrame()
# df = pd.DataFrame(columns=[voxel])
# df = pd.DataFrame(columns=["deep", "middle", "superficial", "area"])
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302


for a in area:
    df = pd.DataFrame()

    if a == "PPA":
        tmap = "spmT_0010.nii"
    elif a == "FFA":
        tmap = "spmT_0006.nii"

<<<<<<< HEAD
=======
    # acc = []
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
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
            X = apply_mask(betas, mask)

<<<<<<< HEAD
            cv = int(len(betas)/2)

            grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
=======
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

            # cv = KFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(svc, param_grid, cv=2, scoring='accuracy')
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
            grid.fit(X, y)
            acc.append(grid.best_score_)

        df[v] = acc
<<<<<<< HEAD
    df["layer"] = layers
    df.to_csv(join(output_dir, '{0}_img_decoding_cat_demean_trial_cv-max.tsv'.format(a)), sep='\t', index=True)
=======
        # df.loc[len(df.index)] = acc + [a]
    df["layer"] = [1, 2, 3]
    df.to_csv(join(input_dir, '{0}_decoding_no_devein_per.tsv'.format(a)), sep='\t', index=True)
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
