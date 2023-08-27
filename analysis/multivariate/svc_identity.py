import sys
import numpy as np
from os import listdir
from os.path import join
from nilearn.masking import apply_mask
from sklearn.svm import LinearSVC
import random as rd
from sklearn.model_selection import GridSearchCV
import image_stats
from nilearn.image import new_img_like, load_img
import pandas as pd


def beta_x_cond(inputdir, conditions, run_list):

    all_betas = [[], [], [], []]
    # for run in run_list:
    for e, k in enumerate(conditions.keys()):
        id1 = conditions[k][0]
        id2 = conditions[k][1]
        for x in listdir(inputdir):
            if ("._" not in x and "_deveinDeconv" in x and "beta_" in x) and (id1 in x or id2 in x):
                all_betas[e].append(join(inputdir, x))
        all_betas[e].sort()
    return all_betas


input_dir = sys.argv[1]
mask_dir = sys.argv[2]
tmap_dir = sys.argv[3]
layer_mask_file = sys.argv[4]
residuals_dir = sys.argv[5]

# input_dir = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/derivatives/sub-18/analysis/no_smoothing/beta/decoding"
# residuals_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/beta"
# mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask"
# tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/localizer"
# layer_mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask/rim_layers_equivol.nii"

run_list = []
[run_list.append(join(residuals_dir, f)) for f in listdir(residuals_dir) if "run-" in f and '._' not in f
 and "00" not in f and "sm_" not in f]
run_list.sort()

# Initialize variables
mask_file = join(mask_dir, "mask.nii")
left_mask_file = join(mask_dir, "left_mask.nii")
right_mask_file = join(mask_dir, "right_mask.nii")
conditions = ['face', 'place']

conditions = {"FFA_per": ['0009', '0010'],  # perception faces
              "FFA_img": ['0004', '0005'],  # imagery faces
              "PPA_per": ['0008', '0011'],  # perception scenes
              "PPA_img": ['0003', '0006']}  # imagery scenes

layers = ["deep", "middle", "superficial"]
area = ["PPA"]
voxel = np.arange(50, 800, 50)
mask_layer_list = image_stats.get_layer(layer_mask_file)
svc = LinearSVC(penalty='l2', loss='hinge', C=1, multi_class='ovr', fit_intercept=True, max_iter=10000)
c = [1]
param_grid = dict(C=c)
nperm = 1000
num_subsamples = 5
cv = 5

pre_mask_right = np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(right_mask_file) != 0)
pre_mask_left = np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(left_mask_file) != 0)

# get betas per condition
betas = beta_x_cond(input_dir, conditions, run_list)
y = np.tile([0, 1], int(len(betas[0]) / 2))

for beta_group in betas:
    for file in beta_group:
        print(file)
    print("---------------------------------------------------------------------------------------------------")
print(y)

order = np.arange(0, int(len(y) / 2))

for x, c in enumerate(conditions.keys()):
    df = pd.DataFrame()

    if "PPA" in c:
        tmap = "spmT_0010.nii"
    elif "FFA" in c:
        tmap = "spmT_0006.nii"

    tmap_file = load_img(join(tmap_dir, tmap))

    for v in voxel:
        acc = []
        for k, layer_mask in enumerate(mask_layer_list):

            mask_left = np.logical_and(pre_mask_left != 0, layer_mask)
            mask_right = np.logical_and(pre_mask_right != 0, layer_mask)

            tval_right = image_stats.img_mask(tmap_file, mask_right)
            tval_right.sort()
            tval_right = tval_right[::-1]
            t_right = tval_right[v]
            mask_right = image_stats.threshold_mask(tmap_file, t_right, mask_right)

            tval_left = image_stats.img_mask(tmap_file, mask_left)
            tval_left.sort()
            tval_left = tval_left[::-1]
            t_left = tval_left[v]
            mask_left = image_stats.threshold_mask(tmap_file, t_left, mask_left)

            mask = np.logical_or(mask_left, mask_right)
            mask = new_img_like(betas[0], mask)  # compute mask

            X = apply_mask(betas[x], mask)

            # grid = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy')
            # grid.fit(X, y)
            # acc.append(grid.best_score_)

            _acc = []

            for n in range(0, nperm):

                pseudo_X = []
                rd.shuffle(order)
                sub_indexes = np.array_split(order, num_subsamples)

                for class_ in np.unique(y):

                    class_index = y == class_

                    sub_X = X[class_index, :]

                    for i, sub_index in enumerate(sub_indexes):

                        pseudo_X.append(np.mean(sub_X[sub_index], 0))

                pseudo_X = np.array(pseudo_X)
                pseudo_y = np.repeat([0, 1], int(len(pseudo_X) / 2))

                # Split the data into training and testing sets
                # X_train, X_test, y_train, y_test = train_test_split(pseudo_X, pseudo_y, test_size=0.5, random_state=42)
                #
                # # Create the AdaBoost classifier
                # adaboost = AdaBoostClassifier(base_estimator=svc, n_estimators=100, random_state=42, algorithm='SAMME')
                #
                # # Train the AdaBoost classifier
                # adaboost.fit(X_train, y_train)
                #
                # # Make predictions on the test set
                # y_pred = adaboost.predict(X_test)
                #
                # # Evaluate the accuracy of the model
                # accuracy = accuracy_score(y_test, y_pred)

                grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
                grid.fit(pseudo_X, pseudo_y)
                _acc.append(grid.best_score_)
                # _acc.append(accuracy)
            acc.append(np.mean(_acc))
        df[v] = acc
    df["layer"] = ["deep", "middle", "superficial"]
    df.to_csv(join(input_dir, '{0}_decoding_id_devein_no-mnn_pseudo-{1}_cv-{2}.tsv'.format(c, num_subsamples, cv)), sep='\t', index=True)
