import sys
import numpy as np
import pandas as pd
import nibabel as nb
from os import listdir
from os.path import join
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
tmap_file = sys.argv[4]
tsv_dir = sys.argv[5]

# get epi and data
modality = ['perception', 'imagery']  # imagery or perception
category = ['face', 'place']
voxels = np.arange(100, 8000, 100) - 1
df = pd.DataFrame()
svc = SVC()
c = [1]
param_grid = dict(C=c)
grid = GridSearchCV(svc, param_grid, cv=2, scoring='accuracy')
df = pd.DataFrame(columns=['v' + str(i) for i in voxels + 1] + ['Modality'])

run_list = []
[run_list.append(join(input_dir, f)) for f in listdir(input_dir) if "run-" in f and '._' not in f
 and "00" not in f and "sm_" not in f]
run_list.sort()

# load mask and get threshold values according to the number of voxels
mask = nb.load(mask_file)
mask = mask.get_fdata()
tmap = nb.load(tmap_file)
tmap = tmap.get_fdata()
print(sum(sum(sum(mask != 0))))
tval = tmap[mask != 0]

tval.sort()
tval = tval[::-1]
threshold = tval[voxels]

for m in modality:

    if m == 'imagery':

        # a, b = ['0003', '0004']
        a, b = ['0004', '0005']
        c, d = ['0003', '0006']

    elif m == 'perception':

        # a, b = ['0006', '0007']
        a, b = ['0009', '0010']
        c, d = ['0008', '0011']

    acc = []

    # concatenate all functional volumes across runs in a 4D matrix
    beta = nb.load(join(input_dir, run_list[0], 'beta_0001.nii'))
    beta = beta.get_fdata()
    features = beta.reshape(-1)
    # features = features[new_mask]
    # features = features[np.logical_not(np.isnan(features))]
    feature_matrix = np.zeros(shape=(len(features), 4 * len(run_list)), dtype='float')
    beta_list = []
    for run in run_list:
        [beta_list.append(join(run, x)) for x in listdir(run) if "._" not in x and a in x or b in x
         or c in x or d in x]
    beta_list.sort()
    y = np.tile([1, 2, 2, 1], int(len(run_list)))
    index = 0
    for beta_file in beta_list:
        print(beta_file)
        beta = nb.load(beta_file)
        beta = beta.get_fdata()
        features = beta.reshape(-1)
        feature_matrix[:, index: index + 1] = np.reshape(features, [features.shape[0], 1])
        index = index + 1
        # print(feature_matrix[100, index:index + 1])

    print(feature_matrix.shape)
    for t in threshold:
        ffa_ppa = tmap >= t
        new_mask = np.logical_and(ffa_ppa, mask)
        new_mask = new_mask.reshape(-1)
        X = feature_matrix[new_mask]

        # remove rows with nan
        rows_with_nan, columns_with_nan = np.where(X != X)
        rows_with_nan = list(set(rows_with_nan))
        if rows_with_nan:
            X = np.delete(X, rows_with_nan, 0)

        X = X.T
        # y = np.tile([1, 2], int(X.shape[0] / 2))
        print(y)
        grid.fit(X, y)
        effect = grid.best_score_
        acc.append(effect)

    df.loc[len(df.index)] = acc + [m]
    df.to_csv(join(output_dir, 'glm_svc_accuracy_devein_linear.tsv'), sep='\t', index=True)
