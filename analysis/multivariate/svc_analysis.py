import sys
import numpy as np
import pandas as pd
import random as rd
import nibabel as nb
from math import ceil
from os import listdir
from os.path import join
from sklearn.svm import SVC
from sklearn import preprocessing
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV

# directories
nifti_dir = sys.argv[1]
output_dir = sys.argv[2]
working_dir = sys.argv[3]
mask_file = sys.argv[4]
tmap_file = sys.argv[5]
tsv_dir = sys.argv[6]

# initialize
svc = SVC()
c = [1]
param_grid = dict(C=c)
TR = 3
newTR = 0.5
trial_x_run = 4
time = 16  # peak response 9 sec after cue onset
modality = ['perception', 'imagery']
category = ['face', 'place']
voxels = np.array([1, 10, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1700, 2000, 2500, 4000]) - 1
df = pd.DataFrame()
trial_type = ["img_berlin", "img_paris", "img_obama", "img_merkel", "seen_berin", "seen_paris", "seen_obama",
              "seen_merkel"]

nifti_files = []
[nifti_files.append(f) for f in listdir(nifti_dir) if "cr_rsub" in f and f.endswith(".nii")
 and "._" not in f and "00" not in f and "sm_" not in f]
nifti_files.sort()

tsv_files = []
[tsv_files.append(t) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
tsv_files.sort()

# get indexes of the event of interest
frames = []
[frames.append(pd.read_table(join(tsv_dir, tsv))) for tsv in tsv_files]
datainfo = pd.concat(frames)
df_grp = datainfo.groupby('trial_type')

index_run = pd.read_table(join(working_dir, 'vol_x_run.tsv'))
index_run = index_run.to_numpy()[:, 1]

# load mask and get the n threshold (n. voxel)
mask = nb.load(mask_file)
mask = mask.get_fdata()
tmap = nb.load(tmap_file)
tmap = tmap.get_fdata()
tval = tmap[mask == 1]

tval.sort()
tval = tval[::-1]
threshold = tval[voxels]
vector = np.vectorize(ceil)

for j, t in enumerate(threshold):
    ffa_ppa = tmap >= t
    new_mask = np.logical_and(ffa_ppa, mask)
    new_mask = new_mask.reshape(-1)

    # concatenate functional volumes across runs in a 2D matrix
    epi4D = nb.load(join(nifti_dir, nifti_files[0]))
    epi4D = epi4D.get_fdata()
    features = epi4D.reshape(-1, epi4D.shape[-1])
    features = features[new_mask]
    features = features[np.logical_not(np.isnan(features.T[0][:]))]
    time_points = np.arange(features.shape[1]) * TR
    new_time_points = np.arange(0, (features.shape[1] - 1) * TR + newTR, newTR)
    interpolator = interp1d(time_points, features, kind='linear')
    interpolated_features = interpolator(new_time_points)
    feature_matrix = np.zeros(
        shape=(len(interpolated_features), len(new_time_points) * (len(nifti_files) + 1)))  # plus one
    feature_matrix[:, 0:interpolated_features.shape[1]] = interpolated_features
    last_index = np.array([0])

    for i, nifti_file in enumerate(nifti_files[1::]):
        epi4D = nb.load(join(nifti_dir, nifti_file))
        epi4D = epi4D.get_fdata()
        features = epi4D.reshape(-1, epi4D.shape[-1])
        features = features[new_mask]
        features = features[np.logical_not(np.isnan(features.T[0][:]))]
        time_points = np.arange(features.shape[1]) * TR
        new_time_points = np.arange(0, (features.shape[1] - 1) * TR + newTR, newTR)
        interpolator = interp1d(time_points, features, kind='linear')
        interpolated_features = interpolator(new_time_points)
        last_index = np.concatenate((last_index, np.array([last_index[i] + interpolated_features.shape[1]])), axis=0)
        feature_matrix[:, last_index[i + 1]: last_index[i + 1] + interpolated_features.shape[1]] \
            = interpolated_features
    feature_matrix = feature_matrix.T

    acc = []

    for m in modality:

        for c in category:

            if m == 'imagery':

                index = np.array(np.tile(index_run, (trial_x_run, 1))).T.flatten()

                a, b = [0, 1]
                c, d = [2, 3]

                # if c == 'face':
                #
                #     a, b = [2, 3]
                #
                # elif c == 'place':
                #
                #     a, b = [0, 1]

            elif m == 'perception':

                index = index_run

                a, b = [4, 5]
                c, d = [6, 7]
                # if c == 'face':
                #
                #     a, b = [6, 7]
                #
                # elif c == 'place':
                #
                #     a, b = [4, 5]

            c1 = trial_type[a]
            c2 = trial_type[b]
            c3 = trial_type[c]
            c4 = trial_type[d]

            # get indexes of the event of interest
            ind = []
            [ind.append(vector(np.array(df_grp.get_group(g).onset) / newTR) + time + index)
             for g in [c1, c2, c3, c4]]

            cond1_index = list(np.concatenate((ind[0], ind[1]), axis=0))
            cond2_index = list(np.concatenate((ind[2], ind[3]), axis=0))
            # cond1_index = ind[0]
            # cond2_index = ind[1]
            cond1_index.sort()
            cond2_index.sort()

            label = np.zeros(feature_matrix.shape[0]).astype(int)
            label[cond1_index] = 1
            label[cond2_index] = 2

            # get the relevant samples
            X = feature_matrix[label != 0]
            y = label[label != 0]

            # scale the data mean 0 and sdv 1
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

            grid = GridSearchCV(svc, param_grid, cv=2, scoring='accuracy')
            grid.fit(X, y)
            effect = grid.best_score_
            acc.append(effect)

            # acc_ = []
            # repetitions = int(len(y) / len(tsv_files))
            # for p in range(len(tsv_files)):
            #     mask_train = np.ones(len(y), dtype=bool)
            #     mask_train[p * repetitions:(p + 1) * repetitions] = False
            #     X_train = X[mask_train]
            #     y_train = y[mask_train]
            #     X_test = X[mask_train == False]
            #     y_test = y[mask_train == False]
            #     svc.fit(X_train, y_train)
            #     acc_.append(svc.score(X_test, y_test))
            # acc.append(np.mean(acc_))

    df['vox{}'.format(voxels[j] + 1)] = acc
df['modality'] = ['perception/face', 'perception/place', 'imagery/face', 'imagery/place']
df.to_csv(join(output_dir, 'acc_identity_vs_voxel_tr-16.tsv'), sep='\t', index=True)
