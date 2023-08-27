import sys
import numpy as np
import random as rd
import pandas as pd
import stats
from os import listdir
from os.path import join
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

# directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
tsv_dir = sys.argv[3]

# input_dir = "/Users/carricarte/PhD/Debugging/sub-06/analysis"
# output_dir = "/Users/carricarte/PhD/Debugging/sub-06/analysis"
# tsv_dir = "/Users/carricarte/PhD/Debugging/behdata/sub-06"

# get subject id
subject_list = ['sub-01', 'sub-02', 'sub-03', 'sub-06', 'sub-07']

voxels = np.arange(20, 2700, 150)
# voxels = np.array([320])
svc = SVC(C=1)
param_grid = dict(C=[1])
grid = GridSearchCV(svc, param_grid, cv=4, scoring='accuracy')
newTR = 0.5
tr = 14
random = [False]
trial = ['single']
modality = ["perception", "imagery"]  # imagery or perception
trial_type = ["img_berlin", "img_paris", "img_obama", "img_merkel", "seen_berin",
              "seen_paris", "seen_obama", "seen_merkel"]

tsvfiles = []
[tsvfiles.append(join(tsv_dir, t)) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
tsvfiles.sort()

last_index = pd.read_table(join(input_dir, 'vol_x_run.tsv'))
last_index = last_index.to_numpy()[:, 1]

df = pd.DataFrame()
df['subject'] = [np.tile(sub, 3) for sub in subject_list if sub in input_dir][0]
df['layer'] = np.array(['1', '2', '3'])

for v in np.array(voxels):

    feature_matrix = pd.read_table(join(input_dir, 'features_FFA_wb_layer-3_vox-{}.tsv'.format(v)), index_col=0)
    feature_matrix = feature_matrix.to_numpy()

    layer_mask = pd.read_table(join(input_dir, 'layer_mask_FFA_wb_layer-3_vox-{}.tsv').format(v))
    layer_mask = layer_mask.to_numpy()[:, 1]

    acc = []
    for ly in np.unique(layer_mask):
        # mean_acc = []
        X_contrast_mean = np.array([])
        for m in modality:

            if m == 'imagery':
                index_train = np.array(np.tile(last_index, (4, 1))).T.flatten()
                index_test = np.array(np.tile(last_index, (4, 1))).T.flatten()
                conds_training = trial_type[0:4]
                conds_testing = trial_type[0:4]

            elif m == 'perception':
                index_test = last_index
                index_train = last_index
                conds_training = trial_type[4:8]
                conds_testing = trial_type[4:8]

            cond1_index_train, cond2_index_train = stats.beh_index(tsvfiles, newTR, index_train, conds_training)

            cond1_index_train.sort()
            cond2_index_train.sort()

            label = np.zeros(feature_matrix.shape[0]).astype(int)
            label[cond1_index_train] = 1
            label[cond2_index_train] = 2

            offset = list(np.zeros(abs(tr), dtype=int))
            label_ = np.array(offset + list(label[:-tr]))

            # reduce feature matrix and label vector to the relevant TRs
            feature_layer = feature_matrix[:, layer_mask == ly]
            X = feature_layer[label_ != 0]
            y = label_[label_ != 0]

            y_train_all = y[np.logical_or(y == 1, y == 2)]
            y_test_all = y_train_all

            X_layer = X

            # scale the data mean 0 and sdv 1

            # scaler = preprocessing.StandardScaler().fit(X_layer)
            # X_layer = scaler.transform(X_layer)
            X_train_all = X_layer[np.logical_or(y == 1, y == 2)]
             # for some reason there are nan values
            X_test_all = X_train_all

            total = 0
            X_contrast = np.zeros(shape=(1, X_train_all.shape[1]))
            X_1 = X_train_all[y == 1]
            X_2 = X_train_all[y == 2]

            if m == "perception":
                samples = int(len(y) / 2)
            avg_X_1 = np.mean(np.array(X_1[0: samples]), 0)
            avg_X_2 = np.mean(np.array(X_2[0: samples]), 0)
            ratio = avg_X_1 / avg_X_2
            # for cond_1 in range(0, samples):
            #     for cond_2 in range(0, samples):
            #         total += 1
            #         X_contrast += X_1[cond_1] / X_2[cond_2]
            # X_contrast_mean = np.append(X_contrast_mean, X_contrast / total).reshape(-1, 1)
            X_contrast_mean = np.append(X_contrast_mean, ratio).reshape(-1, 1)
        no_nan_vector = X_contrast_mean == X_contrast_mean
        y_contrast = np.zeros(len(X_contrast_mean)).astype(int)
        y_contrast[:-int(len(X_contrast_mean) / 2)] = 1

        X_contrast_mean = X_contrast_mean[no_nan_vector[:, 0]]
        y_contrast = y_contrast[no_nan_vector[:, 0]]

        for r in random:

            if r:
                rd.shuffle(y_contrast)
            grid.fit(X_contrast_mean, y_contrast)
            acc_ = grid.best_score_
            acc.append(acc_)
            print("Hello World")

    df['vox-' + str(v)] = acc
df.to_csv(join(output_dir, 'svc_contrast_all_layer_FFA.tsv'), sep='\t', index=True)