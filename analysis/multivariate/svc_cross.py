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

# get subject id
subject_list = ['sub-01', 'sub-02', 'sub-03', 'sub-06', 'sub-07']
for s in subject_list:
    if s in input_dir:
        sub = s

layer = True
percent = np.array([95])
# percent = np.array([97, 94, 91, 88, 85, 82, 79, 76, 73, 70])
svc = SVC(C=1)
param_grid = dict(C=[1])
grid = GridSearchCV(svc, param_grid, cv=4, scoring='accuracy')
newTR = 0.5
tr = 14
random = [False, True]
# random = [False]
trial = ['single']
modality = ["PP", "II", 'IP', 'PI']  # imagery or perception
# modality = ["PP"]  # imagery or perception
# df = pd.DataFrame(
#     columns=['Subject', 'P/P', 'P/P_null', 'I/I', 'I/I_null', 'I/P', 'I/P_null', 'P/I', 'P/I_null', ])

trial_type = ["img_berlin", "img_paris", "img_obama", "img_merkel", "seen_berin",
              "seen_paris", "seen_obama", "seen_merkel"]

tsvfiles = []
[tsvfiles.append(join(tsv_dir, t)) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
tsvfiles.sort()

feature_matrix = pd.read_table(join(input_dir, 'features_FFA_wb_layer-3_vox-470_devein_deconv.tsv'), index_col=0)
feature_matrix = feature_matrix.to_numpy()

layer_mask = pd.read_table(join(input_dir, 'layer_mask_FFA_wb_layer-3_vox-470.tsv'))
layer_mask = layer_mask.to_numpy()[:, 1]

voxels = feature_matrix.shape[1]
print(voxels)

last_index = pd.read_table(join(input_dir, 'vol_x_run.tsv'))
last_index = last_index.to_numpy()[:, 1]

df = pd.DataFrame(
    columns=['Subject', 'layer', 'P/P', 'P/P_null', 'I/I', 'I/I_null', 'I/P', 'I/P_null', 'P/I', 'P/I_null', ])
# df = pd.DataFrame(
#     columns=['Subject', 'layer', '97', '94', '91', '88', '85', '82', '79', '76', '73', '70'])
for ly in np.unique(layer_mask):
# for ly in np.unique([0]):
#     df = pd.DataFrame(
#         columns=['Subject', 'layer', 'P/P', 'P/P_null', 'I/I', 'I/I_null', 'I/P', 'I/P_null', 'P/I', 'P/I_null', ])
    mean_acc = []
    for m in modality:

        if m == 'IP':
            index_train = np.array(np.tile(last_index, (4, 1))).T.flatten()
            index_test = last_index
            conds_training = trial_type[0:4]
            conds_testing = trial_type[4:8]

        elif m == 'PI':
            index_test = np.array(np.tile(last_index, (4, 1))).T.flatten()
            index_train = last_index
            conds_training = trial_type[4:8]
            conds_testing = trial_type[0:4]

        if m == 'II':
            index_train = np.array(np.tile(last_index, (4, 1))).T.flatten()
            index_test = np.array(np.tile(last_index, (4, 1))).T.flatten()
            conds_training = trial_type[0:4]
            conds_testing = trial_type[0:4]

        elif m == 'PP':
            index_test = last_index
            index_train = last_index
            conds_training = trial_type[4:8]
            conds_testing = trial_type[4:8]

        cond1_index_train, cond2_index_train = stats.beh_index(tsvfiles, newTR, index_train, conds_training)
        cond1_index_test, cond2_index_test = stats.beh_index(tsvfiles, newTR, index_test, conds_testing)

        cond1_index_train.sort()
        cond1_index_test.sort()
        cond2_index_train.sort()
        cond2_index_test.sort()

        label = np.zeros(feature_matrix.shape[0]).astype(int)
        label[cond1_index_train] = 1
        label[cond2_index_train] = 2

        if conds_training != conds_testing:
            label[cond1_index_test] = -1
            label[cond2_index_test] = -2

        offset = list(np.zeros(abs(tr), dtype=int))
        label_ = np.array(offset + list(label[:-tr]))

        feature_layer = feature_matrix[:, layer_mask == ly]

        # reduce feature matrix and label vector to the relevant TRs
        X = feature_layer[label_ != 0]

        y = label_[label_ != 0]
        y_train_all = y[np.logical_or(y == 1, y == 2)]
        y_test_all = y_train_all

        # reduce feature matrix to voxels with low variance/mean intensity
        # for f in percent:
        # val = np.var(feature_layer, axis=0) / np.mean(feature_layer, axis=0)
        # filt = np.percentile(val, percent[0])
        # feature_matrix = feature_matrix[:, val <= filt]

        # X_layer = X[:, layer_mask == ly]
        # X_layer = X[:, val <= filt]
        X_layer = X
        # X_layer = X[:, layer_mask != 0]
        print(m + '-' + str(ly))
        # print(sum(val <= filt)/len(val))

        # remove superficial vein contribution

        # scale the data mean 0 and sdv 1
        scaler = preprocessing.StandardScaler().fit(X_layer)
        X_layer = scaler.transform(X_layer)

        X_train_all = X_layer[np.logical_or(y == 1, y == 2)]
        X_test_all = X_train_all

        if conds_training != conds_testing:
            X_test_all = X_layer[np.logical_or(y == -1, y == -2)]
            y_test_all = y[np.logical_or(y == -1, y == -2)]
            y_test_all[y_test_all == -1] = 1
            y_test_all[y_test_all == -2] = 2

        for r in random:

            if r:
                rd.shuffle(y_train_all)

            # acc_ = []
            # X_train_all = X_train_all.T
            scaler = preprocessing.StandardScaler().fit(X_train_all)
            X_train_all = scaler.transform(X_train_all)
            # X_train_all = X_train_all.T
            print(X_train_all.shape)
            print(y_train_all.shape)
            grid.fit(X_train_all, y_train_all)
            acc = grid.best_score_
            if conds_training != conds_testing:
                acc = grid.score(X_test_all, y_test_all)
            mean_acc.append(acc)
            # print(grid.cv_results_["mean_test_score"])
            print(acc)
            # rep_train = int(len(y_train_all) / len(tsvfiles))
            # rep_test = int(len(y_test_all) / len(tsvfiles))
            #
            # for p in range(len(tsvfiles)):
            #     mask_train = np.ones(len(y_train_all), dtype=bool)
            #     mask_train[p * rep_train:(p + 1) * rep_train] = False
            #     mask_test = np.zeros(len(y_test_all), dtype=bool)
            #     mask_test[p * rep_test:(p + 1) * rep_test] = True
            #     X_train = X_train_all[mask_train]
            #     y_train = y_train_all[mask_train]
            #     X_test = X_test_all[mask_test]
            #     y_test = y_test_all[mask_test]
            #     svc.fit(X_train, y_train)
            #     acc_.append(svc.score(X_test, y_test))
            # mean_acc.append(np.mean(acc_))

    df.loc[len(df.index)] = [sub] + [ly] + mean_acc
    # df.loc[len(df.index)] = [sub] + [ly] + mean_acc
    # ly = 'ribbon'
df.to_csv(join(output_dir, 'cross_svc_all_layer_FFA_{}_devein_deconv.tsv'.format(sub)), sep='\t', index=True)