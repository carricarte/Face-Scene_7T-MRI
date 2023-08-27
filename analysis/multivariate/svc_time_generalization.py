import sys
import numpy as np
import pandas as pd
from math import ceil
from os import listdir
from os.path import join
from sklearn.svm import SVC
from sklearn import preprocessing
from itertools import combinations_with_replacement as comb

# directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
tsv_dir = sys.argv[3]

# input_dir = '/Users/carricarte/PhD/Debugging/sub-01/analysis/multivariate'
# output_dir = '/Users/carricarte/Desktop'
# tsv_dir = '/Users/carricarte/PhD/Debugging/sub-01/behdat'

# get subject id
subject_list = ['sub-01', 'sub-02', 'sub-03', 'sub-06', 'sub-07']
for s in subject_list:
    if s in input_dir:
        sub = s

svc = SVC(C=1)

# get epi and data
TR = 0.5
trial_x_run = 4
cv = 4  # cross validation
modality = ['PP', 'II', 'PI', 'IP']  # imagery or perception
time_points = np.arange(-12, 37)
combined_times = list(comb(time_points, 2))
columns = []
[columns.append(str(name)) for name in combined_times]
df = pd.DataFrame(columns=columns)
trial_type = ["img_berlin", "img_paris", "img_obama", "img_merkel", "seen_berin", "seen_paris", "seen_obama",
              "seen_merkel"]
tsvfiles = []
[tsvfiles.append(t) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
tsvfiles.sort()

feature_matrix = pd.read_table(join(input_dir, 'features_FFA_wb_layer-3_vox-1970.tsv'))
feature_matrix = feature_matrix.to_numpy()
feature_matrix = feature_matrix[:, 1:feature_matrix.shape[1]]

voxels = feature_matrix.shape[1]

index_train = pd.read_table(join(input_dir, 'vol_x_run.tsv'))
run_index = index_train.to_numpy()

for m in modality:

    index_train = list(run_index[:, 1])
    index_test = list(run_index[:, 1])

    if m == 'PP':
        a, b = [4, 8]
        c, d = [4, 8]
    elif m == 'II':
        a, b = [0, 4]
        c, d = [0, 4]
        index_train = np.array(np.tile(index_train, (trial_x_run, 1))).T.flatten()
        index_test = np.array(np.tile(index_test, (trial_x_run, 1))).T.flatten()
    elif m == 'PI':
        a, b = [4, 8]
        c, d = [0, 4]
        index_test = np.array(np.tile(index_test, (trial_x_run, 1))).T.flatten()
    elif m == 'IP':
        a, b = [0, 4]
        c, d = [4, 8]
        index_train = np.array(np.tile(index_train, (trial_x_run, 1))).T.flatten()

    p1_train, p2_train, f1_train, f2_train = trial_type[a:b]
    p1_test, p2_test, f1_test, f2_test = trial_type[c:d]

    acc = []

    X_train = ''
    y_train = ''
    X_test = ''
    y_test = ''

    for times in combined_times:

        # read tsv files and create label vector with three condition (2 conditions of interest + rest)
        vector = np.vectorize(ceil)
        ind_train = []
        ind_test = []

        frames = []
        [frames.append(pd.read_table(join(tsv_dir, tsv))) for tsv in tsvfiles]
        datainfo = pd.concat(frames)
        df_grp = datainfo.groupby('trial_type')

        [ind_train.append(vector(np.array(df_grp.get_group(g).onset) / TR) + times[0] + index_train)
         for g in [p1_train, p2_train, f1_train, f2_train]]
        [ind_test.append(vector(np.array(df_grp.get_group(k).onset) / TR) + times[1] + index_test)
         for k in [p1_test, p2_test, f1_test, f2_test]]

        places_index_train = list(np.concatenate((ind_train[0], ind_train[1]), axis=0))
        faces_index_train = list(np.concatenate((ind_train[2], ind_train[3]), axis=0))
        places_index_test = list(np.concatenate((ind_test[0], ind_test[1]), axis=0))
        faces_index_test = list(np.concatenate((ind_test[2], ind_test[3]), axis=0))

        places_index_train.sort()
        faces_index_train.sort()
        places_index_test.sort()
        faces_index_test.sort()

        label_train = np.zeros(feature_matrix.shape[0]).astype(int)
        label_test = np.zeros(feature_matrix.shape[0]).astype(int)
        label_train[faces_index_train] = 1
        label_train[places_index_train] = 2
        label_test[faces_index_test] = 1
        label_test[places_index_test] = 2

        # reduce feature matrix and label vector to the relevant TRs
        X_train_all = feature_matrix[np.logical_or(label_train == 1, label_train == 2)]
        y_train_all = label_train[np.logical_or(label_train == 1, label_train == 2)]
        X_test_all = feature_matrix[np.logical_or(label_test == 1, label_test == 2)]
        y_test_all = label_test[np.logical_or(label_test == 1, label_test == 2)]

        # transform data to mean 0 varaince 1
        scaler_train = preprocessing.StandardScaler().fit(X_train_all)
        scaler_test = preprocessing.StandardScaler().fit(X_test_all)
        X_train_all = scaler_train.transform(X_train_all)
        X_test_all = scaler_test.transform(X_test_all)

        acc_ = []
        training_x_run = int(len(y_train_all)/len(tsvfiles))
        testing_x_run = int(len(y_test_all) / len(tsvfiles))
        for p in range(len(tsvfiles)):
            mask_train = np.ones(len(y_train_all), dtype=bool)
            mask_train[p * training_x_run:(p + 1) * training_x_run] = False
            mask_testing = np.ones(len(y_test_all), dtype=bool)
            mask_testing[p * testing_x_run:(p + 1) * testing_x_run] = False
            X_train = X_train_all[mask_train]
            y_train = y_train_all[mask_train]
            X_test = X_test_all[mask_testing == False]
            y_test = y_test_all[mask_testing == False]
            svc.fit(X_train, y_train)
            acc_.append(svc.score(X_test, y_test))
        acc.append(np.mean(acc_))

    df.loc[len(df.index)] = acc

df['modality'] = modality
df.to_csv(join(output_dir, 'acc_FFA_time_generalization.tsv'), sep='\t', index=True)
