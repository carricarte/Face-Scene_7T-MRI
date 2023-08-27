import sys
import numpy as np
import pandas as pd
from math import ceil
import random as rd
from os import listdir
from os.path import join
from sklearn.svm import SVC
from sklearn import preprocessing
import stats
from sklearn.model_selection import GridSearchCV

# directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
tsv_dir = sys.argv[3]

# input_dir = "/Users/carricarte/scratch/projects/imagery/pilot_07/derivatives/sub-01/analysis"
# output_dir = '/Users/carricarte/Desktop'
# tsv_dir = "/Users/carricarte/PhD/Debugging/sub-01/behdat"

# get subject id
subject_list = ['sub-01', 'sub-02', 'sub-03', 'sub-06', 'sub-07']

# initialize some variables
svc = SVC(C=1)
param_grid = dict(C=[1])
grid = GridSearchCV(svc, param_grid, cv=4, scoring='accuracy')
# param_grid = dict(C=c)
TR = 0.5
modality = ['perception', 'imagery']  # imagery or perception
category = ['face', 'place']
trial_x_run = 4
times = [14]
# times = np.arange(-12, 37)
# voxels = [199, 299, 399, 499, 999, 1499, 1999, 2499, 2999, 3499, 3999, 4499, 4999, 5499, 5999, 6499, 6999, 7499, 7999]
voxels = np.arange(20, 2000, 150) - 1
# voxels = np.array([620]) - 1
# voxels = np.array([1]) - 1
# columns = []
# [columns.append("v-" + str(ti)) for ti in voxels]
# df = pd.DataFrame(columns=['subject', 'modality'] + columns)
df = pd.DataFrame()
df['subject'] = [np.tile(sub, 6) for sub in subject_list if sub in input_dir][0]
df['layer'] = np.tile(['1', '2', '3'], 2)
df['modality'] = np.concatenate((np.tile(['perception'], 3), np.tile(['imagery'], 3)))

df_vox = pd.DataFrame()
df_vox['subject'] = [np.tile(sub, 3) for sub in subject_list if sub in input_dir][0]
df_vox['layer'] = ['1', '2', '3']

# df = pd.DataFrame(
#     columns=['subject'] + ['vox-' + str(p) for p in voxels + 1] + ['modality', 'category'])
# df = pd.DataFrame(
#     columns=['subject'] + ['l1', 'l2', 'l3'] + ['modality'])
# df = pd.DataFrame(
#    columns=['subject', 'perception/faces', 'perception/places', 'imagery/faces', 'imagery/places'])
trial_type = ["img_berlin", "img_paris", "img_obama", "img_merkel", "seen_berin", "seen_paris", "seen_obama",
              "seen_merkel"]

# load tables with the onset of the events
tsvfiles = []
[tsvfiles.append(join(tsv_dir, t)) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
tsvfiles.sort()

# load array with number of volumes per run
index_run = pd.read_table(join(input_dir, 'vol_x_run.tsv'))
index_run = index_run.to_numpy()[:, 1]
# index_run = index_run[random_runs]

vector = np.vectorize(ceil)
# acc = []
for v in np.array(voxels) + 1:

    # load layer mask
    layer_mask = pd.read_table(join(input_dir, 'layer_mask_FFA_wb_layer-3_vox-{}.tsv'.format(v)), index_col=0)
    # layer_PPA_mask = pd.read_table(join(input_dir, 'layer_mask_PPA_wb_layer-3_vox-{}.tsv'.format(v)), index_col=0)
    # layer_frames = [layer_FFA_mask, layer_PPA_mask]
    # layer_mask = np.array(pd.concat(layer_frames, axis=0)).astype(int).squeeze()
    layer_mask = np.array(layer_mask).astype(int).squeeze()

    # load features
    feature_matrix = pd.read_table(join(input_dir, 'features_FFA_wb_layer-3_vox-{}_devein.tsv'.format(v)), index_col=0)
    # feature_PPA_matrix = pd.read_table(join(input_dir, 'features_PPA_wb_layer-3_vox-{}.tsv'.format(v)), index_col=0)
    # feature_frames = [feature_FFA_matrix, feature_PPA_matrix]
    # feature_matrix = pd.concat(feature_frames, axis=1)
    feature_matrix = feature_matrix.to_numpy()
    # feature_matrix = feature_matrix[:, 1:feature_matrix.shape[1]]

    acc = []

    for m in modality:
        # acc = []
        if m == 'imagery':

            index = np.array(np.tile(index_run, (trial_x_run, 1))).T.flatten()

            a, b = [0, 1]
            c, d = [2, 3]

        elif m == 'perception':

            index = index_run

            a, b = [4, 5]
            c, d = [6, 7]

        conds = [trial_type[a], trial_type[b], trial_type[c], trial_type[d]]
        cond1_index, cond2_index = stats.beh_index(tsvfiles, TR, index, conds)
        cond1_index.sort()
        cond2_index.sort()

        # create vector of label
        label_ = np.zeros(feature_matrix.shape[0]).astype(int)
        label_[cond1_index] = 1
        label_[cond2_index] = 2

        # all_label = []
        for tr in times:

            offset = list(np.zeros(abs(tr), dtype=int))
            if tr < 0:
                label = np.array(list(label_[abs(tr):]) + offset)
            elif tr > 0:
                label = np.array(offset + list(label_[:-tr]))
            else:
                label = label_
        # get the relevant samples
            X = feature_matrix[label != 0]
            y = label[label != 0]
            # print(y)
        # X = feature_matrix[label_ != 0]
        # y = label_[label_ != 0]
            # scale the data mean 0 and sdv 1
            X = X[:, X[0] == X[0]]  # for some reason there are nan values
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

            n_voxel = []

            for l in np.array([1, 2, 3]):

                sub_X = X[:, layer_mask == l]
                # scaling within layer
                # sub_X = sub_X.T
                # scaler = preprocessing.StandardScaler().fit(sub_X)
                # sub_X = scaler.transform(sub_X)
                # sub_X = sub_X.T
                n_voxel.append(np.size(sub_X, 1))
                acc_ = []
                grid.fit(sub_X, y)
                acc_ = grid.best_score_
                acc.append(acc_)

        # df.loc[len(df.index)] = [[sub] + acc + [m, c] for sub in subject_list if sub in input_dir][0]
        # df.loc[len(df.index)] = [[sub] + [m] + acc for sub in subject_list if sub in input_dir][0]
    df['vox-' + str(v)] = acc

    df_vox['vox-' + str(v)] = n_voxel
df.to_csv(join(output_dir, 'acc_FFA_wb_layer-3_devein.tsv'.format(v)), sep='\t', index=True)
df_vox.to_csv(join(output_dir, 'n_voxel_category_FFA_ribbon.tsv'), sep='\t', index=True)
