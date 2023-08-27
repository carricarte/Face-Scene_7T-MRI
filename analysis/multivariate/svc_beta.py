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


def get_divisors(X):
    return np.unique([d for d in np.arange(2, int(X/2) + 1) if X % d == 0])


input_dir = sys.argv[1]
output_dir = sys.argv[2]
subject = sys.argv[3]

if "01" == subject:
    no_runs = 9
elif "02" == subject:
    no_runs = 10
elif "03" == subject:
    no_runs = 10
elif "04" == subject:
    no_runs = 12
elif "05" == subject:
    no_runs = 12
elif "06" == subject:
    no_runs = 10
elif "07" == subject:
    no_runs = 12
elif "08" == subject:
    no_runs = 10
elif "09" == subject:
    no_runs = 10
elif "10" == subject:
    no_runs = 10
elif "11" == subject:
    no_runs = 10
elif "12" == subject:
    no_runs = 10
elif "13" == subject:
    no_runs = 10
elif "14" == subject:
    no_runs = 10
elif "15" == subject:
    no_runs = 10
elif "16" == subject:
    no_runs = 10
elif "17" == subject:
    no_runs = 12
elif "18" == subject:
    no_runs = 10

# con = ["PPA_per", "PPA_img", "FFA_per", "FFA_img"]
con = ["per", "img"]
c = [1]
param_grid = dict(C=c)
nperm = 1000
# d = np.arange(2, 7)
# cv_arr = [2]
# cv_arr = np.arange(3, 10)
layers = ["1", "2", "3"]
voxel = np.arange(50, 550, 50)
svc = LinearSVC(penalty='l2', loss='hinge', C=1, multi_class='ovr', fit_intercept=True, max_iter=10000)

no_runs = no_runs*2

for d in get_divisors(no_runs):

    for c in con:

        df_acc = pd.DataFrame()

        for v in voxel:

            acc = []

            for l in layers:

                beta_files = []
                [beta_files.append(join(input_dir, f)) for f in listdir(input_dir) if
                 "vox-{0}_".format(v) in f and '._' not in f
                 and c in f and f.endswith(".tsv") and "layer-{0}".format(l) in f]
                beta_files.sort()

                betas = pd.read_table(beta_files[0], sep='\t', index_col=0)
                for i, beta in enumerate(beta_files[1::]):
                    df = pd.read_table(beta, sep='\t', index_col=0)
                    frames = [betas, df]
                    betas = pd.concat(frames, ignore_index=True)

                # get betas per condition
                y = np.tile([0, 1], int(betas.shape[0] / 2))
                order = np.arange(0, int(len(y) / 2))
                # cv = int(len(y) / 2)
                # grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
                # grid.fit(betas.to_numpy(), y)
                # acc.append(grid.best_score_)

                _acc = []
                for n in range(0, nperm):

                    pseudo_X = []
                    rd.shuffle(order)
                    sub_indexes = np.array_split(order, d)

                    for class_ in np.unique(y):

                        class_index = y == class_
                        sub_X = betas.to_numpy()[class_index, :]

                        for i, sub_index in enumerate(sub_indexes):
                            pseudo_X.append(np.mean(sub_X[sub_index], 0))

                    pseudo_X = np.array(pseudo_X)
                    pseudo_y = np.repeat([0, 1], int(len(pseudo_X) / 2))
                    cv = int(len(pseudo_y) / 2)
                    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
                    grid.fit(pseudo_X, pseudo_y)
                    _acc.append(grid.best_score_)
                acc.append(np.mean(_acc))

            df_acc[v] = acc
        df_acc["layer"] = ["deep", "middle", "superficial"]
        cv = "max"
        df_acc.to_csv(
            join(output_dir, '{0}_decoding_cat_devein_mnn_pseudotrial-{1}_cv-{2}.tsv'.format(c, d, cv)),
            sep='\t', index=True)
        # df_acc.to_csv(join(output_dir, '{0}_decoding_id_devein_mnn_pseudo-{1}_cv-{2}.tsv'.format(c, num_subsamples, cv)), sep='\t', index=True)

# df["layer"] = ["deep", "middle", "superficial"]
# df.to_csv(join(input_dir, '{0}_decoding_id_devein_no-mnn_pseudo-{1}_cv-{2}.tsv'.format(c, num_subsamples, cv)), sep='\t', index=True)
