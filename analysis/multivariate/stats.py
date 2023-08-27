from os.path import join, split
from os import listdir
import numpy as np
import pandas as pd
from math import ceil, floor


def nifti_list(input_dir, f1, f2):
    beta_files = []
    [beta_files.append(join(input_dir, file)) for file in listdir(input_dir) if
     file.endswith('.nii') and '._' not in file]
    beta_files.sort()

    if f1 == "imagery" and f2 == "face":
        i = 3
    elif f1 == "imagery" and f2 == "place":
        i = 4
    elif f1 == "perception" and f2 == "face":
        i = 6
    elif f1 == "perception" and f2 == "place":
        i = 7
    beta_ind = np.arange(i, len(beta_files) - 12, 14)
    return np.array(beta_files)[beta_ind]


def beh_index(tsv_files, tr, index, groups):

    vector = np.vectorize(floor)
    # print(c1 + " " + c2 + " " + c3 + " " + c4)
    # get indexes of the event of interest
    frames = []
    [frames.append(pd.read_table(tsv)) for tsv in tsv_files]
    datainfo = pd.concat(frames)

    df_grp = datainfo.groupby('trial_type')
    ind = []
    # [ind.append(vector(np.array(df_grp.get_group(g).onset) / tr) + index)
    #  for g in groups]

    # for g in [c1, c2, c3, c4]:
    for g in groups:
        print(g)
        if "seen" in g:
            index_run = np.ndarray.flatten(np.vstack((index, index)).T)
        else:
            index_run = np.ndarray.flatten(np.vstack((index, index, index, index)).T)
        # index_run = np.ndarray.flatten(np.vstack((index, index, index, index, index, index)).T)

        ind.append(vector(np.array(df_grp.get_group(g).onset) / tr) + index_run)
        print("Hello Wolrd")

    cond1_index = ind[0]
    cond2_index = ind[1]
    cond3_index = ind[2]
    cond4_index = ind[3]

    # cond1_index = list(np.concatenate((ind[0], ind[1]), axis=0))
    # cond2_index = list(np.concatenate((ind[2], ind[3]), axis=0))

    # return [cond1_index, cond2_index]
    return [cond1_index, cond2_index, cond3_index, cond4_index]

    # return ind

# from os import listdir
# from os.path import join

# tsv_dir = "/Users/carricarte/scratch/projects/imagery/pilot_07/behdata/sub-02"
# tsvfiles = []
# [tsvfiles.append(join(tsv_dir, t)) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
# tsvfiles.sort()
# trial_type = ["img_berlin", "img_paris", "img_obama", "img_merkel", "seen_berin",
#               "seen_paris", "seen_obama", "seen_merkel"]
# conds_training = trial_type[4:8]
#
# beh_index(tsvfiles, 0.5, 2, conds_training)
