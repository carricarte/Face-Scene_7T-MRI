import sys
import numpy as np
from os import listdir
from os.path import join
from nilearn.image import mean_img, math_img


def beta_x_cond(inputdir, conditions, inx):
    zscore = ['0001', '0001']

    run_list = []
    [run_list.append(join(inputdir, f)) for f in listdir(inputdir) if "run-" in f and '._' not in f
     and "00" not in f and "sm_" not in f]

    run_list.sort()
    all_betas = [[], []]

    for j, i in enumerate(inx):
        sub_string_1 = zscore[j]
        runs = []
        [runs.append(run_list[e]) for e in np.nonzero(i)[0]]
        for run in runs:
            for x in listdir(run):
                if ("._" not in x and "spmT" in x) and (sub_string_1 in x):
                    all_betas[j].append(join(run, x))

    return all_betas


# input_dir = sys.argv[1]
input_dir = "/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-09/analysis/bold"

conditions = ['img_place', 'seen_place']
img_ind = [0, 1, 0, 1, 1, 1, 0, 1]
ind = [img_ind, list(np.array(img_ind) == 0)]
[cond1, cond2] = beta_x_cond(input_dir, conditions, ind)

beta_files = []
for c, b in enumerate([cond1, cond2]):
    for file in b:
        print(file)
    print("-----------------------------------------------------------------------------------------------")
    beta_mean = mean_img(b)
    filename = join(input_dir, 'mean_beta_{}.nii'.format(conditions[c]))
    beta_mean.to_filename(filename)
    beta_files.append(filename)
