import sys
import numpy as np
from os import listdir
from os.path import join
from nilearn.image import mean_img, math_img


def beta_x_cond(inputdir, conditions, inx):
    beta = ['0002', '0002', '0009', '0009']

    run_list = []
    [run_list.append(join(inputdir, f, "beta")) for f in listdir(inputdir) if "run-" in f and '._' not in f
     and "00" not in f and "sm_" not in f]

    run_list.sort()
    all_betas = [[], [], [], []]

    for j, i in enumerate(inx):
        sub_string_1 = beta[j]
        runs = []
        [runs.append(run_list[e]) for e in np.nonzero(i)[0]]
        for run in runs:
            for x in listdir(run):
                if ("._" not in x and "SPM" not in x) and (sub_string_1 in x):
                    all_betas[j].append(join(run, x))

    return all_betas


# input_dir = sys.argv[1]
input_dir = "/Users/carricarte/scratch/projects/imagery/main/vaso/derivatives/sub-09/analysis/vaso"

conditions = ['img_place', 'seen_place', 'constant', 'constant']
img_ind = [0, 1, 0, 1, 1, 1, 0, 1]
ind = [img_ind, list(np.array(img_ind) == 0), img_ind, list(np.array(img_ind) == 0)]
[cond1, cond2, cond3, cond4] = beta_x_cond(input_dir, conditions, ind)

beta_files = []
for c, b in enumerate([cond1, cond2, cond3, cond4]):
    for file in b:
        print(file)
    print("-----------------------------------------------------------------------------------------------")
    beta_mean = mean_img(b)
    filename = join(input_dir, 'mean_beta_{}.nii'.format(conditions[c]))
    beta_mean.to_filename(filename)
    beta_files.append(filename)

print('calculating percentage signal change')

constant = beta_files[2:4]
for c, beta in enumerate(beta_files[0:2]):
    psc = math_img("(img1/img2)*100", img1=beta, img2=constant[c])
    psc.to_filename(join(input_dir, "psc_{}.nii".format(conditions[c])))
