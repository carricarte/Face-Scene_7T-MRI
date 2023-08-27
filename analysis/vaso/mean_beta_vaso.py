import sys
import numpy as np
from os import listdir
from os.path import join
from nilearn.image import mean_img, math_img


def beta_x_cond(inputdir, conditions):
    # beta = [['0002', '0003', '0004'], ['0005', '0006', '0007'], ['0014', '0014', '0014']]
    beta = [['0002', '0002', '0002'], ['0003', '0003', '0003'], ['0010', '0010', '0010']]

    run_list = []
    [run_list.append(join(inputdir, f, 'beta')) for f in listdir(inputdir) if "run-" in f and '._' not in f
     and "00" not in f and "sm_" not in f]

    run_list.sort()
    all_betas = [[], [], []]
    for run in run_list:
        for j, cond in enumerate(conditions):
            sub_string_1, sub_string_2, sub_string_3 = beta[j]
            for x in listdir(run):
                if ("._" not in x and "beta_" in x) and (sub_string_1 in x or sub_string_2 in x or sub_string_3 in x):
                    all_betas[j].append(join(run, x))

    return all_betas


input_dir = sys.argv[1]

conditions = ['img_place', 'seen_place', 'constant']
[cond1, cond2, cond3] = beta_x_cond(input_dir, conditions)

beta_files = []
for c, b in enumerate([cond1, cond2, cond3]):
    for file in b:
        print(file)
    print("-----------------------------------------------------------------------------------------------")
    beta_mean = mean_img(b)
    filename = join(input_dir, 'mean_beta_{}.nii'.format(conditions[c]))
    beta_mean.to_filename(filename)
    beta_files.append(filename)

print('calculating percent signal change')

constant = beta_files[2]
for c, beta in enumerate(beta_files[0:2]):
    psc = math_img("(img1/img2)*100", img1=beta, img2=constant)
    psc.to_filename(join(input_dir, "psc_{}.nii".format(conditions[c])))
