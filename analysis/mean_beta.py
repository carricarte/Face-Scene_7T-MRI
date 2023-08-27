import sys
<<<<<<< HEAD
from os import listdir
from os.path import join
from nilearn.image import mean_img

def beta_x_cond(inputdir, conditions):
    # beta = [['0004', '0005'], ['0003', '0006'], ['0009', '0010'], ['0008', '0011']]
    beta = [['0004', '0005'], ['0003', '0006'], ['0009', '0010'], ['0008', '0011']]
=======
import numpy as np
from os import listdir
from os.path import join
from nilearn.image import mean_img, math_img

def beta_x_cond(inputdir, conditions):
    # beta = [['0004', '0005'], ['0003', '0006'], ['0009', '0010'], ['0008', '0011']]
    beta = [['0004', '0005'], ['0003', '0006'], ['0009', '0010'], ['0008', '0011'], ['0019', 'EXCLUDE']]
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302

    run_list = []
    [run_list.append(join(inputdir, f)) for f in listdir(inputdir) if "run-" in f and '._' not in f
     and "00" not in f and "sm_" not in f]

    run_list.sort()
<<<<<<< HEAD
    all_betas = [[], [], [], []]
=======
    all_betas = [[], [], [], [], []]
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
    for run in run_list:
        for j, cond in enumerate(conditions):
            sub_string_1, sub_string_2 = beta[j]
            for x in listdir(run):
<<<<<<< HEAD
                if ("._" not in x and "_deveinDeconv" in x and "beta_" in x) and (sub_string_1 in x or sub_string_2 in x)\
                        and "rh" not in x and "lh" not in x:
                    all_betas[j].append(join(run, x))
                if j == 4:
                    if ("._" not in x and "_deveinDeconv" in x and "beta_" in x) and (
=======
                if ("._" not in x and "_deveinDeconv" not in x and "beta_" in x) and (sub_string_1 in x or sub_string_2 in x)\
                        and "rh" not in x and "lh" not in x:
                    all_betas[j].append(join(run, x))
                if j == 4:
                    if ("._" not in x and "_deveinDeconv" not in x and "beta_" in x) and (
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
                            sub_string_1 in x or sub_string_2 in x) \
                            and "rh" not in x and "lh" not in x:
                        all_betas[j].append(join(run, x))


    return all_betas

<<<<<<< HEAD

input_dir = sys.argv[1]

conditions = ['img_face', 'img_place', 'seen_face', 'seen_place']
# conditions = ['img_place', 'seen_place']
[cond1, cond2, cond3, cond4] = beta_x_cond(input_dir, conditions)
# all_means = []
for c, b in enumerate([cond1, cond2, cond3, cond4]):
=======
# def beta_x_cond(inputdir, conditions):
#
#     all_betas = [[], []]
#     betas = []
#     [betas.append(join(inputdir, x)) for x in listdir(inputdir) if
#      "._" not in x and "beta_" in x and x.endswith(".nii") and "mean" not in x]
#     betas.sort()
#     for j, cond in enumerate(conditions):
#         if "img_place" in cond:
#             s = 1
#         elif "seen_place" in cond:
#             s = 2
#         all_betas[j].append(betas[s:90:9])
#
#
#     return all_betas

input_dir = sys.argv[1]
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place', 'constant']
# conditions = ['img_place', 'seen_place']
[cond1, cond2, cond3, cond4, cond5] = beta_x_cond(input_dir, conditions)
all_means = []
for c, b in enumerate([cond1, cond2, cond3, cond4, cond5]):
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
    for file in b:
        print(file)
    print("-----------------------------------------------------------------------------------------------")
    beta_mean = mean_img(b)
<<<<<<< HEAD
    beta_mean.to_filename(join(input_dir, 'mean_beta_{}_deveinDeconv.nii'.format(conditions[c])))
    # all_means.append(beta_mean)

# mean_constant = all_means[4]

# print('calculating % signal change...')
# for a, vitamine in enumerate(all_means):
#     psc = math_img("(img1/img2)*100", img1=vitamine, img2=mean_constant)
#     psc.to_filename(join(input_dir, "psc_{}.nii".format(conditions[a])))
# print('done')
=======
    beta_mean.to_filename(join(input_dir, 'mean_beta_{}.nii'.format(conditions[c])))
    all_means.append(beta_mean)

mean_constant = all_means[4]

print('calculating % signal change...')
for a, vitamine in enumerate(all_means):
    psc = math_img("(img1/img2)*100", img1=vitamine, img2=mean_constant)
    psc.to_filename(join(input_dir, "psc_{}.nii".format(conditions[a])))
print('done')
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
