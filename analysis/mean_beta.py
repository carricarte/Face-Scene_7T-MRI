import sys
from os import listdir
from os.path import join
from nilearn.image import mean_img

def beta_x_cond(inputdir, conditions):
    # beta = [['0004', '0005'], ['0003', '0006'], ['0009', '0010'], ['0008', '0011']]
    beta = [['0004', '0005'], ['0003', '0006'], ['0009', '0010'], ['0008', '0011']]

    run_list = []
    [run_list.append(join(inputdir, f)) for f in listdir(inputdir) if "run-" in f and '._' not in f
     and "00" not in f and "sm_" not in f]

    run_list.sort()
    all_betas = [[], [], [], []]
    for run in run_list:
        for j, cond in enumerate(conditions):
            sub_string_1, sub_string_2 = beta[j]
            for x in listdir(run):
                if ("._" not in x and "_deveinDeconv" in x and "beta_" in x) and (sub_string_1 in x or sub_string_2 in x)\
                        and "rh" not in x and "lh" not in x:
                    all_betas[j].append(join(run, x))
                if j == 4:
                    if ("._" not in x and "_deveinDeconv" in x and "beta_" in x) and (
                            sub_string_1 in x or sub_string_2 in x) \
                            and "rh" not in x and "lh" not in x:
                        all_betas[j].append(join(run, x))


    return all_betas


input_dir = sys.argv[1]

conditions = ['img_face', 'img_place', 'seen_face', 'seen_place']
# conditions = ['img_place', 'seen_place']
[cond1, cond2, cond3, cond4] = beta_x_cond(input_dir, conditions)
# all_means = []
for c, b in enumerate([cond1, cond2, cond3, cond4]):
    for file in b:
        print(file)
    print("-----------------------------------------------------------------------------------------------")
    beta_mean = mean_img(b)
    beta_mean.to_filename(join(input_dir, 'mean_beta_{}_deveinDeconv.nii'.format(conditions[c])))
    # all_means.append(beta_mean)

# mean_constant = all_means[4]

# print('calculating % signal change...')
# for a, vitamine in enumerate(all_means):
#     psc = math_img("(img1/img2)*100", img1=vitamine, img2=mean_constant)
#     psc.to_filename(join(input_dir, "psc_{}.nii".format(conditions[a])))
# print('done')
