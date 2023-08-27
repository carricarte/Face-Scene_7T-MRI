import sys
from os.path import join, split
import numpy as np
import pandas as pd
import image_stats
from nilearn.image import new_img_like
from mean_beta import beta_x_cond
from nilearn.masking import apply_mask

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
layer_mask_file = sys.argv[4]

mask_dir, mask_filename = split(mask_file)


def decoding_analysis(X, y, mask):

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    svc = SVC()
    c = [1]
    param_grid = dict(C=c)
    masked_data = apply_mask(X, mask)
    cv = int(len(X) / 4)
    if "sub-08" in input_dir or "sub-06" in input_dir or "sub-10" in input_dir or "sub-12" in input_dir or "sub-13" in input_dir or "sub-17" in input_dir or "sub-18" in input_dir:
        cv = int(len(X) / 2)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
    grid.fit(masked_data, y)

    return grid


mask_layer_list = image_stats.get_layer(layer_mask_file)

df = pd.DataFrame()
r = 1
cols = ["accuracy"]
area = ["PPA", "FFA"]
hemisphere = ["whole"]
modality = ['IP', 'PI', 'II', 'PP']
# modality = ['II', 'PP']
column_names = ["deep", "middle", "superficial"]
[cond1, cond2, cond3, cond4] = beta_x_cond(input_dir, ['img_face', 'img_place', 'seen_face', 'seen_place'])

voxel = 1000

for a in area:

    for h in hemisphere:

        if a == "FFA" and h == "whole":
            pre_mask = join(mask_dir, "mask_whole_FFA.nii")
        elif a == "FFA" and h == "right":
            pre_mask = join(mask_dir, "mask_right_FFA.nii")
        elif a == "FFA" and h == "left":
            pre_mask = join(mask_dir, "mask_left_FFA.nii")
        elif a == "PPA" and h == "whole":
            pre_mask = join(mask_dir, "mask_whole_PPA.nii")
        elif a == "PPA" and h == "right":
            pre_mask = join(mask_dir, "mask_right_PPA.nii")
        elif a == "PPA" and h == "left":
            pre_mask = join(mask_dir, "mask_left_PPA.nii")

        pre_mask = np.array(image_stats.load_nifti(pre_mask))
        da = pd.DataFrame()

        for k, layer_mask in enumerate(mask_layer_list):

            mask = np.logical_and(pre_mask, layer_mask)
            mask = new_img_like(cond1[0], mask)
            vx = []

            acc = []
            for m in modality:

                if m == "IP":
                    train_data = np.concatenate((np.array(cond1), np.array(cond2)), 0)
                    test_data = np.concatenate((np.array(cond3), np.array(cond4)), 0)
                elif m == "PI":
                    test_data = np.concatenate((np.array(cond1), np.array(cond2)), 0)
                    train_data = np.concatenate((np.array(cond3), np.array(cond4)), 0)
                if m == "II":
                    train_data = np.concatenate((np.array(cond1), np.array(cond2)), 0)
                elif m == "PP":
                    train_data = np.concatenate((np.array(cond3), np.array(cond4)), 0)

                y = np.concatenate((np.zeros(len(cond1)), np.ones(len(cond2))), 0)
                print('{}_{}_{}_{}'.format(h, a, k + 1, m))
                for d in train_data:
                    print(d)
                svm_grid = decoding_analysis(train_data, y, mask)

                score = svm_grid.best_score_
                if m == "IP" or m == "PI":
                    X_test = apply_mask(test_data, mask)
                    score = svm_grid.score(X_test, y)
                acc.append(score)
            print(acc)
            da[column_names[k]] = acc
        da.to_csv(join(output_dir, 'cross_acc_{}_{}_devein_deconv.tsv').format(h, a), sep='\t', index=True)
