import sys
from os.path import join, split
import numpy as np
import pandas as pd
import image_stats
from nilearn.image import new_img_like
from mean_beta import beta_x_cond

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
layer_mask_file = sys.argv[4]

mask_dir, mask_filename = split(mask_file)


def decoding_analysis(X, y, mask):
    from nilearn.masking import apply_mask
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    svc = SVC()
    c = [1]
    param_grid = dict(C=c)
    masked_data = apply_mask(X, mask)

    for k in np.arange(2, int(len(X) / 2) + 1):
        grid = GridSearchCV(svc, param_grid, cv=k, scoring='accuracy')
        grid.fit(masked_data, y)
        print('-> folds=%d, accuracy=%.3f' % (k, grid.best_score_))

    cv = int(len(X) / 4)
    # if "sub-06" in input_dir or "sub-07" in input_dir:
    if "sub-08" in input_dir or "sub-06" in input_dir or "sub-10" in input_dir or "sub-12" in input_dir or "sub-13" in input_dir or "sub-17" in input_dir or "sub-18" in input_dir:
        cv = int(len(X) / 2)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
    grid.fit(masked_data, y)
    return grid.best_score_


mask_layer_list = image_stats.get_layer(layer_mask_file)

df = pd.DataFrame()
r = 1
cols = ["accuracy"]
area = ["PPA", "FFA"]
modality = ["per", "img"]
hemisphere = ["whole", "right", "left"]
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place']
column_names = ["deep", "middle", "superficial"]
[cond1, cond2, cond3, cond4] = beta_x_cond(input_dir, conditions)

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
            acc = []
            vx = []

            acc = []
            for m in modality:

                if m == "per":
                    cond_a = cond3
                    cond_b = cond4
                elif m == "img":
                    cond_a = cond1
                    cond_b = cond2
                data = np.concatenate((np.array(cond_a), np.array(cond_b)), 0)
                # for d in data:
                #     print(d)
                y = np.concatenate((np.zeros(len(cond_a)), np.ones(len(cond_b))), 0)
                print('{}_{}_{}_{}.tsv'.format(h, a, k + 1, m))
                acc.append(decoding_analysis(data, y, mask))

            da[column_names[k]] = acc
        da.to_csv(join(output_dir, 'acc_{}_{}_no_devein.tsv').format(h, a), sep='\t', index=True)
