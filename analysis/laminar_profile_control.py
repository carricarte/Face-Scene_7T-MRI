import sys
import numpy as np
from nilearn.masking import apply_mask
import pandas as pd
import image_stats
from nilearn.image import new_img_like
from os.path import join, split
from os import listdir
from math import isnan


def get_column_mask(column_niimg, column_ind):
    # column_niimg = load_img(column_file).get_fdata()
    column_mask = np.zeros(shape=np.shape(column_niimg))
    for ind in column_ind:
        if ind != 0 and not isnan(ind):
            column_mask = np.logical_or(column_mask == 1, column_niimg == ind)
    print(sum(sum(sum(column_mask))))
    return column_mask


input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
layer_mask_file = sys.argv[4]
tmap_dir = sys.argv[5]
hemisphere_mask_dir = sys.argv[6]

# input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/test"
# output_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/test"
# mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask/mask.nii"
# layer_mask_file = "/Users/carricarte/PhD/Debugging/bold/sub-18/anat/rim_layers_equivol.nii"
# tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/analysis/localizer"
# hemisphere_mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-18/mask"

column_dir, lf = split(layer_mask_file)
mask_layer_list = image_stats.get_layer(layer_mask_file)

# column_niimg_right = load_img(join(column_dir, "lh_ribbon_columns20000.nii")).get_fdata()
# column_niimg_left = load_img(join(column_dir, "rh_ribbon_columns20000.nii")).get_fdata()

df = pd.DataFrame()
r = 1
cols = ["accuracy"]
area = ["FFA", "PPA"]
# area = ["PPA", "FFA"]
hemisphere = ["whole", "right", "left"]
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place']
layers = ["deep", "middle", "superficial"]
# layers = ["l1", "l2", "l3"]
left_mask_file = join(hemisphere_mask_dir, "left_mask.nii")
right_mask_file = join(hemisphere_mask_dir, "right_mask.nii")

avg_betas = []
[avg_betas.append(join(input_dir, beta_vol)) for beta_vol in listdir(input_dir) if "CBV" not in beta_vol and
 "psc" not in beta_vol and "constant" not in beta_vol and "Linear" not in beta_vol and "econv" not in beta_vol and
 beta_vol.endswith(".nii") and "lh" not in beta_vol and "rh" not in beta_vol and "._" not in beta_vol and "lambda"
 not in beta_vol]
avg_betas.sort()

for file in avg_betas:
    print(file)

# voxel = 50  # 1000 final
voxel = np.arange(50, 1100, 50)  # 1000 final
# voxel = np.arange(50, 1050, 50)  # 1000 final


for v in voxel:
    for a in area:
        if a == "PPA":
            tmap = "spmT_0010.nii"
        elif a == "FFA":
            tmap = "spmT_0006.nii"

        for h in hemisphere:

            pf = pd.DataFrame(columns=conditions)
            vx = pd.DataFrame(columns=layers)

            tmap_file = join(tmap_dir, tmap)

            for k, layer_mask in enumerate(mask_layer_list):

                pre_mask_right = np.logical_and(
                    np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(right_mask_file) != 0)
                    , layer_mask)

                pre_mask_left = np.logical_and(
                    np.logical_and(image_stats.load_nifti(mask_file) != 0, image_stats.load_nifti(left_mask_file) != 0)
                    , layer_mask)

                tval_right = image_stats.img_mask(tmap_file, pre_mask_right)
                tval_right.sort()
                tval_right = tval_right[::-1]
                t_right = tval_right[v]
                mask_right = image_stats.threshold_mask(tmap_file, t_right, pre_mask_right)
                # column_index_right = column_niimg_right[mask_right]
                # print(column_index_right)
                # mask_right = get_column_mask(column_niimg_right, column_index_right)

                tval_left = image_stats.img_mask(tmap_file, pre_mask_left)
                tval_left.sort()
                tval_left = tval_left[::-1]
                t_left = tval_left[v]
                mask_left = image_stats.threshold_mask(tmap_file, t_left, pre_mask_left)
                # column_index_left = column_niimg_left[mask_left]
                # print(column_index_left)
                # mask_left = get_column_mask(column_niimg_left, column_index_left)

                if h == "whole":
                    mask = np.logical_or(mask_left, mask_right)
                elif h == "right":
                    mask = mask_right
                elif h == "left":
                    mask = mask_left

                vx.loc[k] = sum(sum(sum(mask)))
                mask = new_img_like(avg_betas[0], mask)
                # if k == 0 and v == 1500 and h == "whole":
                #     mask.to_filename(join(hemisphere_mask_dir, "mask1502_{}_{}.nii".format(h, a)))
                acc = []
                lp = []

                for avg_b in avg_betas:
                    print(avg_b)
                    roi_mean = apply_mask(avg_b, mask)
                    outlayer_mask = np.logical_or(roi_mean < -500, roi_mean > 500)
                    lp.append(np.mean(roi_mean[outlayer_mask == 0]))
                print(lp)
                pf.loc[len(pf)] = lp

            pf["layer"] = ["deep", "middle", "superficial"]
            # pf["layer"] = ["l1", "l2" "l3" "l4" "l5", "l6"]
            # vx.to_csv(join(output_dir, 'voxels_{}_{}.tsv').format(h, a), sep='\t', index=True)
            pf.to_csv(join(output_dir, '@lamprofile_{}_{}_{}_no-devein.tsv').format(h, a, v), sep='\t',
                      index=True)
