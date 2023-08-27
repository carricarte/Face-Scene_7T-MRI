import sys
import numpy as np
from nilearn.masking import apply_mask
import pandas as pd
import image_stats
from nilearn.image import new_img_like
from os.path import join, exists
from os import listdir

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
layer_mask_file = sys.argv[4]
tmap_dir = sys.argv[5]
hemisphere_mask_dir = sys.argv[6]

mask_layer_list = image_stats.get_layer(layer_mask_file)
layer_mask = image_stats.load_nifti(layer_mask_file)

df = pd.DataFrame()
r = 1
cols = ["accuracy"]
area = ["PPA", "FFA"]
hemisphere = ["whole", "right", "left"]
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place']
layers = ["deep", "middle", "superficial"]
# layers = ["l1", "l2", "l3"]
avg_betas = []

[avg_betas.append(join(input_dir, beta_vol)) for beta_vol in listdir(input_dir) if "CBV" not in beta_vol and
"psc" in beta_vol and "constant" not in beta_vol and "Linear" not in beta_vol and "deveinDeconv" not in beta_vol and
 beta_vol.endswith(".nii") and "lh" not in beta_vol and "rh" not in beta_vol and "._" not in beta_vol]
avg_betas.sort()

# voxel = np.arange(600, 2500, 100)  # 1000 final
# voxel = np.arange(2400, 2500, 100)  # 1000 final
# voxel = [2400]  # 2400 final (previous value)
voxel = [750]  # 750 visualization 31.05.23

for v in voxel:
    for a in area:
        if a == "PPA":
            tmap = "spmT_0010.nii"
        elif a == "FFA":
            tmap = "spmT_0006.nii"

        for h in hemisphere:

            pre_mask = join(hemisphere_mask_dir, "mask_{}_{}.nii".format(h, a))
            pf = pd.DataFrame(columns=conditions)
            vx = pd.DataFrame(columns=layers)

            # if exists(pre_mask):
            if True:
                left_mask_file = join(hemisphere_mask_dir, "left_mask.nii")
                right_mask_file = join(hemisphere_mask_dir, "right_mask.nii")

                tmap_file = join(tmap_dir, tmap)

                tval_right = image_stats.img_mask(tmap_file, mask_file, layer_mask_file, right_mask_file)
                tval_right.sort()
                tval_right = tval_right[::-1]
                t_right = tval_right[v]
                pre_mask_right = image_stats.threshold_mask(tmap_file, t_right,
                                                            np.logical_and(image_stats.load_nifti(right_mask_file) != 0,
                                                                           np.logical_and(
                                                                               layer_mask != 0,
                                                                               image_stats.load_nifti(mask_file) != 0)))

                tval_left = image_stats.img_mask(tmap_file, mask_file, layer_mask_file, left_mask_file)
                tval_left.sort()
                tval_left = tval_left[::-1]
                t_left = tval_left[v]
                pre_mask_left = image_stats.threshold_mask(tmap_file, t_left,
                                                           np.logical_and(image_stats.load_nifti(left_mask_file) != 0,
                                                                          np.logical_and(
                                                                              image_stats.load_nifti(mask_file) != 0,
                                                                              layer_mask != 0)))

                if h == "whole":
                    pre_mask = np.logical_or(pre_mask_right, pre_mask_left)
                elif h == "right":
                    pre_mask = pre_mask_right
                elif h == "left":
                    pre_mask = pre_mask_left

                # pre_mask = np.logical_and(pre_mask, layer_mask != 0)
                pre_mask_nii = new_img_like(avg_betas[0], pre_mask)
                pre_mask_nii.to_filename(join(hemisphere_mask_dir, "mask_{}_{}.nii".format(h, a)))

            # for k, layer_mask in enumerate(mask_layer_list):
            #
            #     mask = np.logical_and(pre_mask, layer_mask)
            #     vx.loc[k] = sum(sum(sum(mask)))
            #
            #     mask = new_img_like(avg_betas[0], mask)
            #
            #     acc = []
            #     lp = []
            #
            #     for avg_b in avg_betas:
            #         print(avg_b)
            #         roi_mean = apply_mask(avg_b, mask)
            #         lp.append(np.mean(roi_mean))
            #     print(lp)
            #     pf.loc[len(pf)] = lp
            #
            # pf["layer"] = ["deep", "middle", "superficial"]
            # pf["layer"] = ["l1", "l2" "l3" "l4" "l5", "l6"]
            # vx.to_csv(join(output_dir, 'voxels_{}_{}.tsv').format(h, a), sep='\t', index=True)
            # pf.to_csv(join(output_dir, 'profile_{}_{}_{}_devein_linear.tsv').format(h, a, v), sep='\t', index=True)
