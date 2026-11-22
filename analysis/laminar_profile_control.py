import sys
import numpy as np
from nilearn.masking import apply_mask
import pandas as pd
import image_stats
from nilearn.image import new_img_like
from os.path import join
from os import listdir


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

mask_layer_list = image_stats.get_layer(layer_mask_file)

# column_niimg_right = load_img(join(column_dir, "lh_ribbon_columns20000.nii")).get_fdata()
# column_niimg_left = load_img(join(column_dir, "rh_ribbon_columns20000.nii")).get_fdata()

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


TMAP_FILES = {"FFA": "spmT_0006.nii", "PPA": "spmT_0010.nii"}
tmaps = {a: image_stats.load_nifti(join(tmap_dir, TMAP_FILES[a])) for a in area}

# ROI ∧ hemisphere ∧ layer masks do not depend on the voxel count, area or hemisphere
# being profiled, so build them once instead of reloading the masks in every iteration.
roi_mask = image_stats.load_nifti(mask_file) != 0
hemi_masks = {side: np.logical_and(roi_mask, image_stats.load_nifti(f) != 0)
              for side, f in (("right", right_mask_file), ("left", left_mask_file))}
pre_masks = [{side: np.logical_and(m, layer_mask) for side, m in hemi_masks.items()}
             for layer_mask in mask_layer_list]


def top_v_mask(tmap, pre_mask, v):
    """Voxels in `pre_mask` whose t-value is at least the (v+1)-th largest."""
    t_thr = np.sort(image_stats.img_mask(tmap, pre_mask))[-v - 1]
    return image_stats.threshold_mask(tmap, t_thr, pre_mask)


for v in voxel:
    for a in area:
        tmap = tmaps[a]

        for h in hemisphere:

            pf = pd.DataFrame(columns=conditions)

            for k, pre_mask in enumerate(pre_masks):

                mask_right = top_v_mask(tmap, pre_mask["right"], v)
                mask_left = top_v_mask(tmap, pre_mask["left"], v)

                if h == "whole":
                    mask = np.logical_or(mask_left, mask_right)
                elif h == "right":
                    mask = mask_right
                elif h == "left":
                    mask = mask_left

                mask = new_img_like(avg_betas[0], mask)
                # if k == 0 and v == 1500 and h == "whole":
                #     mask.to_filename(join(hemisphere_mask_dir, "mask1502_{}_{}.nii".format(h, a)))
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
            pf.to_csv(join(output_dir, '@lamprofile_{}_{}_{}_no-devein.tsv').format(h, a, v), sep='\t',
                      index=True)
