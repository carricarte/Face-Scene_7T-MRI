import sys
import numpy as np
from nilearn.image import mean_img
from nilearn.masking import apply_mask
import pandas as pd
import image_stats
from nilearn.image import new_img_like
from os.path import join
from nilearn.image import math_img
from os import listdir
from nipype.interfaces.ants import ImageMath
from sklearn.ensemble import IsolationForest

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_file = sys.argv[3]
layer_mask_file = sys.argv[4]
tmap_dir = sys.argv[5]
hemisphere_mask_dir = sys.argv[6]


def truncate_masker(img, mask_f):
    val = img[mask_f]
    # val.sort()
    # n = len(val)
    # min_val = val[int(0.025*n)] - 1
    # max_val = val[int(0.975*n)]

    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(np.array(val).reshape(-1, 1))
    # select all rows that are not outliers
    vmask = yhat != -1
    mask = np.zeros(shape=img.shape).astype(bool)
    mask[mask_f] = vmask
    # return np.logical_and(img > min_val, img < max_val)
    return mask

def contrast_ratio(inputdir):

    modality = ["perception", "imagery"]
    area = ["FFA", "PPA"]

    nifti_con = []
    for a in area:
        for m in modality:

            run_list = []
            [run_list.append(join(inputdir, f)) for f in listdir(inputdir) if "run-" in f and '._' not in f]
            run_list.sort()

            if m == "perception":
                if a == "FFA":
                    c1 = "con_0010.nii"
                    c2 = "con_0011.nii"
                elif a == "PPA":
                    c1 = "con_0011.nii"
                    c2 = "con_0010.nii"
            elif m == "imagery":
                if a == "FFA":
                    c1 = "con_0012.nii"
                    c2 = "con_0013.nii"
                elif a == "PPA":
                    c1 = "con_0013.nii"
                    c2 = "con_0012.nii"

            con1_files = []
            for run in run_list:
                [con1_files.append(join(run, x)) for x in listdir(run) if
                 "._" not in x and c1 in x]
            con1_files.sort()

            con2_files = []
            for run in run_list:
                [con2_files.append(join(run, x)) for x in listdir(run) if
                 "._" not in x and c2 in x]
            con2_files.sort()
            c1_data = mean_img(con1_files)
            c2_data = mean_img(con2_files)
            c1.to_filename(join(input_dir, "mean_{}".format(c1)))
            c2.to_filename(join(input_dir, "mean_{}".format(c2)))
            nifti_con.append([c1_data, c2_data])
            # nifti_con.append(math_img("img1 + img2", img1=c1, img2=c2))
    return nifti_con


def profile_activation(contrast_par, mask, layer_mask):

    contrast_c1 = contrast_par[0].get_fdata()
    contrast_c2 = contrast_par[1].get_fdata()
    profile_act = []
    n_layer = max(np.unique(layer_mask)).astype(int)

    for l in range(1, n_layer + 1):
        lay_mask = np.logical_and(np.logical_and(mask != 0, layer_mask == l),
                                  np.logical_and(np.logical_not(np.isnan(contrast_c1)), np.logical_not(np.isnan(contrast_c2))))
        # masked_con = apply_mask(contrast_map, lay_mask)
        truncate_mask = truncate_masker(contrast_c1/contrast_c2, lay_mask)
        final_mask = np.logical_and(lay_mask, truncate_mask)
        print(sum(sum(sum(final_mask))))
        profile_act.append(np.median(contrast_c1[final_mask])/np.median(contrast_c2[final_mask]))
        # profile_act.append(np.mean(masked_con))

    return profile_act


df = pd.DataFrame()

r = 1

hemisphere = ["whole", "left", "right"]

for h in hemisphere:
    if h == "whole":
        hmask = "mask.nii"
        voxel = 3000
    elif h == "left":
        hmask = "left_mask.nii"
        voxel = 1500
    elif h == "right":
        hmask = "right_mask.nii"
        voxel = 1500

    hemisphere_mask_file = join(hemisphere_mask_dir, hmask)
    layer = ["deep", "middle", "superficial"]
    area = ["FFA", "PPA"]

    [con_FFA_per, con_FFA_img, con_PPA_per, con_PPA_img] = contrast_ratio(input_dir)

    mask_layer_list = image_stats.get_layer(layer_mask_file)

    for a in area:
        if a == "FFA":
            tmap_file = join(tmap_dir, "spmT_0006.nii")
            contrast_pair = [con_FFA_per, con_FFA_img]
        elif a == "PPA":
            tmap_file = join(tmap_dir, "spmT_0010.nii")
            contrast_pair = [con_PPA_per, con_PPA_img]

        tval = image_stats.img_mask(tmap_file, mask_file, layer_mask_file, hemisphere_mask_file)
        # tval = image_stats.img_mask(tmap_file, mask_file, layer_mask_file)
        tval.sort()
        tval = tval[::-1]
        t = tval[voxel]

        pre_mask = image_stats.threshold_mask(tmap_file, t, np.logical_and(image_stats.load_nifti(hemisphere_mask_file),
                                                                           np.logical_and(
                                                                               image_stats.load_nifti(mask_file) != 0,
                                                                               layer_mask_file != 0)))
        pf = pd.DataFrame(columns=layer)
        for c in contrast_pair:
            pf.loc[len(pf)] = profile_activation(c, pre_mask, image_stats.load_nifti(layer_mask_file))
        pf["modality"] = ["perception", "imagery"]
        # pf.to_csv(join(output_dir, 'profile_{}_{}_devein_contrast.tsv'.format(h, a)), sep='\t', index=True)
