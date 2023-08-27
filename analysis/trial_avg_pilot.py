import sys
import numpy as np
from nilearn.image import new_img_like, load_img
import pandas as pd
import image_stats
from os.path import join

# MEM: 20GB

nifti_dir = sys.argv[1]
output_dir = sys.argv[2]
tsv_dir = sys.argv[3]
mask_file = sys.argv[4]
subject = sys.argv[5]

# nifti_file = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/func/mean4d_bold.nii"
# output_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/func/"
# tsv_file = "/Users/carricarte/scratch/projects/imagery/main/vaso/behdata/sub-04/glm_block_sub04_run01.tsv"
# mask_file = "/Users/carricarte/PhD/Debugging/vaso/main/sub-04/mask/loc_mask.nii"

condition = ["perception", "imagery"]
sequence = ["bold", "vaso"]

for c in condition:

    for s in sequence:

        output_dir = join(output_dir, s)

        if c == "perception":
            cond = ['seen_place']
            tsv_file = join(tsv_dir, "glm_block_sub" + subject + "_run01.tsv")

            if s == "bold":

                nifti_file = join(nifti_dir, "up_mean4d_bold_per.nii")

            elif s == "vaso":

                nifti_file = join(nifti_dir, "up_mean4d_vaso_per.nii")

        if c == "imagery":
            cond = ['img_place']
            tsv_file = join(tsv_dir, "glm_block_sub" + subject + "_run02.tsv")

            if s == "bold":

                nifti_file = join(nifti_dir, "up_mean4d_bold_img.nii")

            elif s == "vaso":

                nifti_file = join(nifti_dir, "up_mean4d_vaso_img.nii")

        trigger = pd.read_table(tsv_file)
        tr = np.tile(np.array([1, 2, 3]), len(trigger['trial_type']))
        trigger = np.repeat(np.array(trigger['trial_type']), 3)

        if subject == "08":
            trigger = np.append(np.array(['waiting']), trigger)  # sub-08
            tr = np.append(0, tr)
        elif subject == "09":
            trigger = np.append(trigger, np.array(['waiting']))  #sub-09
            tr = np.append(tr, 0)

        time_series = pd.DataFrame(image_stats.time_series_to_mat(nifti_file).T)
        print(np.array(trigger).shape)
        print(np.array(time_series).shape)
        time_series['condition'] = trigger
        time_series['order'] = tr
        ori_shape = load_img(mask_file).get_fdata().shape

        groups = time_series.groupby(['order', 'condition'])
        # cond = ['seen_place', 'img_place']

        for tr in [1, 2, 3]:
            for c in cond:
                print(tr)
                print(np.array(groups.get_group((tr, 'baseline')))[:, :-2])
                print(np.mean(np.array(groups.get_group((tr, 'baseline')))[:, :-2]), 0)
                volume = np.mean(np.array(groups.get_group((tr, c)))[:, :-2], 0) - \
                         np.mean(np.array(groups.get_group((tr, 'baseline')))[:, :-2], 0)
                volume = np.array(volume).reshape(ori_shape).astype(np.float32)
                mean_file = new_img_like(mask_file, volume)
                mean_file.to_filename(join(output_dir, "mean-{0}_tr-{1}.nii".format(c, tr)))
