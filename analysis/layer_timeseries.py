import numpy as np
from nilearn.image import load_img
import pandas as pd
from os.path import join
from image_stats import time_series_to_matrix
import pandas as pd
import matplotlib.pyplot as plt
import sys

n_loc = [100, 200, 300, 400, 500, 600, 700, 800]
color = ['crimson', 'crimson', 'darkgoldenrod', '#666699']

layer = ['deep', 'middle', 'superficial']
condition_color = ['blue', 'green', 'red']
nifti_dir = sys.argv[1]
output = sys.argv[2]
up_loc_tmap_file = sys.argv[3]
layer_mask_file = sys.argv[4]
tsv_dir = sys.argv[5]
subject = sys.argv[6]

condition = ["per", "img"]
sequence = ["bold", "vaso"]

for s in sequence:

    output_dir = join(output, s)

    for c in condition:

        # data = pd.DataFrame(columns=["deep", "middle", "superficial"])
        # vox = pd.DataFrame(columns=["deep", "middle", "superficial"])

        if c == "per":
            cond = ['seen_place']
            tsv_file = join(tsv_dir, "glm_block_sub" + subject + "_run01.tsv")

            if s == "bold":

                mean4d = join(nifti_dir, "up_mean4d_bold_per.nii")

            elif s == "vaso":

                mean4d = join(nifti_dir, "up_mean4d_vaso_per.nii")

        if c == "img":
            cond = ['img_place']
            tsv_file = join(tsv_dir, "glm_block_sub" + subject + "_run02.tsv")

            if s == "bold":

                mean4d = join(nifti_dir, "up_mean4d_bold_img.nii")

            elif s == "vaso":

                mean4d = join(nifti_dir, "up_mean4d_vaso_img.nii")

        for n in n_loc:

            df = pd.DataFrame()
            fig = plt.figure()
            ax = fig.add_subplot(111)

            trigger = pd.read_table(tsv_file)
            trigger = np.repeat(np.array(trigger['trial_type']), 3)

            if subject == "08":
                trigger = np.append(np.array(['error']), trigger)  # sub-08
            elif subject == "09":
                trigger = np.append(trigger, np.array(['baseline']))  # sub-09

            up_t_map_loc = load_img(up_loc_tmap_file).get_fdata()
            layer_mask = load_img(layer_mask_file).get_fdata()

            for a, l in enumerate(np.unique(layer_mask)):
                if l != 0:
                    lmask = layer_mask == int(l)
                    gmasked_t_map_loc = up_t_map_loc[lmask]
                    gmasked_t_map_loc.sort()
                    gsorted_masked_t_map_loc = gmasked_t_map_loc[::-1]
                    gt_loc = gsorted_masked_t_map_loc[n]
                    mask = np.logical_and(up_t_map_loc > gt_loc, lmask)
                    # print(sum(sum(sum(mask))))
                    mean_ = np.mean(time_series_to_matrix(mean4d, mask), 0)
                    df[layer[a - 1]] = mean_
                    ax.plot(mean_, color=color[a], linewidth=0.5)

            df = df.T
            df.to_csv(join(output_dir, 'layer_timeseries_{0}_{1}.tsv'.format(c, n)), sep='\t', index=True)
            previous_t = trigger[0]
            v = 0
            w = 0
            y = plt.ylim()
            for e, t in enumerate(trigger):
                if t != previous_t:
                    w = e
                    if previous_t == "baseline":
                        col = "green"
                    elif previous_t == "seen_place":
                        col = "blue"
                    elif previous_t == "img_place":
                        col = "red"
                    else:
                        col = "white"
                    ax.fill_betweenx(y, v, w, facecolor=col, alpha=0.1)
                    v = w
                    previous_t = t
            ax.fill_betweenx(y, w, len(trigger), facecolor='green', alpha=0.1)
            # ax.fill_between((w, len(trigger)), facecolor="green", alpha=0.1)
            plt.savefig(join(output_dir, "layer_timeseries_{0}_{1}.png".format(c, n)))
