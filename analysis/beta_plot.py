import sys
import numpy as np
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from os.path import split
import pandas as pd

sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/multivariate')
import image_stats

# directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_dir = sys.argv[3]
tmap_file = sys.argv[4]
hemisphere_mask_dir = sys.argv[5]
tmap_dir, file = split(tmap_file)

# input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-02/analysis"
# output_dir = '/Users/carricarte/PhD/Debugging/bold/sub-02/analysis'
# mask_dir = "/Users/carricarte/PhD/Debugging/bold/sub-02/mask/mask.nii"
# tmap_dir = "/Users/carricarte/PhD/Debugging/bold/sub-02/analysis/localizer/"
areas = ["FFA", "PPA"]
# v = 2600
v = 1300

betas = []
[betas.append(join(input_dir, f)) for f in listdir(input_dir) if "mean" in f and "no_devein" in f and
 f.endswith(".nii")]
betas.sort()

color_list = ['steelblue', 'steelblue', '#ff7f0e', '#ff7f0e']
alpha_list = [0.3, 1, 0.3, 1]
label_list = ['imagery', 'perception']
# label_list = trial_type
posX = [-1.5, - 0.5, 0.5, 1.5]
mean_beta = []
areas = ["right", "left", "both"]

for a in areas:

    # if a == "FFA":
    #     f = "spmT_0006.nii"
    # elif a == "PPA":
    #     f = "spmT_0010.nii"
    if a == "right":
        f = "right_mask.nii"
    elif a == "left":
        f = "left_mask.nii"
    elif a == "both":
        f = "mask.nii"
        v = 2600

    # posX = [-1.5, - 0.5, 0.5, 1.5]
    hemisphere_mask = join(hemisphere_mask_dir, f)
    tmap_file = join(tmap_dir, "spmT_0006.nii")
    for b in betas:
        # tmap_file = join(tmap_dir, f)
        # tval = image_stats.img_mask(tmap_file, mask_dir)
        tval = image_stats.img_mask(tmap_file, mask_dir, hemisphere_mask)
        tval.sort()
        tval = tval[::-1]
        t = tval[v]
        new_mask = image_stats.threshold_mask(tmap_file, t)
        mask = image_stats.load_nifti(mask_dir)
        # mean_beta.append(image_stats.mean_roi(b, np.logical_and(mask, new_mask)))
        mean_beta.append(image_stats.mean_roi(b, np.logical_and(image_stats.load_nifti(hemisphere_mask),
                                                                np.logical_and(mask, new_mask))))
        print(b)
    print(mean_beta)

df = pd.DataFrame(columns=["img_face", "img_place", "seen_face", "seen_place"])
df.loc[len(df)] = mean_beta[0:4]
df.loc[len(df)] = mean_beta[4:8]
df.loc[len(df)] = mean_beta[8:12]
# df["area"] = ["FFA", "PPA"]
df["area"] = ["right", "left", "whole-brain"]
df.to_csv(join(output_dir, 'betas_FFA.tsv'), sep='\t', index=True)
# all_mean = np.vstack((mean_beta[0:4], mean_beta[4:8])).T
# # labels = ['FFA', 'PPA', 'EVC']
# labels = ['FFA', 'PPA']
# # # x = np.array([1, 4, 7])  # the label locations
# x = np.array([1, 4])  # the label locations
# width = 0.5  # the width of the bars
# fig, ax = plt.subplots()
# ax.bar(x - 1.5 * width, all_mean[0], width, label='img_face', color='steelblue', alpha=0.3)
# ax.bar(x - 0.5 * width, all_mean[1], width, label='img_face', color='#ff7f0e', alpha=0.3)
# ax.bar(x + 0.5 * width, all_mean[2], width, label='img_face', color='steelblue', alpha=1)
# ax.bar(x + 1.5 * width, all_mean[3], width, label='img_face', color='#ff7f0e', alpha=1)
# # # plt.ylim(0, 1)
# ax.set_ylabel('mean beta')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# plt.show()
# plt.savefig(join(output_dir, "beta_plot"))
# plt.clf()
# print("hello world")
