import sys
import numpy as np
import pandas as pd
from math import ceil
from os import listdir
from os.path import join
from sklearn.svm import SVC
from mpl_toolkits.axes_grid1 import host_subplot
from nilearn.image import index_img
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/multivariate')
import image_stats
# import stats
from analysis.multivariate import stats
from sklearn.model_selection import GridSearchCV
import nibabel as nb
from nilearn.image import new_img_like
import matplotlib.pyplot as plt
from analysis import image_stats
from sklearn.preprocessing import StandardScaler

# directories
# input_dir = sys.argv[1]
# output_dir = sys.argv[2]
# tsv_dir = sys.argv[3]
# mask_dir = sys.argv[4]
# tmap_file = sys.argv[5]

input_dir = "/Users/carricarte/PhD/Debugging/vaso/sub-02/func"
output_dir = '/Users/carricarte/PhD/Debugging/vaso/sub-02'
tsv_dir = "/Users/carricarte/scratch/projects/imagery/pilot/vaso/behdata/sub-02"
mask_dir = "/Users/carricarte/PhD/Debugging/vaso/sub-02/mask"
tmap_dir = "/Users/carricarte/PhD/Debugging/vaso/sub-02/analysis/"
bmap_dir = "/Users/carricarte/PhD/Debugging/vaso/sub-02/analysis/"
tmap_bold_dir = "/Users/carricarte/PhD/Debugging/vaso/sub-02/analysis/univariate/imagery/"
# areas = ["FFA", "EVC"]
areas = ["FFA", "PPA", "EVC"]
analysis_mode = ["raw", "beta"]
# v = 99
v = 199

fnifti = []
[fnifti.append(join(input_dir, f)) for f in listdir(input_dir) if "corr" in f and "sm" not in f and
 f.endswith(".nii") and "._" not in f and "cr_" not in f and "raw" not in f and "mean" not in f and "loc" not in f]
fnifti.sort()

index = [0]
for nifti in fnifti[:-1]:
    print(nifti)
    data = image_stats.load_nifti(nifti)
    index.append(index[len(index) - 1] + data.shape[3])

ref_vol = nb.load(fnifti[0])
ori_shape = ref_vol.shape
features = image_stats.time_series_to_mat(fnifti[0])
feature_matrix = np.zeros(
    shape=(len(features), features.shape[1] * (len(fnifti))))  # if there are different repetitions per run we recommend
# to increase plus one (len(x) + 1)
feature_matrix[:, 0:features.shape[1]] = features

last_index = np.array([0])

for i, nifti in enumerate(fnifti[1:]):
    features = image_stats.time_series_to_mat(nifti)
    feature_matrix[:, index[i + 1]: index[i + 1] + features.shape[1]] \
        = features

feature_matrix = feature_matrix.T

scaler = StandardScaler().fit(feature_matrix)
X = scaler.transform(feature_matrix)

X = feature_matrix

# initialize some variables
svc = SVC(C=1)
param_grid = dict(C=[1])
grid = GridSearchCV(svc, param_grid, cv=4, scoring='accuracy')
# param_grid = dict(C=c)
TR = 4.826
modality = ['perception', 'imagery']  # imagery or perception
category = ['face', 'place']
trial_x_run = 4
times = [14]

# trial_type = ["img_face", "seen_face"]
trial_type = ["img_face", "seen_face", "img_place", "seen_place"]
# trial_type = ["Faces", "Places"]

# load tables with the onset of the events
tsvfiles = []
[tsvfiles.append(join(tsv_dir, t)) for t in listdir(tsv_dir) if "glm" in t
 and "._" not in t and "loc" not in t and t.endswith(".tsv")]
tsvfiles.sort()

vector = np.vectorize(ceil)

acc = []

a, b = [0, 1]
c, d = [2, 3]
color_list = ['steelblue', 'steelblue', '#ff7f0e', '#ff7f0e']
alpha_list = [0.3, 1, 0.3, 1]
# label_list = ['imagery', 'perception']
label_list = trial_type
# posX = [- 0.5, 0.5]
posX = [-1.5, - 0.5, 0.5, 1.5]
# conds = [trial_type[a], trial_type[b]]
conds = [trial_type[a], trial_type[b], trial_type[c], trial_type[d]]
# cond1_index, cond2_index = stats.beh_index(tsvfiles, TR, index, conds)
cond1_index, cond2_index,  cond3_index, cond4_index = stats.beh_index(tsvfiles, TR, index, conds)
# cond1_index, cond2_index = stats.beh_index(tsvfiles, TR, index, conds)
cond1_index.sort()
cond2_index.sort()
cond3_index.sort()
cond4_index.sort()

avg_cond = []
for c in [cond1_index, cond2_index, cond3_index, cond4_index]:
# for c in [cond1_index, cond2_index]:

    # get average for each time windows for each condition
    sum_X = np.zeros(shape=(9, X.shape[1]))
    total = 0
    for j in c:
        if j > 3 and j < 545:
            total = total + 1
            t1 = j - 3
            t2 = j + 6
            sum_X = sum_X + X[t1:t2]
            avg = sum_X / total

    avg_cond.append(avg)

new_shape = np.array(ori_shape)
new_shape[3] = 9
for i, vol in enumerate(avg_cond):
    vol = vol.T
    vol = vol.reshape(tuple(new_shape))
    new_vol = new_img_like(ref_vol, vol)
    new_vol.to_filename(join(output_dir, 'avg_{}'.format(conds[i])))

mean_beta = []

for m in analysis_mode:

    for area in areas:

        host = host_subplot(111)
        host.set_xlabel("TRs")
        host.set_ylabel("Normalized signal")

        if area != "EVC":

            if area == "FFA":
                f = "spmT_0006.nii"
                ff = "spmT_0001.nii"
            elif area == "PPA":
                f = "spmT_0010.nii"
                # ff = "spmT_0005.nii"
                ff = "spmT_0001.nii"

            mask_file = join(mask_dir, "mask.nii")
            hemisphere_mask = join(mask_dir, "right_mask.nii")
            manual_mask = np.logical_and(image_stats.load_nifti(mask_file),
                                         image_stats.load_nifti(hemisphere_mask))
            # manual_mask = mask_file
            t_map = join(tmap_dir, f)
            # t_map = join(tmap_bold_dir, ff)
            # tval = image_stats.img_mask(t_map, manual_mask,
            #                             np.logical_not(np.isnan(index_img(fnifti[0], 0).get_fdata())))
            tval = image_stats.img_mask(t_map, manual_mask)
            tval.sort()
            tval = tval[::-1]
            t = tval[v]
            loc_mask = image_stats.threshold_mask(t_map, t, manual_mask)
            # loc_mask = image_stats.threshold_mask(t_map, t, manual_mask, fnifti[0])
            # ff = "spmT_0001.nii"  # for subject 2 the contrast is "spmT_0001.nii"
            t_map = join(tmap_bold_dir, ff)

            # tval = image_stats.img_mask(t_map, manual_mask, loc_mask,
            #                             np.logical_not(np.isnan(index_img(fnifti[0], 0).get_fdata())))

            tval = image_stats.img_mask(t_map, manual_mask, loc_mask)

            tval.sort()
            tval = tval[::-1]
            t = tval[149]
            # bold_mask = image_stats.threshold_mask(t_map, t, np.logical_and(image_stats.load_nifti(manual_mask),
            #                                                                 loc_mask))
            bold_mask = image_stats.threshold_mask(t_map, t, np.logical_and(manual_mask,
                                                                            loc_mask))
            mask_ = np.logical_and(loc_mask, bold_mask)

        elif area == "EVC":
            mask_file = join(mask_dir, "early_visual_cortex.nii")
            mask_ = np.array(image_stats.load_nifti(mask_file)).astype(bool)

        for i in range(0, len(conds)):

            bmap = join(bmap_dir, conds[i] + ".nii")

            if m == "raw":

                maski = mask_.reshape(-1)
                avg = avg_cond[i]
                mean_avg_time = np.mean(avg[:, maski], 1)
                p1, = host.plot(np.arange(-3, 6), mean_avg_time, label=conds[i], color=color_list[i], alpha = alpha_list[i])
                host.yaxis.get_label().set_color(p1.get_color())

            elif m == "beta":

                roi_betas = image_stats.img_mask(bmap, mask_)
                print(sum(sum(sum(mask_))))
                mean_beta.append(np.mean(roi_betas))

        if m == "raw":
            # leg = plt.legend()
            plt.savefig(join(output_dir, "univariate_{}".format(area)))
            # del plt
            # plt.show()
            plt.clf()

if m == "beta":
    all_mean = np.vstack((mean_beta[0:4], mean_beta[4:8], mean_beta[8:12])).T
    # all_mean = np.vstack((mean_beta[0:2], mean_beta[2:4], mean_beta[4:6])).T
    labels = ['FFA', 'PPA', 'EVC']
    # labels = ['FFA', 'EVC']
    x = np.array([1, 4, 7])  # the label locations
    # x = np.array([1, 3])  # the label locations
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots()
    # ax.bar(x - 1.5 * width, all_mean[0], width, label='img_face', color='steelblue', alpha=0.3)
    for i in np.arange(len(all_mean)):
        ax.bar(x + posX[i] * width, all_mean[i], width, label=label_list[i],
               color=color_list[i], alpha = alpha_list[i])
    # plt.ylim(0, 1)
    ax.set_ylabel('mean beta')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.savefig(join(output_dir, "beta_plot_right"))
    plt.clf()
    print("hello world")











