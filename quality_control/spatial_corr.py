import sys
import numpy as np
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from nilearn.image import mean_img

input_dir = sys.argv[1]
output_dir = sys.argv[2]

runs = []

# BOLD project
# [runs.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
#  and "rsub" in _ and "sm_cr_rsub" not in _ and "wcr_rsub" not in _
#  and "cr_rsub" not in _ and "mean" not in _
#  and "4D" not in _]

[runs.append(join(input_dir, _)) for _ in listdir(input_dir) if _.endswith(".nii") and "._" not in _
 and "rsub" in _ and "sm_cr_rsub" not in _ and "wcr_rsub" not in _
 and "cr_rsub" not in _ and "mean" not in _
 and "vaso" not in _]

runs.sort()

mean_r = np.array([])

for r in runs:
    run = mean_img(r).get_fdata()
    flatted_run = run.flatten()
    flatted_run = flatted_run[np.logical_not(np.isnan(flatted_run))]
    mean_r = np.vstack([mean_r, flatted_run]) if mean_r.size else flatted_run

corr_mat = np.tril(np.ma.corrcoef(mean_r))
print(corr_mat)

fig, ax = plt.subplots()
ax.matshow(corr_mat, cmap='seismic')
for (i, j), z in np.ndenumerate(corr_mat):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

plt.savefig(join(output_dir, "spatial_corr_matrix.png"))
print("saving: " + join(output_dir, "spatial_corr_matrix.png"))