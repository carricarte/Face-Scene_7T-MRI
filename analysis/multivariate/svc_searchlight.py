import sys
import numpy as np
from nilearn.decoding import SearchLight
from sklearn import svm
import random
from os.path import join, exists
from nilearn.image import new_img_like, get_data
from sklearn.model_selection import KFold
from stats import nifti_list
import pandas as pd

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_img = sys.argv[3]
iter = sys.argv[4]
script_dir = sys.argv[5]

# input_dir = "/Users/carricarte/PhD/Debugging/sub-03/analysis/imagery/beta/run_all"
# output_dir = "/Users/carricarte/PhD/Debugging/sub-03/analysis/imagery/beta/run_all"
# mask_img = "/Users/carricarte/PhD/Debugging/sub-03/analysis/imagery/beta/mask_meanepi_normalized.nii"
# anat_img = ""

modality = ['perception', 'imagery']
category = ['face', 'place']
c1, c2 = ['face', 'place']

# mask = get_data(mask_img)
# mask[mask == 1] = 0
# mask[mask != 0] = 1
# mask_img = new_img_like(mask_img, mask)

for m in modality:

    X_cond1 = nifti_list(input_dir, m, c1)
    X_cond2 = nifti_list(input_dir, m, c2)

    X = np.empty((X_cond1.size + X_cond2.size,), dtype=X_cond1.dtype)
    X[::2] = X_cond1
    X[1::2] = X_cond2

    y = np.tile([1, 2], int(len(X) / 2))

    searchlight = SearchLight(mask_img=mask_img,
                              radius=5, scoring='accuracy', estimator=svm.SVC(C=1),
                              cv=KFold(n_splits=2),
                              verbose=1,
                              n_jobs=-1,
                              )

    # random.shuffle(y)
    searchlight.fit(X, y)
    score_vector = list(np.array(searchlight.scores_).reshape(-1))

    filename = "searchlight_{}_score_null_{}.tsv".format(m, iter)

    # if not exists(join(output_dir, filename)):
    df = pd.DataFrame(columns=list(np.arange(len(score_vector))))
    # df.to_csv(join(output_dir, filename), index=False)

    # df = pd.read_csv(join(output_dir, filename))

    df.loc[len(df)] = score_vector
    df.to_csv(join(output_dir, filename), index=False)

    # searchlight_img = new_img_like(anat_img, searchlight.scores_)
    # searchlight_img.to_filename(join(output_dir, 'searchlight_{}.nii'.format(m)))
