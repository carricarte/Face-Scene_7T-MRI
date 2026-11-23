from os.path import join
import sys
from os.path import split
import numpy as np
from nilearn.image import load_img, new_img_like
from sklearn import preprocessing

input_dir = sys.argv[1]

pathname, filename = split(input_dir)

nifti = np.array(load_img(input_dir).get_fdata())
ori_shape = nifti.shape
features = nifti.reshape(-1, nifti.shape[-1])

# scale the data mean 0 and sdv 1
scaler = preprocessing.StandardScaler().fit(features)
std_features = scaler.transform(features) * 100
std_nifti = std_features.reshape(ori_shape)

new_nifti = new_img_like(load_img(input_dir), std_nifti)
new_nifti.to_filename(join(pathname, "std_" + filename))
