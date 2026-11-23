from scipy import ndimage
from scipy.spatial import distance
from os.path import join
import nibabel as nb
import numpy as np
import pandas as pd
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

right_FFA = join(input_dir, "mask_right_FFA.nii")
right_PPA = join(input_dir, "mask_right_PPA.nii")
left_FFA = join(input_dir, "mask_left_FFA.nii")
left_PPA = join(input_dir, "mask_left_PPA.nii")

c_left_FFA = np.round(ndimage.center_of_mass(np.array(nb.load(left_FFA).get_fdata()))).astype(int)
c_right_FFA = np.round(ndimage.center_of_mass(np.array(nb.load(right_FFA).get_fdata()))).astype(int)
c_left_PPA = np.round(ndimage.center_of_mass(np.array(nb.load(left_PPA).get_fdata()))).astype(int)
c_right_PPA = np.round(ndimage.center_of_mass(np.array(nb.load(right_PPA).get_fdata()))).astype(int)

left_dist = distance.euclidean(c_left_FFA, c_left_PPA)
right_dist = distance.euclidean(c_right_FFA, c_right_PPA)
dist = pd.DataFrame(columns=["left", "right"])
dist.loc[len(dist)] = [left_dist, right_dist]
dist.to_csv(join(output_dir, 'dist_FFA-PPA.tsv'), sep='\t', index=True)
