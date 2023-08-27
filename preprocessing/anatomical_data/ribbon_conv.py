import nibabel as nib
import numpy as np
from os.path import join
import sys
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
import image_stats as st

input_dir = sys.argv[1]
output_dir = sys.argv[2]
# mask_dir = sys.argv[3]
#
# ribbon_lh = st.vol_to_vector(join(input_dir, "lh_ribbon.nii"))
# ribbon_rh = st.vol_to_vector(join(input_dir, "rh_ribbon.nii"))
# mask_left = st.vol_to_vector(join(mask_dir, "mask_left.nii"))
# mask_right = st.vol_to_vector(join(mask_dir, "mask_right.nii"))
# mask = st.vol_to_vector(join(mask_dir, "mask.nii"))
#
# reference_vol = nib.load(join(input_dir, "lh_ribbon.nii"))
# original_shape = reference_vol.shape
#
# mask_left = np.logical_and(mask, mask_left)
# mask_right = np.logical_and(mask, mask_right)
#
# rh_ribbon = np.array(ribbon_rh * mask_left).astype(int)
# lh_ribbon = np.array(ribbon_lh * mask_right).astype(int)
#
# rh_new_vol = nib.Nifti1Image(rh_ribbon.reshape(original_shape), reference_vol.affine, reference_vol.header)
# lh_new_vol = nib.Nifti1Image(lh_ribbon.reshape(original_shape), reference_vol.affine, reference_vol.header)
# nib.save(rh_new_vol, join(output_dir, "rh_ribbon.nii"))
# nib.save(lh_new_vol, join(output_dir, "lh_ribbon.nii"))
#
# print("saving output in: " + output_dir)

# ribbon = rh_ribbon
# ribbon[lh_ribbon != 0] = lh_ribbon[lh_ribbon != 0]

nii_1 = st.load_nifti(join(input_dir, "rh_ribbon_columns10000.nii"))
nii_2 = st.load_nifti(join(input_dir, "lh_ribbon_columns10000.nii"))

reference_vol = nib.load(join(input_dir, "rh_ribbon_columns10000.nii"))
original_shape = reference_vol.shape

nii_1[nii_1 != 0] = nii_1[nii_1 != 0] + 10000
nii_3 = nii_1
nii_3[nii_2 != 0] = nii_2[nii_2 != 0]

new_vol = nib.Nifti1Image(nii_3, reference_vol.affine, reference_vol.header)
nib.save(new_vol, join(output_dir, "ribbon_columns10000.nii"))
