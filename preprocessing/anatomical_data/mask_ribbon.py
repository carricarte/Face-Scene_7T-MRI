import nibabel as nb
from nilearn.image import new_img_like
import numpy as np
from os.path import split, join
import sys

midGM = sys.argv[1]
ribbon = sys.argv[2]
prefix = sys.argv[3]
hem_mask = sys.argv[4]
manual_mask = sys.argv[5]

ribbonpath, ribbonname = split(ribbon)
midGMpath, midGMname = split(midGM)

ribbon_vol = nb.load(ribbon).get_fdata()
midGM_vol = np.array(nb.load(midGM).get_fdata()).astype(bool)
hem_mask_vol = np.array(nb.load(hem_mask).get_fdata()).astype(bool)
manual_mask_vol = np.array(nb.load(manual_mask).get_fdata()).astype(bool)

mask_vol = np.logical_and(hem_mask_vol, manual_mask_vol)
midGM_sub_mask = np.logical_and(midGM_vol, mask_vol)
print("masking ribbon")
ribbon_vol[mask_vol == 0] = 0
# print(np.unique(ribbon_vol))
midGM_vol[midGM_sub_mask == 0] = 0
# print(np.unique(midGM_vol))

ribbon_masked = new_img_like(ribbon, ribbon_vol)
midGM_masked = new_img_like(manual_mask, midGM_vol)

ribbon_masked.to_filename(join(ribbonpath, prefix + ribbonname))
midGM_masked.to_filename(join(midGMpath, prefix + midGMname))
