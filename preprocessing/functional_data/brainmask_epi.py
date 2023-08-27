from os.path import split, join
import nibabel as nb
import subprocess
import sys
import numpy as np

meanepi = sys.argv[1]  # change path to mean epi if coregistering
# meanepi = "/Users/carricarte/PhD/Debugging/bold/sub-16/func/registered_Warp.nii"  # change path to mean epi if coregistering
# mask = "/Users/carricarte/PhD/Debugging/bold/sub-16/mask/brainmask.nii"  # change path to mean epi if coregistering
mask = sys.argv[2]  # change path to mean epi if coregistering

path, file = split(meanepi)

epi_obj = nb.load(meanepi)
epi = epi_obj.get_fdata()
mask_obj = nb.load(mask)
mask = mask_obj.get_fdata()
try:
    # mask[np.where(mask < 0.09)] = 0
    # mask[np.where(mask != 0)] = 1
    # masked_epi = np.zeros(shape=epi.shape)
    masked_epi = epi*mask
    epi_brainmask = masked_epi > 0
    # masked_epi[np.where(mask != 0)] = epi[np.where(mask != 0)]
    epi_brainmask = nb.Nifti1Image(epi_brainmask, epi_obj.affine, epi_obj.header)
    nb.save(epi_brainmask, join(path, 'epimask.nii'))
except:
    print("file not found, re-run")

