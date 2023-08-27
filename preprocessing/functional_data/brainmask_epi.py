from os.path import split, join
import nibabel as nb
import subprocess
import sys
import numpy as np

<<<<<<< HEAD
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
=======
c1 = sys.argv[1]
c2 = sys.argv[2]
output_dir = sys.argv[3]
meanepi = sys.argv[4]  # change path to mean epi if coregistering
script_dir = sys.argv[5]
affine_orig2epi = sys.argv[6]

# c1_obj = nb.load(c1)
# c2_obj = nb.load(c2)
#
# gm = c1_obj.get_fdata()
# wm = c2_obj.get_fdata()
#
# brainmask = gm + wm
brainmask_img = join(output_dir, 'brain_mask.nii')
# masked_epi = nb.Nifti1Image(brainmask, c1_obj.affine, c1_obj.header)
# nb.save(masked_epi, brainmask_img)

path, file = split(meanepi)
output_img = join(path, "epi_brain_mask.nii")

# not all co-registration were calculated with the orig image (just the ones that needed brain masking)
# I was just lazy to change the name "affine_orig2epi" to a generic name
# this is a rough transformation.

subprocess.call(
    ['sbatch', join(script_dir, "ants_apply_transform"), meanepi, brainmask_img, affine_orig2epi,
     output_img])

epi_obj = nb.load(meanepi)
try:
    mask_obj = nb.load(output_img)
    epi = epi_obj.get_fdata()
    mask = mask_obj.get_fdata()
    mask[np.where(mask < 0.09)] = 0
    mask[np.where(mask != 0)] = 1
    # masked_epi = np.zeros(shape=epi.shape)
    masked_epi = epi*mask
    # masked_epi[np.where(mask != 0)] = epi[np.where(mask != 0)]
    masked_epi = nb.Nifti1Image(masked_epi, epi_obj.affine, epi_obj.header)
    nb.save(masked_epi, join(path, 'masked_' + file))
except:
    print("file not found, re-run")

# epi brainmask in orig space (after coregistration)
# brainmask[brainmask != 0] = 1
# epi = nb.load(meanepi)
# epi = epi.get_fdata()
# epi_brainmask = np.array(np.logical_and(brainmask, epi != 0)).astype(int)
# epi_brainmask_img = join(output_dir, 'epi_brain_mask_orig_space.nii')
# epi_mask = nb.Nifti1Image(epi_brainmask, c1_obj.affine, c1_obj.header)
# nb.save(epi_mask, epi_brainmask_img)

>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
