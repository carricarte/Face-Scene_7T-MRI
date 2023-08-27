import sys
from os.path import split
import nibabel as nib
from os.path import join

nifti1_dir = sys.argv[1]
nifti2_dir = sys.argv[2]

# input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-12/analysis/lh_mean_beta_img_face_deveinDeconv.nii"
# output_dir = "/Users/carricarte/PhD/Debugging/bold/sub-12/analysis/rh_mean_beta_img_face_deveinDeconv.nii"

nii_1 = nib.load(nifti1_dir).get_fdata()
nii_2 = nib.load(nifti2_dir).get_fdata()

reference_vol = nib.load(nifti1_dir)
original_shape = reference_vol.shape

nii_1[nii_1 == 0] = nii_2[nii_1 == 0]

pathfile, namefile = split(nifti1_dir)

new_vol = nib.Nifti1Image(nii_1, reference_vol.affine, reference_vol.header)
<<<<<<< HEAD
nib.save(new_vol, join(pathfile, namefile[3::]))
=======
nib.save(new_vol, join(pathfile, "manual_" + namefile[3::]))
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
