from nipype.interfaces import spm
from nipype.interfaces.freesurfer import MRIConvert
from nibabel import Nifti1Image
from nibabel import load, save
import sys
from os.path import join
from os import rename

input_dir = sys.argv[1]
output_dir = sys.argv[2]
spm_dir = sys.argv[3]
file = "orig.nii"

spm.SPMCommand.set_mlab_paths(paths=spm_dir);
print(spm.SPMCommand().version)

mc = MRIConvert()
mc.inputs.in_file = join(input_dir, "orig.mgz")
mc.inputs.out_file = join(output_dir, file)
mc.inputs.out_type = 'nii'
mc.run()

target_img = join(output_dir, file)

seg = spm.NewSegment()
seg.inputs.channel_files = target_img
tissue1 = ((join(spm_dir, 'tpm/TPM.nii'), 1), 2, (True,False), (False, False))
tissue2 = ((join(spm_dir, 'tpm/TPM.nii'), 2), 2, (True,False), (False, False))
tissue3 = ((join(spm_dir, 'tpm/TPM.nii'), 3), 2, (True,False), (False, False))
tissue4 = ((join(spm_dir, 'tpm/TPM.nii'), 4), 2, (False,False), (False, False))
tissue5 = ((join(spm_dir, 'tpm/TPM.nii'), 5), 2, (False,False), (False, False))
tissue6 = ((join(spm_dir, 'tpm/TPM.nii'), 6), 2, (False,False), (False, False))
seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
seg.inputs.write_deformation_fields = [False, True]
seg.run()

tissue1 = load(join(output_dir, "c1" + file))
tissue2 = load(join(output_dir, "c2" + file))

gm = tissue1.get_fdata()
wm = tissue2.get_fdata()

brain_mask = gm + wm

anat_vol = load(target_img)
masked_brain = anat_vol.dataobj*brain_mask

stripped_brain = Nifti1Image(masked_brain, anat_vol.affine, anat_vol.header)
save(stripped_brain, join(output_dir, "brainmask_spm.nii"))

# rename(join(input_dir, "brainmask.mgz"), join(output_dir, "brainmask_freesurfer.mgz"))

mc.inputs.in_file = join(output_dir, "brainmask_spm.nii")
mc.inputs.out_file = join(input_dir, "brainmask.mgz")
mc.inputs.out_type = 'mgz'
mc.run()

