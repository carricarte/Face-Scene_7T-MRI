import nipype.interfaces.spm as spm
from os import listdir
from os.path import join
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]
working_dir = sys.argv[3]
spm_dir = sys.argv[4]

spm.SPMCommand.set_mlab_paths(paths=spm_dir);
print(spm.SPMCommand().version)

nifti_files = []
[nifti_files.append(join(output_dir, _)) for _ in listdir(output_dir) if "cr_rsub" in _ and "sm_" not in _ and "._" not in _]
nifti_files.sort()

norm12 = spm.Normalize12()
norm12.inputs.image_to_align = input_dir
# norm12.inputs.deformation_file = working_dir
norm12.inputs.apply_to_files = nifti_files
norm12.run()