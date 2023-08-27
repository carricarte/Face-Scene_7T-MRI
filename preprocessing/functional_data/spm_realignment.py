from nipype.interfaces.spm import Realign
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.freesurfer import MRIConvert
from os.path import join
from os.path import splitext
from os import rename, listdir, remove
import sys
from nipype.interfaces import spm

input_dir = sys.argv[1]
output_dir = sys.argv[2]
spm_dir = sys.argv[3]
matlab_dir = sys.argv[4]

spm.SPMCommand.set_mlab_paths(paths=spm_dir)

nifti_list = []
[nifti_list.append(_) for _ in listdir(input_dir) if _.endswith(".gz") and "._" not in _ and "Warp" not in _]
nifti_list.sort(reverse=True)

for nifti_file in nifti_list:

    filename, extension = splitext(nifti_file)

    mc = MRIConvert()
    mc.inputs.in_file = join(input_dir, nifti_file)
    mc.inputs.out_file = join(output_dir, filename)
    mc.inputs.out_type = 'nii'
    mc.run()

# sequence = ["vaso", "bold"]
sequence = ["bold"]

del_epi = []
[del_epi.append(join(output_dir, _)) for _ in listdir(output_dir) if _.endswith(".nii") and "._" not in _
 and "sub" in _ and "wcr_rsub" not in _ and "cr_rsub" not in _ and "rsub" in _
 and "mean" not in _ and "loc" in _ and "corr" not in _ and "con" not in _ and "cr_2" not in _ and "bc" not in _
 and "tr" not in _]
for epi in del_epi:
    remove(epi)


for s in sequence:
    epi_list = []
    [epi_list.append(join(output_dir, _)) for _ in listdir(output_dir) if _.endswith(".nii") and "._" not in _
     and "sub" in _ and "wcr_rsub" not in _ and "cr_rsub" not in _ and "rsub" not in _
     and "mean" not in _ and s in _ and "loc" in _]
    epi_list.sort()

    print("initializing list")
    for epi in epi_list:
        print(epi)

    # Estimate and reslice - SPM12
    ra = Realign()
    ra.inputs.jobtype = 'estwrite'
    ra.inputs.quality = 1
    ra.inputs.fwhm = 2
    ra.inputs.separation = 2
    ra.inputs.register_to_mean = True
    ra.inputs.interp = 4
    ra.inputs.write_interp = 4
    ra.inputs.in_files = epi_list
    ra.run()
