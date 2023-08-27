from nipype.interfaces.fsl import TOPUP, ApplyTOPUP
from nipype.interfaces.io import DataSink
from nipype import Workflow, Node
import subprocess
from nilearn import image
from os.path import join
import pandas as pd
import json
from os import listdir, chdir
import sys
import numpy as np

nifti_dir = sys.argv[1]
fmap_dir = sys.argv[2]
json_file = sys.argv[3]
# script_dir = sys.argv[4]

# create the acq_params.txt file. IMPORTANT: IM USING THE JSON FILE FROM THE PRF MAPPING. IT MIGHT NOT BE THE
# ENCODING DIRECTION AS IN THE VASO RUNS

with open(json_file) as j_file:
    data = json.load(j_file)
    total_readout_time = data['TotalReadoutTime']
    encoding_direction = data['PhaseEncodingDirection']

if encoding_direction == "j":
    e_dir = [0, 1, 0, 0]
elif encoding_direction == "j-" or "-j":
    e_dir = [0, -1, 0, 0]

e_dir[3] = total_readout_time
e_dir = np.tile(e_dir, (10, 1))
e_dir[0:5, 1] = -1 * e_dir[0:5, 1]
params = pd.DataFrame(e_dir)
topup_file = join(fmap_dir, "acq_params.txt")
np.savetxt(topup_file, params.values, fmt='%1.6f')

# seq = ["vaso"]
seq = ["bold", "vaso"]
task = "task-img"
task = "task-img"
topup = Node(TOPUP(encoding_file=topup_file, output_type="NIFTI"), name="topup")
# applytopup = Node(ApplyTOPUP(encoding_file=topup_file, output_type="NIFTI"), name="applytopup")
# applytopup = ApplyTOPUP(encoding_file=topup_file, output_type="NIFTI")

from nipype.interfaces.fsl import ApplyTOPUP
fieldcorr_dir = join(fmap_dir, "field_corr")
for s in seq:

    nifti_list = []
    [nifti_list.append(join(nifti_dir, nifti)) for nifti in listdir(nifti_dir) if task in nifti and "._" not in nifti
     and s in nifti and nifti.endswith(".nii") and "rsub" not in nifti and "mean" not in nifti]
    nifti_list.sort()

    topup_list = []
    [topup_list.append(join(fmap_dir, nifti)) for nifti in listdir(fmap_dir) if task in nifti and "._" not in nifti
     and s in nifti and "dir-down" in nifti and nifti.endswith(".nii") and "rsub" not in nifti and "mean" not in nifti]
    topup_list.sort()

    # Check length of the two list
    if len(topup_list) == len(nifti_list):

        for i in range(0, len(nifti_list)):

            # Estimate TOPUP distortions
            niimg = image.load_img(nifti_list[i])
            fmap_niimg = image.load_img(topup_list[i])

            selected_volumes = image.index_img(niimg, slice(0, 5))
            combined_volumnes = image.concat_imgs([selected_volumes, fmap_niimg])
            comb_vol_file = join(fmap_dir, "cmb_niimg_run-{:02d}_{}.nii".format(i + 1, s))
            combined_volumnes.to_filename(comb_vol_file)
            topup.inputs.in_file = comb_vol_file
            run = i + 1

            datasink = Node(DataSink(base_directory=fmap_dir), name='datasink')
            wf = Workflow(name='nipype', base_dir=fmap_dir)

            wf.connect([
                (topup, datasink, [("out_corrected", 'field_corr'),
                                   ("out_enc_file", 'field_corr.@File'),
                                   ("out_field", 'field_corr.@Field'),
                                   ("out_fieldcoef", 'field_corr.@Coef'),
                                   ("out_logfile", 'field_corr.@Log'),
                                   ("out_movpar", 'field_corr.@Mov'),
                                   ]),
            ])

            # Apply distortion estimated TOPUP maps
            # applytopup.inputs.in_files = [nifti_list[i], topup_list[i]]
            # applytopup.inputs.in_topup_fieldcoef = join(fieldcorr_dir, "cmb_niimg_run-{:02d}_base_fieldcoef.nii".format(i + 1))
            # applytopup.inputs.in_topup_movpar = join(fieldcorr_dir, "cmb_niimg_run-{:02d}_base_movpar.txt".format(i + 1))
            # applytopup.run()

            # chdir(fieldcorr_dir)
            # subprocess.call(
            #     ['applytopup', '--imain={0}'.format(nifti_list[i]), '--inindex=1',
            #      '--datain={0}'.format(topup_file), '--topup=cmb_niimg_run-{:02d}_base'.format(i + 1),
            #      '--out=topup_corr_vaso_run-{:02d}'.format(i + 1), '--method=jac', '--interp=spline'])


            # wf.connect([
            #     (applytopup, datasink, [("out_corrected", 'field_corr')
            #                        ]),
            # ])
            #
            wf.run()

    else:
        for n in nifti_list:
            print(n)
        for t in topup_list:
            print(t)
