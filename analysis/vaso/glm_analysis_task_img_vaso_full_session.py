from nipype.interfaces.spm import Level1Design, EstimateModel, EstimateContrast
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.io import DataSink, DataGrabber
from nipype import Workflow, Node
import pandas as pd
import sys
from os import listdir
from os.path import join
from nipype.interfaces import spm
from nipype.interfaces.base import Bunch
import numpy as np

input_dir = sys.argv[1]
output_dir = sys.argv[2]
glm_dir = sys.argv[3]
subject_id = sys.argv[4]
spm_dir = sys.argv[5]
matlab_dir = sys.argv[6]
epi_mask = sys.argv[7]

spm.SPMCommand.set_mlab_paths(paths=spm_dir)
print(spm.SPMCommand().version)

if "01" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
elif "02" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
elif "03" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "04" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8]
elif "05" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "06" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8]
elif "07" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "08" == subject_id:
    run_list = [1, 3, 7]  # perception run
    # run_list = [2, 4, 5, 6, 8]  # imagery run
    run_type = [1, 0, 1, 0, 0, 0, 1, 0]
    # run_type = [0, 1, 0, 1, 1, 1, 0, 1]
elif "09" == subject_id:
    run_list = [1, 3, 6]  # perception run
    # run_list = [2, 4, 5, 7, 8, 9]  # imagery run
    run_type = [1, 0, 1, 0, 0, 1, 0, 0, 0]
    # run_type = [0, 1, 0, 1, 1, 0, 1, 1, 1]


# TR of functional images
TR = 4.826
seq = "vaso"
# Condition names
# condition_names = ['seen_berlin', 'seen_paris', 'seen_pisa', 'img_berlin', 'img_paris', 'img_pisa', 'baseline']

# Contrasts
# cont01 = ['stimulation > baseline',                     'T', condition_names, [1, 1, 1, 1, 1, 1, -1]]
# cont02 = ['seen_places > baseline',                     'T', condition_names, [1, 1, 1, 0, 0, 0, -1]]
# cont03 = ['imagined_places > baseline',                 'T', condition_names, [0, 0, 0, 1, 1, 1, -1]]

condition_names = ['seen_place', 'img_place', 'baseline']

# Contrasts
cont01 = ['stimulation > baseline',                     'T', condition_names, [1, 1, -1]]
cont02 = ['seen_places > baseline',                     'T', condition_names, [1, 0, -1]]
cont03 = ['imagined_places > baseline',                 'T', condition_names, [0, 1, -1]]

contrast_list = [cont01, cont02, cont03]

glm_files = []
[glm_files.append(_) for _ in listdir(glm_dir) if _.endswith(".tsv") and "block" in _ and "._" not in _
 and "trial" not in _]
glm_files.sort()

if subject_id == "08" or subject_id == "09":

    condition_names = ["seen_place", 'baseline']

    # Contrasts
    cont01 = ['stimulation > baseline', 'T', condition_names, [1, -1]]

    contrast_list = [cont01]

    glms = []
    [glms.append(glm_files[e]) for e in np.nonzero(run_type)[0]]
    glm_files = glms

subject_info = []

for glm in glm_files:

    trialinfo = pd.read_table(join(glm_dir, glm))
    trialinfo.head()
    conditions = []
    onsets = []
    durations = []

    for group in trialinfo.groupby('trial_type'):
        conditions.append(group[0])
        onsets.append(group[1].onset.tolist())
        durations.append(group[1].duration.tolist())

    subject_info.append(
        Bunch(conditions=conditions,
              onsets=onsets,
              durations=durations,
              amplitudes=None,
              tmod=None,
              pmod=None,
              regressor_names=None,
              regressors=None
              )
    )

scans = Node(DataGrabber(infields=['subject_id', 'run'], outfields=['func']), name="scans")
scans.inputs.base_directory = input_dir
scans.inputs.template = 'corr_rsub-%s_task-img_run-%02d_vaso.nii'
# scans.inputs.template = 'intemp_rsub-%s_task-img_run-%02d_bold.nii'
scans.inputs.sort_filelist = True
scans.inputs.subject_id = subject_id
scans.inputs.run = run_list

mc_param = Node(DataGrabber(infields=['subject_id', 'run'], outfields=['param']), name="mc_param")
mc_param.inputs.base_directory = input_dir
mc_param.inputs.template = 'rp_sub-%s_task-img_run-%02d_{0}.txt'.format(seq)
mc_param.inputs.sort_filelist = True
mc_param.inputs.subject_id = subject_id
mc_param.inputs.run = run_list

# Node specification
# SpecifyModel - Generates SPM-specific Model
modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                 input_units='secs',
                                 output_units='secs',
                                 time_repetition=TR,
                                 high_pass_filter_cutoff=128), name="modelspec")

# Get Subject Info - get subject specific condition information
modelspec.inputs.subject_info = subject_info

# Level1Design - Generates an SPM design matrix
level1design = Node(Level1Design(bases={'gamma': {'length': 32, 'order': 1}},
                                 timing_units='secs',
                                 interscan_interval=TR,
                                 model_serial_correlations='FAST',
                                 mask_image=epi_mask,
                                 mask_threshold='-Inf'
                                 ), name='level1design')

# EstimateModel - estimate the parameters of the model
level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}), name='level1estimate')

# EstimateContrast - estimates contrasts
level1conest = Node(EstimateContrast(), name="level1conest")
level1conest.inputs.contrasts = contrast_list

datasink = Node(DataSink(base_directory=output_dir), name='datasink')

wf = Workflow(name='nipype', base_dir=output_dir)

wf.connect([
    (scans, modelspec, [("func", "functional_runs")]),
    (mc_param, modelspec, [("param", "realignment_parameters")]),
    (modelspec, level1design, [("session_info", "session_info")]),
    (level1design, level1estimate, [("spm_mat_file", "spm_mat_file")]),
    (level1estimate, level1conest, [("spm_mat_file", "spm_mat_file"),
                                    ("beta_images", "beta_images"),
                                    ("residual_image", "residual_image")]),
    (level1conest, datasink, [("spm_mat_file", "{0}.gamma".format(seq)),
                              ("spmT_images", "{0}.gamma.@T".format(seq)),
                              ("con_images", "{0}.gamma.@con".format(seq)),
                              ("spmF_images", "{0}.gamma.@F".format(seq)),
                              ("ess_images", "{0}.gamma.@ess".format(seq)),
                              ]),
    (level1estimate, datasink, [("spm_mat_file", '{0}.gamma.beta'.format(seq)),
                                ("beta_images", '{0}.gamma.beta.@B'.format(seq)),
                                ]),
])
wf.run()