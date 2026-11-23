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

input_dir = sys.argv[1]
output_dir = sys.argv[2]
glm_dir = sys.argv[3]
subject_id = sys.argv[4]
spm_dir = sys.argv[5]
matlab_dir = sys.argv[6]

# input_dir = "/Users/carricarte/PhD/Debugging/bold/sub-10/online_localizer"
# output_dir = "/Users/carricarte/PhD/Debugging/bold/sub-10/online_localizer"
# glm_dir = "/Users/carricarte/PhD/Debugging/bold/sub-10/online_localizer"
# subject_id = "10"
# spm_dir = "/Users/carricarte/toolbox/spm12"
#matlab_dir = sys.argv[6]

spm.SPMCommand.set_mlab_paths(paths=spm_dir);
print(spm.SPMCommand().version)

run_list = 1

# TR of functional images
TR = 3
#with open('/Users/carricarte/PhD/Others/sub-02/func/sub-02_task-imagery_run-01_bold.json', 'rt') as fp:
#    task_info = json.load(fp)
#TR = task_info['RepetitionTime']

# Condition names
condition_names = ['Faces', 'Places', 'Objects', 'Scrambled', 'Baseline']

# Contrasts
cont01 = ['average',                               'T', condition_names, [1/3., 1/3., 1/3., 1/3., 1/3.]]
cont02 = ['visual_stimulation > baseline',         'T', condition_names, [1, 1, 1, 1, -1]]
cont03 = ['faces + places > baseline',             'T', condition_names, [1, 1, 0, 0, -1]]
cont04 = ['faces + places > scrambled',            'T', condition_names, [1, 1, 0, -1, 0]]
cont05 = ['faces + places > objects',              'T', condition_names, [1, 1, -1, 0, 0]]
cont06 = ['faces > places',                        'T', condition_names, [1, -1, 0, 0, 0]]
cont07 = ['faces > objects',                       'T', condition_names, [1, 0, -1, 0, 0]]
cont08 = ['faces > scrambled',                     'T', condition_names, [1, 0, 0, -1, 0]]
cont09 = ['faces > baseline',                      'T', condition_names, [1, 0, 0, 0, -1]]
cont10 = ['places > faces',                        'T', condition_names, [-1, 1, 0, 0, 0]]
cont11 = ['places > objects',                      'T', condition_names, [0, 1, -1, 0, 0]]
cont12 = ['places > scrambled',                    'T', condition_names, [0, 1, 0, -1, 0]]
cont13 = ['places > baseline',                     'T', condition_names, [0, 1, 0, 0, -1]]
cont14 = ['objects > scrambled',                   'T', condition_names, [0, 0, 1, -1, 0]]

contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08, cont09, cont10, cont11, cont12, cont13,
                 cont14]

glm_files = []
[glm_files.append(_) for _ in listdir(glm_dir) if _.endswith(".tsv") and "-loc" in _ and "._" not in _]
glm_files.sort()

subject_info = []

for glm_file in glm_files:

    trialinfo = pd.read_table(join(glm_dir,glm_file))
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

scans = Node(DataGrabber(infields=['subject_id', 'run'],  outfields=['func']), name="scans")
scans.inputs.base_directory = input_dir
scans.inputs.template = 'sm_cr_rsub-%s_task-loc_run-%02d_bold.nii'
# scans.inputs.template = 'rsub-%s_task-img_run-%02d_bold.nii'
scans.inputs.sort_filelist = True
scans.inputs.subject_id = subject_id
scans.inputs.run = run_list

mc_param = Node(DataGrabber(infields=['subject_id', 'run'], outfields=['param']), name="mc_param")
mc_param.inputs.base_directory = input_dir
mc_param.inputs.template = 'rp_sub-%s_task-loc_run-%02d_bold.txt'
mc_param.inputs.sort_filelist = True
mc_param.inputs.subject_id = subject_id
mc_param.inputs.run = run_list

#Node specification
# SpecifyModel - Generates SPM-specific Model
modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                 input_units='secs',
                                 output_units='secs',
                                 time_repetition=TR,
                                 high_pass_filter_cutoff=128), name="modelspec")

# Get Subject Info - get subject specific condition information
modelspec.inputs.subject_info = subject_info

# Level1Design - Generates an SPM design matrix
level1design = Node(Level1Design(bases={'hrf': {'derivs': [1, 0]}},
                                 timing_units='secs',
                                 interscan_interval=TR,
                                 model_serial_correlations='FAST'), name='level1design')

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
            (level1conest, datasink, [("spm_mat_file", "analysis.localizer"),
                                      ("spmT_images", "analysis.localizer.@T"),
                                      ("con_images", "analysis.localizer.@con"),
                                      ("spmF_images", "analysis.localizer.@F"),
                                      ("ess_images", "analysis.localizer.@ess"),
                                      ]),
            ])
wf.run()
