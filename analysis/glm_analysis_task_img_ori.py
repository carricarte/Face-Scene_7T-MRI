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
epi_mask = sys.argv[7]

spm.SPMCommand.set_mlab_paths(paths=spm_dir);
print(spm.SPMCommand().version)


if "01" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
elif"02" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "03" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif"04" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
elif "05" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
elif "06" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "07" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
elif "08" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "09" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "10" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "11" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "12" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "13" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "14" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "15" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "16" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "17" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
elif "18" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# TR of functional images
TR = 3
#with open('/Users/carricarte/PhD/Others/sub-02/func/sub-02_task-imagery_run-01_bold.json', 'rt') as fp:
#    task_info = json.load(fp)
#TR = task_info['RepetitionTime']

# Condition names
condition_names = ['seen_merkel', 'seen_obama', 'seen_berin', 'seen_paris', 'img_merkel', 'img_obama', 'img_berlin', 'img_paris', 'baseline']

# Contrasts
cont01 = ['average',                                           'T', condition_names, [1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3.]]
cont02 = ['visual_stimulation > baseline',                     'T', condition_names, [1, 1, 1, 1, 0, 0, 0, 0, -1]]
cont03 = ['imagining > baseline',                              'T', condition_names, [0, 0, 0, 0, 1, 1, 1, 1, -1]]
cont04 = ['faces > places',                                    'T', condition_names, [1, 1, -1, -1, 1, 1, -1, -1, 0]]
cont05 = ['places > faces',                                    'T', condition_names, [-1, -1, 1, 1, -1, -1, 1, 1, 0]]
cont06 = ['visual > imaging',                                  'T', condition_names, [1, 1, 1, 1, -1, -1, -1, -1, 0]]
cont07 = ['imaging > visual',                                  'T', condition_names, [-1, -1, -1, -1, 1, 1, 1, 1, 0]]
cont08 = ['merkel > obama',                                    'T', condition_names, [1, -1, 0, 0, 1, -1, 0, 0, 0]]
cont09 = ['obama > merkel',                                    'T', condition_names, [-1, 1, 0, 0, -1, 1, 0, 0, 0]]
cont10 = ['berlin > paris',                                    'T', condition_names, [0, 0, 1, -1, 0, 0, 1, -1, 0]]
cont11 = ['paris > berlin',                                    'T', condition_names, [0, 0, -1, 1, 0, 0, -1, 1, 0]]
cont12 = ['seen_faces > imagined_faces',                       'T', condition_names, [1, 1, 0, 0, -1, -1, 0, 0, 0]]
cont13 = ['seen_places > imagined_places',                     'T', condition_names, [0, 0, 1, 1, 0, 0, -1, -1, 0]]
cont14 = ['imagined_faces > seen_faces',                       'T', condition_names, [-1, -1, 0, 0, 1, 1, 0, 0, 0]]
cont15 = ['imagined_places > seen_places',                     'T', condition_names, [0, 0, -1, -1, 0, 0, 1, 1, 0]]
cont16 = ['seen_faces > seen_places',                          'T', condition_names, [1, 1, -1, -1, 0, 0, 0, 0, 0]]
cont17 = ['seen_places > seen_faces',                          'T', condition_names, [-1, -1, 1, 1, 0, 0, 0, 0, 0]]
cont18 = ['imagined_faces > imagined_places',                  'T', condition_names, [0, 0, 0, 0, 1, 1, -1, -1, 0]]
cont19 = ['imagined_places > imagined_faces',                  'T', condition_names, [0, 0, 0, 0, -1, -1, 1, 1, 0]]

contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08, cont09, cont10, cont11, cont12, cont13, cont14, cont15, cont16, cont17, cont18, cont19]

glm_files = []
[glm_files.append(_) for _ in listdir(glm_dir) if _.endswith(".tsv") and "loc" not in _ and "._" not in _]
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
scans.inputs.template = 'sm_cr_rsub-%s_task-img_run-%02d_bold.nii'
scans.inputs.sort_filelist = True
scans.inputs.subject_id = subject_id
scans.inputs.run = run_list

mc_param = Node(DataGrabber(infields=['subject_id', 'run'], outfields=['param']), name="mc_param")
mc_param.inputs.base_directory = input_dir
mc_param.inputs.template = 'rp_sub-%s_task-img_run-%02d_bold.txt'
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
                                 # mask_image=epi_mask,
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
            (level1conest, datasink, [("spm_mat_file", "analysis.univariate"),
                                      ("spmT_images", "analysis.univariate.@T"),
                                      ("con_images", "analysis.univariate.@con"),
                                      ("spmF_images", "analysis.univariate.@F"),
                                      ("ess_images", "analysis.univariate.@ess"),
                                      ("residual_image", "analysis.univariate.@eres"),
                                      ]),
            ])
wf.run()