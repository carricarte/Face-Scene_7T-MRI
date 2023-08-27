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
<<<<<<< HEAD
epi_mask = sys.argv[5]
spm_dir = sys.argv[6]
matlab_dir = sys.argv[7]
=======
spm_dir = sys.argv[5]
matlab_dir = sys.argv[6]
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302

spm.SPMCommand.set_mlab_paths(paths=spm_dir);
print(spm.SPMCommand().version)

if "01" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
elif "02" == subject_id:
    # run_list = [1, 2, 3, 4, 5, 6, 7, 8]
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif "03" == subject_id:
    run_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # run_list = [1, 2, 3, 4, 5, 6, 7]
elif "04" == subject_id:
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

# Condition names
condition_names = ['seen_merkel', 'seen_obama', 'seen_berin', 'seen_paris', 'img_merkel', 'img_obama', 'img_berlin', 'img_paris', 'baseline']

# condition_names = ['seen_face', 'img_face', 'baseline']
# condition_names = ['seen_face', 'img_face', 'seen_place', 'img_place', 'baseline']

# Contrasts
# cont01 = ['visual_stimulation > baseline',                     'T', condition_names, [1, 0, -1]]
# cont02 = ['imagining > baseline',                              'T', condition_names, [0, 1, -1]]
# cont03 = ['seen_faces > imagined_faces',                       'T', condition_names, [1, -1, 0]]
# cont04 = ['stim > baseline',                                   'T', condition_names, [1, 1, -1]]

# cont01 = ['stimulation > baseline', 'T', condition_names, [1, 1, 1, 1, -1]]
# cont02 = ['visual > baseline', 'T', condition_names, [1, 0, 1, 0, -1]]
# cont03 = ['imagining > baseline', 'T', condition_names, [0, 1, 0, 1, -1]]
# cont04 = ['faces > places', 'T', condition_names, [1, 1, -1, -1, 0]]
# cont05 = ['places > faces', 'T', condition_names, [-1, -1, 1, 1, 0]]

cont01 = ['average',                                           'T', condition_names, [1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3.]]
cont02 = ['visual_stimulation > baseline',                     'T', condition_names, [1, 1, 1, 1, 0, 0, 0, 0, -1]]
cont03 = ['imagining > baseline',                              'T', condition_names, [0, 0, 0, 0, 1, 1, 1, 1, -1]]
cont04 = ['faces > places',                                    'T', condition_names, [1, 1, -1, -1, 1, 1, -1, -1, 0]]
cont05 = ['places > faces',                                    'T', condition_names, [-1, -1, 1, 1, -1, -1, 1, 1, 0]]
cont06 = ['seen_faces > seen_places',                          'T', condition_names, [1, 1, -1, -1, 0, 0, 0, 0, 0]]
cont07 = ['seen_places > seen_faces',                          'T', condition_names, [-1, -1, 1, 1, 0, 0, 0, 0, 0]]
cont08 = ['imagined_faces > imagined_places',                  'T', condition_names, [0, 0, 0, 0, 1, 1, -1, -1, 0]]
cont09 = ['imagined_places > imagined_faces',                  'T', condition_names, [0, 0, 0, 0, -1, -1, 1, 1, 0]]
<<<<<<< HEAD
cont10 = ['seen_faces > baseline',                          'T', condition_names, [1, 1, 0, 0, 0, 0, 0, 0, -1]]
=======
cont10 = ['seen_faces > seen_baseline',                          'T', condition_names, [1, 1, 0, 0, 0, 0, 0, 0, -1]]
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
cont11 = ['seen_places > baseline',                          'T', condition_names, [0, 0, 1, 1, 0, 0, 0, 0, -1]]
cont12 = ['imagined_faces > baseline',                  'T', condition_names, [0, 0, 0, 0, 1, 1, 0, 0, -1]]
cont13 = ['imagined_places > baseline',                  'T', condition_names, [0, 0, 0, 0, 0, 0, 1, 1, -1]]

contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08, cont09, cont10, cont11, cont12, cont13]
# contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08]
# contrast_list = [cont01, cont02, cont03, cont04, cont05]

glm_files = []
[glm_files.append(_) for _ in listdir(glm_dir) if _.endswith(".tsv") and "loc" not in _ and "._" not in _]
glm_files.sort()

subject_info = []

for glm_file in glm_files:

    trialinfo = pd.read_table(join(glm_dir, glm_file))
    # trialinfo = trialinfo.replace('img_paris', 'img_place')
    # trialinfo = trialinfo.replace('img_berlin', 'img_place')
    # trialinfo = trialinfo.replace('seen_paris', 'seen_place')
    # trialinfo = trialinfo.replace('seen_berin', 'seen_place')
    # trialinfo = trialinfo.replace('img_merkel', 'img_face')
    # trialinfo = trialinfo.replace('img_obama', 'img_face')
    # trialinfo = trialinfo.replace('seen_merkel', 'seen_face')
    # trialinfo = trialinfo.replace('seen_obama', 'seen_face')
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
<<<<<<< HEAD
# scans.inputs.template = 'cr_rsub-%s_task-img_run-%02d_bold.nii'
scans.inputs.template = 'sm_cr_rsub-%s_task-img_run-%02d_bold.nii'
=======
scans.inputs.template = 'cr_rsub-%s_task-img_run-%02d_bold.nii'
# scans.inputs.template = 'sm_rsub-%s_task-imagery_run-%02d_bold.nii'
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
scans.inputs.sort_filelist = True
scans.inputs.subject_id = subject_id
scans.inputs.run = run_list

mc_param = Node(DataGrabber(infields=['subject_id', 'run'], outfields=['param']), name="mc_param")
mc_param.inputs.base_directory = input_dir
mc_param.inputs.template = 'rp_sub-%s_task-img_run-%02d_bold.txt'
# mc_param.inputs.template = 'rp_sub-%s_task-imagery_run-%02d_bold.txt'
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
<<<<<<< HEAD
                                 # mask_image=epi_mask,
                                 # mask_threshold='-Inf',
=======
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
                                 model_serial_correlations='FAST'), name='level1design')

# EstimateModel - estimate the parameters of the model
level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}), name='level1estimate')

# EstimateContrast - estimates contrasts
level1conest = Node(EstimateContrast(), name="level1conest")
level1conest.inputs.contrasts = contrast_list

datasink = Node(DataSink(base_directory=output_dir), name='datasink')

wf = Workflow(name='nipype', base_dir=output_dir)

# previous path "analysis.univariate.non_sm_data.non_nm_data.imagery.category"
wf.connect([
    (scans, modelspec, [("func", "functional_runs")]),
    (mc_param, modelspec, [("param", "realignment_parameters")]),
    (modelspec, level1design, [("session_info", "session_info")]),
    (level1design, level1estimate, [("spm_mat_file", "spm_mat_file")]),
    (level1estimate, level1conest, [("spm_mat_file", "spm_mat_file"),
                                    ("beta_images", "beta_images"),
                                    ("residual_image", "residual_image")]),
<<<<<<< HEAD
    (level1conest, datasink, [("spm_mat_file", "analysis.imagery"),
                              ("spmT_images", "analysis.imagery.@T"),
                              ("con_images", "analysis.imagery.@con"),
                              ("spmF_images", "analysis.imagery.@F"),
                              ("ess_images", "analysis.imagery.@ess"),
                              ]),
    (level1estimate, datasink, [("spm_mat_file", 'analysis.imagery.beta'),
                                ("beta_images", 'analysis.imagery.beta.@B'),
=======
    (level1conest, datasink, [("spm_mat_file", "analysis.univariate.no_smooth"),
                              ("spmT_images", "analysis.univariate.no_smooth.@T"),
                              ("con_images", "analysis.univariate.no_smooth.@con"),
                              ("spmF_images", "analysis.univariate.no_smooth.@F"),
                              ("ess_images", "analysis.univariate.no_smooth.@ess"),
                              ]),
    (level1estimate, datasink, [("spm_mat_file", 'analysis.univariate.no_smooth.beta'),
                                ("beta_images", 'analysis.univariate.no_smooth.beta.@B'),
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
                                ]),
])
wf.run()
