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
from os.path import split

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

condition_names = ['seen_merkel', 'seen_obama', 'seen_berin', 'seen_paris', 'img_merkel', 'img_obama', 'img_berlin', 'img_paris', 'baseline']

cont01 = ['average',                                           'T', condition_names, [1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1/3.]]
cont02 = ['visual_stimulation > baseline',                     'T', condition_names, [1, 1, 1, 1, 0, 0, 0, 0, -1]]
cont03 = ['imagining > baseline',                              'T', condition_names, [0, 0, 0, 0, 1, 1, 1, 1, -1]]
cont04 = ['faces > places',                                    'T', condition_names, [1, 1, -1, -1, 1, 1, -1, -1, 0]]
cont05 = ['places > faces',                                    'T', condition_names, [-1, -1, 1, 1, -1, -1, 1, 1, 0]]
cont06 = ['seen_faces > seen_places',                          'T', condition_names, [1, 1, -1, -1, 0, 0, 0, 0, 0]]
cont07 = ['seen_places > seen_faces',                          'T', condition_names, [-1, -1, 1, 1, 0, 0, 0, 0, 0]]
cont08 = ['imagined_faces > imagined_places',                  'T', condition_names, [0, 0, 0, 0, 1, 1, -1, -1, 0]]
cont09 = ['imagined_places > imagined_faces',                  'T', condition_names, [0, 0, 0, 0, -1, -1, 1, 1, 0]]
cont10 = ['seen_faces > seen_baseline',                          'T', condition_names, [1, 1, 0, 0, 0, 0, 0, 0, -1]]
cont11 = ['seen_places > baseline',                          'T', condition_names, [0, 0, 1, 1, 0, 0, 0, 0, -1]]
cont12 = ['imagined_faces > baseline',                  'T', condition_names, [0, 0, 0, 0, 1, 1, 0, 0, -1]]
cont13 = ['imagined_places > baseline',                  'T', condition_names, [0, 0, 0, 0, 0, 0, 1, 1, -1]]

contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08, cont09, cont10, cont11, cont12, cont13]

glm_files = []
[glm_files.append(_) for _ in listdir(glm_dir) if _.endswith(".tsv") and "loc" not in _ and "._" not in _]
glm_files.sort()
subject_info = []

for r in run_list:

    subject_info = []
    tsv_file = glm_files[r - 1]
    print(tsv_file)

    trialinfo = pd.read_table(join(glm_dir, tsv_file))
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
    scans.inputs.template = 'cr_rsub-%s_task-img_run-%02d_bold.nii'
    # scans.inputs.template = 'sm_cr_rsub-%s_task-img_run-%02d_bold.nii'
    scans.inputs.sort_filelist = True
    scans.inputs.subject_id = subject_id
    scans.inputs.run = r

    mc_param = Node(DataGrabber(infields=['subject_id', 'run'], outfields=['param']), name="mc_param")
    mc_param.inputs.base_directory = input_dir
    mc_param.inputs.template = 'rp_sub-%s_task-img_run-%02d_bold.txt'
    mc_param.inputs.sort_filelist = True
    mc_param.inputs.subject_id = subject_id
    mc_param.inputs.run = r

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
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                                     timing_units='secs',
                                     interscan_interval=TR,
                                     model_serial_correlations='FAST',
                                     mask_image=epi_mask,
                                     mask_threshold='-Inf',
                                     ), name='level1design')
    # mask_image=epi_mask,
    # EstimateModel - estimate the parameters of the model
    level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}, write_residuals=True),
                          name='level1estimate')

    # EstimateContrast - estimates contrasts
    # level1conest = Node(EstimateContrast(), name="level1conest")
    # level1conest.inputs.contrasts = contrast_list


    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    wf = Workflow(name='nipype', base_dir=output_dir)

    wf.connect([
        (scans, modelspec, [("func", "functional_runs")]),
        (mc_param, modelspec, [("param", "realignment_parameters")]),
        (modelspec, level1design, [("session_info", "session_info")]),
        (level1design, level1estimate, [("spm_mat_file", "spm_mat_file")]),

    # (level1estimate, level1conest, [("spm_mat_file", "spm_mat_file"),
    #                                 ("beta_images", "beta_images"),
    #                                 ("residual_image", "residual_image")]),
    (level1estimate, datasink, [("spm_mat_file", 'analysis.identity.run-{}'.format('%02d' % r)),
                                    ("beta_images", 'analysis.identity.run-{}.@B'.format('%02d' % r)),
                                    ("residual_images", 'analysis.identity.run-{}.@res'.format('%02d' % r)),
                                    ]),
    # (level1conest, datasink, [("spm_mat_file", "analysis.imagery"),
    #                           ("spmT_images", 'analysis.imagery.run-{}.@T'.format('%02d' % r)),
    #                           ("con_images", 'analysis.imagery.run-{}.@con'.format('%02d' % r)),
    #                           ("spmF_images", 'analysis.imagery.run-{}.@F'.format('%02d' % r)),
    #                           ("ess_images", 'analysis.imagery.run-{}.@ess'.format('%02d' % r)),
    #                           ]),
    ])
    # wf.connect([
    #     (scans, modelspec, [("func", "functional_runs")]),
    #     (mc_param, modelspec, [("param", "realignment_parameters")]),
    #     (modelspec, level1design, [("session_info", "session_info")]),
    #     (level1design, level1estimate, [("spm_mat_file", "spm_mat_file")]),
    #     (level1estimate, datasink, [("spm_mat_file", 'analysis.no_smoothing.no_devein.beta'),
    #                                 ("beta_images", 'analysis.no_smoothing.no_devein.beta.@B'),
    #                                 ]),
    # ])
    wf.run()
