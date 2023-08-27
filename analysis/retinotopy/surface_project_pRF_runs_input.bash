#!/bin/bash

SUBJECT=sub-01
project=project-01-LaminarCrowdpRF
FSDIR=/home/mayajas/scratch/${project}/derivatives/wf_advanced_skullstrip/_subject_id_${SUBJECT}/autorecon1
OUTDIR=/home/mayajas/scratch/${project}/pRF/data
REG_MAT=/home/mayajas/scratch/${project}/output/func/coreg/_subject_id_${SUBJECT}/registered_0GenericAffine.mat
MEAN_FUNC=/home/mayajas/scratch/${project}/output/func/meanFunc/_sess_id_task-bar_run-01_sess_nr_0/_subject_id_${SUBJECT}/meanatask-bar_run-01_roi.nii
PRF_BAR1=/home/mayajas/scratch/${project}/output/func/realign/_sess_id_task-bar_run-01_sess_nr_0/_subject_id_${SUBJECT}/ratask-bar_run-01_roi.nii
PRF_BAR2=/home/mayajas/scratch/${project}/output/func/realign/_sess_id_task-bar_run-02_sess_nr_1/_subject_id_${SUBJECT}/ratask-bar_run-02_roi.nii
OCC=/home/mayajas/scratch/${project}/manualcorr/${SUBJECT}/occipital.nii

if [[ ! -d $OUTDIR/${SUBJECT} ]]
then
	mkdir -p $OUTDIR/${SUBJECT}
fi
./surface_project_pRF_runs.bash $SUBJECT $FSDIR $OUTDIR $REG_MAT $MEAN_FUNC $PRF_BAR1 $PRF_BAR2 $OCC

PRF_FSDIR=/home/mayajas/scratch/${project}/pRF/data_FS
if [[ ! -d $PRF_FSDIR/${SUBJECT} ]]
then
	mkdir -p $PRF_FSDIR/${SUBJECT}
	cp -r $FSDIR/${SUBJECT} $PRF_FSDIR
fi
