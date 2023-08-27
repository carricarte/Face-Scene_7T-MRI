#!/bin/bash

# e.g., ./surface_project_pRF_runs.sh sub-01 /usr/local/freesurfer/subjects /home/mayajas/Documents/project-00-7t-pipeline-dev/data/pRF/data /home/mayajas/Documents/project-00-7t-pipeline-dev/data/output/func/coreg/_subject_id_sub-01/registered_0GenericAffine.mat /home/mayajas/Documents/project-00-7t-pipeline-dev/data/output/func/sliceTimeCorr/_subject_id_sub-01/_sess_id_task-bar_run-01_sess_nr_0_sess_nvol_124/atask-bar_run-01_roi_warp4D.nii /home/mayajas/Documents/project-00-7t-pipeline-dev/data/output/func/sliceTimeCorr/_subject_id_sub-01/_sess_id_task-bar_run-02_sess_nr_1_sess_nvol_124/atask-bar_run-02_roi_warp4D.nii /home/mayajas/scratch/project-00-7t-pipeline-dev/manualcorr/sub-01/occipital.nii

SUBJECT=$1
FSDIR=$2
OUTDIR=$3
REG_MAT=$4
MEAN_FUNC=$5
PRF_BAR1=$5
PRF_BAR2=$6
OCC=$7
#OCC=${12}

export SUBJECTS_DIR=$FSDIR
ANAT=$FSDIR/$SUBJECT/mri/T1.mgz
cd $OUTDIR/$SUBJECT


echo "Occipital file: ${OCC}"
echo "Global1 file: ${PRF_GLOBAL1}"
echo "Global2 file: ${PRF_GLOBAL2}"

# Convert ANTS/ITK transform to LTA (FreeSurfer):
# First convert the ANTS binary mat file to ITK text file format and then to lta (adding src and trg geometry info, from images that were used to create the transform in ANTS):
$ANTSPATH/ConvertTransformFile 3 $REG_MAT $OUTDIR/$SUBJECT/registered_0GenericAffine.txt
lta_convert --initk $OUTDIR/$SUBJECT/registered_0GenericAffine.txt --outlta $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --src $MEAN_FUNC --trg $ANAT --invert --subject $SUBJECT

# Coregister functional runs to structural space (volume)
# mean functional
mri_vol2vol --mov $MEAN_FUNC --lta $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --o $OUTDIR/$SUBJECT/reg_meanFunc.nii --trilin --fstarg --no-resample
# bar sess 1
mri_vol2vol --mov $PRF_BAR1 --lta $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --o $OUTDIR/$SUBJECT/reg_prf_bar1.nii --trilin --fstarg --no-resample
# bar sess 2
mri_vol2vol --mov $PRF_BAR2 --lta $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --o $OUTDIR/$SUBJECT/reg_prf_bar2.nii --trilin --fstarg --no-resample
# occipital mask
mri_vol2vol --mov $OCC --lta $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --o $OUTDIR/$SUBJECT/reg_occ.nii --trilin --fstarg --no-resample


# Coregistration and surface sampling 
for hemis in lh rh
do
	method=trilinear
	# bar sess 1
	mri_vol2surf --mov $PRF_BAR1 --srcreg $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --hemi ${hemis} --cortex \
	--out $OUTDIR/$SUBJECT/${hemis}_bar_sess1.mgh --interp ${method} --projfrac 0.8
	# bar sess 2
	mri_vol2surf --mov $PRF_BAR2 --srcreg $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --hemi ${hemis} --cortex \
	--out $OUTDIR/$SUBJECT/${hemis}_bar_sess2.mgh --interp ${method} --projfrac 0.8

	method=nearest
	# occipital label
	mri_vol2surf --mov $OCC --srcreg $OUTDIR/$SUBJECT/registered_0GenericAffine.lta --hemi ${hemis} --cortex \
	--out $OUTDIR/$SUBJECT/${hemis}_occ.mgh --interp ${method} --projfrac 0.8
	mri_vol2label --i $OUTDIR/$SUBJECT/${hemis}_occ.mgh --id 1  --surf ${SUBJECT} ${hemis}  \
	--l $OUTDIR/$SUBJECT/${hemis}_occ.label
done
