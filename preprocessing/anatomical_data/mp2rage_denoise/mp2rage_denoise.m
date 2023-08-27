function mp2rage_denoise(inputdir, workingdir)

fprintf(inputdir);
fprintf(workingdir);
%load anatomical files

MP2RAGE.filenameINV1 = fullfile(inputdir, 'INV1.nii')
MP2RAGE.filenameINV2 = fullfile(inputdir, 'INV2.nii')
MP2RAGE.filenameUNI = fullfile(inputdir, 'UNI.nii')
MP2RAGE.filenameOUT = fullfile(inputdir, 'dUNI.nii');

%this should be checked interactively in SPM
regularization  = 7;

addpath(genpath(workingdir));
RobustCombination(MP2RAGE,regularization);

end