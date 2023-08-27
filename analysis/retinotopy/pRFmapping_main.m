function pRFmapping_main()
dbstop if error

project_name    = 'project-01-LaminarCrowdpRF';

scripts_path    = ['/home/mayajas/Documents/' project_name '/scripts/'];
data_path       = ['/home/mayajas/scratch/' project_name '/pRF/data/'];
FS_path         = ['/home/mayajas/scratch/' project_name '/pRF/data_FS/'];
programs_path   = '/home/mayajas/Documents/programs';
anat_path       = ['/home/mayajas/scratch/' project_name '/output/func/anat/_subject_id_'];

% add scripts path
addpath(scripts_path)

% add paths to required software
addpath(genpath(fullfile(programs_path,'vistasoft')))
addpath(genpath(fullfile(programs_path,'spm12')))
rmpath(genpath(fullfile(programs_path,'spm12','external','fieldtrip')))
addpath(genpath(fullfile(programs_path,'samsrf')))

% iterables
% get fMRI and pRF parameters
params = fMRIparameters();
params = pRFparameters(params);  

for sub = 1
    thisSubject = ['sub-0' num2str(sub)];
    databasedir = [data_path thisSubject];
    fsdir       = [FS_path thisSubject];

    % copy T1_out.nii to fsdir
    if ~exist([fsdir filesep 'mri' filesep 'orig' filesep params.fMRI.struct.name '.nii'],'file')
        copyfile([anat_path thisSubject filesep params.fMRI.struct.name '.nii'],[fsdir filesep 'mri' filesep 'orig' filesep])
    end

    runPRFmapping(databasedir,fsdir,scripts_path,params);
end
                                            