function pRFmapping_curta(sub,ncpus)
p = parpool('local',ncpus);
project_name    = 'project-01-LaminarCrowdpRF';

% set paths
scripts_path    = ['/home/mayaaj90/scripts/' project_name '/'];
data_path       = ['/scratch/mayaaj90/' project_name '/pRF/data/'];
FS_path         = ['/scratch/mayaaj90/' project_name '/pRF/data_FS/'];
programs_path   = '/home/mayaaj90/programs';
anat_path       = ['/scratch/mayaaj90/' project_name '/output/func/anat/_subject_id_'];

% add scripts path
addpath(scripts_path)

% add paths to required software
addpath(genpath(fullfile(programs_path,'vistasoft')))
addpath(genpath(fullfile(programs_path,'spm12')))
rmpath(genpath(fullfile(programs_path,'spm12','external','fieldtrip')))
addpath(genpath(fullfile(programs_path,'samsrf')))

% iterables
cd ..
% get fMRI and pRF parameters
params = fMRIparameters();
params = pRFparameters(params);                                      


% get the folder contents
thisSubject = ['sub-0' num2str(sub)];
databasedir = [data_path thisSubject];
fsdir       = [FS_path thisSubject];

% copy T1_out.nii to fsdir
if ~exist([fsdir filesep 'mri' filesep 'orig' filesep params.fMRI.struct.name '.nii'],'file')
    copyfile([anat_path thisSubject filesep params.fMRI.struct.name '.nii'],[fsdir filesep 'mri' filesep 'orig' filesep])
end

% run prf mapping
runPRFmapping(databasedir,fsdir,scripts_path,params);

end                                            
                                            