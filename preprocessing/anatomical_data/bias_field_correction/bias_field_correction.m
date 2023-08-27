function bias_field_correction(inputdir, workingdir, spmdir)

% Pre-processes high resolution high field MRI data intended for
% quantification of cortical (and sub-cortical) structures.
% 
% Run bias field correction of SPM12. Allows to extract WM, GM, and CSF
% segmentation and generation of a brainmask based on these segmentations.
%
% Falk Luesebrink (22.11.2017)
%
% Minor changes
% Falk Lüsebrink (12.02.2019)

param.seg     = false;      % Write segmentation of GM, WM, CSF
param.mask    = false;      % Create brainmask (requires param.seg = true)
param.gzip    = false;      % Write compressed NIfTIs after processing

% Parameters for bias field correction:
%
fwhm = [30];
reg = [0.001];
samp = [2];
% See 'bias_correction.m' for in-depth explantion of parameters.
param.path    = char(fullfile(spmdir, "/tpm/TPM.nii")); % Set path to tissue probability model
% param.fwhm    = fwhm;        % FWHM (default: 60)
% param.reg     = reg;     % Regularization (default: 0.001)
% param.samp    = samp;         % Sampling distance (default: 3)

% Parameters for brainmask
param.dilate  = 10;       % Dilate brainmask by cube with edge length of param.dilate
param.erode   = 10;       % If greater than 0, erode brainmask after dilation

files = dir(fullfile(inputdir, '*.nii'))
data = rmfield(files, ["folder", "date", "bytes", "isdir", "datenum"])

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Bias correction, segmentation, and brain mask creation %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for id=1:length(data)
    data(id).file = [inputdir data(id).name];
   
    
    fprintf('************************************\n')
    fprintf('************************************\n')
    fprintf('* Bias correction and segmentation  \n')
    fprintf('* Processing file: %s\n', data(id).file);
    fprintf('************************************\n\n')
    
    [path, name, ext] = fileparts(data(id).file);

    if strcmp(ext, '.gz')
        gunzip(data(id).file);
        parma.gzip = true;
    end
    
    addpath(genpath(spmdir));
    addpath(genpath(workingdir));
    
    for f=1:length(fwhm)
        
        param.fwhm    = fwhm(f);
        
        for r=1:length(reg)
            param.reg     = reg(r);     % Regularization (default: 0.001)

            for s=1:length(samp)
                param.samp    = samp(s);         % Sampling distance (default: 3)
                bias_correction(data(id).file, param);
                
                data(id).name = [data(id).file(length(inputdir)+1:end-4) '_fwhm-' num2str(param.fwhm) '_reg-' num2str(param.reg) '_samp-' num2str(param.samp) '.nii'];
                fprintf('Output: %s\n\n', data(id).name);
                fprintf('Done.\n')
            end
        end
    end
end
