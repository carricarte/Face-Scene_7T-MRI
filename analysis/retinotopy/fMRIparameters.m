function [params] = fMRIparameters()
% general
params.fMRI.B0                   = 7;        % magnetic field strength (Tesla)

% fMRI
params.fMRI.EPI.num_runs         = [2];        % number of runs per stimulus type
params.fMRI.EPI.TR               = 2;        % repetition time
params.fMRI.EPI.fMRI_name        = {['lh_bar_sess1.mgh'],['lh_bar_sess2.mgh'],...
                                        ['rh_bar_sess1.mgh'],['rh_bar_sess2.mgh']};
params.fMRI.EPI.total_num_runs   = sum(params.fMRI.EPI.num_runs);

% struct
params.fMRI.struct.name          = 'T1_out'; % anatomical image name
params.fMRI.struct.isMT          = 0;        % is the anatomical image an MT map
                                            % if not, it's assumed to be a T1 mprage
end