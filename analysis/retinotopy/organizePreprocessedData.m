function [databasedir] = organizePreprocessedData(databasedir, fsdir, params)

disp('Now organizing the preprocessed data...')
savedir = [fsdir filesep 'fMRI'];
if ~exist(fullfile(savedir))
    mkdir(fullfile(savedir))
end

% pRF mapping functional runs
hems = 1:2;
for stim = 1:length(params.pRF.stimulus_names)
    stimulus_name = params.pRF.stimulus_names{stim};
    for hem = hems
        if hem == 1
            hem_txt = 'lh';
        elseif hem == 2
            hem_txt = 'rh';
        end
        for sess = 1:params.fMRI.EPI.num_runs(stim)
            pRF_file   = fullfile(databasedir,...
                [hem_txt '_' params.pRF.stimulus_names{stim} '_sess' num2str(sess) '.mgh']);
            
            % make dir if doesn't yet exist
            if ~exist(fullfile(savedir,[stimulus_name]))
                mkdir(fullfile(savedir,[stimulus_name]))
            end

            % copy functional files 
            if ~exist(fullfile(savedir,[stimulus_name],[hem_txt '_surf_sess' num2str(sess) '.mgh']),'file')
                copyfile(pRF_file,fullfile(savedir,[stimulus_name],[hem_txt '_surf_sess' num2str(sess) '.mgh']))
            end
        end
    end
end