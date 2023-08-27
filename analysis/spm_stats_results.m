
subjects = ["01", "02", "03", "04", "05", "06", "07"];
contrasts = [5];

for s=1:length(subjects)
    
    cd("/Users/carricarte/scratch/projects/imagery/pilot_07/derivatives/sub-" + subjects(s) + "/analysis/localizer")
    
    for c=1:length(contrasts)
        
        spm('defaults', 'FMRI');
        spm_jobman('initcfg');
        matlabbatch{1}.spm.stats.results.spmmat = {'SPM.mat'};
        matlabbatch{1}.spm.stats.results.conspec.titlestr = '';
        matlabbatch{1}.spm.stats.results.conspec.contrasts = contrasts(c);
        matlabbatch{1}.spm.stats.results.conspec.threshdesc = 'none';
        matlabbatch{1}.spm.stats.results.conspec.thresh = 0.05;
        matlabbatch{1}.spm.stats.results.conspec.extent = 10;
        matlabbatch{1}.spm.stats.results.conspec.conjunction = 1;
        matlabbatch{1}.spm.stats.results.conspec.mask.none = 1;
        matlabbatch{1}.spm.stats.results.units = 1;
        matlabbatch{1}.spm.stats.results.export{1}.ps = true;
        matlabbatch{1}.spm.stats.results.export{2}.nary.basename = '_nary_map';
        
        spm_jobman('run', matlabbatch);
    end
end