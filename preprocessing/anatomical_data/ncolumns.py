from os.path import join
import numpy as np
import sys
from nilearn.image import load_img, new_img_like

sub = ['01', '02', '03', '06', '07', '08', '09', "10", "11", "12", "13", "14", "15", '16', '17', "18"]

rh_column = []
lh_column = []
for s in sub:
    lh_column_file = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/derivatives/sub-{}/anat/lh_rim_columns20000.nii.gz".format(s)
    rh_column_file = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/derivatives/sub-{}/anat/rh_rim_columns20000.nii.gz".format(s)
    rh_column.append(int(len(np.unique(load_img(rh_column_file).get_fdata())) - 1))
    lh_column.append(int(len(np.unique(load_img(lh_column_file).get_fdata())) - 1))

total_columns = np.array(rh_column) + np.array(lh_column)
print(np.mean(total_columns))
print(np.std(total_columns))
print(np.max(total_columns))
print(np.min(total_columns))
