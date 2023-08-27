from nilearn.masking import apply_mask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

func_filename = "/Users/carricarte/PhD/Others/sub-02/func/sub-02_task-loc_run-01_bold.nii"
mask = "/Users/carricarte/Documents/GitHub/layer_mri/pipeline/analysis/restrictive_mask.nii"
masked_data = apply_mask(func_filename, mask)

avg_masked_data = np.mean(masked_data, axis=1)

avg_masked_data

trialinfo = pd.read_table("/Users/carricarte/PhD/Others/sub-02/glm_localizer.tsv")
onset = trialinfo.iloc[:]['onset']
cond = trialinfo.iloc[:]['trial_type']
onset2TR = onset/3

name = []
for c in cond:
    if c =='Faces':
        name.append('f')
    elif c == 'Objects':
        name.append('o')
    elif c == 'Places':
        name.append('p')
    elif c == 'Baseline':
        name.append('b')
    elif c == 'Scrambled':
        name.append('s')

plt.plot(avg_masked_data)
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.vlines(onset2TR, ymin=avg_masked_data.min(), ymax=avg_masked_data.max(), label=name, linestyles='solid', colors='black', linewidth=0.1)
plt.show()