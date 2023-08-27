import scipy.io
import numpy as np
import pandas as pd
import os
import sys
from os import listdir
from os.path import join

# input_dir = sys.argv[1]
input_dir = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/behdata/sub-05"

# output_dir = sys.argv[2]
output_dir = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/behdata/sub-05"


def data2glm(input_file, output_file):
    trial_type = ''
    data_struct = scipy.io.loadmat(input_file)
    data = data_struct['dat']
    condition_order = np.concatenate(data['order_blocks'][0])

    trial_type = np.where(condition_order != 1, trial_type, 'Faces')
    trial_type = np.where(condition_order != 2, trial_type, 'Places')
    trial_type = np.where(condition_order != 3, trial_type, 'Objects')
    trial_type = np.where(condition_order != 4, trial_type, 'Scrambled')
    trial_type = np.where(condition_order != 5, trial_type, 'Baseline')

    condition_onset = data['blockOnset'][0][0]
    condition_dur = data['blockDur'][0][0]
    weight = np.ones(condition_order.shape)

    model = {
        'onset': np.concatenate(condition_onset, axis=0),
        'duration': np.concatenate(condition_dur, axis=0),
        'weight': np.concatenate(weight, axis=0),
        'trial_type': np.concatenate(trial_type, axis=0),
    }
    glm_model = pd.DataFrame(model, columns=['onset', 'duration', 'weight', 'trial_type'])
    glm_model.to_csv(os.path.join(output_file, 'glm_task-loc.tsv'), sep='\t', index=True)

behfiles = []
[behfiles.append(b) for b in listdir(input_dir) if "standard" in b and "._" not in b and ".tsv" not in b]

for file in behfiles:
    data2glm(join(input_dir, file), output_dir)
