import scipy.io
import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import join

# input_dir = sys.argv[1]
# output_dir = sys.argv[2]
input_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-01/beh"
output_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-01/beh"

def data2glm(input_file, output_file):

    import os
    path, file = os.path.split(input_file)
    filename, ext = os.path.splitext(file)
    data = scipy.io.loadmat(input_file)['data']['design'][0][0][0][0]

    condition = data['blocks'][0][0:np.array(data['blocks']).size]
    # onset = data['block_onset'] - (data['trigger'][0][0] - 2.077)
    onset = data['block_onset'] - (data['block_onset'][0][0]) + 14.482  # + 14.482 becuase I didnt remove the first
    # three non-staedy state volumes
    onset = onset[0]
    duration = onset[1:onset.size] - onset[0:onset.size - 1]
    duration = np.concatenate((duration[:-1], np.array([14.482, 14.482])), 0)
    # onset = onset[1:onset.size - 1]
    # trigger = data['trigger'][0]
    trial_type = np.zeros(condition.shape)
    weight = np.ones(condition.shape)
    trial_type = np.where(condition != 1, trial_type, 'seen_place')
    trial_type = np.where(condition != 3, trial_type, 'baseline')
    trial_type = np.where(condition != 2, trial_type, 'img_place')

    a = 1

    model = {
        'onset': onset,
        'duration': duration,
        'weight': weight,
        'trial_type': trial_type,
    }
    glm_model = pd.DataFrame(model, columns=['onset', 'duration', 'weight', 'trial_type'])
    glm_model.to_csv(os.path.join(output_file, 'glm_{0}.tsv'.format(filename)), sep='\t', index=True)


behfiles = []
[behfiles.append(b) for b in listdir(input_dir) if "run" in b and "._" not in b and ".tsv" not in b]

for file in behfiles:
    data2glm(join(input_dir, file), output_dir)
