import scipy.io
import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import join

# input_dir = sys.argv[1]
# output_dir = sys.argv[2]
sub = ["09"]
for s in sub:
    print("--------------------------")
    input_dir = "/Users/carricarte/scratch/projects/imagery/main/vaso/behdata/sub-{0}".format(s)
    output_dir = "/Users/carricarte/scratch/projects/imagery/main/vaso/behdata/sub-{0}".format(s)

    # input_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-01/beh"
    # output_dir = "/Users/carricarte/PhD/Debugging/vaso/main/sub-01/beh"


    def data2glm(input_file, output_file):

        import os
        path, file = os.path.split(input_file)
        filename, ext = os.path.splitext(file)
        data = scipy.io.loadmat(input_file)['data']['design'][0][0][0][0]

        # condition = data['events'][0][0:np.array(data['events']).size]
        condition = data['blocks'][0][0:np.array(data['blocks']).size]
        # onset = data['upd_screen_time'][0] - (data['trigger'][0][0])
        onset = data['block_onset'][0] - (data['block_onset'][0][0])
        # onset = data['block_onset'][0] - (data['block_onset'][0][0] - 2.077)  # for sub-08
        duration = onset[1:onset.size] - onset[0:onset.size - 1]
        # duration = np.concatenate((duration[:-1], np.array([4.832, 14.482])), 0)
        # duration = np.concatenate((duration[:-1], np.array([14.482, 14.482])), 0)
        duration = np.concatenate((duration, np.array([14.482])), 0)

        # patch
        onset[-1] = onset[-2] + duration[-2]
        # duration = np.concatenate((duration, np.array([14.482])), 0)
        trial_type = np.zeros(condition.shape)
        weight = np.ones(condition.shape)
        # trial_type = np.where(condition != 1, trial_type, 'seen_berlin')
        # trial_type = np.where(condition != 3, trial_type, 'seen_paris')
        # trial_type = np.where(condition != 2, trial_type, 'seen_pisa')
        # trial_type = np.where(condition != -1, trial_type, 'img_berlin')
        # trial_type = np.where(condition != -2, trial_type, 'img_paris')
        # trial_type = np.where(condition != -3, trial_type, 'img_pisa')
        # trial_type = np.where(condition != 0, trial_type, 'baseline')
        trial_type = np.where(condition != 1, trial_type, 'seen_place')
        trial_type = np.where(condition != 2, trial_type, 'img_place')
        trial_type = np.where(condition != 3, trial_type, 'baseline')

        model = {
            'onset': onset,
            'duration': duration,
            'weight': weight,
            'trial_type': trial_type,
        }
        glm_model = pd.DataFrame(model, columns=['onset', 'duration', 'weight', 'trial_type'])
        # glm_model = glm_model.drop(len(glm_model) - 1, 0)  # for sub-08
        glm_model.to_csv(os.path.join(output_file, 'glm_block_{0}.tsv'.format(filename)), sep='\t', index=True)


    behfiles = []
    [behfiles.append(b) for b in listdir(input_dir) if "run" in b and "._" not in b and ".tsv" not in b]

    for file in behfiles:
        data2glm(join(input_dir, file), output_dir)
