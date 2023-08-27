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

    import os
    path, file = os.path.split(input_file)
    filename, ext = os.path.splitext(file)
    trial_type = ''
    data_struct = scipy.io.loadmat(input_file)
    events = data_struct['data'][0][0][2][0][0][0][0]
    upd_screen_time = data_struct['data'][0][0][2][0][0][23][0]
    # upd_screen_time = data_struct['data'][0][0][2][0][0][24][0]
    experiment_on = data_struct['data'][0][0][2][0][0][25][0]
    # experiment_on = data_struct['data'][0][0][2][0][0][26][0]
    # experiment_off = data_struct['data'][0][0][2][0][0][27][0]
    experiment_off = data_struct['data'][0][0][2][0][0][26][0]
    # upd_screen_time = data_struct['data'][0][0][2][0][0][15][0]
    # experiment_on = data_struct['data'][0][0][2][0][0][12][0]
    # experiment_off = data_struct['data'][0][0][2][0][0][17][0]
    upd_screen_time = upd_screen_time - experiment_on

    events = events[1:]
    upd_screen_time = upd_screen_time[1:]
    condition_order = events
    condition_onset = upd_screen_time

    dindex = np.ones(shape=events.shape)

    for e in range(0, len(events)):

        if events[e] == 10 or (events[e] == 1 and events[e - 1] == 10) or (events[e] == 2 and events[e - 1] == 10) or (events[e] == 3 and events[e - 1] == 10) or (events[e] == 4 and events[e - 1] == 10) or (events[e] == 0 and events[e - 1] == 31) or ((events[e] >= 20 and events[e] <= 30) and events[e - 1] >= 20):
            dindex[e] = 0
        elif events[e] == -4 or events[e] == -3 or events[e] == -2 or events[e] == -1:
            condition_order[e] = -1
        elif events[e] == -5 or events[e] == -6 or events[e] == -7 or events[e] == -8:
            condition_order[e] = -2

    condition_order = condition_order[dindex == 1]
    condition_onset = condition_onset[dindex == 1]
    condition_order = np.where(condition_order != 31, condition_order, 0)
    condition_order = np.where(condition_order < 20, condition_order, 9)

    condition_order = condition_order[0:len(condition_order) - 1]
    condition_onset = condition_onset[0:len(condition_onset) - 1]

    temp = np.append(condition_onset[1:], [experiment_off])
    condition_dur = temp - condition_onset

    trial_type = np.where(condition_order != 1, trial_type, 'seen_merkel')
    trial_type = np.where(condition_order != 2, trial_type, 'seen_obama')
    trial_type = np.where(condition_order != 3, trial_type, 'seen_berin')
    trial_type = np.where(condition_order != 4, trial_type, 'seen_paris')
    trial_type = np.where(condition_order != 5, trial_type, 'img_merkel')
    trial_type = np.where(condition_order != 6, trial_type, 'img_obama')
    trial_type = np.where(condition_order != 7, trial_type, 'img_berlin')
    trial_type = np.where(condition_order != 8, trial_type, 'img_paris')
    trial_type = np.where(condition_order != 9, trial_type, 'rvsp')
    trial_type = np.where(condition_order != -1, trial_type, 'visual_cue')
    trial_type = np.where(condition_order != -2, trial_type, 'auditory_cue')
    trial_type = np.where(condition_order != 0, trial_type, 'baseline')

    weight = np.ones(condition_order.shape)

    model = {
        'onset': condition_onset,
        'duration': condition_dur,
        'weight': weight,
        'trial_type': trial_type,
    }
    glm_model = pd.DataFrame(model, columns=['onset', 'duration', 'weight', 'trial_type'])
    glm_model.to_csv(os.path.join(output_file, 'glm_{0}.tsv'.format(filename)), sep='\t', index=True)


behfiles = []
[behfiles.append(b) for b in listdir(input_dir) if "run" in b and "._" not in b and ".tsv" not in b]

for file in behfiles:
    data2glm(join(input_dir, file), output_dir)
