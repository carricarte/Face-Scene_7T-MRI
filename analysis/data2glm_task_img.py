import scipy.io
import numpy as np
import pandas as pd
import os
import sys
from os import listdir
from os.path import join

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# Event code -> trial_type label ("seen_berin" is the label the existing GLMs use)
TRIAL_TYPES = {1: 'seen_merkel', 2: 'seen_obama', 3: 'seen_berin', 4: 'seen_paris',
               5: 'img_merkel', 6: 'img_obama', 7: 'img_berlin', 8: 'img_paris',
               9: 'rvsp', -1: 'visual_cue', -2: 'auditory_cue', 0: 'baseline'}


def data2glm(input_file, output_file):

    filename, ext = os.path.splitext(os.path.basename(input_file))
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

        ev = events[e]
        prev = events[e - 1] if e > 0 else np.nan  # no event before the first one
        if (ev == 10 or (ev in (1, 2, 3, 4) and prev == 10) or (ev == 0 and prev == 31)
                or (20 <= ev <= 30 and prev >= 20)):
            dindex[e] = 0
        elif ev in (-4, -3, -2, -1):
            condition_order[e] = -1
        elif ev in (-5, -6, -7, -8):
            condition_order[e] = -2

    condition_order = condition_order[dindex == 1]
    condition_onset = condition_onset[dindex == 1]
    condition_order = np.where(condition_order != 31, condition_order, 0)
    condition_order = np.where(condition_order < 20, condition_order, 9)

    condition_order = condition_order[0:len(condition_order) - 1]
    condition_onset = condition_onset[0:len(condition_onset) - 1]

    temp = np.append(condition_onset[1:], [experiment_off])
    condition_dur = temp - condition_onset

    trial_type = np.vectorize(lambda c: TRIAL_TYPES.get(c, ''), otypes=[str])(condition_order)

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
