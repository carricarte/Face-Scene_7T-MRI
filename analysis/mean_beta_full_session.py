import sys
import numpy as np
from os import listdir
from os.path import join, dirname
from nilearn.image import mean_img, math_img, new_img_like

input_dir = sys.argv[1]
subject_id = sys.argv[2]
output_dir = dirname(input_dir)

if "01" == subject_id:
    n_runs = 9
elif "02" == subject_id:
    n_runs = 10
elif "03" == subject_id:
    n_runs = 10
elif "04" == subject_id:
    n_runs = 10
elif "05" == subject_id:
    n_runs = 10
elif "06" == subject_id:
    n_runs = 10
elif "07" == subject_id:
    n_runs = 12
elif "08" == subject_id:
    n_runs = 10
elif "09" == subject_id:
    n_runs = 10
elif "10" == subject_id:
    n_runs = 10
elif "11" == subject_id:
    n_runs = 10
elif "12" == subject_id:
    n_runs = 10
elif "13" == subject_id:
    n_runs = 10
elif "14" == subject_id:
    n_runs = 10
elif "15" == subject_id:
    n_runs = 10
elif "16" == subject_id:
    n_runs = 10
elif "17" == subject_id:
    n_runs = 12
elif "18" == subject_id:
    n_runs = 10

betas = []
[betas.append(join(input_dir, b)) for b in listdir(input_dir) if b.endswith(".nii") and "._" not in b]
betas.sort()
betas = np.array(betas)
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place']

img_face_index = np.append(np.arange(4, len(betas) - n_runs, 18), np.arange(5, len(betas) - n_runs, 18))
img_place_index = np.append(np.arange(3, len(betas) - n_runs, 18), np.arange(6, len(betas) - n_runs, 18))
seen_face_index = np.append(np.arange(9, len(betas) - n_runs, 18), np.arange(10, len(betas) - n_runs, 18))
seen_place_index = np.append(np.arange(8, len(betas) - n_runs, 18), np.arange(11, len(betas) - n_runs, 18))

ind_constant = np.arange(len(betas) - n_runs, len(betas))

betas_img_face = mean_img(list(betas[img_face_index]))
betas_img_place = mean_img(list(betas[img_place_index]))
betas_seen_face = mean_img(list(betas[seen_face_index]))
betas_seen_place = mean_img(list(betas[seen_place_index]))
mean_constant = mean_img(list(betas[ind_constant]))

for a, mean_beta in enumerate([betas_img_face, betas_img_place, betas_seen_face, betas_seen_place]):
    mean_beta.to_filename(join(output_dir, "mean_{}.nii".format(conditions[a])))
    psc_seen_place = math_img("(img1/img2)*100", img1=mean_beta, img2=mean_constant)
    psc_seen_place.to_filename(join(output_dir, "psc_{}.nii".format(conditions[a])))
