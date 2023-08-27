import sys
from os import listdir
from os.path import join
from nilearn.image import math_img

<<<<<<< HEAD
# input_dir = sys.argv[1]
input_dir = "/Users/carricarte/scratch/projects/imagery/main/bold/pilot_07/derivatives/sub-01/analysis/no_smoothing/beta"
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place', 'constant']

betas = []
[betas.append(join(input_dir, beta_vol)) for beta_vol in listdir(input_dir) if "CBV" not in beta_vol and
 "mean_beta" in beta_vol and "constant" not in beta_vol and "Linear" not in beta_vol and "Deconv" not in beta_vol and
 beta_vol.endswith(".nii") and "lh" not in beta_vol and "rh" not in beta_vol and "._" not in beta_vol and "lambda"
 not in beta_vol and "manual" not in beta_vol]
=======
input_dir = sys.argv[1]
conditions = ['img_face', 'img_place', 'seen_face', 'seen_place', 'constant']

betas = []
[betas.append(join(input_dir, _)) for _ in listdir(input_dir) if "lh" not in _ and "rh" not in _ and "._" not in _
 and "deveinDeconv" in _ and "psc" not in _ and "CBV" not in _]
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
betas.sort()

constant = join(input_dir, "mean_beta_constant.nii")

print('calculating % signal change...')
for a, b in enumerate(betas):
    psc = math_img("(img1/img2)*100", img1=b, img2=constant)
<<<<<<< HEAD
    psc.to_filename(join(input_dir, "manual_psc_{}_check.nii".format(conditions[a])))
=======
    psc.to_filename(join(input_dir, "manual_psc_{}_deveinDeconv.nii".format(conditions[a])))
>>>>>>> e9c38f8738dd8dbe89fa92894a87eb1d36cd8302
print('done')
