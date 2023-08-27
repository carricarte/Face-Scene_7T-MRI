from os.path import join, split, splitext
import nibabel as nb
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
import image_stats

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mask_dir = sys.argv[3]
layer_dir = sys.argv[4]

# input_dir = "/Users/carricarte/scratch/projects/imagery/pilot/bold/pilot_07/derivatives/sub-11/mask"
# output_dir = "/Users/carricarte/scratch/projects/imagery/pilot/bold/pilot_07/derivatives/sub-11/analysis"

right_FFA = join(mask_dir, "mask_right_FFA.nii")
right_PPA = join(mask_dir, "mask_right_PPA.nii")
left_FFA = join(mask_dir, "mask_left_FFA.nii")
left_PPA = join(mask_dir, "mask_left_PPA.nii")

mask = [left_FFA, right_FFA, left_PPA, right_PPA]

mean_tsnr = []
for m in mask:
    mean_tsnr.append(np.mean(image_stats.img_mask(input_dir, m, layer_dir)))

dist = pd.DataFrame(columns=["left_FFA", "right_FFA", "left_PPA", "right_PPA"])
dist.loc[len(dist)] = mean_tsnr
dist.to_csv(join(output_dir, 'tsnr.tsv'), sep='\t', index=True)
print("Hello World")
