from os.path import join
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/carricarte/layer_mri/pipeline/analysis/')
import image_stats


input_dir = sys.argv[1]
output_dir = sys.argv[2]
layer_file = sys.argv[3
]
rois = ["mask_left_FFA.nii", "mask_right_FFA.nii", "mask_left_PPA.nii", "mask_right_PPA.nii"]
mask_layer_list = image_stats.get_layer(layer_file)

voxels = []
for r in rois:
    for layer_mask in mask_layer_list:
        mask = np.logical_and(image_stats.load_nifti(join(input_dir, r)), layer_mask)
        voxels.append(sum(sum(sum(mask))))

columns = ["deep_left_FFA", "middle_left_FFA", "superficial_left_FFA", "deep_right_FFA", "middle_right_FFA",
           "superficial_right_FFA", "deep_left_PPA", "middle_left_PPA", "superficial_left_PPA", "deep_right_PPA",
           "middle_right_PPA", "superficial_right_PPA"]
vx = pd.DataFrame(columns=columns)
vx.loc[len(vx)] = voxels

vx.to_csv(join(output_dir, 'voxels.tsv'), sep='\t', index=True)
print("done")
